"""QLoRA fine-tune a model on XL-Sum v2.0 (Sinhala or English) train, pick the
best checkpoint by val loss, then generate summaries for the test articles.

Training mirrors benchmark/downstream (nf4 4-bit + double quant, bf16 compute,
LoRA r16/a16x2/0.05 on all 7 projections, lr 2e-4 cosine, warmup 5%, eff.
batch 16, loss on the answer only, best by eval_loss). Lengths and decoding
mirror /ml/xl-sum/seq2seq/evaluate.sh: article <= 512 tokens, summary <= 84
tokens, beams 4, length_penalty 0.6. XL-Sum also uses no_repeat_ngram_size 2,
but for a decoder-only model HF applies it over prompt + output, which forbids
copying any token bigram from the article (in mT5 it only sees the decoder
output); so it is off by default here (NO_REPEAT=0).
The loss only projects summary positions through lm_head (same value as the
model's own loss, but no full-sequence 128k-vocab logits), so llama-3-8b's
~6k-token Sinhala sequences fit in 16 GB.

Env overrides (defaults = the original Sinhala SinLlama runs):
  XL_LANG=sinhala|english   DATA=dir with <lang>_{train,val,test}.jsonl
  MAX_SOURCE/MAX_TARGET     token caps on article/summary (0 = none, data pre-cut)
  MAX_NEW, NO_REPEAT        generation length cap, no_repeat_ngram_size (default 0)
  TRAIN_BS, ACCUM, EVAL_BS, GEN_BS
  ADAPTER=dir               skip training, generate with this saved adapter
  TRAIN_ONLY=1              train + save the best adapter, no generation
  PREDS=name                predictions file name (default test_preds.jsonl)
Usage: python xlsum_ft.py MODEL_DIR OUT_DIR      (SMOKE=1: 10 steps on the longest articles)
"""
import gc, json, os, sys, time
import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

env = os.environ.get
LANG = env("XL_LANG", "sinhala")
DATA = os.path.expanduser(env("DATA", f"~/xlsum_eval/{LANG}_XLSum_v2.0"))
PROMPTS = {
    "sinhala": ("පහත පුවත් ලිපිය එක් වාක්‍යයකින් සාරාංශ කරන්න.\n\nලිපිය: ", "\nසාරාංශය:"),
    "english": ("Summarize the following news article in one sentence.\n\nArticle: ", "\nSummary:"),
}
PREFIX, SUFFIX = PROMPTS[LANG]
MAX_SOURCE, MAX_TARGET = int(env("MAX_SOURCE", 512)), int(env("MAX_TARGET", 84))
MAX_NEW, NO_REPEAT = int(env("MAX_NEW", 84)), int(env("NO_REPEAT", 0))
TRAIN_BS, ACCUM = int(env("TRAIN_BS", 2)), int(env("ACCUM", 8))
EVAL_BS, GEN_BS = int(env("EVAL_BS", 4)), int(env("GEN_BS", 2))
ADAPTER, PREDS = env("ADAPTER"), env("PREDS", "test_preds.jsonl")
TRAIN_ONLY = env("TRAIN_ONLY") == "1"      # stop after saving the best adapter
SMOKE = env("SMOKE") == "1"


def read(split):
    return [json.loads(l) for l in open(f"{DATA}/{LANG}_{split}.jsonl")]


def cap(ids, n):
    return ids[:n] if n > 0 else ids


def prompt_ids(tok, text):
    art = cap(tok(text, add_special_tokens=False).input_ids, MAX_SOURCE)
    return ([tok.bos_token_id] + tok(PREFIX, add_special_tokens=False).input_ids + art
            + tok(SUFFIX, add_special_tokens=False).input_ids)


def build(tok, rows):
    out = []
    for r in rows:
        p = prompt_ids(tok, r["text"])
        a = cap(tok(" " + r["summary"], add_special_tokens=False).input_ids, MAX_TARGET) + [tok.eos_token_id]
        out.append({"input_ids": p + a, "labels": [-100] * len(p) + a})
    return out


def collate(pad):
    def f(feats):
        n = max(len(x["input_ids"]) for x in feats)
        return {
            "input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in feats]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in feats]),
            "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in feats]),
        }
    return f


class LabelOnlyLossTrainer(Trainer):
    """Causal-LM loss with lm_head applied only at positions that predict a label."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.get_base_model()
        h = base.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        tgt = labels[:, 1:]
        keep = tgt != -100
        logits = base.lm_head(h[:, :-1][keep]).float()
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss = F.cross_entropy(logits, tgt[keep], reduction="sum") / num_items_in_batch
        else:
            loss = F.cross_entropy(logits, tgt[keep])
        return (loss, {"loss": loss}) if return_outputs else loss


def load_base(model_dir):
    return AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))


def train(model_dir, out_dir, tok, test_n):
    model = load_base(model_dir)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train_rows, val_rows = read("train"), read("val")
    if SMOKE:
        # longest articles first: the smoke run doubles as a memory stress test
        longest = lambda rows, n: sorted(rows, key=lambda r: -len(r["text"]))[:n]
        train_rows, val_rows = longest(train_rows, 64), longest(val_rows, 16)
    train_ds, val_ds = build(tok, train_rows), build(tok, val_rows)
    lens = sorted(len(x["input_ids"]) for x in train_ds)
    print(f"{LANG}: train={len(train_ds)} val={len(val_ds)} test={test_n}; "
          f"train len p50 {lens[len(lens) // 2]} max {lens[-1]}", flush=True)

    args = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", overwrite_output_dir=True,
        num_train_epochs=3, max_steps=10 if SMOKE else -1,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0,
        per_device_train_batch_size=TRAIN_BS, gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=EVAL_BS, prediction_loss_only=True,
        gradient_checkpointing=True, bf16=True,
        logging_steps=10, logging_first_step=True,
        eval_strategy="steps", eval_steps=5 if SMOKE else 50,
        save_strategy="steps", save_steps=5 if SMOKE else 50, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], seed=42, group_by_length=True, dataloader_num_workers=2)
    trainer = LabelOnlyLossTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                                   data_collator=collate(tok.pad_token_id),
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    t0 = time.time()
    trainer.train()
    print(f"train {((time.time() - t0) / 60):.1f} min, best {trainer.state.best_model_checkpoint} "
          f"eval_loss {trainer.state.best_metric:.4f}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    trainer.save_model(f"{out_dir}/best")
    json.dump({"best_checkpoint": trainer.state.best_model_checkpoint,
               "best_eval_loss": trainer.state.best_metric,
               "log_history": trainer.state.log_history},
              open(f"{out_dir}/train_state.json", "w"), indent=1)
    # best adapter is loaded by load_best_model_at_end; free optimizer state before beam search
    trainer.optimizer = trainer.lr_scheduler = None
    del trainer
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    return model


def main(model_dir, out_dir):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test = sorted(read("test"), key=lambda r: -len(r["text"]))[:8] if SMOKE else read("test")
    if ADAPTER:
        model = PeftModel.from_pretrained(load_base(model_dir), ADAPTER)
        print(f"generate-only with adapter {ADAPTER}", flush=True)
    else:
        model = train(model_dir, out_dir, tok, len(test))
        if TRAIN_ONLY:
            return
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    print(f"decoding: beams 4, length_penalty 0.6, no_repeat_ngram_size {NO_REPEAT}, max_new_tokens {MAX_NEW}", flush=True)
    t0 = time.time()
    with open(f"{out_dir}/{PREDS}", "w") as f:
        for i in range(0, len(test), GEN_BS):
            batch = test[i:i + GEN_BS]
            ids = [prompt_ids(tok, r["text"]) for r in batch]
            n = max(map(len, ids))
            enc = {"input_ids": torch.tensor([[tok.pad_token_id] * (n - len(x)) + x for x in ids]).cuda(),
                   "attention_mask": torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids]).cuda()}
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, num_beams=4, length_penalty=0.6,
                                     no_repeat_ngram_size=NO_REPEAT, early_stopping=True, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            for r, g in zip(batch, gen[:, n:]):
                raw = tok.decode(g, skip_special_tokens=True)
                new = int((g != tok.pad_token_id).sum())
                f.write(json.dumps({
                    "id": r["id"], "prediction": raw.strip().split("\n")[0].strip(),
                    "reference": r["summary"], "raw": raw,
                    "new_tokens": new, "hit_cap": new >= MAX_NEW,
                }, ensure_ascii=False) + "\n")
            f.flush()
            k = i + len(batch)
            if k % 20 == 0 or k == len(test):
                print(f"[{k}/{len(test)}] {(time.time() - t0) / 60:.1f} min | "
                      f"{tok.decode(gen[0, n:], skip_special_tokens=True)[:80]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
