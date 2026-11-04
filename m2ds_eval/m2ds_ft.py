"""QLoRA fine-tune a model on M2DS (multi-document summarisation, Sinhala or
English), pick the best checkpoint by val loss, then generate summaries for the
test clusters.

Same recipe as ~/xlsum_eval/xlsum_ft.py (nf4 4-bit + double quant, bf16
compute, LoRA r16/a32/0.05 on all 7 projections, lr 2e-4 cosine, warmup 5%,
eff. batch 16, loss on the summary only, eval every EVAL_STEPS, early stop
patience 3, best by eval_loss), adapted to multi-document input:
  * a cluster's articles are split on the M2DS separator "|||||" and given to
    the model as "Article 1: ...\n\nArticle 2: ..." under one instruction;
  * MAX_SOURCE is the token budget for all article text of a cluster, shared
    max-min fairly (short articles kept whole, the rest split equally among
    the longer ones, each cut at its end), so no article is ever dropped --
    the PRIMERA-style per-document truncation;
  * prompt + summary never exceed CTX tokens (llama-3's 8192): the summary
    target is cut and max_new_tokens shrunk to fit.
Decoding: beams 4, length_penalty 1.0, no n-gram blocking (on a decoder-only
LM it would also forbid copying from the articles).

Env overrides:
  M2_LANG=sinhala|english   DATA=dir with <lang>/{train,validation,test}.json
  MAX_SOURCE, MAX_TARGET, MAX_NEW, CTX, TRAIN_CTX (training-only length cap)
  EPOCHS, MAX_STEPS, EVAL_STEPS, VAL_N (val subsample for eval_loss, 0 = all)
  TRAIN_BS, ACCUM, EVAL_BS, GEN_BS, GEN_TOK
  ADAPTER=dir   skip training, generate with this saved adapter
  TRAIN_ONLY=1  train + save the best adapter, no generation
  PREDS=name    predictions file name (default test_preds.jsonl)
Usage: python m2ds_ft.py MODEL_DIR OUT_DIR   (SMOKE=1: 6 steps on the longest clusters)
"""
import gc, glob, json, os, random, sys, time
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

env = os.environ.get
LANG = env("M2_LANG", "sinhala")
DATA = os.path.expanduser(env("DATA", "~/m2ds"))
PROMPTS = {
    "sinhala": ("පහත දැක්වෙන එකිනෙකට සම්බන්ධ පුවත් ලිපි කිහිපය වාක්‍ය කිහිපයකින් සාරාංශ කරන්න.",
                "ලිපිය", "සාරාංශය:"),
    "english": ("Summarize the following related news articles in a few sentences.",
                "Article", "Summary:"),
}
INSTR, ART, SUM = PROMPTS[LANG]
MAX_SOURCE, MAX_TARGET = int(env("MAX_SOURCE", 4096)), int(env("MAX_TARGET", 512))
MAX_NEW, CTX = int(env("MAX_NEW", MAX_TARGET)), int(env("CTX", 8192))
TRAIN_CTX = int(env("TRAIN_CTX", CTX))     # training sequence cap (activation memory); decoding uses CTX
EPOCHS, EVAL_STEPS, VAL_N = float(env("EPOCHS", 3)), int(env("EVAL_STEPS", 50)), int(env("VAL_N", 0))
MAX_STEPS = int(env("MAX_STEPS", -1))      # optimizer-step cap (cosine schedule spans it)
TRAIN_BS, ACCUM = int(env("TRAIN_BS", 1)), int(env("ACCUM", 16))
EVAL_BS, GEN_BS = int(env("EVAL_BS", 1)), int(env("GEN_BS", 1))
GEN_TOK = int(env("GEN_TOK", 0))   # >0: batch up to GEN_BS prompts while batch x longest <= GEN_TOK
ADAPTER, PREDS = env("ADAPTER"), env("PREDS", "test_preds.jsonl")
TRAIN_ONLY = env("TRAIN_ONLY") == "1"
SMOKE = env("SMOKE") == "1"
SEP = "|||||"


def read(split):
    rows = [json.loads(l) for l in open(f"{DATA}/{LANG}/{split}.json")]
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


def fair_budget(lens, budget):
    """Max-min fair split of `budget` tokens over documents of length `lens`."""
    alloc, left, todo = [0] * len(lens), budget, sorted(range(len(lens)), key=lambda i: lens[i])
    for k, i in enumerate(todo):
        share = left // (len(todo) - k)
        alloc[i] = min(lens[i], share)
        left -= alloc[i]
    return alloc


def prompt_ids(tok, document):
    docs = [d.strip() for d in document.split(SEP) if d.strip()]
    ids = [tok(d, add_special_tokens=False).input_ids for d in docs]
    alloc = fair_budget([len(x) for x in ids], MAX_SOURCE) if MAX_SOURCE > 0 else [len(x) for x in ids]
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    out = [tok.bos_token_id] + enc(INSTR)
    for k, (x, a) in enumerate(zip(ids, alloc), 1):
        out += enc(f"\n\n{ART} {k}: ") + x[:a]
    return out + enc(f"\n\n{SUM}"), sum(alloc), sum(map(len, ids))


def build(tok, rows):
    out = []
    for r in rows:
        p, _, _ = prompt_ids(tok, r["document"])
        a = tok(" " + r["summary"].strip(), add_special_tokens=False).input_ids
        a = a[:max(1, min(MAX_TARGET, TRAIN_CTX - len(p) - 1))] + [tok.eos_token_id]
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


LOSS_CHUNK = int(env("LOSS_CHUNK", 1024))


def _chunk_ce(lm_head, h, y):
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum")


class LabelOnlyLossTrainer(Trainer):
    """Causal-LM loss with lm_head applied only at positions that predict a label.

    The summary positions go through lm_head in chunks of LOSS_CHUNK, each
    recomputed in the backward pass (checkpointed), so at most one chunk's fp32
    logits exist at a time: llama-3-8b's Sinhala references are up to ~4k tokens,
    and 4k x 128k-vocab fp32 logits (+ grads) alone are ~4 GiB. Same loss value."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        base = model.get_base_model()
        h = base.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        tgt = labels[:, 1:]
        keep = tgt != -100
        hk, yk = h[:, :-1][keep], tgt[keep]
        total = sum(torch.utils.checkpoint.checkpoint(_chunk_ce, base.lm_head, hk[i:i + LOSS_CHUNK],
                                                      yk[i:i + LOSS_CHUNK], use_reentrant=False)
                    for i in range(0, len(yk), LOSS_CHUNK))
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss = total / num_items_in_batch
        else:
            loss = total / len(yk)
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

    train_rows, val_rows = read("train"), read("validation")
    if VAL_N and len(val_rows) > VAL_N:
        val_rows = random.Random(42).sample(val_rows, VAL_N)
    if SMOKE:
        # longest clusters first: the smoke run doubles as a memory stress test
        longest = lambda rows, n: sorted(rows, key=lambda r: -len(r["document"]) - len(r["summary"]))[:n]
        train_rows, val_rows = longest(train_rows, 32), longest(val_rows, 8)
    train_ds, val_ds = build(tok, train_rows), build(tok, val_rows)
    lens = sorted(len(x["input_ids"]) for x in train_ds)
    print(f"{LANG}: train={len(train_ds)} val={len(val_ds)} test={test_n}; "
          f"train len p50 {lens[len(lens) // 2]} p90 {lens[int(len(lens) * .9)]} max {lens[-1]}", flush=True)

    args = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", overwrite_output_dir=True,
        num_train_epochs=EPOCHS, max_steps=6 if SMOKE else MAX_STEPS,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0,
        per_device_train_batch_size=TRAIN_BS, gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=EVAL_BS, prediction_loss_only=True,
        gradient_checkpointing=True, bf16=True,
        logging_steps=1 if SMOKE else 10, logging_first_step=True,
        eval_strategy="steps", eval_steps=3 if SMOKE else EVAL_STEPS,
        save_strategy="steps", save_steps=3 if SMOKE else EVAL_STEPS, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], seed=42, group_by_length=True, dataloader_num_workers=2)
    trainer = LabelOnlyLossTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                                   data_collator=collate(tok.pad_token_id),
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    t0 = time.time()
    resume = bool(glob.glob(f"{out_dir}/ckpt/checkpoint-*")) and not SMOKE   # crash / reboot recovery
    print(f"resume from checkpoint: {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    print(f"train {((time.time() - t0) / 60):.1f} min, best {trainer.state.best_model_checkpoint} "
          f"eval_loss {trainer.state.best_metric:.4f}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    trainer.save_model(f"{out_dir}/best")
    json.dump({"best_checkpoint": trainer.state.best_model_checkpoint,
               "best_eval_loss": trainer.state.best_metric,
               "train_minutes": round((time.time() - t0) / 60, 1),
               "log_history": trainer.state.log_history},
              open(f"{out_dir}/train_state.json", "w"), indent=1)
    trainer.optimizer = trainer.lr_scheduler = None
    del trainer
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_disable()
    return model


def _gen(model, tok, ids, mask, max_new, beams):
    with torch.no_grad():
        return model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new,
                              num_beams=beams, length_penalty=1.0, early_stopping=beams > 1, do_sample=False,
                              pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)


def generate(model, tok, ids, mask, max_new, single):
    """Beam 4; on CUDA OOM return None for a batch (caller re-queues its clusters
    one by one); a single cluster falls back to 2 beams, then greedy."""
    for beams in ((4, 2, 1) if single else (4,)):
        try:
            return _gen(model, tok, ids, mask, max_new, beams), beams
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"OOM at {beams} beam(s), batch {ids.shape[0]} x {ids.shape[1]}", flush=True)
    return None, 0


def main(model_dir, out_dir):
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test = read("test")
    if SMOKE:
        test = sorted(test, key=lambda r: -len(r["document"]))[:4]
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

    path = f"{out_dir}/{PREDS}"
    done = set()                                   # resumable; drop a half-written last line
    if os.path.exists(path):
        good = []
        for l in open(path):
            try:
                done.add(json.loads(l)["id"]); good.append(l if l.endswith("\n") else l + "\n")
            except json.JSONDecodeError:
                pass
        open(path, "w").writelines(good)
    enc_test = [(r, *prompt_ids(tok, r["document"])) for r in test if r["id"] not in done]
    enc_test.sort(key=lambda x: -len(x[1]))          # longest first: OOM shows up at once
    print(f"decoding {len(enc_test)} (already {len(done)}): beams 4, length_penalty 1.0, "
          f"no n-gram blocking, max_new_tokens <= {MAX_NEW}, ctx {CTX}", flush=True)
    t0 = time.time()
    with open(path, "a") as f:
        batches, cur = [], []
        for x in enc_test:                       # sorted longest first
            if cur and (len(cur) == GEN_BS or (GEN_TOK and (len(cur) + 1) * len(cur[0][1]) > GEN_TOK)):
                batches.append(cur); cur = []
            cur.append(x)
        if cur:
            batches.append(cur)
        k, skipped = 0, []
        while batches:
            batch = batches.pop(0)
            n = max(len(b[1]) for b in batch)
            max_new = max(16, min(MAX_NEW, CTX - n))
            ids = torch.tensor([[tok.pad_token_id] * (n - len(b[1])) + b[1] for b in batch]).cuda()
            mask = torch.tensor([[0] * (n - len(b[1])) + [1] * len(b[1]) for b in batch]).cuda()
            gen, beams = generate(model, tok, ids, mask, max_new, single=len(batch) == 1)
            if gen is None:
                if len(batch) > 1:
                    batches = [[x] for x in batch] + batches
                else:                # scored as an empty summary (ROUGE 0), flagged by beams 0
                    r, p, used, total = batch[0]
                    skipped.append(r["id"])
                    print(f"SKIP id {r['id']}: OOM even greedy", flush=True)
                    f.write(json.dumps({"id": r["id"], "prediction": "", "reference": r["summary"].strip(),
                                        "raw": "", "prompt_tokens": len(p), "source_tokens_used": used,
                                        "source_tokens_total": total, "new_tokens": 0, "max_new": max_new,
                                        "hit_cap": False, "beams": 0}, ensure_ascii=False) + "\n")
                    f.flush()
                    k += 1
                continue
            for (r, p, used, total), g in zip(batch, gen[:, n:]):
                raw = tok.decode(g, skip_special_tokens=True)
                new = int((g != tok.pad_token_id).sum())
                f.write(json.dumps({
                    "id": r["id"], "prediction": " ".join(raw.split()),
                    "reference": r["summary"].strip(), "raw": raw,
                    "prompt_tokens": len(p), "source_tokens_used": used, "source_tokens_total": total,
                    "new_tokens": new, "max_new": max_new, "hit_cap": new >= max_new, "beams": beams,
                }, ensure_ascii=False) + "\n")
            f.flush()
            k += len(batch)
            if k % 10 < len(batch) or k == len(enc_test):
                print(f"[{k}/{len(enc_test)}] {(time.time() - t0) / 60:.1f} min, prompt {n}, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | "
                      f"{tok.decode(gen[0, n:], skip_special_tokens=True)[:80]!r}", flush=True)
    if skipped:
        print(f"{len(skipped)} clusters could not be decoded: {skipped}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
