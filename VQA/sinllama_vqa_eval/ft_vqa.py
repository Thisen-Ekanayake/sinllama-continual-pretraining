#!/usr/bin/env python3
"""QLoRA fine-tune SinLlama_wiki_uc_gen_mix on the COCO VQA set, one adapter per language.

  python ft_vqa.py split                      # -> ft/split.json, ft/test_ids.json
  python ft_vqa.py train --lang en            # -> ft/en/{best,train_state.json}
  python ft_vqa.py train --lang si            # -> ft/si/...
  add --smoke for a 4-step run on the 32 longest rows (memory check, -> ft_smoke/<lang>)

Split: 80/10/10 by image_id (seed 42) -> 4000 train / 500 val / 500 test, the SAME image_ids
for English and Sinhala, so the two test sets are paired. Train rows whose answer repeats a
sentence (in either language; Qwen generation loops, carried into Sinhala by NLLB) are then
dropped from train only -> 3994; val and test are untouched.

Hyperparameters are textbook_sft/config.yaml's: LoRA r8 / alpha 16 / dropout 0.05 on all 7
projections, lr 2e-4 cosine, warmup 3%, weight decay 0, max grad norm 1.0, batch 4 x accum 16
= 64, 3-epoch cap, early stopping patience 3 on val loss (best checkpoint kept), max length
2048, loss on the assistant answer only. Differences, forced by the 16 GB lab box or the data:
  * QLoRA: base in nf4 (double quant, bf16 compute) instead of bf16;
  * eval/save every 10 steps instead of 50 (4000 rows = ~63 steps/epoch, so 50 would give 3
    evals and early stopping could never fire);
  * no liger kernel (not installed), no wandb (log history -> train_state.json).
Prompt = the same SFT chat template and wording gen_sinllama.py uses at test time.
"""
import argparse, glob, json, os, random, re, time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from gen_sinllama import PROMPTS, user_turn

DATA = {"en": "../vqa_english.json", "si": "../vqa_sinhala.json"}
MODEL = os.path.expanduser("~/sinllama-continual-pretraining/models/SinLlama_wiki_uc_gen_mix")
MAX_LEN = 2048
LOSS_CHUNK = 1024


def looping(text):
    """True if a sentence of 20+ characters occurs twice in the text."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 20]
    return len(sents) != len(set(sents))


def make_split(out_dir, seed=42):
    en = json.load(open(DATA["en"], encoding="utf-8"))
    si = json.load(open(DATA["si"], encoding="utf-8"))
    ids = sorted(r["image_id"] for r in en)
    assert ids == sorted(r["image_id"] for r in si), "en/si image_ids differ"
    random.Random(seed).shuffle(ids)
    n_tr, n_va = int(len(ids) * .8), int(len(ids) * .1)
    split = {"seed": seed, "train": sorted(ids[:n_tr]), "val": sorted(ids[n_tr:n_tr + n_va]),
             "test": sorted(ids[n_tr + n_va:])}
    loops = {r["image_id"] for d in (en, si) for r in d if looping(r["answer"])}
    split["train_dropped_looping_answer"] = sorted(loops & set(split["train"]))
    split["train"] = sorted(set(split["train"]) - loops)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(split, open(f"{out_dir}/split.json", "w"), indent=0)
    json.dump(split["test"], open(f"{out_dir}/test_ids.json", "w"))
    print({k: len(v) for k, v in split.items() if k != "seed"}, split["train_dropped_looping_answer"])


def build(tok, lang, rows):
    """input = chat-template prompt (incl. BOS) + answer + <|end_of_text|>; labels on the answer only.
    The answer is tokenized on its own, exactly as the model continues the prompt at test time."""
    out = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": user_turn(lang, r)}],
                                    tokenize=False, add_generation_prompt=True)
        p = tok(p, add_special_tokens=False).input_ids
        a = tok(r["answer"].strip(), add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = (p + a)[:MAX_LEN]
        out.append({"input_ids": ids, "labels": ([-100] * len(p) + a)[:MAX_LEN]})
    return out


def collate(pad):
    def f(feats):
        n = max(len(x["input_ids"]) for x in feats)
        return {
            "input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in feats]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"]))
                                            for x in feats]),
            "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in feats]),
        }
    return f


def _chunk_ce(lm_head, h, y):
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum")


class LabelOnlyLossTrainer(Trainer):
    """Causal-LM loss with lm_head applied only at answer positions, in checkpointed chunks
    (as m2ds_ft.py): the same loss value without full 139k-vocab fp32 logits for the prompt."""

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


def train(lang, model_dir, out_dir, smoke):
    set_seed(42)
    split = json.load(open("ft/split.json"))
    by_id = {r["image_id"]: r for r in json.load(open(DATA[lang], encoding="utf-8"))}
    train_rows = [by_id[i] for i in split["train"]]
    val_rows = [by_id[i] for i in split["val"]]
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds, val_ds = build(tok, lang, train_rows), build(tok, lang, val_rows)
    if smoke:   # longest rows first: the smoke run doubles as a memory stress test
        train_ds = sorted(train_ds, key=lambda x: -len(x["input_ids"]))[:32]
        val_ds = sorted(val_ds, key=lambda x: -len(x["input_ids"]))[:8]
    lens = sorted(len(x["input_ids"]) for x in train_ds)
    print(f"{lang}: train={len(train_ds)} val={len(val_ds)}; train len p50 {lens[len(lens) // 2]} "
          f"p99 {lens[int(len(lens) * .99)]} max {lens[-1]}", flush=True)
    print("--- example ---\n" + tok.decode(train_ds[0]["input_ids"]) + "\n---", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=f"{out_dir}/ckpt", overwrite_output_dir=True,
        num_train_epochs=3, max_steps=4 if smoke else -1,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        max_grad_norm=1.0, per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=2 if smoke else 16, prediction_loss_only=True,
        gradient_checkpointing=True, bf16=True, group_by_length=True, seed=42,
        logging_steps=1 if smoke else 10, logging_first_step=True,
        eval_strategy="steps", eval_steps=2 if smoke else 10,
        save_strategy="steps", save_steps=2 if smoke else 10, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], dataloader_num_workers=4)
    trainer = LabelOnlyLossTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                                   data_collator=collate(tok.pad_token_id),
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=3,
                                                                    early_stopping_threshold=0.0)])
    resume = bool(glob.glob(f"{out_dir}/ckpt/checkpoint-*")) and not smoke   # crash / reboot recovery
    print(f"resume from checkpoint: {resume}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    mins = (time.time() - t0) / 60
    print(f"train {mins:.1f} min, best {trainer.state.best_model_checkpoint} eval_loss "
          f"{trainer.state.best_metric:.4f}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    trainer.save_model(f"{out_dir}/best")
    json.dump({"lang": lang, "base_model": os.path.basename(os.path.normpath(model_dir)),
               "n_train": len(train_ds), "n_val": len(val_ds),
               "best_checkpoint": trainer.state.best_model_checkpoint,
               "best_eval_loss": trainer.state.best_metric, "global_step": trainer.state.global_step,
               "epochs_done": trainer.state.epoch, "train_minutes": round(mins, 1),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "prompt_format": PROMPTS[lang], "log_history": trainer.state.log_history},
              open(f"{out_dir}/train_state.json", "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "train"])
    ap.add_argument("--lang", choices=["en", "si"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.cmd == "split":
        make_split("ft")
        return
    if not args.lang:
        ap.error("--lang is required for train")
    train(args.lang, args.model, f"{'ft_smoke' if args.smoke else 'ft'}/{args.lang}", args.smoke)


if __name__ == "__main__":
    main()
