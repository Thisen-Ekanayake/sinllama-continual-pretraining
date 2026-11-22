#!/usr/bin/env python3
"""QLoRA instruct-tuning of SinLlama_wiki_uc_gen_mix on the bilingual mix (build_instruct_mix.py), lab box.

  python train_instruct.py --out runs/r64                       # full run (resumes from runs/r64/ckpt)
  python train_instruct.py --out runs/smoke_r64 --smoke         # memory stress (longest rows) + throughput sample

Data: mix/train.parquet + mix/val.parquet (messages, source, lang). Each dialogue is rendered with the model's
chat template; loss on every assistant turn (content + <|end_of_text|>), nothing else. Each assistant answer is
tokenized on its own, after its prompt, exactly as the model continues a prompt at generation time.

Recipe (see instruct_sft/README.md): LoRA on all 7 projections, nf4 double-quant base, bf16 compute,
lr cosine with 3% warmup, 1 epoch, effective batch = bsz x accum, eval + checkpoint every --eval-steps with the
overall and per-source validation loss, best checkpoint by overall val loss, early stopping patience 4 as a snapshot, not a stop: the first time 4 evals
in a row fail to beat the best, the best adapter so far is copied to <out>/early_stop (+ READY) and training goes on
to the full epoch. At the end <out>/best = best over the epoch, <out>/final = last-step weights.
Logged to Weights & Biases (key from ~/.netrc); the run id is kept in <out>/wandb_run_id so a restart after a
crash resumes the same W&B run.
"""
import argparse, collections, glob, json, os, time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

H = os.path.expanduser("~")
MODEL = f"{H}/sinllama-continual-pretraining/models/SinLlama_wiki_uc_gen_mix"
MIX = f"{H}/sinhala_instruct_datasets/mix"
MAX_LEN = 2048
LOSS_CHUNK = 1024
_tok = None


def _init(model_dir):
    global _tok
    _tok = AutoTokenizer.from_pretrained(model_dir)


def encode(msgs):
    """-> (input_ids, labels) int32 arrays. Prompt segments are labelled -100."""
    tok = _tok
    full = tok.apply_chat_template(msgs, tokenize=False)
    ids, labels, pos = [], [], 0
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(msgs[:k], tokenize=False, add_generation_prompt=True)
        assert full.startswith(prefix) and full[len(prefix):].startswith(m["content"]), "template not prefix-stable"
        p = tok(prefix[pos:], add_special_tokens=False).input_ids
        a = tok(m["content"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids += p + a
        labels += [-100] * len(p) + a
        pos = len(prefix) + len(m["content"])
        if full.startswith(tok.eos_token, pos):
            pos += len(tok.eos_token)
    tail = tok(full[pos:], add_special_tokens=False).input_ids   # whitespace after the last answer: no loss
    ids += tail
    labels += [-100] * len(tail)
    return np.array(ids[:MAX_LEN], np.int32), np.array(labels[:MAX_LEN], np.int32)


def load_split(path, model_dir, cache, sources):
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    t = pq.read_table(path, columns=["source", "messages"]).to_pylist()
    msgs = [[{"role": x["role"], "content": x["content"]} for x in r["messages"]] for r in t]
    with Pool(16, initializer=_init, initargs=(model_dir,)) as pool:
        enc = pool.map(encode, msgs, chunksize=256)
    rows = [{"input_ids": a, "labels": b, "src": sources.index(r["source"])} for (a, b), r in zip(enc, t)]
    torch.save(rows, cache)
    return rows


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(pad):
    def f(feats):
        n = max(len(x["input_ids"]) for x in feats)
        ids = torch.full((len(feats), n), pad, dtype=torch.long)
        lab = torch.full((len(feats), n), -100, dtype=torch.long)
        att = torch.zeros((len(feats), n), dtype=torch.long)
        for i, x in enumerate(feats):
            L = len(x["input_ids"])
            ids[i, :L] = torch.from_numpy(x["input_ids"].astype(np.int64))
            lab[i, :L] = torch.from_numpy(x["labels"].astype(np.int64))
            att[i, :L] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab,
                "src": torch.tensor([x["src"] for x in feats])}
    return f


class TokenBudgetBatches(torch.utils.data.Sampler):
    """Micro-batches of up to `budget` padded tokens (rows x longest row): ~8 short rows or 2 rows of 2048.
    QLoRA dequantizes every weight on each forward/backward, so a micro-batch of 2 short rows wastes most of a step.
    Rows are shuffled, cut into pools of 2048, sorted by length inside a pool, packed greedily, and the batches are
    shuffled. Deterministic in (seed, epoch), so a resumed run replays the same order. shuffle=False keeps the row
    order (smoke: longest rows first)."""

    def __init__(self, lengths, budget, seed=42, shuffle=True, pool=2048):
        self.lengths, self.budget, self.seed, self.shuffle, self.pool = np.asarray(lengths), budget, seed, shuffle, pool
        self.epoch = 0
        self.batches = self._make()

    def _make(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        idx = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for p in range(0, len(idx), self.pool):
            chunk = idx[p:p + self.pool]
            if self.shuffle:
                chunk = chunk[np.argsort(-self.lengths[chunk], kind="stable")]
            cur, mx = [], 0
            for i in chunk:
                L = int(self.lengths[i])
                if cur and max(mx, L) * (len(cur) + 1) > self.budget:
                    batches.append(cur)
                    cur, mx = [], 0
                cur.append(int(i))
                mx = max(mx, L)
            if cur:
                batches.append(cur)
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch = epoch
            self.batches = self._make()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _chunk_ce(lm_head, h, y):
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum")


class MixTrainer(Trainer):
    """Answer-only causal-LM loss (lm_head only at labelled positions, checkpointed chunks), per-source eval."""

    sources = []
    tokens_seen = 0
    sequential = False   # smoke: keep the row order (longest rows first)
    token_budget = 4096
    _t_last, _n_last = None, 0

    def get_train_dataloader(self):
        lengths = [len(r["input_ids"]) for r in self.train_dataset.rows]
        self.budget_sampler = TokenBudgetBatches(lengths, self.token_budget, seed=self.args.seed,
                                                 shuffle=not self.sequential)
        dl = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.budget_sampler,
                                         collate_fn=self.data_collator, num_workers=self.args.dataloader_num_workers,
                                         pin_memory=True)
        return self.accelerator.prepare(dl)

    def log(self, logs, *a, **kw):
        """Adds training throughput (non-pad tokens/s since the last training log) and peak GPU memory."""
        if "loss" in logs:
            now = time.time()
            if self._t_last is not None:
                logs["tokens_per_sec"] = round((self.tokens_seen - self._n_last) / (now - self._t_last), 1)
            logs["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            logs["tokens_seen"] = self.tokens_seen
            self._t_last, self._n_last = now, self.tokens_seen
        return super().log(logs, *a, **kw)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("src", None)
        labels = inputs.pop("labels")
        self.tokens_seen += int(inputs["attention_mask"].sum())
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

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dl = self.get_eval_dataloader(eval_dataset)
        model = self.model
        was_training = model.training
        model.eval()
        base = model.get_base_model()
        s_loss, s_n = collections.defaultdict(float), collections.defaultdict(int)
        t0 = time.time()
        for batch in dl:
            batch = self._prepare_inputs(batch)
            src, labels = batch.pop("src"), batch.pop("labels")
            h = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            tgt = labels[:, 1:]
            keep = tgt != -100
            row = keep.nonzero()[:, 0]
            hk, yk = h[:, :-1][keep], tgt[keep]
            per = torch.cat([F.cross_entropy(base.lm_head(hk[i:i + LOSS_CHUNK]).float(), yk[i:i + LOSS_CHUNK],
                                             reduction="none") for i in range(0, len(yk), LOSS_CHUNK)])
            rl = torch.zeros(len(src), device=per.device).index_add_(0, row, per)
            rn = torch.bincount(row, minlength=len(src))
            for s, l, n in zip(src.tolist(), rl.tolist(), rn.tolist()):
                s_loss[s] += l
                s_n[s] += n
        lang = lambda s: "en" if self.sources[s].endswith("_en") else "si"  # noqa: E731
        m = {f"{metric_key_prefix}_loss": sum(s_loss.values()) / sum(s_n.values())}
        for g in ("si", "en"):
            ss = [s for s in s_loss if lang(s) == g]
            if ss:
                m[f"{metric_key_prefix}_{g}_loss"] = sum(s_loss[s] for s in ss) / sum(s_n[s] for s in ss)
        for s in sorted(s_loss):
            m[f"{metric_key_prefix}_{self.sources[s]}_loss"] = s_loss[s] / s_n[s]
        m[f"{metric_key_prefix}_runtime"] = round(time.time() - t0, 1)
        self.log(m)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, m)
        if was_training:
            model.train()
        return m


class WandbExtras(TrainerCallback):
    def __init__(self, extra):
        self.extra = extra

    def on_train_begin(self, args, state, control, **kw):
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(self.extra, allow_val_change=True)
        except Exception as e:  # logging must never kill training
            print("wandb extras:", e)


class EarlyStopSnapshot(TrainerCallback):
    """Early stopping that records instead of stopping. on_evaluate runs before the Trainer updates best_metric, so
    state.best_metric / best_model_checkpoint still hold the previous best (that checkpoint is on disk, and checkpoint
    rotation never deletes it). The patience counter is rebuilt from log_history, so it survives a resume."""
    ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

    def __init__(self, out, patience=4):
        self.dir, self.patience, self.bad = f"{out}/early_stop", patience, 0

    def on_train_begin(self, args, state, control, **kw):
        best = None
        for x in state.log_history:
            if "eval_loss" in x:
                if best is None or x["eval_loss"] < best:
                    best, self.bad = x["eval_loss"], 0
                else:
                    self.bad += 1
        print(f"early-stop snapshot: patience counter {self.bad}/{self.patience}, "
              f"snapshot {'taken' if os.path.exists(self.dir + '/READY') else 'not taken'}", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        cur = (metrics or {}).get("eval_loss")
        if cur is None:
            return
        if state.best_metric is None or cur < state.best_metric:
            self.bad = 0
            return
        self.bad += 1
        print(f"early-stop snapshot: eval_loss {cur:.4f} >= best {state.best_metric:.4f}, "
              f"{self.bad}/{self.patience}", flush=True)
        if self.bad < self.patience or os.path.exists(self.dir + "/READY") or not args.should_save:
            return
        import shutil
        src = state.best_model_checkpoint
        os.makedirs(self.dir, exist_ok=True)
        for f in self.ADAPTER_FILES:
            shutil.copy2(f"{src}/{f}", f"{self.dir}/{f}")
        json.dump({"early_stop_at_step": state.global_step, "epoch": state.epoch,
                   "best_checkpoint": src, "best_step": int(src.rsplit("-", 1)[1]),
                   "best_eval_loss": state.best_metric, "patience": self.patience,
                   "eval_history": [x for x in state.log_history if "eval_loss" in x] + [{"step": state.global_step, **metrics}]},
                  open(f"{self.dir}/early_stop.json", "w"), indent=1)
        open(self.dir + "/READY", "w").close()
        print(f"EARLY STOP HIT at step {state.global_step}: best {src} ({state.best_metric:.4f}) -> {self.dir}; "
              f"continuing to the end of the epoch", flush=True)
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.summary.update({"early_stop_step": state.global_step,
                                          "early_stop_best_step": int(src.rsplit("-", 1)[1])})
        except Exception as e:
            print("wandb early-stop:", e)


class ResumeCheckpoints(TrainerCallback):
    """Extra checkpoints every `every` steps between evals, so a crash loses at most that many steps. The Trainer
    requires save_steps to be a multiple of eval_steps when load_best_model_at_end is set, so these are requested
    via control.should_save instead. Rotation (save_total_limit=2) always keeps the best checkpoint plus the newest."""

    def __init__(self, every):
        self.every = every

    def on_step_end(self, args, state, control, **kw):
        if self.every and state.global_step % self.every == 0:
            control.should_save = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mix", default=MIX)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--token-budget", type=int, default=4096, help="max padded tokens per micro-batch")
    ap.add_argument("--bsz", type=int, default=4, help="eval batch size only (training uses --token-budget)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--ckpt-steps", type=int, default=100, help="extra resume checkpoints between evals (0 = off)")
    ap.add_argument("--wandb-project", default="sinllama-instruct-mix")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=24)
    args = ap.parse_args()
    set_seed(42)
    os.makedirs(args.out, exist_ok=True)

    stats = json.load(open(f"{args.mix}/stats.json"))
    sources = sorted(stats["summary"])
    MixTrainer.sources = sources
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    train_rows = load_split(f"{args.mix}/train.parquet", args.model, f"{args.mix}/train.tok.pt", sources)
    val_rows = load_split(f"{args.mix}/val.parquet", args.model, f"{args.mix}/val.tok.pt", sources)
    lens = np.array([len(r["input_ids"]) for r in train_rows])
    n_lab = sum(int((r["labels"] != -100).sum()) for r in train_rows)
    print(f"tokenized in {(time.time() - t0) / 60:.1f} min: train {len(train_rows)} rows, {lens.sum():,} tokens "
          f"({n_lab:,} answer tokens), len p50 {int(np.median(lens))} p99 {int(np.percentile(lens, 99))} "
          f"max {lens.max()}; val {len(val_rows)}", flush=True)
    ex = train_rows[0]
    print("--- example (answer tokens in [[ ]]) ---")
    print("".join(tok.decode(ex["input_ids"][i:i + 1]) if ex["labels"][i] == -100 else
                  "[[" + tok.decode(ex["input_ids"][i:i + 1]) + "]]" for i in range(min(len(ex["input_ids"]), 400))))
    print("---", flush=True)

    if args.smoke:   # 1/3 of the steps on the longest rows (memory), the rest on random rows (throughput)
        rng = np.random.default_rng(0)
        n_long = (args.smoke_steps // 3) * args.accum * 2          # 2 rows of ~2048 per micro-batch
        longest = sorted(train_rows, key=lambda r: -len(r["input_ids"]))[:n_long]
        n_rand = (args.smoke_steps - args.smoke_steps // 3) * args.accum * 8   # ~ all consumed in 16 steps
        rand = [train_rows[i] for i in rng.choice(len(train_rows), n_rand, replace=False)]
        rand.sort(key=lambda r: -len(r["input_ids"]))                # pack like the real sampler does
        train_rows = longest + rand
        per = collections.defaultdict(list)
        for r in val_rows:
            per[r["src"]].append(r)
        val_rows = [r for rs in per.values() for r in rs[:16]]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    run_name = args.run_name or os.path.basename(os.path.normpath(args.out))
    idf = f"{args.out}/wandb_run_id"
    if not os.path.exists(idf):
        import wandb
        open(idf, "w").write(wandb.util.generate_id())
    os.environ.update(WANDB_PROJECT=args.wandb_project, WANDB_RUN_ID=open(idf).read().strip(),
                      WANDB_RESUME="allow", WANDB_LOG_MODEL="false", WANDB_WATCH="false")

    targs = TrainingArguments(
        output_dir=f"{args.out}/ckpt", run_name=run_name, report_to=["wandb"],
        num_train_epochs=args.epochs, max_steps=args.smoke_steps if args.smoke else -1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0, max_grad_norm=1.0,
        per_device_train_batch_size=1, per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum, gradient_checkpointing=True, bf16=True,
        seed=42, logging_steps=1 if args.smoke else 10, logging_first_step=True,
        eval_strategy="steps", eval_steps=args.smoke_steps // 2 if args.smoke else args.eval_steps,
        save_strategy="steps", save_steps=args.smoke_steps // 2 if args.smoke else args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, dataloader_num_workers=2, remove_unused_columns=False)
    trainer = MixTrainer(model=model, args=targs, train_dataset=Rows(train_rows), eval_dataset=Rows(val_rows),
                         data_collator=collate(tok.pad_token_id),
                         callbacks=[EarlyStopSnapshot(args.out, patience=4),
                                    ResumeCheckpoints(0 if args.smoke else args.ckpt_steps)])
    trainer.sequential = args.smoke
    trainer.token_budget = args.token_budget
    trainer.add_callback(WandbExtras({"lora_rank": args.rank, "lora_alpha": args.alpha, "mix": stats["total"],
                                      "mix_sources": stats["summary"], "smoke": args.smoke}))
    resume = bool(glob.glob(f"{args.out}/ckpt/checkpoint-*")) and not args.smoke
    print(f"resume from checkpoint: {resume}", flush=True)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    mins = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"train {mins:.1f} min, best {trainer.state.best_model_checkpoint} eval_loss {trainer.state.best_metric}, "
          f"peak {peak:.2f} GiB", flush=True)
    if args.smoke:
        steps = [x for x in trainer.state.log_history if "tokens_per_sec" in x]
        rand_steps = [x for x in steps if x["step"] > args.smoke_steps // 3 + 1 and x["step"] != args.smoke_steps // 2 + 1]
        tps = float(np.median([x["tokens_per_sec"] for x in rand_steps])) if rand_steps else float("nan")
        full_tokens = int(lens.sum())
        print(f"SMOKE: peak {peak:.2f} GiB; random-row throughput {tps:.0f} tok/s -> 1 epoch of "
              f"{full_tokens / 1e6:.1f}M tokens ~ {full_tokens / tps / 3600:.1f} h (+ evals)", flush=True)
    trainer.save_model(f"{args.out}/best")
    json.dump({"base_model": os.path.basename(os.path.normpath(args.model)), "args": vars(args),
               "n_train": len(train_rows), "n_val": len(val_rows),
               "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
               "global_step": trainer.state.global_step, "epochs_done": trainer.state.epoch,
               "train_minutes": round(mins, 1), "peak_gib": round(peak, 2),
               "log_history": trainer.state.log_history},
              open(f"{args.out}/train_state.json", "w"), ensure_ascii=False, indent=1)
    last = f"{args.out}/ckpt/checkpoint-{trainer.state.global_step}"   # the Trainer always saves the last step
    try:
        import shutil
        os.makedirs(f"{args.out}/final", exist_ok=True)
        for f in EarlyStopSnapshot.ADAPTER_FILES:
            shutil.copy2(f"{last}/{f}", f"{args.out}/final/{f}")
    except Exception as e:
        print(f"WARNING: could not copy last-step adapter {last} -> <out>/final: {e}", flush=True)


if __name__ == "__main__":
    main()
