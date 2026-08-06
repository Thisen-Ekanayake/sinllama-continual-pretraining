#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for the Global-PIQA (parallel / non-parallel, Sinhala / English)
evaluation scripts in this directory.

Method mirrors mmlu/evaluate_sinhala_mmlu.py and mmlu/evaluate_english_mmlu.py:
scoring = highest next-token probability among the answer-option tokens (digit
1..N for Sinhala, letter A..D for English), from a single forward pass over a
context ending in the answer cue. base/CPT models are scored raw; the
Bactrian-X instruct model is re-wrapped in the Alpaca template it was SFT'd on.

Difference from the MMLU scripts: models are loaded in bf16 by default here
(--quant bf16/4bit/8bit — see load_model()), since this benchmark commonly
runs on VRAM-constrained boxes as well as full-size cloud GPUs. There is no
dev/fewshot split for Global-PIQA, so few-shot exemplars are drawn from the
benchmark itself: pick_fewshot() selects a balanced set (equal count per gold
answer key) and those rows are EXCLUDED from the test set (never scored). Each
eval script defaults to --kshot 8 (few-shot); pass --kshot 0 for a zero-shot run.
"""
import os, re, csv, json, time, gc, random, shutil, subprocess
from collections import defaultdict

# Reduce allocator fragmentation overhead across the sequential model loads in
# a single process (must be set before the first CUDA call).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          PreTrainedTokenizerFast)

try:
    from tqdm import tqdm
except Exception:                                    # pragma: no cover
    def tqdm(x, **k): return x

LETTERS = "ABCD"


def clean(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def pick_fewshot(golds, per_key, seed=0):
    """Choose few-shot exemplar row indices with a BALANCED answer-key mix:
    `per_key` exemplars for each distinct gold label. Selection is deterministic
    (seeded), and the final order is shuffled so the exemplar answer keys are
    interleaved (not grouped A,A,B,B,…) — otherwise the model could pick up a
    positional pattern instead of learning the format. Returns a list of indices;
    the caller EXCLUDES these from the test set so nothing is scored on a question
    the model already saw the answer to."""
    rng = random.Random(seed)
    by_key = defaultdict(list)
    for i, g in enumerate(golds):
        by_key[g].append(i)
    picked = []
    for g in sorted(by_key):
        idxs = by_key[g][:]
        rng.shuffle(idxs)
        picked.extend(idxs[:per_key])
    rng.shuffle(picked)
    return picked


# ----------------------------------------------------------------------------- #
# Prompt templates
# ----------------------------------------------------------------------------- #
# Sinhala: same subject-less structure as mmlu's SinhalaMMLU template (PIQA
# questions have no clean subject field), but the cue markers are English
# ("Question:"/"Answer:", matching the English templates below); block_to_alpaca_si
# splits on those same markers.
TEMPLATE_SI = (
    "Choose the correct or most appropriate answer from these [NUMS] to the following question.\n"
    "Question: [QUESTION]\n"
    "[OPTIONS]\n"
    "Answer:"
)

# English, parallel set (has a real question + N options).
TEMPLATE_EN = (
    "The following is a multiple choice question (with answer).\n"
    "[QUESTION]\n"
    "[OPTIONS]\n"
    "Answer:"
)

# English, non-parallel set: there is no separate translated question — each
# option (eng_translated0/1) is already a full, self-contained statement (the
# Sinhala question + that option's answer translated together). So the model
# is asked to pick the correct full statement rather than answer a question.
TEMPLATE_EN_NONPARALLEL = (
    "The following are candidate statements about everyday life in Sri Lanka; "
    "exactly one is correct.\n"
    "[OPTIONS]\n"
    "Answer:"
)


def render_si(question, choices, answer=None):
    """One Sinhala block. answer=None -> block ends in the answer cue (predict the
    option digit); otherwise the block is a rendered few-shot exemplar ending in
    the gold answer digit (used to build the few-shot prefix)."""
    n = len(choices)
    nums = ", ".join(str(i) for i in range(1, n + 1))
    opts = "\n".join(f"{i + 1}. {clean(c)}" for i, c in enumerate(choices))
    block = (TEMPLATE_SI
             .replace("[NUMS]", nums)
             .replace("[QUESTION]", clean(question))
             .replace("[OPTIONS]", opts))
    if answer is None:
        return block + " "
    return block + " " + str(answer)


def render_en(question, choices, answer_letter=None):
    """One English block (parallel set: real question + options)."""
    opts = "\n".join(f"{LETTERS[i]}. {clean(c)}" for i, c in enumerate(choices))
    block = (TEMPLATE_EN
             .replace("[QUESTION]", clean(question))
             .replace("[OPTIONS]", opts))
    if answer_letter is None:
        return block
    return block + " " + answer_letter


def render_en_nonparallel(choices, answer_letter=None):
    """One English block (non-parallel set: full self-contained statements)."""
    opts = "\n".join(f"{LETTERS[i]}. {clean(c)}" for i, c in enumerate(choices))
    block = TEMPLATE_EN_NONPARALLEL.replace("[OPTIONS]", opts)
    if answer_letter is None:
        return block
    return block + " " + answer_letter


# ----------------------------------------------------------------------------- #
# Alpaca-format wrapping (for SinLlama_Bactrianx_Instruct, SFT'd on this
# template). Adapted from mmlu/evaluate_{sinhala,english}_mmlu.py — the split
# logic depends only on the "Question:"/"Answer:" markers (block_to_alpaca_si)
# and the first-line/"Answer:" markers (block_to_alpaca_en), which every template
# above uses.
# ----------------------------------------------------------------------------- #
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")


def block_to_alpaca_si(block):
    before, cue, after = block.rpartition("Answer:")
    instr, _, qbody = before.partition("\nQuestion:")
    instruction = instr.strip()
    input_text = ("Question:" + qbody).strip()
    response = cue + after
    return ALPACA_PROMPT.format(instruction, input_text, response)


def block_to_alpaca_en(block):
    before, cue, after = block.rpartition("Answer:")
    instruction, _, body = before.partition("\n")
    response = cue + after
    return ALPACA_PROMPT.format(instruction.strip(), body.strip(), response)


def to_alpaca(raw_prompt, block_fn):
    """Re-wrap every block of a (possibly few-shot) prompt. Blocks are joined
    by blank lines and clean() strips internal newlines, so the split is safe."""
    return "\n\n".join(block_fn(b) for b in raw_prompt.split("\n\n"))


# ----------------------------------------------------------------------------- #
# Model loading — bf16 by default (cloud GPU, e.g. 20GB RTX 4000 Ada: an 8B
# model in bf16 is ~16GB resident, comfortably clear of a quantized load's
# accuracy trade-off). 4-bit/8-bit (bitsandbytes) remain available via --quant
# for VRAM-constrained boxes (e.g. an 8GB RTX 4060 laptop).
# Attention: sdpa (faster than eager). CAUTION — on the ROCm/MI300X stack the
# MMLU evals found sdpa mis-handled left-padding masks in batched inference and
# collapsed every prediction to the first option (eager scored correctly); on
# NVIDIA/CUDA sdpa handles the masks fine. After switching stacks, sanity-check
# the first model's pred_label spread in its *_predictions.csv — an all-"A"/
# all-"1" llama-3-8b run means the mask bug is back: revert to eager there.
# ----------------------------------------------------------------------------- #
def load_model(path, quant="bf16"):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for inference.")
    # Squeeze out any cached-but-unused memory from a prior model (in the same
    # process) before loading the next one.
    gc.collect()
    torch.cuda.empty_cache()
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except ValueError as e:
        # SinLlama_Bactrianx_Instruct's tokenizer_config.json declares
        # tokenizer_class="TokenizersBackend" (not a real transformers class,
        # an artifact of how that checkpoint was saved) — same vocab/fast
        # tokenizer.json as the other SinLlama models, so load it directly.
        if "does not exist or is not currently imported" not in str(e):
            raise
        tok = PreTrainedTokenizerFast.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # keep answer cue at the sequence end
    tok.truncation_side = "left"       # drop earliest few-shot exemplars first if over length

    kwargs = dict(device_map={"": 0}, attn_implementation="sdpa")
    if quant == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"unknown quant mode: {quant}")

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model, tok


# ----------------------------------------------------------------------------- #
# Scoring
# ----------------------------------------------------------------------------- #
def digit_token_ids(tok):
    return {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)}


def letter_token_ids(tok):
    # single-token id for each option letter (leading space is part of the token)
    return [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]


@torch.no_grad()
def _score_batch(model, tok, batch, cand_ids, pred_labels, max_len):
    """Score one batch of records in place (sets r['pred'], r['pred_label'],
    r['correct']). cand_ids(r) -> list of candidate token ids for record r;
    pred_labels[i] -> the label ("1".."4" or "A".."D") for candidate index i.
    On CUDA OOM, free memory and recursively split the batch (then shrink
    max_len) until it fits."""
    enc = out = None
    try:
        enc = tok([r["prompt"] for r in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=max_len).to(model.device)
        try:
            out = model(**enc, logits_to_keep=1)
        except TypeError:                               # older transformers
            out = model(**enc)
        logits = out.logits[:, -1, :].float().cpu()     # next-token logits
        for j, r in enumerate(batch):
            cand = cand_ids(r)
            r["pred"] = int(torch.argmax(logits[j, cand]).item())
            r["pred_label"] = pred_labels[r["pred"]]
            r["correct"] = (r["pred"] == r["gold"])
    except torch.cuda.OutOfMemoryError:
        enc = out = None
        torch.cuda.empty_cache()
        if len(batch) == 1:
            if max_len <= 256:
                batch[0]["pred"] = -1
                batch[0]["pred_label"] = "?"
                batch[0]["correct"] = False
            else:
                _score_batch(model, tok, batch, cand_ids, pred_labels, max_len // 2)
        else:
            mid = len(batch) // 2
            _score_batch(model, tok, batch[:mid], cand_ids, pred_labels, max_len)
            _score_batch(model, tok, batch[mid:], cand_ids, pred_labels, max_len)


@torch.no_grad()
def evaluate_digits(model, tok, records, batch_size, max_len):
    digit_id = digit_token_ids(tok)
    labels = [str(d) for d in range(1, 10)]

    def cand_ids(r):
        return [digit_id[d] for d in range(1, r["n_choices"] + 1)]

    todo = [r for r in records if r["valid"]]
    for r in todo:                      # gold label known upfront, independent of scoring
        r["gold_label"] = labels[r["gold"]]
    for i in tqdm(range(0, len(todo), batch_size), desc="  eval"):
        _score_batch(model, tok, todo[i:i + batch_size], cand_ids, labels, max_len)
    return records


@torch.no_grad()
def evaluate_letters(model, tok, records, batch_size, max_len):
    letter_id = letter_token_ids(tok)
    labels = list(LETTERS)

    def cand_ids(r):
        return letter_id[:r["n_choices"]]

    todo = [r for r in records if r["valid"]]
    for r in todo:                      # gold label known upfront, independent of scoring
        r["gold_label"] = labels[r["gold"]]
    for i in tqdm(range(0, len(todo), batch_size), desc="  eval"):
        _score_batch(model, tok, todo[i:i + batch_size], cand_ids, labels, max_len)
    return records


# ----------------------------------------------------------------------------- #
# Aggregation / reporting
# ----------------------------------------------------------------------------- #
def acc_table(records, keyfn):
    agg = defaultdict(lambda: [0, 0])
    for r in records:
        if not r.get("valid"):
            continue
        k = keyfn(r)
        agg[k][1] += 1
        agg[k][0] += int(r["correct"])
    return {k: (c, n, 100.0 * c / n if n else 0.0) for k, (c, n) in agg.items()}


def overall(records):
    ev = [r for r in records if r.get("valid")]
    c = sum(int(r["correct"]) for r in ev)
    return c, len(ev), (100.0 * c / len(ev) if ev else 0.0)


def compute_metrics(records, group_keys):
    """group_keys: dict of {metrics_key: keyfn} extra breakdowns, e.g.
    {'by_category': lambda r: r['category']}."""
    c, n, a = overall(records)
    m = dict(
        overall=dict(correct=c, total=n, accuracy=a),
        skipped=sum(1 for r in records if not r.get("valid")),
    )
    for key, keyfn in group_keys.items():
        m[key] = acc_table(records, keyfn)
    return m


def fmt_rows(table, order=None):
    lines = [f"{'':<38}{'acc%':>8}{'correct':>10}{'total':>8}", "-" * 64]
    keys = order if order else sorted(table, key=lambda k: -table[k][2])
    for k in keys:
        if k not in table:
            continue
        c, n, a = table[k]
        lines.append(f"{str(k):<38}{a:>8.2f}{c:>10}{n:>8}")
    return "\n".join(lines)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_txt(path, title, name, m, sections):
    """sections: list of (heading, metrics_key, order|None)."""
    L = ["=" * 64, f"{title} — {name}", "=" * 64]
    o = m["overall"]
    L.append(f"\nOVERALL ACCURACY : {o['accuracy']:.2f}%  "
             f"({o['correct']}/{o['total']})   skipped={m['skipped']}\n")
    for heading, key, order in sections:
        L.append(f"\n## {heading}")
        L.append(fmt_rows(m[key], order=order))
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


QUANT_LABELS = {"bf16": "bf16", "4bit": "4-bit NF4 (bitsandbytes)",
                "8bit": "8-bit (bitsandbytes)"}


def shot_label(m_or_meta):
    k = m_or_meta.get("kshot", 0)
    return f"{k}-shot" if k else "zero-shot"


def write_md(path, title, name, m, sections):
    o = m["overall"]
    L = [f"# {title} — {name}\n",
         f"**Overall accuracy: {o['accuracy']:.2f}%** "
         f"({o['correct']}/{o['total']}) · {m['skipped']} skipped · "
         f"{shot_label(m)} · {QUANT_LABELS.get(m.get('quant'), m.get('quant', 'bf16'))} · "
         f"{m.get('format', 'raw')} prompt\n"]

    def section(title_s, table, order=None):
        L.append(f"\n## {title_s}\n")
        keys = order or sorted(table, key=lambda k: -table[k][2])
        rows = [[k, f"{table[k][2]:.2f}%", table[k][0], table[k][1]]
                for k in keys if k in table]
        L.append(md_table(["", "Accuracy", "Correct", "Total"], rows))

    for heading, key, order in sections:
        section(heading, m[key], order=order)
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_results_md(path, title, all_metrics, meta, sections):
    """sections: list of (heading, metrics_key) for the combined cross-model report."""
    names = list(all_metrics)
    L = [f"# {title} — Evaluation Results\n"]
    L.append(f"- Benchmark: **{meta['bench']}** ({meta['total']} evaluated MCQs; "
             f"{meta.get('n_shots', 0)} balanced exemplars held out as few-shot)")
    L.append(f"- Setting: **{shot_label(meta)}** (exemplars balanced across answer "
             f"keys, excluded from test), answer chosen by highest option-token "
             f"probability")
    L.append(f"- Precision: **{QUANT_LABELS.get(meta.get('quant'), meta.get('quant', 'bf16'))}** "
             f"on {meta['gpu']}")
    L.append(f"- Prompt format: base/CPT models scored **raw**; the instruct "
             f"model scored in its **Alpaca SFT template** (see the *Format* column)")
    L.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    L.append("## Overall accuracy\n")
    L.append(md_table(["Model", "Accuracy", "Correct", "Total", "Format"],
                      [[n, f"{all_metrics[n]['overall']['accuracy']:.2f}%",
                        all_metrics[n]['overall']['correct'],
                        all_metrics[n]['overall']['total'],
                        all_metrics[n].get('format', 'raw')] for n in names]))

    for heading, key in sections:
        L.append(f"\n## {heading}\n")
        cats = sorted({k for n in names for k in all_metrics[n].get(key, {})})
        rows = []
        for c in cats:
            any_n = next(n for n in names if c in all_metrics[n].get(key, {}))
            tot = all_metrics[any_n][key][c][1]
            rows.append([c, tot] + [f"{all_metrics[n].get(key, {}).get(c, (0, 0, 0))[2]:.2f}%"
                                     for n in names])
        col = key.replace("by_", "").replace("_", " ").capitalize()
        L.append(md_table([col, "N"] + names, rows))

    L.append("\n---\nPer-model detail: see each `<model>_results.md` "
             "(and `*_results.txt` / `*_metrics.json`) in this directory.\n")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def write_predictions_csv(path, records, extra_cols=()):
    """One human-readable row per question: idx, example_id, extra_cols (e.g.
    category / cultural_score), the question text, each option's raw text,
    the gold and predicted answer (label + text), and correctness."""
    max_n = max((r.get("n_choices", 0) for r in records), default=0)
    fieldnames = (["idx", "example_id"] + [h for h, _ in extra_cols] + ["question"]
                  + [f"option_{i + 1}" for i in range(max_n)]
                  + ["gold_label", "gold_text", "pred_label", "pred_text", "correct", "valid"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            choices = r.get("choices", [])
            gold, pred = r.get("gold"), r.get("pred")
            row = {
                "idx": r.get("idx"), "example_id": r.get("example_id", ""),
                "question": r.get("question", ""),
                "gold_label": r.get("gold_label", ""),
                "gold_text": choices[gold] if isinstance(gold, int) and 0 <= gold < len(choices) else "",
                "pred_label": r.get("pred_label", ""),
                "pred_text": choices[pred] if isinstance(pred, int) and 0 <= pred < len(choices) else "",
                "correct": r.get("correct", ""), "valid": r.get("valid", ""),
            }
            for h, fn in extra_cols:
                row[h] = fn(r)
            for i in range(max_n):
                row[f"option_{i + 1}"] = choices[i] if i < len(choices) else ""
            w.writerow(row)


# ----------------------------------------------------------------------------- #
# GCS upload (optional; no-op unless --bucket is passed)
# ----------------------------------------------------------------------------- #
def per_model_files(out_dir, name):
    sfx = ("_metrics.json", "_results.md", "_results.txt", "_predictions.csv")
    return [os.path.join(out_dir, name + s) for s in sfx
            if os.path.exists(os.path.join(out_dir, name + s))]


def gcs_cp(files, dest):
    files = [f for f in files if f]
    if not files:
        return
    if shutil.which("gsutil") is None:
        print("  (gsutil not found — skipping upload)")
        return
    try:
        subprocess.run(["gsutil", "-m", "cp", *files, dest], check=True)
        print(f"  uploaded {len(files)} file(s) -> {dest}")
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: gsutil upload failed -> {dest}: {e}")


def free_model():
    """Call after `del model` (and `del tok`) in the CALLER's own scope.
    `del` only drops a name binding in the frame it executes in — a helper
    that received `model` as a parameter and did `del model` internally would
    only drop its own local reference, leaving the caller's variable (and the
    GPU tensors behind it) alive until the next iteration overwrites it. On a
    multi-model loop that means the next model's weights get loaded while the
    previous model is still fully resident in VRAM."""
    gc.collect()
    torch.cuda.empty_cache()
