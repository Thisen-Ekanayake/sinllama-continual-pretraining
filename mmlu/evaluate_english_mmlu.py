#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SinLlama / Llama models on the English MMLU benchmark (Hendrycks test)
to measure catastrophic forgetting of the base model's English knowledge after
Sinhala continued-pretraining / instruction-tuning.

Mirrors evaluate_sinhala_mmlu.py: same scoring protocol, same per-model output
files (*_results.txt, *_results.md, *_metrics.json, *_predictions.jsonl) and a
combined results.md, same parallel/upload plumbing (--bucket, --combine-only,
--alpaca-models).

Method
------
* Data: HF `cais/mmlu` parquet — all/test-*.parquet (14,042 Qs, 57 subjects),
  few-shot exemplars from all/dev-*.parquet (5 per subject).
* Prompt: per-block template (below), 5-shot by default, exemplars drawn from
  the matching subject's dev rows.
* Scoring: highest probability among the answer options. The context ends in
  "Answer:" and, from a single forward pass, we compare the next-token logits of
  the single-token option letters " A"/" B"/" C"/" D" (leading space is part of
  the token for this tokenizer). argmax -> predicted option.
* Prompt format: base/CPT/llama scored raw; a model named via --alpaca-models
  (the Bactrian-X instruct model) is re-wrapped in the Alpaca template it was
  SFT'd on. Digit/letter scoring is identical either way.
"""
import os, re, json, glob, time, argparse, gc, shutil, subprocess
from collections import defaultdict

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:                                    # pragma: no cover
    def tqdm(x, **k): return x

LETTERS = "ABCD"

# ----------------------------------------------------------------------------- #
# Prompt template (per block, so few-shot exemplars and the test question share
# one structure — this makes the Alpaca re-wrap below a clean block-by-block map).
# ----------------------------------------------------------------------------- #
TEMPLATE = (
    "The following is a multiple choice question (with answer) about [SUBJECT].\n"
    "[QUESTION]\n"
    "[OPTIONS]\n"
    "Answer:"
)

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")

# ----------------------------------------------------------------------------- #
# Standard MMLU subject -> subcategory -> 4 super-categories (STEM / humanities /
# social sciences / other), from the Hendrycks MMLU repo's categories.py.
# ----------------------------------------------------------------------------- #
SUBCATEGORIES = {
    "abstract_algebra": "math", "anatomy": "health", "astronomy": "physics",
    "business_ethics": "business", "clinical_knowledge": "health",
    "college_biology": "biology", "college_chemistry": "chemistry",
    "college_computer_science": "computer science", "college_mathematics": "math",
    "college_medicine": "health", "college_physics": "physics",
    "computer_security": "computer science", "conceptual_physics": "physics",
    "econometrics": "economics", "electrical_engineering": "engineering",
    "elementary_mathematics": "math", "formal_logic": "philosophy",
    "global_facts": "other", "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history", "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics", "high_school_mathematics": "math",
    "high_school_microeconomics": "economics", "high_school_physics": "physics",
    "high_school_psychology": "psychology", "high_school_statistics": "math",
    "high_school_us_history": "history", "high_school_world_history": "history",
    "human_aging": "health", "human_sexuality": "culture",
    "international_law": "law", "jurisprudence": "law",
    "logical_fallacies": "philosophy", "machine_learning": "computer science",
    "management": "business", "marketing": "business",
    "medical_genetics": "health", "miscellaneous": "other",
    "moral_disputes": "philosophy", "moral_scenarios": "philosophy",
    "nutrition": "health", "philosophy": "philosophy", "prehistory": "history",
    "professional_accounting": "other", "professional_law": "law",
    "professional_medicine": "health", "professional_psychology": "psychology",
    "public_relations": "politics", "security_studies": "politics",
    "sociology": "culture", "us_foreign_policy": "politics", "virology": "health",
    "world_religions": "philosophy",
}
CATEGORIES = {
    "stem": ["physics", "chemistry", "biology", "computer science", "math",
             "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social_sciences": ["politics", "culture", "economics", "geography",
                        "psychology"],
    "other": ["other", "business", "health"],
}


def clean(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def subject_pretty(subj):
    return subj.replace("_", " ")


def subject_to_domain(subj):
    sub = SUBCATEGORIES.get(subj, "other")
    for cat, subs in CATEGORIES.items():
        if sub in subs:
            return cat
    return "other"


def render(subject, question, choices, answer_letter=None):
    """Render one block. If answer_letter is given (few-shot exemplar) it is
    appended after 'Answer:' with a leading space; otherwise the block ends in
    'Answer:' and the model's next token is the option letter."""
    opts = "\n".join(f"{LETTERS[i]}. {clean(c)}" for i, c in enumerate(choices))
    block = (TEMPLATE
             .replace("[SUBJECT]", subject)
             .replace("[QUESTION]", clean(question))
             .replace("[OPTIONS]", opts))
    if answer_letter is None:
        return block                            # ends 'Answer:' -> predict letter
    return block + " " + answer_letter          # exemplar 'Answer: A'


# Canonical Hendrycks MMLU format: a single header per subject, then bare Q/A
# blocks (no per-block instruction). Gives leaderboard-comparable numbers.
CANON_HEADER = "The following are multiple choice questions (with answers) about {}.\n\n"


def render_canonical(question, choices, answer_letter=None):
    opts = "\n".join(f"{LETTERS[i]}. {clean(c)}" for i, c in enumerate(choices))
    block = f"{clean(question)}\n{opts}\nAnswer:"
    if answer_letter is None:
        return block
    return block + " " + answer_letter


# ----------------------------------------------------------------------------- #
# Alpaca-format wrapping (for the instruct model SFT'd on the Alpaca template).
# ----------------------------------------------------------------------------- #
def block_to_alpaca(block):
    """Re-wrap one rendered block: instruction = the first (task) line, input =
    question+options, response = the 'Answer:...' tail (no trailing space, so
    the model still predicts the ' A' letter token)."""
    before, cue, after = block.rpartition("Answer:")   # cue only at answer slot
    instruction, _, body = before.partition("\n")      # first line = instruction
    response = cue + after                              # 'Answer:' or 'Answer: A'
    return ALPACA_PROMPT.format(instruction.strip(), body.strip(), response)


def to_alpaca(raw_prompt):
    """Re-wrap every block of a (few-shot) prompt. Blocks are joined by blank
    lines and clean() strips internal newlines, so the split is safe."""
    return "\n\n".join(block_to_alpaca(b) for b in raw_prompt.split("\n\n"))


# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
def _read_split(data_root, split):
    fs = glob.glob(os.path.join(data_root, "all", f"{split}-*.parquet"))
    if not fs:
        raise SystemExit(f"missing {split} parquet under {data_root}/all/")
    return pd.read_parquet(fs[0])


def build_examples(data_root, k, limit=0, canonical=False):
    """Flat list of test records with fully-rendered k-shot prompts.

    canonical=True uses the standard Hendrycks format (one header per subject,
    then bare Q/A blocks); otherwise the per-block-instruction template that
    mirrors the Sinhala eval (and supports the Alpaca re-wrap)."""
    dev = _read_split(data_root, "dev")
    test = _read_split(data_root, "test")
    fewshot = {subj: sub.head(k) for subj, sub in dev.groupby("subject")}

    records = []
    seen = defaultdict(int)
    for row in test.itertuples(index=False):
        subj = row.subject
        if limit and seen[subj] >= limit:
            continue
        seen[subj] += 1
        pretty = subject_pretty(subj)
        choices = [clean(c) for c in list(row.choices)]
        gold = int(row.answer)                          # 0..3
        n = len(choices)
        valid = (n == 4 and 0 <= gold < n)
        shots = fewshot.get(subj)
        shots = list(shots.itertuples(index=False)) if shots is not None else []
        if canonical:
            blocks = [render_canonical(s.question, [clean(c) for c in list(s.choices)],
                                       LETTERS[int(s.answer)]) for s in shots]
            blocks.append(render_canonical(row.question, choices, None))
            prompt = CANON_HEADER.format(pretty) + "\n\n".join(blocks)
        else:
            blocks = [render(pretty, s.question, [clean(c) for c in list(s.choices)],
                             LETTERS[int(s.answer)]) for s in shots]
            blocks.append(render(pretty, row.question, choices, None))
            prompt = "\n\n".join(blocks)
        records.append(dict(
            subject=subj, subject_pretty=pretty, domain=subject_to_domain(subj),
            n_choices=n, gold=gold, gold_letter=LETTERS[gold] if valid else "?",
            valid=valid, prompt=prompt))
    return records


# Content-free prompt per subject (same few-shot prefix + a blank test Q), used
# to estimate each model's option-letter prior for contextual calibration
# (Zhao et al. 2021) — this removes the strong first-option/"A" bias.
CF_QUESTION = "N/A"
CF_CHOICES = ["N/A", "N/A", "N/A", "N/A"]


def build_cf_prompts(data_root, k, canonical=False):
    dev = _read_split(data_root, "dev")
    fewshot = {subj: sub.head(k) for subj, sub in dev.groupby("subject")}
    cf = {}
    for subj, shots in fewshot.items():
        pretty = subject_pretty(subj)
        shots = list(shots.itertuples(index=False))
        if canonical:
            blocks = [render_canonical(s.question, [clean(c) for c in list(s.choices)],
                                       LETTERS[int(s.answer)]) for s in shots]
            blocks.append(render_canonical(CF_QUESTION, CF_CHOICES, None))
            cf[subj] = CANON_HEADER.format(pretty) + "\n\n".join(blocks)
        else:
            blocks = [render(pretty, s.question, [clean(c) for c in list(s.choices)],
                             LETTERS[int(s.answer)]) for s in shots]
            blocks.append(render(pretty, CF_QUESTION, CF_CHOICES, None))
            cf[subj] = "\n\n".join(blocks)
    return cf


# ----------------------------------------------------------------------------- #
# Model / scoring
# ----------------------------------------------------------------------------- #
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # keep answer cue at the sequence end
    tok.truncation_side = "left"       # drop few-shot first if over length
    # IMPORTANT: eager, not sdpa. On this ROCm/transformers stack the SDPA kernel
    # mis-handles left-padding masks in batched inference — it collapses every
    # prediction to option "A" (verified: sdpa acc 0.32/all-A vs eager acc 0.70).
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    model.eval()
    return model, tok


def _letter_tok(tok):
    # single-token id for each option letter (leading space is part of the token)
    return [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]


@torch.no_grad()
def compute_cf_bias(model, tok, cf_prompts, max_len, batch_size=16):
    """Per-subject option-letter logits for the content-free prompts → the model's
    positional prior, to be subtracted from each real question's letter logits."""
    letter_tok = _letter_tok(tok)
    items = list(cf_prompts.items())
    bias = {}
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        try:
            out = model(**enc, logits_to_keep=1)
        except TypeError:
            out = model(**enc)
        logits = out.logits[:, -1, :].float().cpu()
        for j, (subj, _) in enumerate(chunk):
            bias[subj] = logits[j, letter_tok]          # 4-vector of prior logits
    return bias


@torch.no_grad()
def _score_batch(model, tok, batch, letter_tok, max_len, cf_bias=None):
    """Score one batch; on OOM, free and recursively split (then shrink max_len).
    If cf_bias is given, subtract the per-subject prior before argmax (calibration)."""
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
            cand = letter_tok[:r["n_choices"]]
            scores = logits[j, cand].clone()
            if cf_bias is not None:                     # contextual calibration
                scores = scores - cf_bias[r["subject"]][:r["n_choices"]]
            r["pred"] = int(torch.argmax(scores).item())              # 0-indexed
            r["pred_letter"] = LETTERS[r["pred"]]
            r["correct"] = (r["pred"] == r["gold"])
    except torch.cuda.OutOfMemoryError:
        enc = out = None
        torch.cuda.empty_cache()
        if len(batch) == 1:
            if max_len <= 512:
                batch[0]["pred"] = -1
                batch[0]["pred_letter"] = "?"
                batch[0]["correct"] = False
            else:
                _score_batch(model, tok, batch, letter_tok, 512, cf_bias)
        else:
            mid = len(batch) // 2
            _score_batch(model, tok, batch[:mid], letter_tok, max_len, cf_bias)
            _score_batch(model, tok, batch[mid:], letter_tok, max_len, cf_bias)


@torch.no_grad()
def evaluate(model, tok, records, batch_size, max_len, cf_bias=None):
    letter_tok = _letter_tok(tok)
    todo = [r for r in records if r["valid"]]
    for i in tqdm(range(0, len(todo), batch_size), desc="  eval"):
        _score_batch(model, tok, todo[i:i + batch_size], letter_tok, max_len, cf_bias)
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


def compute_metrics(records):
    c, n, a = overall(records)
    return dict(
        overall=dict(correct=c, total=n, accuracy=a),
        skipped=sum(1 for r in records if not r.get("valid")),
        by_domain=acc_table(records, lambda r: r["domain"]),
        by_subject=acc_table(records, lambda r: r["subject_pretty"]),
    )


def fmt_rows(table, order=None):
    lines = [f"{'':<34}{'acc%':>8}{'correct':>10}{'total':>8}", "-" * 60]
    keys = order if order else sorted(table, key=lambda k: -table[k][2])
    for k in keys:
        if k not in table:
            continue
        c, n, a = table[k]
        lines.append(f"{str(k):<34}{a:>8.2f}{c:>10}{n:>8}")
    return "\n".join(lines)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


DOMAIN_ORDER = ["stem", "humanities", "social_sciences", "other"]


def write_md(path, name, m):
    o = m["overall"]
    L = [f"# English MMLU — {name}\n",
         f"**Overall accuracy: {o['accuracy']:.2f}%** "
         f"({o['correct']}/{o['total']}) · {m['skipped']} skipped · "
         f"5-shot · bf16 · {m.get('format', 'raw')} prompt"
         f"{' · calibrated' if m.get('calibrated') else ''}\n"]

    def section(title, table, order=None):
        L.append(f"\n## {title}\n")
        keys = order or sorted(table, key=lambda k: -table[k][2])
        rows = [[k, f"{table[k][2]:.2f}%", table[k][0], table[k][1]]
                for k in keys if k in table]
        L.append(md_table(["", "Accuracy", "Correct", "Total"], rows))

    section("By domain", m["by_domain"], order=DOMAIN_ORDER)
    section("By subject", m["by_subject"])
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_txt(path, name, m):
    L = ["=" * 60, f"English MMLU evaluation — {name}", "=" * 60]
    o = m["overall"]
    L.append(f"\nOVERALL ACCURACY : {o['accuracy']:.2f}%  "
             f"({o['correct']}/{o['total']})   skipped={m['skipped']}\n")
    L.append("\n## Accuracy by domain")
    L.append(fmt_rows(m["by_domain"], order=DOMAIN_ORDER))
    L.append("\n\n## Accuracy by subject")
    L.append(fmt_rows(m["by_subject"]))
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def write_results_md(path, all_metrics, meta):
    names = list(all_metrics)
    L = ["# English MMLU — Evaluation Results\n"]
    L.append(f"- Benchmark: **MMLU** (Hendrycks; {meta['total']} test MCQs, "
             f"57 subjects)")
    L.append(f"- Purpose: **catastrophic-forgetting check** — English knowledge "
             f"retained after Sinhala CPT / instruction-tuning")
    L.append(f"- Setting: **{meta['k']}-shot**, answer chosen by highest "
             f"option-letter probability")
    L.append(f"- Precision: **bf16** on {meta['gpu']}")
    L.append(f"- Prompt format: base/CPT scored **raw**; instruct scored in its "
             f"**Alpaca SFT template** (see the *Format* column)")
    if any(all_metrics[n].get("calibrated") for n in names):
        L.append(f"- **Contextual calibration ON**: per-subject content-free "
                 f"option-letter prior subtracted before argmax (removes first-option bias)")
    L.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    L.append("## Overall accuracy\n")
    L.append(md_table(["Model", "Accuracy", "Correct", "Total", "Format"],
                      [[n, f"{all_metrics[n]['overall']['accuracy']:.2f}%",
                        all_metrics[n]['overall']['correct'],
                        all_metrics[n]['overall']['total'],
                        all_metrics[n].get('format', 'raw')] for n in names]))

    L.append("\n## Accuracy by domain\n")
    rows = [[d] + [f"{all_metrics[n]['by_domain'].get(d,(0,0,0))[2]:.2f}%"
                   for n in names] for d in DOMAIN_ORDER]
    L.append(md_table(["Domain"] + names, rows))

    L.append("\n## Accuracy by subject\n")
    subs = sorted({k for n in names for k in all_metrics[n]["by_subject"]})
    rows = []
    for s in subs:
        any_n = next(n for n in names if s in all_metrics[n]["by_subject"])
        tot = all_metrics[any_n]["by_subject"][s][1]
        rows.append([s, tot] + [f"{all_metrics[n]['by_subject'].get(s,(0,0,0))[2]:.2f}%"
                                 for n in names])
    L.append(md_table(["Subject", "N"] + names, rows))

    L.append("\n---\nPer-model detail: see each `<model>_results.md` "
             "(and `*_results.txt` / `*_metrics.json`) in this directory.\n")
    open(path, "w", encoding="utf-8").write("\n".join(L))


# ----------------------------------------------------------------------------- #
# GCS upload
# ----------------------------------------------------------------------------- #
def per_model_files(out_dir, name):
    sfx = ("_metrics.json", "_results.md", "_results.txt", "_predictions.jsonl")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="english_mmlu")
    ap.add_argument("--models", nargs="+",
                    default=["llama-3-8b", "SinLlama_v01",
                             "SinLlama_cpt_merged", "SinLlama_Backtrianx_instruct"])
    ap.add_argument("--alpaca-models", nargs="*", default=[],
                    help="model-name substrings to score in the Alpaca template")
    ap.add_argument("--out-dir", default="results_english")
    ap.add_argument("--kshot", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap questions per subject (0=all) for a smoke test")
    ap.add_argument("--canonical", action="store_true",
                    help="use the standard Hendrycks MMLU format (one header per "
                         "subject) for ALL models — leaderboard-comparable numbers; "
                         "--alpaca-models is ignored in this mode")
    ap.add_argument("--calibrate", action="store_true",
                    help="contextual calibration: subtract each model's per-subject "
                         "option-letter prior (from a content-free prompt) before "
                         "argmax, removing the first-option (A) bias")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--bucket", default="",
                    help="GCS bucket/prefix; each model's files upload to "
                         "<bucket>/<model_name>/ as soon as it finishes")
    ap.add_argument("--combine-only", action="store_true",
                    help="just rebuild (and upload) the combined results.md")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records0 = build_examples(args.data_root, args.kshot, args.limit,
                              canonical=args.canonical)
    cf_prompts = (build_cf_prompts(args.data_root, args.kshot, canonical=args.canonical)
                  if args.calibrate else None)
    n_valid = sum(r["valid"] for r in records0)
    n_skip = len(records0) - n_valid
    print(f"Loaded {len(records0)} questions ({n_valid} valid, {n_skip} skipped) "
          f"across {len({r['subject'] for r in records0})} subjects.")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    if args.combine_only:
        all_metrics = {}
        for path in args.models:
            name = os.path.basename(path.rstrip("/"))
            mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
            if os.path.exists(mpath):
                all_metrics[name] = json.load(open(mpath, encoding="utf-8"))
            else:
                print(f"  (combine) missing {mpath} — skipping")
        if not all_metrics:
            raise SystemExit("combine-only: no *_metrics.json found in --out-dir")
        meta = dict(k=args.kshot, gpu=gpu, total=n_valid)
        rpath = os.path.join(args.out_dir, "results.md")
        write_results_md(rpath, all_metrics, meta)
        if args.bucket:
            gcs_cp([rpath], args.bucket.rstrip("/") + "/")
        print(f"Wrote combined {rpath}")
        return

    all_metrics = {}
    for path in args.models:
        name = os.path.basename(path.rstrip("/"))
        use_alpaca = (not args.canonical) and any(s in name for s in args.alpaca_models)
        fmt = "canonical" if args.canonical else ("alpaca" if use_alpaca else "raw")
        mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
        if args.skip_existing and os.path.exists(mpath):
            m = json.load(open(mpath, encoding="utf-8"))
            m["format"] = fmt
            m["calibrated"] = bool(args.calibrate)
            print(f"\n=== {name}: reusing cached metrics "
                  f"({m['overall']['accuracy']:.2f}%) ===")
        else:
            print(f"\n=== Evaluating {name}  (prompt format: {fmt}) ===")
            t0 = time.time()
            model, tok = load_model(path)
            records = [dict(r) for r in records0]      # fresh copy per model
            if use_alpaca:
                for r in records:
                    r["prompt"] = to_alpaca(r["prompt"])
            cf_bias = None
            if args.calibrate:
                cfp = {s: to_alpaca(p) for s, p in cf_prompts.items()} if use_alpaca \
                      else cf_prompts
                cf_bias = compute_cf_bias(model, tok, cfp, args.max_len)
            evaluate(model, tok, records, args.batch_size, args.max_len, cf_bias)
            m = compute_metrics(records)
            m["format"] = fmt
            m["calibrated"] = bool(args.calibrate)
            print(f"  {name}: {m['overall']['accuracy']:.2f}%  "
                  f"({m['overall']['correct']}/{m['overall']['total']}) "
                  f"in {time.time()-t0:.0f}s")
            json.dump(m, open(mpath, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            with open(os.path.join(args.out_dir, f"{name}_predictions.jsonl"),
                      "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps({k: r.get(k) for k in
                             ("subject", "domain", "n_choices", "gold",
                              "gold_letter", "valid", "pred", "pred_letter",
                              "correct")}, ensure_ascii=False) + "\n")
            del model
            gc.collect(); torch.cuda.empty_cache()

        all_metrics[name] = m
        write_txt(os.path.join(args.out_dir, f"{name}_results.txt"), name, m)
        write_md(os.path.join(args.out_dir, f"{name}_results.md"), name, m)
        if args.bucket:
            gcs_cp(per_model_files(args.out_dir, name),
                   args.bucket.rstrip("/") + f"/{name}/")

    meta = dict(k=args.kshot, gpu=gpu, total=n_valid)
    rpath = os.path.join(args.out_dir, "results.md")
    write_results_md(rpath, all_metrics, meta)
    if args.bucket and len(args.models) > 1:
        gcs_cp([rpath], args.bucket.rstrip("/") + "/")
    print(f"\nWrote results to {args.out_dir}/  (results.md + per-model files)")


if __name__ == "__main__":
    main()
