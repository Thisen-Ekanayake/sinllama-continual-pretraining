#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SinLlama models on the SinhalaMMLU benchmark.

Method
------
* Prompt: the official SinhalaMMLU template (mmlu/prompt.txt), with the
  answer-number list adapted to the question's option count (4 or 5).
* Few-shot: 3-shot. The 3 exemplars come from the matching subject file in
  SinhalaMMLU/fewshot/<difficulty>/, matched to each TEST file by
  (difficulty, subject) parsed from the filename.
* Scoring: "highest probability among the answer options" (same protocol the
  SinhalaMMLU paper uses for open models). We build the context ending in
  "පිළිතුර: " (trailing space) and, from a single forward pass, compare the
  next-token logits of the single-token digits 1..N. argmax -> predicted option.
* Prompt format: each model is scored in the format it expects. Base/CPT models
  use the raw template above; any model named via --alpaca-models (the Bactrian-X
  instruct model, which was SFT'd on the Alpaca template) is re-wrapped in the
  "### Instruction / ### Input / ### Response" scaffold. The digit-scoring step is
  identical either way — only the surrounding text changes.

Outputs (per model): a human-readable *_results.txt, a machine-readable
*_metrics.json and a *_predictions.jsonl. Finally a combined results.md.
"""
import os, re, json, glob, time, argparse, gc
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:                                    # pragma: no cover
    def tqdm(x, **k): return x

# ----------------------------------------------------------------------------- #
# Prompt template (exactly mmlu/prompt.txt, with [NUMS] where it hard-coded
# "1, 2, 3, 4" so the list can grow to "1, 2, 3, 4, 5" for A/L questions).
# ----------------------------------------------------------------------------- #
DEFAULT_TEMPLATE = (
    "මෙය [SUBJECT] විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි. පහත ප්‍රශ්නයට [NUMS] යන "
    "පිළිතුරු වලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන හෝ පිළිතුර තෝරන්න.\n"
    "ප්‍රශ්නය: [QUESTION]\n"
    "[OPTIONS]\n"
    "පිළිතුර:"
)

KNOWN_CATEGORIES = [
    "social_science", "business studies", "business_studies",
    "humanities", "language", "stem", "other",
]


def load_template(path):
    if path and os.path.exists(path):
        text = open(path, encoding="utf-8").read().strip("\n")
        # normalise the hard-coded numeral list into a placeholder
        text = re.sub(r"1\s*,\s*2\s*,\s*3\s*,\s*4", "[NUMS]", text, count=1)
        return text
    return DEFAULT_TEMPLATE


def norm_cat(c):
    return (c or "").strip().lower().replace(" ", "_")


def clean(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def subject_from_filename(fname):
    """Return a normalised (key) and display subject parsed from a data
    filename, ignoring the noisy category token and the difficulty/suffix."""
    base = os.path.basename(fname)
    base = re.sub(r"\.json$", "", base, flags=re.I)
    base = re.sub(r"_fewshot$", "", base, flags=re.I)
    base = re.sub(r"_f_\d+[a-z]?$", "", base, flags=re.I)      # TEST suffix
    base = re.sub(r"_(easy|medium|hard)$", "", base, flags=re.I)
    # strip exactly one trailing category token (longest first), on a
    # separator boundary, and never empty the subject
    for cat in sorted(KNOWN_CATEGORIES, key=len, reverse=True):
        m = re.search(r"[_ ]" + re.escape(cat) + r"$", base, flags=re.I)
        if m and m.start() > 0:
            base = base[:m.start()]
            break
    display = base.replace("_", " ").strip()
    key = re.sub(r"[^a-z0-9]", "", display.lower())
    return key, display


def render(template, subject, question, choices, answer=None):
    """Render one (few-shot or test) block. If answer is given it is appended
    after 'පිළිතුර:' with a leading space; otherwise a single trailing space is
    added so the next token the model predicts is the answer digit."""
    n = len(choices)
    nums = ", ".join(str(i) for i in range(1, n + 1))
    opts = "\n".join(f"{i+1}. {clean(c)}" for i, c in enumerate(choices))
    block = (template
             .replace("[SUBJECT]", clean(subject) or "විෂය")
             .replace("[NUMS]", nums)
             .replace("[QUESTION]", clean(question))
             .replace("[OPTIONS]", opts))
    if answer is None:
        return block + " "                     # test question -> predict digit
    return block + " " + str(answer)           # exemplar with gold answer


# ----------------------------------------------------------------------------- #
# Alpaca-format wrapping (for instruction-tuned models fine-tuned on the Alpaca
# template, e.g. SinLlama_Backtrianx_instruct — see 158_P_Finetunning_BacterianX).
# to_alpaca() re-wraps an already-built (few-shot) raw prompt block-by-block; the
# trailing "පිළිතුර: " of the test block is preserved so digit-scoring is unchanged.
# ----------------------------------------------------------------------------- #
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")


def block_to_alpaca(block):
    """Re-wrap one rendered MMLU block into the Alpaca template: instruction =
    task line, input = question+options, response = the 'පිළිතුර: ...' tail."""
    before, cue, after = block.rpartition("පිළිතුර:")   # cue only at the answer slot
    instr, _, qbody = before.partition("\nප්‍රශ්නය:")
    instruction = instr.strip()
    input_text = ("ප්‍රශ්නය:" + qbody).strip()
    response = cue + after                               # "පිළිතුර: " or "පිළිතුර: 3"
    return ALPACA_PROMPT.format(instruction, input_text, response)


def to_alpaca(raw_prompt):
    """Re-wrap every block of a (few-shot) prompt. Blocks are joined by blank
    lines and clean() strips internal newlines, so the split is safe; the
    trailing space of the final (test) block is preserved for digit-scoring."""
    return "\n\n".join(block_to_alpaca(b) for b in raw_prompt.split("\n\n"))


def build_fewshot_index(data_root, k):
    """difficulty -> subject_key -> formatted-exemplar list (raw dicts)."""
    idx = {}
    for diff in ("easy", "medium", "hard"):
        idx[diff] = {}
        for f in glob.glob(os.path.join(data_root, "fewshot", diff, "*.json")):
            key, _ = subject_from_filename(f)
            data = json.load(open(f, encoding="utf-8"))
            idx[diff][key] = data[:k]
    return idx


def subject_original(item):
    md = item.get("metadata", {}) or {}
    so = clean(md.get("subject_original"))
    return so or clean(item.get("subject"))


def build_examples(data_root, template, k):
    """Return a flat list of question records (with fully-rendered prompt)."""
    fewshot = build_fewshot_index(data_root, k)
    records, missing_fewshot = [], set()
    for diff in ("easy", "medium", "hard"):
        for f in sorted(glob.glob(os.path.join(data_root, "TEST", diff, "*.json"))):
            skey, sdisp = subject_from_filename(f)
            if not sdisp:                              # safety: never blank
                sdisp = re.sub(r"\.json$", "", os.path.basename(f)).split("_")[0]
                skey = re.sub(r"[^a-z0-9]", "", sdisp.lower())
            shots = fewshot.get(diff, {}).get(skey)
            if not shots:                              # fuzzy fallback
                cand = [kk for kk in fewshot.get(diff, {}) if kk and (kk in skey or skey in kk)]
                if cand:
                    shots = fewshot[diff][sorted(cand, key=len)[-1]]
                else:
                    missing_fewshot.add((diff, skey))
                    shots = []
            # pre-render the shared few-shot prefix for this file
            shot_blocks = [render(template, subject_original(s), s["question"],
                                  s["choices"], s["answer"]) for s in shots]
            data = json.load(open(f, encoding="utf-8"))
            for it in data:
                choices = it.get("choices") or []
                n = len(choices)
                gold = it.get("answer")
                valid = isinstance(gold, int) and 1 <= gold <= n and n >= 2
                test_block = render(template, subject_original(it),
                                    it["question"], choices, answer=None)
                prompt = "\n\n".join(shot_blocks + [test_block])
                records.append(dict(
                    file=os.path.basename(f), difficulty=diff,
                    category=norm_cat(it.get("category")),
                    subject_key=skey, subject_display=sdisp,
                    subject_original=subject_original(it),
                    n_choices=n, gold=gold, valid=valid, prompt=prompt))
    # one canonical display label per subject_key (merges case-only variants
    # such as "Buddhism"/"buddhism", "Geography"/"geography")
    disp_by_key = {}
    for r in records:
        k = r["subject_key"] or "unknown"
        r["subject_key"] = k
        d = r["subject_display"] or k
        cur = disp_by_key.get(k)
        score = (sum(c.isupper() for c in d), len(d))
        if cur is None or score > (sum(c.isupper() for c in cur), len(cur)):
            disp_by_key[k] = d
    for r in records:
        r["subject_label"] = disp_by_key[r["subject_key"]]
    return records, missing_fewshot


def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # keep answer cue at the sequence end
    tok.truncation_side = "left"       # drop few-shot first if over length
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()
    return model, tok


@torch.no_grad()
def _score_batch(model, tok, batch, digit_id, max_len):
    """Score one batch. On CUDA OOM, free memory and recursively split the
    batch (and finally shrink max_len) until it fits — base Llama-3 byte-falls
    back on Sinhala, so some batches are very long."""
    enc = out = None
    try:
        enc = tok([r["prompt"] for r in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=max_len).to(model.device)
        # logits for the LAST position only — avoids a [B, seq, vocab] tensor
        try:
            out = model(**enc, logits_to_keep=1)
        except TypeError:                               # older transformers
            out = model(**enc)
        logits = out.logits[:, -1, :].float().cpu()     # next-token logits
        for j, r in enumerate(batch):
            cand = [digit_id[d] for d in range(1, r["n_choices"] + 1)]
            r["pred"] = int(torch.argmax(logits[j, cand]).item()) + 1  # 1-indexed
            r["correct"] = (r["pred"] == r["gold"])
    except torch.cuda.OutOfMemoryError:
        enc = out = None                                # drop refs, then reclaim
        torch.cuda.empty_cache()
        if len(batch) == 1:
            if max_len <= 768:                          # give up on one huge item
                batch[0]["pred"] = -1
                batch[0]["correct"] = False
            else:
                _score_batch(model, tok, batch, digit_id, 768)
        else:
            mid = len(batch) // 2
            _score_batch(model, tok, batch[:mid], digit_id, max_len)
            _score_batch(model, tok, batch[mid:], digit_id, max_len)


@torch.no_grad()
def evaluate(model, tok, records, batch_size, max_len):
    # single-token id for each answer digit (1.."9")
    digit_id = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)}
    todo = [r for r in records if r["valid"]]
    for i in tqdm(range(0, len(todo), batch_size), desc="  eval"):
        _score_batch(model, tok, todo[i:i + batch_size], digit_id, max_len)
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
    m = dict(
        overall=dict(correct=c, total=n, accuracy=a),
        skipped=sum(1 for r in records if not r.get("valid")),
        by_difficulty=acc_table(records, lambda r: r["difficulty"]),
        by_domain=acc_table(records, lambda r: r["category"]),
        by_subject=acc_table(records, lambda r: r["subject_label"]),
        by_domain_difficulty=acc_table(
            records, lambda r: f'{r["category"]}|{r["difficulty"]}'),
    )
    return m


def fmt_rows(table, order=None, label="key"):
    lines = [f"{'':<38}{'acc%':>8}{'correct':>10}{'total':>8}",
             "-" * 64]
    keys = order if order else sorted(table, key=lambda k: -table[k][2])
    for k in keys:
        if k not in table:
            continue
        c, n, a = table[k]
        lines.append(f"{str(k):<38}{a:>8.2f}{c:>10}{n:>8}")
    return "\n".join(lines)


def write_md(path, name, m):
    """Per-model standalone markdown with every breakdown."""
    o = m["overall"]
    L = [f"# SinhalaMMLU — {name}\n",
         f"**Overall accuracy: {o['accuracy']:.2f}%** "
         f"({o['correct']}/{o['total']}) · {m['skipped']} malformed skipped · "
         f"3-shot · bf16 · {m.get('format', 'raw')} prompt\n"]

    def section(title, table, order=None):
        L.append(f"\n## {title}\n")
        keys = order or sorted(table, key=lambda k: -table[k][2])
        rows = [[k, f"{table[k][2]:.2f}%", table[k][0], table[k][1]]
                for k in keys if k in table]
        L.append(md_table(["", "Accuracy", "Correct", "Total"], rows))

    section("By difficulty", m["by_difficulty"], order=["easy", "medium", "hard"])
    section("By domain", m["by_domain"])
    section("By domain × difficulty", m["by_domain_difficulty"])
    section("By subject", m["by_subject"])
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_txt(path, model_name, m):
    L = []
    L.append("=" * 64)
    L.append(f"SinhalaMMLU evaluation — {model_name}")
    L.append("=" * 64)
    o = m["overall"]
    L.append(f"\nOVERALL ACCURACY : {o['accuracy']:.2f}%  "
             f"({o['correct']}/{o['total']})   skipped(malformed)={m['skipped']}\n")
    L.append("\n## Accuracy by difficulty")
    L.append(fmt_rows(m["by_difficulty"], order=["easy", "medium", "hard"]))
    L.append("\n\n## Accuracy by domain")
    L.append(fmt_rows(m["by_domain"]))
    L.append("\n\n## Accuracy by domain x difficulty")
    L.append(fmt_rows(m["by_domain_difficulty"]))
    L.append("\n\n## Accuracy by subject")
    L.append(fmt_rows(m["by_subject"]))
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_results_md(path, all_metrics, meta):
    names = list(all_metrics)
    L = ["# SinhalaMMLU — Evaluation Results\n"]
    L.append(f"- Benchmark: **SinhalaMMLU** ({meta['total']} evaluated MCQs, "
             f"{meta['skipped_note']})")
    L.append(f"- Setting: **{meta['k']}-shot**, official prompt template, "
             f"answer chosen by highest option-digit probability")
    L.append(f"- Precision: **bf16** on {meta['gpu']}")
    L.append(f"- Prompt format: base/CPT models scored **raw**; instruct models "
             f"scored in their **Alpaca SFT template** (see the *Format* column)")
    L.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Overall
    L.append("## Overall accuracy\n")
    L.append(md_table(["Model", "Accuracy", "Correct", "Total", "Format"],
                      [[n, f"{all_metrics[n]['overall']['accuracy']:.2f}%",
                        all_metrics[n]['overall']['correct'],
                        all_metrics[n]['overall']['total'],
                        all_metrics[n].get('format', 'raw')] for n in names]))

    # By difficulty
    L.append("\n## Accuracy by difficulty\n")
    rows = []
    for d in ["easy", "medium", "hard"]:
        rows.append([d] + [f"{all_metrics[n]['by_difficulty'].get(d,(0,0,0))[2]:.2f}%"
                            for n in names])
    L.append(md_table(["Difficulty"] + names, rows))

    # By domain
    L.append("\n## Accuracy by domain\n")
    domains = sorted({k for n in names for k in all_metrics[n]["by_domain"]})
    rows = [[d] + [f"{all_metrics[n]['by_domain'].get(d,(0,0,0))[2]:.2f}%" for n in names]
            for d in domains]
    L.append(md_table(["Domain"] + names, rows))

    # By subject
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="SinhalaMMLU")
    ap.add_argument("--prompt-file", default="prompt.txt")
    ap.add_argument("--models", nargs="+",
                    default=["llama-3-8b", "SinLlama_v01",
                             "SinLlama_cpt_merged", "SinLlama_Backtrianx_instruct"])
    ap.add_argument("--alpaca-models", nargs="*", default=[],
                    help="model-name substrings to score in the Alpaca "
                         "### Instruction/Input/Response template they were SFT'd "
                         "on (e.g. SinLlama_Backtrianx_instruct)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--kshot", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)   # bump on large GPUs
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap questions per TEST file (0=all) for a smoke test")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse an existing <model>_metrics.json instead of "
                         "re-running that model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    template = load_template(args.prompt_file)
    records0, missing = build_examples(args.data_root, template, args.kshot)
    if args.limit:                     # keep first N per file for a quick test
        seen = defaultdict(int)
        kept = []
        for r in records0:
            if seen[r["file"]] < args.limit:
                kept.append(r); seen[r["file"]] += 1
        records0 = kept

    n_valid = sum(r["valid"] for r in records0)
    n_skip = len(records0) - n_valid
    print(f"Loaded {len(records0)} questions "
          f"({n_valid} valid, {n_skip} malformed/skipped).")
    if missing:
        print(f"WARNING: no few-shot match for: {sorted(missing)}")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    all_metrics = {}
    for path in args.models:
        name = os.path.basename(path.rstrip("/"))
        use_alpaca = any(s in name for s in args.alpaca_models)
        fmt = "alpaca" if use_alpaca else "raw"
        mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
        if args.skip_existing and os.path.exists(mpath):
            m = json.load(open(mpath, encoding="utf-8"))
            print(f"\n=== {name}: reusing cached metrics "
                  f"({m['overall']['accuracy']:.2f}%) ===")
        else:
            print(f"\n=== Evaluating {name}  (prompt format: {fmt}) ===")
            t0 = time.time()
            model, tok = load_model(path)
            records = [dict(r) for r in records0]      # fresh copy per model
            if use_alpaca:                             # re-wrap in the SFT template
                for r in records:
                    r["prompt"] = to_alpaca(r["prompt"])
            evaluate(model, tok, records, args.batch_size, args.max_len)
            m = compute_metrics(records)
            print(f"  {name}: {m['overall']['accuracy']:.2f}%  "
                  f"({m['overall']['correct']}/{m['overall']['total']}) "
                  f"in {time.time()-t0:.0f}s")
            json.dump(m, open(mpath, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            with open(os.path.join(args.out_dir, f"{name}_predictions.jsonl"),
                      "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps({k: r.get(k) for k in
                             ("file", "difficulty", "category", "subject_display",
                              "n_choices", "gold", "valid", "pred", "correct")},
                             ensure_ascii=False) + "\n")
            del model
            gc.collect(); torch.cuda.empty_cache()

        m["format"] = fmt
        all_metrics[name] = m
        write_txt(os.path.join(args.out_dir, f"{name}_results.txt"), name, m)
        write_md(os.path.join(args.out_dir, f"{name}_results.md"), name, m)

    meta = dict(k=args.kshot, gpu=gpu, total=n_valid,
                skipped_note=f"{n_skip} malformed answers excluded")
    write_results_md(os.path.join(args.out_dir, "results.md"), all_metrics, meta)
    print(f"\nWrote results to {args.out_dir}/  (results.md + per-model files)")


if __name__ == "__main__":
    main()
