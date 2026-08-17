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
import os, re, json, glob, time, argparse, gc, shutil, subprocess
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


# ----------------------------------------------------------------------------- #
# Chat-format wrapping, for models SFT'd on the UltraChat template in
# sft/config.yaml: "### User:\n...\n\n### Assistant:\n...<|end_of_text|>",
# turns joined by "\n\n". Structurally the same trick as the Alpaca wrapper
# above -- one turn pair per few-shot block, answer cue left in the assistant
# slot so the digit is still the next token predicted.
# ----------------------------------------------------------------------------- #
CHAT_PROMPT = "### User:\n{}\n\n{}\n\n### Assistant:\n{}"
CHAT_EOS = "<|end_of_text|>"


def block_to_chat(block, eos=""):
    """Re-wrap one rendered MMLU block as a user/assistant turn pair."""
    before, cue, after = block.rpartition("පිළිතුර:")   # cue only at the answer slot
    instr, _, qbody = before.partition("\nප්‍රශ්නය:")
    instruction = instr.strip()
    input_text = ("ප්‍රශ්නය:" + qbody).strip()
    response = cue + after                               # "පිළිතුර: " or "පිළිතුර: 3"
    return CHAT_PROMPT.format(instruction, input_text, response) + eos


def to_chat(raw_prompt, eos=CHAT_EOS):
    """Every few-shot exemplar becomes a completed turn pair terminated by EOS,
    exactly as in training. The final (test) block gets NO terminator: it must
    end at 'පිළිතුර: ' so the answer digit is the next token scored."""
    blocks = raw_prompt.split("\n\n")
    wrapped = [block_to_chat(b, eos) for b in blocks[:-1]]
    wrapped.append(block_to_chat(blocks[-1], ""))
    return "\n\n".join(wrapped)


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
            shot_prefix = "\n\n".join(shot_blocks)
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
                    n_choices=n, gold=gold, valid=valid, prompt=prompt,
                    # retained for the de-biasing arms (calibration needs the
                    # exact few-shot prefix; permutation re-renders the options)
                    shot_prefix=shot_prefix, question=it["question"],
                    choices=choices))
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


def assert_eager(model, path):
    """Fail loudly if the SDPA kernel is in effect.

    `attn_implementation="eager"` is a request, not a guarantee -- it can be
    overridden by a config, ignored by some transformers versions, or silently
    fall back. On this ROCm stack SDPA mis-handles left-padding masks in batched
    inference and collapses predictions onto one option, which does NOT raise:
    it just returns a plausible-looking wrong number. That is not hypothetical
    here -- SinLlama_cpt was published at 40.13% English / 35.65% Sinhala from an
    affected run and re-scored at 47.27% / 41.33% once eager was forced, a 6-7pp
    error whose only visible symptom was a near-absent option "B" (543/14042
    predictions against a 24.7% gold rate). Check it rather than trust it."""
    impl = getattr(model.config, "_attn_implementation", None)
    if impl != "eager":
        raise SystemExit(
            f"FATAL: {path} loaded with attn_implementation={impl!r}, not "
            "'eager'. Batched left-padded option scoring is invalid on this "
            "stack with SDPA -- it silently collapses predictions onto a single "
            "option instead of erroring. Refusing to produce numbers.")


# ----------------------------------------------------------------------------- #
# Content-free prompts for contextual calibration (Zhao et al. 2021).
#
# The model is shown the same few-shot prefix and an otherwise-identical test
# block whose question and every option are the placeholder "N/A". Whatever
# probability mass it still puts on the answer digits is pure positional prior:
# there is no content to reason about. Subtracting that prior (in log space)
# before argmax removes the option-position bias.
#
# Unlike English MMLU, SinhalaMMLU is not uniformly 4-way (4353 items are 4-way,
# 2523 are 5-way, 2 are 6-way), so the prior is estimated per option-count as
# well as per subject: a 5-way prompt has a different positional prior than a
# 4-way one. The key is (difficulty, subject_key, n_choices) because the
# few-shot prefix itself varies by difficulty and subject.
# ----------------------------------------------------------------------------- #
CF_QUESTION = "N/A"


def cf_key(r):
    return (r["difficulty"], r["subject_key"], r["n_choices"])


def build_cf_prompts(template, records):
    """One content-free prompt per (difficulty, subject_key, n_choices) combo.

    Reuses each record's stored `shot_prefix` verbatim, so the calibration
    prompt is byte-identical to the real prompt up to the test block -- no
    risk of drifting from build_examples()'s few-shot selection or its fuzzy
    subject fallback."""
    cf = {}
    for r in records:
        if not r.get("valid"):
            continue
        k = cf_key(r)
        if k in cf:
            continue
        n = r["n_choices"]
        block = render(template, r["subject_original"], CF_QUESTION,
                       [CF_QUESTION] * n, answer=None)
        cf[k] = (r["shot_prefix"] + "\n\n" + block) if r["shot_prefix"] else block
    return cf


@torch.no_grad()
def compute_cf_bias(model, tok, cf_prompts, digit_id, max_len, batch_size=8):
    """(difficulty, subject, n_choices) -> tensor of prior logits over the
    answer digits, to be subtracted from each real question's digit logits."""
    items = list(cf_prompts.items())
    bias = {}
    for i in tqdm(range(0, len(items), batch_size), desc="  cf-bias"):
        chunk = items[i:i + batch_size]
        enc = tok([p for _, p in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        try:
            out = model(**enc, logits_to_keep=1)
        except TypeError:
            out = model(**enc)
        logits = out.logits[:, -1, :].float().cpu()
        for j, (key, _) in enumerate(chunk):
            n = key[2]
            cand = [digit_id[d] for d in range(1, n + 1)]
            # log-softmax over just the candidate digits: the prior is a
            # distribution over the options, not raw unnormalised logits
            bias[key] = torch.log_softmax(logits[j, cand], dim=-1)
    return bias


def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # keep answer cue at the sequence end
    tok.truncation_side = "left"       # drop few-shot first if over length
    # IMPORTANT: eager, not sdpa. On this ROCm/transformers stack the SDPA kernel
    # mis-handles left-padding masks in batched inference — it collapses every
    # prediction to the first option (verified on English MMLU: sdpa all-"A" acc
    # 0.32 vs eager acc 0.70). Batched left-padded scoring must use eager here.
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    assert_eager(model, path)
    model.eval()
    return model, tok


@torch.no_grad()
def _score_batch(model, tok, batch, digit_id, max_len, cf_bias=None):
    """Score one batch. On CUDA OOM, free memory and recursively split the
    batch (and finally shrink max_len) until it fits — base Llama-3 byte-falls
    back on Sinhala, so some batches are very long.

    If cf_bias is given, subtract the content-free positional prior before
    argmax (contextual calibration)."""
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
            scores = torch.log_softmax(logits[j, cand], dim=-1)
            if cf_bias is not None:                     # contextual calibration
                scores = scores - cf_bias[cf_key(r)]
            r["pred"] = int(torch.argmax(scores).item()) + 1  # 1-indexed
            r["correct"] = (r["pred"] == r["gold"])
    except torch.cuda.OutOfMemoryError:
        enc = out = None                                # drop refs, then reclaim
        torch.cuda.empty_cache()
        if len(batch) == 1:
            if max_len <= 768:                          # give up on one huge item
                batch[0]["pred"] = -1
                batch[0]["correct"] = False
            else:
                _score_batch(model, tok, batch, digit_id, 768, cf_bias)
        else:
            mid = len(batch) // 2
            _score_batch(model, tok, batch[:mid], digit_id, max_len, cf_bias)
            _score_batch(model, tok, batch[mid:], digit_id, max_len, cf_bias)


def _digit_ids(tok):
    """Single-token id for each answer digit (1.."9"). render() already emits the
    space before the digit, so these are the bare digit tokens.

    Guarded: this takes the FIRST token of each digit's encoding, which is only
    correct when digits are single tokens (true for the Llama-3 BPE that SinLlama
    uses). A SentencePiece tokenizer instead emits a shared leading-space token
    first, so every digit would collapse to the same id and every score would be
    identical -- silently producing meaningless accuracy rather than an error."""
    ids = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)}
    if len(set(ids.values())) != len(ids):
        raise SystemExit(
            "FATAL: answer digits do not map to distinct single tokens with this "
            f"tokenizer (got {ids}). Digit-probability scoring is invalid here; "
            "this eval assumes a Llama-3-style BPE where '1'..'9' are single "
            "tokens.")
    return ids


@torch.no_grad()
def evaluate(model, tok, records, batch_size, max_len, cf_bias=None):
    digit_id = _digit_ids(tok)
    todo = [r for r in records if r["valid"]]
    for i in tqdm(range(0, len(todo), batch_size), desc="  eval"):
        _score_batch(model, tok, todo[i:i + batch_size], digit_id, max_len, cf_bias)
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


# ----------------------------------------------------------------------------- #
# GCS upload
# ----------------------------------------------------------------------------- #
def per_model_files(out_dir, name):
    """The result files a single model produces, that currently exist on disk."""
    sfx = ("_metrics.json", "_results.md", "_results.txt", "_predictions.jsonl")
    return [os.path.join(out_dir, name + s) for s in sfx
            if os.path.exists(os.path.join(out_dir, name + s))]


def gcs_cp(files, dest):
    """Copy files to a GCS dest (e.g. gs://bucket/<model>/ or the bucket root).
    No-op if there is nothing to send or gsutil is unavailable; on failure it
    warns rather than aborting the whole evaluation."""
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
    ap.add_argument("--data-root", default="SinhalaMMLU")
    ap.add_argument("--prompt-file", default="prompt.txt")
    ap.add_argument("--models", nargs="+",
                    default=["llama-3-8b", "SinLlama_v01",
                             "SinLlama_cpt_merged", "SinLlama_Backtrianx_instruct"])
    ap.add_argument("--alpaca-models", nargs="*", default=[],
                    help="model-name substrings to score in the Alpaca "
                         "### Instruction/Input/Response template they were SFT'd "
                         "on (e.g. SinLlama_Backtrianx_instruct)")
    ap.add_argument("--chat-models", nargs="*", default=[],
                    help="model-name substrings to score in the UltraChat "
                         "### User/### Assistant chat template they were SFT'd on "
                         "(e.g. SinLlama_uc_instruct). Ignored for a model that "
                         "also matches --alpaca-models.")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--kshot", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)   # bump on large GPUs
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap questions per TEST file (0=all) for a smoke test")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse an existing <model>_metrics.json instead of "
                         "re-running that model")
    ap.add_argument("--bucket", default="",
                    help="GCS bucket/prefix; each model's files are uploaded to "
                         "<bucket>/<model_name>/ as soon as it finishes")
    ap.add_argument("--calibrate", action="store_true",
                    help="contextual calibration: subtract each model's "
                         "positional prior, estimated from a content-free "
                         "(all-'N/A') prompt, before argmax. Estimated per "
                         "(difficulty, subject, n_choices).")
    ap.add_argument("--combine-only", action="store_true",
                    help="skip evaluation: just rebuild (and upload) the combined "
                         "results.md from existing <model>_metrics.json files")
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

    if args.combine_only:                    # rebuild the combined report only
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
        meta = dict(k=args.kshot, gpu=gpu, total=n_valid,
                    skipped_note=f"{n_skip} malformed answers excluded")
        rpath = os.path.join(args.out_dir, "results.md")
        write_results_md(rpath, all_metrics, meta)
        if args.bucket:
            gcs_cp([rpath], args.bucket.rstrip("/") + "/")
        print(f"Wrote combined {rpath}")
        return

    all_metrics = {}
    for path in args.models:
        name = os.path.basename(path.rstrip("/"))
        use_alpaca = any(s in name for s in args.alpaca_models)
        use_chat = (not use_alpaca) and any(s in name for s in args.chat_models)
        fmt = "alpaca" if use_alpaca else ("chat" if use_chat else "raw")
        mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
        if args.skip_existing and os.path.exists(mpath):
            m = json.load(open(mpath, encoding="utf-8"))
            m["format"] = fmt
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
            elif use_chat:                             # ... or the chat template
                for r in records:
                    r["prompt"] = to_chat(r["prompt"])

            cf_bias = None
            if args.calibrate:
                # the content-free prompt must go through the SAME format
                # wrapper as the real prompts, or the prior is measured in a
                # context the model never sees at scoring time
                cfp = build_cf_prompts(template, records)
                if use_alpaca:
                    cfp = {k: to_alpaca(v) for k, v in cfp.items()}
                elif use_chat:
                    cfp = {k: to_chat(v) for k, v in cfp.items()}
                print(f"  estimating positional prior from {len(cfp)} "
                      f"content-free prompts")
                cf_bias = compute_cf_bias(model, tok, cfp, _digit_ids(tok),
                                          args.max_len)

            evaluate(model, tok, records, args.batch_size, args.max_len, cf_bias)
            m = compute_metrics(records)
            m["format"] = fmt                          # persist into metrics.json
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
                             ("file", "difficulty", "category", "subject_display",
                              "n_choices", "gold", "valid", "pred", "correct")},
                             ensure_ascii=False) + "\n")
            del model
            gc.collect(); torch.cuda.empty_cache()

        all_metrics[name] = m
        write_txt(os.path.join(args.out_dir, f"{name}_results.txt"), name, m)
        write_md(os.path.join(args.out_dir, f"{name}_results.md"), name, m)
        if args.bucket:                        # upload as soon as this model is done
            gcs_cp(per_model_files(args.out_dir, name),
                   args.bucket.rstrip("/") + f"/{name}/")

    meta = dict(k=args.kshot, gpu=gpu, total=n_valid,
                skipped_note=f"{n_skip} malformed answers excluded")
    rpath = os.path.join(args.out_dir, "results.md")
    write_results_md(rpath, all_metrics, meta)
    # only multi-model (and the combine-only) passes own the combined report, so
    # parallel single-model runs don't clobber it in the bucket root
    if args.bucket and len(args.models) > 1:
        gcs_cp([rpath], args.bucket.rstrip("/") + "/")
    print(f"\nWrote results to {args.out_dir}/  (results.md + per-model files)")


if __name__ == "__main__":
    main()
