"""Validate textbook_sft/data/textbook_mix (built by build_textbook_sft_dataset.py).

Checks
------
  structure      user -> assistant turns, non-empty, unique ids, no chat-template
                 markers leaking into content
  characters     U+FFFD, (cid:NN), private-use, unconverted legacy-font glyphs,
                 control chars, orphaned vowel signs / hal, misplaced ZWJ,
                 unbalanced $, raw ^{..} outside math, stray backticks
  failure modes  very short or apparently truncated answers, answer == prompt,
                 one prompt mapped to many different answers, repetition loops,
                 non-Sinhala answers
  contamination  textbook eval vs train, UltraChat eval vs train (and vs the
                 whole UltraChat train split it was sampled from), cross-source
                 eval vs train, ids across splits, and every
                 training row vs the benchmark sets on disk (MMLU-Sinhala test /
                 validation / dev, news / sentiment / writing test, the 100-prompt
                 and benchmark probe prompts) by Sinhala word 8-gram overlap
  tokens         SinLlama token counts with the chat template, per source/split

usage: python textbook_sft/validate_textbook_sft.py [--rebuild]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "textbook_sft" / "data" / "textbook_mix"
TOKENIZERS = [REPO / "models" / "SinLlama_wiki_uc_gen_mix", REPO / "models" / "SinLlama_cpt"]
TEMPLATE = {"user": "### User:\n", "assistant": "### Assistant:\n", "sep": "\n\n", "eos": "<|end_of_text|>"}
N = 8  # word n-gram size for overlap

SIN_WORD = re.compile(r"[඀-෿‍]+")
CHECKS = {
    "replacement_char": re.compile(r"�"),
    "cid_dump": re.compile(r"\(cid:\d+\)"),
    "private_use": re.compile(r"[-]"),
    "legacy_glyph": re.compile(r"[¡-¶¸-ÖØ-öø-ɏ]"),  # not ·×÷
    "control_char": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    "orphan_sign": re.compile(r"(?:^|[\s(“\"'])[්-ෟ]"),
    "misplaced_zwj": re.compile(r"(?<!්)‍|‍(?![යර])"),
    "kombu_no_consonant": re.compile(r"(?:^|[^ක-ෆ‍්ා-ෟ])ෙ"),
    "raw_script_outside_math": re.compile(r"(?<!\$)[\w)]\^\{|(?<!\$)[\w)]_\{"),
    "stray_backtick": re.compile(r"`"),
    "template_marker": re.compile(r"### (?:User|Assistant|System):|<\|(?:end|begin)_of_text\|>"),
}


def load(name):
    return pq.read_table(DATA / name).to_pylist()


def ngrams(text, n=N):
    w = SIN_WORD.findall(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def unbalanced_math(text):
    inline = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    return inline.count("$") % 2 == 1 or text.count("$$") % 2 == 1


def benchmark_items():
    """(benchmark name, item text) for every benchmark/probe item available locally."""
    items = []
    for f in glob.glob(str(REPO / "data/mmlu-sinhala/*/*.json")):
        split = Path(f).name.split("-")[0]
        if split not in ("test", "validation", "dev"):
            continue
        for q in json.load(open(f)):
            choices = " ".join(str(c) for c in q.get("choices", []))
            items.append((f"mmlu_si_{split}", f"{q.get('question', '')} {choices}"))
    for name, path, col in [("news_test", "data/news/news_test.csv", "comments"),
                            ("sentiment_test", "data/sentiment/sentiment_test.csv", None),
                            ("writing_test", "data/Writing/writing_style_test.csv", None)]:
        p = REPO / path
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                r = csv.DictReader(fh)
                key = col or r.fieldnames[0]
                items += [(name, row.get(key) or "") for row in r]
    for name, path in [("probe_100", "inference/100_prompts.json"),
                       ("probe_benchmark", "inference/sinhala_benchmark_prompts.json")]:
        p = REPO / path
        if p.exists():
            d = json.load(open(p))
            d = d if isinstance(d, list) else d.get("prompts", [])
            items += [(name, x.get("prompt", "") if isinstance(x, dict) else str(x)) for x in d]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-run the builder first")
    args = ap.parse_args()
    if args.rebuild:
        subprocess.run([sys.executable, str(REPO / "textbook_sft/build_textbook_sft_dataset.py")], check=True)

    train, ev_tb, ev_uc = load("train_mix.parquet"), load("eval_textbook.parquet"), load("eval_uc.parquet")
    splits = {"train": train, "eval_textbook": ev_tb, "eval_uc": ev_uc}
    report = {"rows": {k: dict(Counter(r["source"] for r in v)) for k, v in splits.items()}}
    problems = defaultdict(list)

    # ---- structure
    ids = Counter(r["id"] for v in splits.values() for r in v)
    report["duplicate_ids"] = sum(c > 1 for c in ids.values())
    bad_turns = 0
    for name, rows in splits.items():
        for r in rows:
            m = r["messages"]
            roles = [x["role"] for x in m]
            ok = (len(m) >= 2 and roles[0] == "user" and roles[-1] == "assistant"
                  and all(a != b for a, b in zip(roles, roles[1:]))
                  and all(x["content"].strip() for x in m))
            if not ok:
                bad_turns += 1
                problems["bad_turns"].append((name, r["id"]))
    report["bad_turn_structure"] = bad_turns

    # ---- characters + failure modes (per source)
    chars = defaultdict(Counter)
    fail = defaultdict(Counter)
    prompt_answers = defaultdict(set)
    for name, rows in splits.items():
        for r in rows:
            src = r["source"]
            text = "\n".join(x["content"] for x in r["messages"])
            no_math = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", text, flags=re.S)
            for k, rx in CHECKS.items():
                n = len(rx.findall(no_math if k == "raw_script_outside_math" else text))
                if n:
                    chars[src][k] += n
                    chars[src][k + "_rows"] += 1
                    if len(problems[k]) < 5:
                        mm = rx.search(no_math if k == "raw_script_outside_math" else text)
                        src_text = no_math if k == "raw_script_outside_math" else text
                        problems[k].append((r["id"], src_text[max(0, mm.start() - 40):mm.end() + 40]))
            if unbalanced_math(text):
                chars[src]["unbalanced_$_rows"] += 1
            if src != "textbook":
                continue
            q, a = r["messages"][0]["content"], r["messages"][-1]["content"]
            prompt_answers[q].add(a)
            if len(a) < 40:
                fail[src]["answer_under_40_chars"] += 1
            last = a.rstrip().split("\n")[-1]
            if not re.search(r"[.?!:;)”\"'’|$]\s*$", last) and not re.match(r"^(\d+[.)]|[-*]) ", last) \
                    and r["kind"] != "verse":
                fail[src]["no_sentence_end"] += 1
                if len(problems["no_sentence_end"]) < 5:
                    problems["no_sentence_end"].append((r["id"], a[-160:]))
            if q.strip() == a.strip() or a.strip() in q:
                fail[src]["answer_equals_prompt"] += 1
            words = a.split()
            if len(words) > 30 and len(set(words)) / len(words) < 0.25:
                fail[src]["repetition_loop"] += 1
            letters = [c for c in a if c.isalpha()]
            if letters and sum("඀" <= c <= "෿" for c in letters) / len(letters) < 0.5:
                fail[src]["mostly_non_sinhala"] += 1
    multi = {q: a for q, a in prompt_answers.items() if len(a) > 1}
    fail["textbook"]["prompts_with_several_answers"] = len(multi)
    fail["textbook"]["rows_sharing_a_prompt"] = sum(len(a) for a in multi.values())
    report["characters"] = {k: dict(v) for k, v in chars.items()}
    report["failure_modes"] = {k: dict(v) for k, v in fail.items()}
    report["top_shared_prompts"] = sorted(((len(a), q) for q, a in multi.items()), reverse=True)[:8]

    # ---- contamination
    tb_train = [r for r in train if r["source"] == "textbook"]
    uc_train = [r for r in train if r["source"] == "ultrachat"]
    train_text = lambda r: "\n".join(x["content"] for x in r["messages"])
    tb_grams = set().union(*(ngrams(train_text(r)) for r in tb_train))
    all_grams = tb_grams | set().union(*(ngrams(train_text(r)) for r in uc_train))
    tb_answers = {re.sub(r"\W+", "", r["messages"][-1]["content"]) for r in tb_train}

    def overlap(rows, grams):
        hits = 0
        for r in rows:
            g = ngrams(train_text(r))
            if g and len(g & grams) / len(g) >= 0.5:
                hits += 1
        return hits

    contam = {
        "textbook_eval_exact_answer_in_train": sum(
            re.sub(r"\W+", "", r["messages"][-1]["content"]) in tb_answers for r in ev_tb),
        "textbook_eval_>=50%_8gram_in_train": overlap(ev_tb, tb_grams),
        "uc_eval_id_in_train": len({r["id"] for r in ev_uc} & {r["id"] for r in uc_train}),
        "uc_eval_>=50%_8gram_in_train": overlap(ev_uc, all_grams),
    }
    # UltraChat: by first user turn, against replay train and the full source split
    first_q = lambda r: re.sub(r"\W+", "", r["messages"][0]["content"])
    uc_eval_q = {first_q(r) for r in ev_uc}
    contam["uc_eval_prompt_in_replay_train"] = len(uc_eval_q & {first_q(r) for r in uc_train})
    cfg = __import__("yaml").safe_load(open(REPO / "textbook_sft/config.yaml"))["textbook_mix"]
    src = pq.read_table(REPO / cfg["ultrachat_train"], columns=["prompt_id", "messages"])
    src_q, src_ids = set(), set(src.column("prompt_id").to_pylist())
    for m in src.column("messages").to_pylist():
        if m:
            src_q.add(re.sub(r"\W+", "", m[0]["content"]))
    contam["uc_eval_prompt_in_full_uc_train_split"] = len(uc_eval_q & src_q)
    contam["uc_eval_id_in_full_uc_train_split"] = len({r["id"] for r in ev_uc} & src_ids)
    # cross-source
    uc_grams = set().union(*(ngrams(train_text(r)) for r in uc_train))
    contam["textbook_eval_>=50%_8gram_in_uc_train"] = overlap(ev_tb, uc_grams)
    contam["uc_eval_>=50%_8gram_in_textbook_train"] = overlap(ev_uc, tb_grams)
    split_ids = {k: {r["id"] for r in v} for k, v in splits.items()}
    contam["ids_shared_between_splits"] = {
        f"{a}&{b}": len(split_ids[a] & split_ids[b])
        for i, a in enumerate(split_ids) for b in list(split_ids)[i + 1:]}

    bench = defaultdict(Counter)
    examples = []
    for name, text in benchmark_items():
        g = ngrams(text)
        bench[name]["items"] += 1
        if not g:
            bench[name]["too_short_to_check"] += 1
            continue
        frac = len(g & all_grams) / len(g)
        if frac >= 0.5:
            bench[name][">=50%_8grams_in_train"] += 1
            if len(examples) < 6:
                examples.append((name, round(frac, 2), text[:160]))
        elif frac > 0:
            bench[name]["some_8gram_overlap"] += 1
    contam["benchmarks"] = {k: dict(v) for k, v in bench.items()}
    contam["benchmark_examples"] = examples
    report["contamination"] = contam

    # ---- tokens
    tok = AutoTokenizer.from_pretrained(next(p for p in TOKENIZERS if p.exists()))
    def templated(r):
        parts = [(TEMPLATE["user"] if x["role"] == "user" else TEMPLATE["assistant"]) + x["content"]
                 for x in r["messages"]]
        return TEMPLATE["sep"].join(parts) + TEMPLATE["eos"]
    toks = defaultdict(lambda: Counter())
    for name, rows in splits.items():
        texts = [templated(r) for r in rows]
        answers = [r["messages"][-1]["content"] for r in rows]
        full = tok(texts, add_special_tokens=True).input_ids
        ans = tok(answers, add_special_tokens=False).input_ids
        for r, f, a in zip(rows, full, ans):
            key = f"{name}/{r['source']}"
            toks[key]["rows"] += 1
            toks[key]["tokens_templated"] += len(f)
            toks[key]["assistant_tokens_last_turn"] += len(a)
            toks[key]["over_2048"] += len(f) > 2048
    report["tokens"] = {k: dict(v) for k, v in toks.items()}
    report["problem_examples"] = {k: v[:5] for k, v in problems.items()}

    out = DATA / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "problem_examples"}, ensure_ascii=False, indent=1))
    print(f"\nfull report (with examples): {out}")


if __name__ == "__main__":
    main()
