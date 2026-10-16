#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate free-text answers to SinhalaMMLU questions and save them next to the
correct answer, for reading what a model actually says rather than only which
answer digit it scores highest.

    python benchmark/mmlu/generate_sinhala_mmlu.py \
        --model models/SinLlama_wiki_clean \
        --benchmark-predictions benchmark/mmlu/SinhalaMMLU_results_final/SinLlama_wiki_clean_predictions.jsonl

Questions, few-shot selection and model loading are imported from
evaluate_sinhala_mmlu.py, so item i here is item i of that script's
*_predictions.jsonl (the scored answer is joined in when
--benchmark-predictions is given). Two prompt formats:

* chat (default): zero-shot, the question in the UltraChat template the
  SinLlama SFT models were trained on ("### User: ... ### Assistant:"), with
  NO answer cue -- the model answers in its own words.
* fewshot: the exact 3-shot raw prompt the benchmark scores, ending at
  "පිළිතුර: ". Output is normally just the digit; generation is cut at the
  next blank line (where the model would start inventing another question).

The chosen option is parsed from the text with a heuristic (see parse_answer)
and reported with the rule that matched, so it can be audited; null means no
option could be read off the answer.

The output JSON is rewritten every --save-every batches and on exit; re-running
with the same --out resumes, skipping items already present.
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_sinhala_mmlu import (  # noqa: E402
    CHAT_EOS, block_to_chat, build_examples, clean, load_model, load_template, tqdm,
)

ANSWER_CUE = "පිළිතුර:"


def chat_prompt(test_prompt):
    """Zero-shot chat prompt: the test block as a user turn, assistant slot
    left empty. block_to_chat puts the 'පිළිතුර: ' cue into the assistant slot
    for digit scoring; here it is removed so the model answers freely."""
    test_block = test_prompt.split("\n\n")[-1]
    wrapped = block_to_chat(test_block)
    cue = ANSWER_CUE + " "
    assert wrapped.endswith(cue), wrapped[-40:]
    return wrapped[: -len(cue)]


def trim(text, fmt):
    """Cut the generation where the answer ends and the model starts a new turn
    or a new question."""
    stops = ["### User:", CHAT_EOS] if fmt == "chat" else ["\n\n", CHAT_EOS]
    for s in stops:
        i = text.find(s)
        if i != -1:
            text = text[:i]
    return text.strip()


def parse_answer(text, choices):
    """Best-effort option number from a free-text answer -> (number|None, rule).

    Tried in order: an explicit 'පිළිතුර: N' / 'answer N'; the answer opening
    with the option number ('3', '3.', '(3)', '3)'); exactly one option's text
    appearing verbatim in the answer. A reply that just re-lists the options is
    reported as 'lists_options'. Anything else is left unparsed rather than
    guessed."""
    n = len(choices)
    in_range = lambda d: 1 <= int(d) <= n  # noqa: E731

    m = re.search(r"(?:පිළිතුර|answer)\s*(?:අංක)?\s*[:：\-]?\s*\(?([1-9])\b", text, flags=re.I)
    if m and in_range(m.group(1)):
        return int(m.group(1)), "cue"
    # A reply that re-lists the options ("1. ...\n2. ...") would otherwise be
    # read as choosing option 1 by the leading-number rule below, and would
    # match several options in the option-text rule.
    listed = set(re.findall(r"(?m)^\s*\(?([1-9])[.)]", text))
    if len(listed) >= 2:
        return None, "lists_options"
    m = re.match(r"\s*\(?([1-9])(?:[.)\s]|$)", text)
    if m and in_range(m.group(1)):
        return int(m.group(1)), "leading_number"
    norm = clean(text)
    # Short options ("10", "අ") would match inside unrelated words or numbers
    # in an explanation, so only options of 3+ characters count here.
    hits = [i + 1 for i, c in enumerate(choices) if len(clean(c)) >= 3 and clean(c) in norm]
    if len(hits) == 1:
        return hits[0], "option_text"
    return None, "unparsed"


@torch.no_grad()
def generate_batch(model, tok, prompts, gen_kwargs, max_len):
    """Greedy-generate for a batch; on OOM, split the batch in half and retry."""
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        out = model.generate(**enc, **gen_kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(new, skip_special_tokens=False), [
            int((row != tok.pad_token_id).sum()) for row in new]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise
        mid = len(prompts) // 2
        a_txt, a_len = generate_batch(model, tok, prompts[:mid], gen_kwargs, max_len)
        b_txt, b_len = generate_batch(model, tok, prompts[mid:], gen_kwargs, max_len)
        return a_txt + b_txt, a_len + b_len


def write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def summarise(items):
    done = [it for it in items if it.get("model_answer") is not None]
    parsed = [it for it in done if it["parsed_answer"] is not None]
    s = {
        "generated": len(done),
        "parsed": len(parsed),
        "parsed_correct": sum(it["parsed_correct"] for it in parsed),
    }
    s["parse_rate"] = round(100 * s["parsed"] / len(done), 2) if done else None
    # accuracy over ALL generated items: an unreadable answer counts as wrong
    s["accuracy_over_generated"] = round(100 * s["parsed_correct"] / len(done), 2) if done else None
    agree = [it for it in parsed if it.get("benchmark_pred") is not None]
    if agree:
        s["agrees_with_benchmark_pred"] = round(
            100 * sum(it["parsed_answer"] == it["benchmark_pred"] for it in agree) / len(agree), 2)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model directory")
    ap.add_argument("--name", default=None, help="label in the output (default: dir name)")
    ap.add_argument("--data-root", default="benchmark/mmlu/SinhalaMMLU")
    ap.add_argument("--prompt-file", default="benchmark/prompts/mmlu_sinhala.txt")
    ap.add_argument("--format", choices=["chat", "fewshot"], default="chat")
    ap.add_argument("--kshot", type=int, default=3, help="fewshot format only")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default 256 for chat, 16 for fewshot")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="first N valid items only (0 = all)")
    ap.add_argument("--benchmark-predictions", default=None,
                    help="*_predictions.jsonl from evaluate_sinhala_mmlu.py to join in")
    ap.add_argument("--out", default=None,
                    help="default benchmark/mmlu/SinhalaMMLU_generations/<name>_<format>.json")
    ap.add_argument("--save-every", type=int, default=20, help="batches between saves")
    ap.add_argument("--reparse-only", action="store_true",
                    help="re-run parse_answer over an existing --out file; no model is loaded")
    args = ap.parse_args()

    name = args.name or os.path.basename(os.path.normpath(args.model))
    out = args.out or os.path.join("benchmark/mmlu/SinhalaMMLU_generations",
                                   f"{name}_{args.format}.json")

    if args.reparse_only:
        data = json.load(open(out, encoding="utf-8"))
        for it in data["items"]:
            choices = [it["choices"][str(k)] for k in range(1, len(it["choices"]) + 1)]
            parsed, rule = parse_answer(it["model_answer"], choices)
            it["parsed_answer"], it["parse_rule"] = parsed, rule
            it["parsed_correct"] = parsed == it["correct_answer"] if parsed is not None else None
        data["meta"]["summary"] = summarise(data["items"])
        write_json(out, data)
        print(f"re-parsed {len(data['items'])} item(s) in {out}: {data['meta']['summary']}",
              file=sys.stderr)
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    max_new = args.max_new_tokens or (256 if args.format == "chat" else 16)

    template = load_template(args.prompt_file)
    records, missing = build_examples(args.data_root, template, args.kshot)
    if missing:
        print(f"note: {len(missing)} (difficulty, subject) combos had no few-shot file", file=sys.stderr)

    bench = None
    if args.benchmark_predictions:
        with open(args.benchmark_predictions, encoding="utf-8") as f:
            bench = [json.loads(l) for l in f if l.strip()]
        if len(bench) != len(records) or any(
                b["file"] != r["file"] or b["gold"] != r["gold"] for b, r in zip(bench, records)):
            raise SystemExit(f"{args.benchmark_predictions} does not line up item-for-item "
                             "with this data root -- different dataset or evaluator version")

    idx = [i for i, r in enumerate(records) if r["valid"]]
    if args.limit:
        idx = idx[:args.limit]

    items = {}
    if os.path.exists(out):
        prev = json.load(open(out, encoding="utf-8"))
        if prev["meta"]["format"] != args.format:
            raise SystemExit(f"{out} holds format={prev['meta']['format']!r}; use another --out")
        items = {it["id"]: it for it in prev["items"] if it.get("model_answer") is not None}
        print(f"resuming: {len(items)} item(s) already in {out}", file=sys.stderr)
    todo = [i for i in idx if i not in items]

    prompts = {}
    for i in todo:
        p = records[i]["prompt"]
        prompts[i] = chat_prompt(p) if args.format == "chat" else p

    model, tok = load_model(args.model)
    eos_ids = [tok.convert_tokens_to_ids(CHAT_EOS)]
    if tok.eos_token_id is not None and tok.eos_token_id not in eos_ids:
        eos_ids.append(tok.eos_token_id)
    gen_kwargs = dict(max_new_tokens=max_new, do_sample=False, eos_token_id=eos_ids,
                      pad_token_id=tok.pad_token_id)

    meta = {
        "model": name, "model_path": args.model, "format": args.format,
        "kshot": args.kshot if args.format == "fewshot" else 0,
        "generation": {"max_new_tokens": max_new, "greedy": True},
        "benchmark_predictions": args.benchmark_predictions,
        "note": "parsed_answer is a heuristic read of model_answer (rule in parse_rule); "
                "null = no option could be read off the text",
    }

    def save():
        ordered = [items[i] for i in idx if i in items]
        write_json(out, {"meta": {**meta, "summary": summarise(ordered)}, "items": ordered})

    # Longest prompts first: similar lengths batch together (less padding) and
    # any OOM surfaces at the start instead of hours in.
    todo.sort(key=lambda i: len(prompts[i]), reverse=True)
    t0 = time.time()
    batches = [todo[k:k + args.batch_size] for k in range(0, len(todo), args.batch_size)]
    try:
        for b, batch in enumerate(tqdm(batches, desc="  generate")):
            texts, n_new = generate_batch(model, tok, [prompts[i] for i in batch],
                                          gen_kwargs, args.max_len)
            for i, raw, n in zip(batch, texts, n_new):
                r = records[i]
                answer = trim(raw.replace(tok.pad_token or "", ""), args.format)
                parsed, rule = parse_answer(answer, r["choices"])
                it = {
                    "id": i,
                    "file": r["file"], "difficulty": r["difficulty"],
                    "category": r["category"], "subject": r["subject_label"],
                    "question": clean(r["question"]),
                    "choices": {str(k + 1): clean(c) for k, c in enumerate(r["choices"])},
                    "correct_answer": r["gold"],
                    "correct_answer_text": clean(r["choices"][r["gold"] - 1]),
                    "model_answer": answer,
                    "generated_tokens": n,
                    "parsed_answer": parsed,
                    "parse_rule": rule,
                    "parsed_correct": parsed == r["gold"] if parsed is not None else None,
                }
                if bench is not None:
                    it["benchmark_pred"] = bench[i].get("pred")
                    it["benchmark_correct"] = bench[i].get("correct")
                items[i] = it
            if (b + 1) % args.save_every == 0:
                save()
    finally:
        save()

    s = summarise([items[i] for i in idx if i in items])
    print(f"wrote {s['generated']} item(s) to {out} in {time.time() - t0:.0f}s; "
          f"parsed {s['parse_rate']}%, accuracy over generated {s['accuracy_over_generated']}%",
          file=sys.stderr)


if __name__ == "__main__":
    main()
