#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SinLlama / Llama models on HellaSwag — both the ORIGINAL English set
and the Sinhala-translated set — to measure commonsense-completion ability and
its retention/degradation after Sinhala CPT / instruction-tuning.

Method (identical to EleutherAI lm-evaluation-harness `hellaswag`, i.e. how the
Llama-3-8B HellaSwag number on the Open-LLM leaderboard is produced)
------------------------------------------------------------------------------
* HellaSwag is a 4-way *sentence-completion* task, NOT a letter-MCQ. There is no
  "A/B/C/D" token to read. Instead, for each of the 4 candidate endings we score
  the continuation with the model and pick the most likely one.
* Per doc, following lm-eval's `utils.process_docs`:
      ctx    = ctx_a + " " + ctx_b.capitalize()
      query  = preprocess(activity_label + ": " + ctx)
      choices= [preprocess(e) for e in endings]
      gold   = int(label)
  `preprocess` strips the WikiHow "[header]" artifacts exactly as lm-eval does.
* For each choice we compute the summed log-prob of the continuation tokens
  (the continuation string is " " + choice, matching lm-eval's target delimiter)
  conditioned on the query, from a single forward pass.
      acc      : argmax of the raw summed log-prob  == gold
      acc_norm : argmax of (summed log-prob / len(choice_chars)) == gold
  The headline HellaSwag metric is **acc_norm** (length-normalised); we report
  both, as lm-eval does.
* Split: HellaSwag's *test* labels are hidden (blank), so — like every public
  HellaSwag number — we evaluate on the **validation** split (10,042 Qs).
* Default 0-shot (the canonical HellaSwag setting). `--kshot K` with a train
  file prepends K fixed correct-ending exemplars.
* Attention: eager + right-padding. On this ROCm/transformers stack SDPA
  mis-handles padded batches (see mmlu/*); eager is correct and HellaSwag
  sequences are short so the O(seq^2) cost is negligible.

All models are scored **raw** (no chat/Alpaca wrapping) — that is the standard
HellaSwag protocol and keeps the base-vs-CPT-vs-instruct comparison fair.

Output mirrors the MMLU evaluators: per-model `*_results.txt`, `*_results.md`,
`*_metrics.json`, `*_predictions.jsonl` and a combined `results.md`; same
`--bucket` / `--combine-only` GCS plumbing.
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


# ----------------------------------------------------------------------------- #
# lm-eval preprocessing (verbatim from lm_eval/tasks/hellaswag/utils.py)
# ----------------------------------------------------------------------------- #
def preprocess(text):
    text = text.strip()
    # Brackets are artifacts of the WikiHow portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def build_query(row):
    ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    return preprocess(row["activity_label"] + ": " + ctx)


# ----------------------------------------------------------------------------- #
# Data loading — parquet (English original) or jsonl (Sinhala translated). Both
# share the exact HellaSwag schema; only the container differs.
# ----------------------------------------------------------------------------- #
def _resolve_split_file(data_path, split):
    """data_path may be a jsonl file, a parquet file, or a dir holding
    <split>-*.parquet or <split>.sinhala.jsonl."""
    if os.path.isfile(data_path):
        return data_path
    if os.path.isdir(data_path):
        pats = (f"{split}-*.parquet", f"{split}.sinhala.jsonl", f"{split}*.jsonl",
                f"{split}*.parquet")
        for base in (data_path, os.path.join(data_path, "data"),
                     os.path.join(data_path, "translated"),
                     os.path.join(data_path, "data", "translated")):
            for pat in pats:
                cands = sorted(glob.glob(os.path.join(base, pat)))
                if cands:
                    return cands[0]
    raise SystemExit(f"could not find the {split} split under {data_path}")


def _read_any(path):
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def _norm_endings(v):
    return [str(e) for e in list(v)]


def build_records(data_path, split, limit=0, kshot=0, train_path=""):
    """Flat list of docs with query/choices/gold and a few-shot prefix string."""
    df = _read_any(_resolve_split_file(data_path, split))

    prefix = ""
    if kshot > 0:
        tp = train_path or data_path
        tdf = _read_any(_resolve_split_file(tp, "train"))
        shots = []
        for r in tdf.itertuples(index=False):
            row = r._asdict()
            lbl = str(row.get("label", "")).strip()
            if not lbl.isdigit():
                continue
            gi = int(lbl)
            ch = _norm_endings(row["endings"])
            if not (0 <= gi < len(ch)):
                continue
            shots.append(build_query(row) + " " + preprocess(ch[gi]))
            if len(shots) >= kshot:
                break
        prefix = ("\n\n".join(shots) + "\n\n") if shots else ""

    records = []
    for r in df.itertuples(index=False):
        row = r._asdict()
        lbl = str(row.get("label", "")).strip()
        choices = [preprocess(e) for e in _norm_endings(row["endings"])]
        valid = lbl.isdigit() and 0 <= int(lbl) < len(choices) and len(choices) >= 2
        gold = int(lbl) if lbl.isdigit() else -1
        records.append(dict(
            ind=int(row.get("ind", len(records))),
            split_type=str(row.get("split_type", "")),
            query=prefix + build_query(row),
            choices=choices,
            n_choices=len(choices),
            gold=gold,
            valid=bool(valid),
        ))
        if limit and len(records) >= limit:
            break
    return records


# ----------------------------------------------------------------------------- #
# Model / scoring
# ----------------------------------------------------------------------------- #
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"         # loglikelihood scoring reads interior
    tok.truncation_side = "left"       # keep the ending; drop old context if long
    # eager, not sdpa: SDPA mis-handles padded batches on this ROCm stack.
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")
    model.eval()
    return model, tok


def _build_items(tok, records):
    """Expand each valid doc into its per-choice continuation-scoring items.
    Uses the prefix-token trick: enc(query + ' ' + choice) with the query enc as
    its prefix, so the continuation token span is [len(q_enc), len(full))."""
    items = []
    for ri, r in enumerate(records):
        if not r["valid"]:
            continue
        q_enc = tok.encode(r["query"], add_special_tokens=True)
        for ci, choice in enumerate(r["choices"]):
            full = tok.encode(r["query"] + " " + choice, add_special_tokens=True)
            cont_start = len(q_enc)
            # guard against rare boundary re-tokenisation (space merged): fall
            # back to re-encoding the continuation alone if the prefix broke.
            if full[:cont_start] != q_enc or cont_start >= len(full):
                cont = tok.encode(" " + choice, add_special_tokens=False)
                full = q_enc + cont
                cont_start = len(q_enc)
            items.append(dict(
                ri=ri, ci=ci, ids=full, cont_start=cont_start,
                char_len=max(len(choice), 1)))
    return items


@torch.no_grad()
def _score_items(model, tok, items, max_len):
    """Fill each item with its continuation log-prob sum. OOM -> split/shrink."""
    if not items:
        return
    try:
        maxlen = min(max(len(it["ids"]) for it in items), max_len)
        pad_id = tok.pad_token_id
        input_ids, attn = [], []
        for it in items:
            ids = it["ids"][-max_len:] if len(it["ids"]) > max_len else it["ids"]
            it["_len"] = len(ids)
            it["_shift"] = len(it["ids"]) - len(ids)      # left-truncation offset
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        input_ids = torch.tensor(input_ids, device=model.device)
        attn = torch.tensor(attn, device=model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1).cpu()
        for j, it in enumerate(items):
            start = max(it["cont_start"] - it["_shift"], 1)   # >=1 (need prev tok)
            end = it["_len"]
            ll = 0.0
            for p in range(start, end):
                tokid = input_ids[j, p].item()
                ll += logprobs[j, p - 1, tokid].item()
            it["ll"] = ll
        del logits, logprobs, input_ids, attn
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) == 1:
            it = items[0]
            if max_len <= 256:
                it["ll"] = float("-inf")
            else:
                _score_items(model, tok, items, max_len // 2)
        else:
            mid = len(items) // 2
            _score_items(model, tok, items[:mid], max_len)
            _score_items(model, tok, items[mid:], max_len)


@torch.no_grad()
def evaluate(model, tok, records, batch_size, max_len):
    items = _build_items(tok, records)
    # longest-first so a batch's pad target is set by its own max (fewer re-pads)
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["ids"]))
    for s in tqdm(range(0, len(order), batch_size), desc="  eval"):
        chunk = [items[i] for i in order[s:s + batch_size]]
        _score_items(model, tok, chunk, max_len)

    # regroup per doc -> pick acc / acc_norm winners
    by_doc = defaultdict(dict)
    for it in items:
        by_doc[it["ri"]][it["ci"]] = it
    for ri, r in enumerate(records):
        if not r["valid"]:
            continue
        cis = by_doc[ri]
        lls = [cis[c]["ll"] for c in range(r["n_choices"])]
        norm = [cis[c]["ll"] / cis[c]["char_len"] for c in range(r["n_choices"])]
        r["pred"] = int(max(range(r["n_choices"]), key=lambda c: lls[c]))
        r["pred_norm"] = int(max(range(r["n_choices"]), key=lambda c: norm[c]))
        r["correct"] = (r["pred"] == r["gold"])
        r["correct_norm"] = (r["pred_norm"] == r["gold"])
    return records


# ----------------------------------------------------------------------------- #
# Aggregation / reporting
# ----------------------------------------------------------------------------- #
def _acc(records, keyfn, field):
    agg = defaultdict(lambda: [0, 0])
    for r in records:
        if not r.get("valid"):
            continue
        k = keyfn(r)
        agg[k][1] += 1
        agg[k][0] += int(r[field])
    return {k: (c, n, 100.0 * c / n if n else 0.0) for k, (c, n) in agg.items()}


def _overall(records, field):
    ev = [r for r in records if r.get("valid")]
    c = sum(int(r[field]) for r in ev)
    return c, len(ev), (100.0 * c / len(ev) if ev else 0.0)


def compute_metrics(records):
    ca, na, aa = _overall(records, "correct")
    cn, nn, an = _overall(records, "correct_norm")
    return dict(
        overall=dict(correct=ca, total=na, acc=aa,
                     correct_norm=cn, acc_norm=an),
        skipped=sum(1 for r in records if not r.get("valid")),
        by_split_type=dict(
            acc=_acc(records, lambda r: r["split_type"] or "unknown", "correct"),
            acc_norm=_acc(records, lambda r: r["split_type"] or "unknown", "correct_norm"),
        ),
    )


def _fmt_rows(table_norm, table_acc):
    lines = [f"{'':<20}{'acc_norm%':>12}{'acc%':>10}{'total':>10}", "-" * 52]
    for k in sorted(table_norm, key=lambda k: -table_norm[k][2]):
        cn, n, an = table_norm[k]
        aa = table_acc.get(k, (0, 0, 0.0))[2]
        lines.append(f"{str(k):<20}{an:>12.2f}{aa:>10.2f}{n:>10}")
    return "\n".join(lines)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_txt(path, name, m, meta):
    o = m["overall"]
    L = ["=" * 60, f"HellaSwag ({meta['dataset']}) evaluation — {name}", "=" * 60,
         f"\nACC_NORM (headline): {o['acc_norm']:.2f}%  "
         f"({o['correct_norm']}/{o['total']})",
         f"ACC             : {o['acc']:.2f}%  ({o['correct']}/{o['total']})",
         f"skipped={m['skipped']}   {meta['k']}-shot   bf16\n",
         "\n## Accuracy by split_type",
         _fmt_rows(m["by_split_type"]["acc_norm"], m["by_split_type"]["acc"]), ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


def write_md(path, name, m, meta):
    o = m["overall"]
    L = [f"# HellaSwag ({meta['dataset']}) — {name}\n",
         f"**acc_norm: {o['acc_norm']:.2f}%** ({o['correct_norm']}/{o['total']}) · "
         f"acc: {o['acc']:.2f}% · {m['skipped']} skipped · "
         f"{meta['k']}-shot · bf16\n",
         "\n## By split_type\n"]
    tn, ta = m["by_split_type"]["acc_norm"], m["by_split_type"]["acc"]
    rows = [[k, f"{tn[k][2]:.2f}%", f"{ta.get(k,(0,0,0))[2]:.2f}%", tn[k][1]]
            for k in sorted(tn, key=lambda k: -tn[k][2])]
    L.append(md_table(["split_type", "acc_norm", "acc", "total"], rows))
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_results_md(path, all_metrics, meta):
    names = list(all_metrics)
    L = [f"# HellaSwag ({meta['dataset']}) — Evaluation Results\n",
         f"- Benchmark: **HellaSwag** (4-way commonsense sentence-completion, "
         f"validation split, {meta['total']} Qs)",
         f"- Dataset: **{meta['dataset']}**"
         + (" (NLLB-200-3.3B translation of the original)"
            if meta['dataset'] == 'sinhala' else " (original English)"),
         f"- Method: lm-eval protocol — per-ending continuation log-likelihood; "
         f"headline metric **acc_norm** (length-normalised), **{meta['k']}-shot**",
         f"- Precision: **bf16** on {meta['gpu']}; all models scored raw (no template)",
         f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
         "## Overall (acc_norm — headline)\n"]
    L.append(md_table(["Model", "acc_norm", "acc", "Correct(norm)", "Total"],
             [[n, f"{all_metrics[n]['overall']['acc_norm']:.2f}%",
               f"{all_metrics[n]['overall']['acc']:.2f}%",
               all_metrics[n]['overall']['correct_norm'],
               all_metrics[n]['overall']['total']] for n in names]))

    L.append("\n## acc_norm by split_type\n")
    sts = sorted({k for n in names
                  for k in all_metrics[n]["by_split_type"]["acc_norm"]})
    rows = [[st] + [f"{all_metrics[n]['by_split_type']['acc_norm'].get(st,(0,0,0))[2]:.2f}%"
                    for n in names] for st in sts]
    L.append(md_table(["split_type"] + names, rows))
    L.append("\n---\nPer-model detail: each `<model>_results.md` "
             "(+ `*_results.txt` / `*_metrics.json` / `*_predictions.jsonl`).\n")
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
    ap.add_argument("--data-path", required=True,
                    help="validation parquet/jsonl file, or a dir containing it")
    ap.add_argument("--dataset", default="english", choices=["english", "sinhala"],
                    help="label for reporting / output naming")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--train-path", default="",
                    help="train file/dir for few-shot exemplars (kshot>0)")
    ap.add_argument("--models", nargs="+",
                    default=["llama-3-8b", "SinLlama_v01",
                             "SinLlama_cpt_merged", "SinLlama_Backtrianx_instruct"])
    ap.add_argument("--out-dir", default="results_hellaswag")
    ap.add_argument("--kshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #questions (0=all) for a smoke test")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--bucket", default="",
                    help="GCS bucket/prefix; each model's files upload to "
                         "<bucket>/<model_name>/ as soon as it finishes")
    ap.add_argument("--combine-only", action="store_true",
                    help="just rebuild (and upload) the combined results.md")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    if args.combine_only:
        all_metrics, total = {}, 0
        for path in args.models:
            name = os.path.basename(path.rstrip("/"))
            mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
            if os.path.exists(mpath):
                all_metrics[name] = json.load(open(mpath, encoding="utf-8"))
                total = all_metrics[name]["overall"]["total"]
            else:
                print(f"  (combine) missing {mpath} — skipping")
        if not all_metrics:
            raise SystemExit("combine-only: no *_metrics.json found in --out-dir")
        meta = dict(k=args.kshot, gpu=gpu, total=total, dataset=args.dataset)
        rpath = os.path.join(args.out_dir, "results.md")
        write_results_md(rpath, all_metrics, meta)
        if args.bucket:
            gcs_cp([rpath], args.bucket.rstrip("/") + "/")
        print(f"Wrote combined {rpath}")
        return

    records0 = build_records(args.data_path, args.split, args.limit,
                             args.kshot, args.train_path)
    n_valid = sum(r["valid"] for r in records0)
    print(f"[{args.dataset}] Loaded {len(records0)} docs "
          f"({n_valid} valid, {len(records0)-n_valid} skipped) from {args.split}.")

    all_metrics = {}
    for path in args.models:
        name = os.path.basename(path.rstrip("/"))
        mpath = os.path.join(args.out_dir, f"{name}_metrics.json")
        if args.skip_existing and os.path.exists(mpath):
            m = json.load(open(mpath, encoding="utf-8"))
            print(f"\n=== {name}: reusing cached "
                  f"(acc_norm {m['overall']['acc_norm']:.2f}%) ===")
        else:
            print(f"\n=== Evaluating {name} on HellaSwag ({args.dataset}) ===")
            t0 = time.time()
            model, tok = load_model(path)
            records = [dict(r) for r in records0]      # fresh copy per model
            evaluate(model, tok, records, args.batch_size, args.max_len)
            m = compute_metrics(records)
            print(f"  {name}: acc_norm {m['overall']['acc_norm']:.2f}%  "
                  f"acc {m['overall']['acc']:.2f}%  "
                  f"({m['overall']['correct_norm']}/{m['overall']['total']}) "
                  f"in {time.time()-t0:.0f}s")
            json.dump(m, open(mpath, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            with open(os.path.join(args.out_dir, f"{name}_predictions.jsonl"),
                      "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps({k: r.get(k) for k in
                             ("ind", "split_type", "n_choices", "gold", "valid",
                              "pred", "pred_norm", "correct", "correct_norm")},
                             ensure_ascii=False) + "\n")
            del model
            gc.collect(); torch.cuda.empty_cache()

        meta = dict(k=args.kshot, gpu=gpu, total=n_valid, dataset=args.dataset)
        all_metrics[name] = m
        write_txt(os.path.join(args.out_dir, f"{name}_results.txt"), name, m, meta)
        write_md(os.path.join(args.out_dir, f"{name}_results.md"), name, m, meta)
        if args.bucket:
            gcs_cp(per_model_files(args.out_dir, name),
                   args.bucket.rstrip("/") + f"/{name}/")

    meta = dict(k=args.kshot, gpu=gpu, total=n_valid, dataset=args.dataset)
    rpath = os.path.join(args.out_dir, "results.md")
    write_results_md(rpath, all_metrics, meta)
    if args.bucket and len(args.models) > 1:
        gcs_cp([rpath], args.bucket.rstrip("/") + "/")
    print(f"\nWrote results to {args.out_dir}/  (results.md + per-model files)")


if __name__ == "__main__":
    main()
