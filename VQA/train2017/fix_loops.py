#!/usr/bin/env python3
"""Repair the repetition loops in the COCO train2017 VQA set (docs: nllb-gaps-research/coco-train2017-loop-issue.md).

Run on the lab box in ~/coco_train (fix_loops.sh drives the steps):

  python3 fix_loops.py select            # loops_100k.jsonl -> fix/ids.json (Qwen-loop vs NLLB-only ids)
  python3 fix_loops.py regen             # llama-server up: regenerate the Qwen-loop records -> fix/en_regen.jsonl
  python3 fix_loops.py en                # fix/en_fix.json = regenerated English + original English of NLLB-only ids
  (translate_nllb.py -> repair_nllb.py --long-loops -> postprocess_si.py on fix/en_fix.json -> fix/si_fix.json)
  python3 fix_loops.py merge             # -> chunks/vqa_train2017_{english,sinhala}_100k_fixed.json + fix/report.json

Qwen loops (the English field loops, NLLB copied it): the whole record (description + question + answer) is
regenerated with the same prompts and captions, but the question/answer call gets the description call's
repetition penalties (presence 0.5, repeat 1.05) instead of none, and a new seed per attempt, until no field
loops (max 6 attempts). NLLB-only loops: the English is kept and re-translated; repair_nllb.py --long-loops
flags the looping sentences so they go through its retry/MBR repair.
The original chunk files are not modified.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scan_loops import FIELDS, loop_info  # noqa: E402

CHUNKS = os.path.join(HERE, "chunks")
FIX = os.path.join(HERE, "fix")
NUMS = [f"{i:02d}" for i in range(10)]          # chunks 00-09 = the 100k release


def load_chunks(lang):
    recs = []
    for x in NUMS:
        recs += json.load(open(f"{CHUNKS}/{lang}_{x}.json", encoding="utf-8"))
    return recs


def record_loops(rec):
    return {k: loop_info(rec[k]) for k in FIELDS if loop_info(rec[k])}


def cmd_select(args):
    en, si = load_chunks("en"), load_chunks("si")
    assert [r["image_id"] for r in en] == [r["image_id"] for r in si]
    qwen, nllb = [], []
    for e, s in zip(en, si):
        le, ls = record_loops(e), record_loops(s)
        if le:
            qwen.append(e["image_id"])
        elif ls:
            nllb.append(e["image_id"])
    os.makedirs(FIX, exist_ok=True)
    json.dump({"qwen": qwen, "nllb": nllb}, open(f"{FIX}/ids.json", "w"))
    print(f"{len(en)} records: {len(qwen)} Qwen-loop (English loops), {len(nllb)} NLLB-only loops")


def cmd_regen(args):
    import generate_vqa as G
    G.QA_SAMPLING = dict(G.QA_SAMPLING, presence_penalty=0.5, repeat_penalty=1.05)
    ids = json.load(open(f"{FIX}/ids.json"))["qwen"]
    out = f"{FIX}/en_regen.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["image_id"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} Qwen-loop records, {len(done)} done, {len(todo)} to regenerate", flush=True)
    caps = G.load_captions(args.captions)

    def one(image_id):
        path = os.path.join(args.images_dir, f"{image_id:012d}.jpg")
        for attempt in range(6):
            a = argparse.Namespace(server=args.server, max_tokens=512, retries=3, seed=50000 + 977 * attempt)
            rec = G.process(path, a, caps)
            bad = record_loops(rec)
            if "error" not in rec["_meta"] and not bad:
                rec["_meta"].update(fix_attempt=attempt + 1, qa_sampling="presence 0.5 / repeat 1.05")
                return rec
        rec["_meta"].update(fix_attempt=6, still_looping=bad)
        return rec

    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(4) as pool:
        for n, rec in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}", flush=True)


def cmd_en(args):
    ids = json.load(open(f"{FIX}/ids.json"))
    regen = {}
    for l in open(f"{FIX}/en_regen.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            regen[r["image_id"]] = r
    still = [i for i, r in regen.items() if r["_meta"].get("still_looping") or "error" in r["_meta"]]
    by_id = {r["image_id"]: r for r in load_chunks("en")}
    keys = ("image_id", "description", "question", "answer")
    recs = [{k: regen[i][k] for k in keys} for i in ids["qwen"] if i in regen]
    recs += [{k: by_id[i][k] for k in keys} for i in ids["nllb"]]
    json.dump(recs, open(f"{FIX}/en_fix.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"en_fix.json: {len(recs)} records ({len(regen)} regenerated, {len(still)} still looping/error: {still})")


def cmd_merge(args):
    ids = json.load(open(f"{FIX}/ids.json"))
    en_fix = {r["image_id"]: r for r in json.load(open(f"{FIX}/en_fix.json", encoding="utf-8"))}
    si_fix = {r["image_id"]: r for r in json.load(open(f"{FIX}/si_fix.json", encoding="utf-8"))}
    en, si = load_chunks("en"), load_chunks("si")
    report = {"qwen_ids": len(ids["qwen"]), "nllb_ids": len(ids["nllb"]), "still_looping": [], "dropped": []}
    out_en, out_si = [], []
    for e, s in zip(en, si):
        i = e["image_id"]
        if i in ids["qwen"] or i in ids["nllb"]:
            if i not in en_fix or i not in si_fix:
                report["dropped"].append(i)
                continue
            e, s = en_fix[i], si_fix[i]
            le, ls = record_loops(e), record_loops(s)
            if le or ls:
                report["still_looping"].append({"image_id": i, "en": le, "si": ls})
        out_en.append(e)
        out_si.append(s)
    # final rescan over everything
    report["final_loop_records"] = sum(bool(record_loops(e) or record_loops(s)) for e, s in zip(out_en, out_si))
    report["records"] = len(out_en)
    for lang, recs in (("english", out_en), ("sinhala", out_si)):
        p = f"{CHUNKS}/vqa_train2017_{lang}_100k_fixed.json"
        json.dump(recs, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("wrote", p, len(recs))
    json.dump(report, open(f"{FIX}/report.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "regen", "en", "merge"])
    ap.add_argument("--images-dir", default=os.path.expanduser("~/coco/train2017"))
    ap.add_argument("--captions", default=os.path.expanduser("~/coco/annotations/captions_train2017.json"))
    ap.add_argument("--server", default="http://localhost:8080")
    args = ap.parse_args()
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
