import json, numpy as np
from transformers import AutoTokenizer
toks = {k: AutoTokenizer.from_pretrained(f"/home/cse_g2/m2ds_eval/tok/{k}") for k in ("sin", "llama")}
def q(a): a=np.array(a); return f"p50 {int(np.median(a))} p90 {int(np.percentile(a,90))} p99 {int(np.percentile(a,99))} max {a.max()} mean {a.mean():.0f}"
for lang in ("sinhala", "english"):
    for split in ("train", "validation", "test"):
        rows = [json.loads(l) for l in open(f"/home/cse_g2/m2ds/{lang}/{split}.json")]
        print(lang, split, len(rows), "keys", list(rows[0].keys()))
        if split == "validation": continue
        rs = rows if split == "test" else rows[:600]
        ndoc = [r["document"].count("|||||") + 1 for r in rs]
        print("  ndocs", q(ndoc))
        print("  doc words", q([len(r["document"].split()) for r in rs]), " sum words", q([len(r["summary"].split()) for r in rs]))
        for k, t in toks.items():
            print(f"  [{k}] doc tok", q([len(t(r['document'], add_special_tokens=False).input_ids) for r in rs]))
            print(f"  [{k}] sum tok", q([len(t(r['summary'], add_special_tokens=False).input_ids) for r in rs]))
        empty = sum(1 for r in rows if not r["summary"].strip() or not r["document"].strip())
        print("  empty", empty)
r = [json.loads(l) for l in open("/home/cse_g2/m2ds/sinhala/test.json")][3]
print(r["summary"][:800]); print("DOC:", r["document"][:600])
r = [json.loads(l) for l in open("/home/cse_g2/m2ds/english/test.json")][3]
print(r["summary"][:800]); print("DOC:", r["document"][:600])
