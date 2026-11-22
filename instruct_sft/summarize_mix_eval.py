#!/usr/bin/env python3
"""Instruct-mix adapter vs 0-shot vs the VQA-only adapters on the 500 VQA test images -> <dir>/summary.json.
Run from ~/sinllama-continual-pretraining/VQA/sinllama_eval (paths are relative to it). Paired Wilcoxon on score /10.
Note: 0-shot used the OLD prompt wording ("image description"); the VQA-only and instruct-mix runs use the current one.
"""
import argparse, json, statistics as st

from scipy.stats import wilcoxon

KEYS = ["question_understanding", "correctness", "completeness", "coherence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    test = set(json.load(open("ft/test_ids.json")))

    def recs(path):
        return {r["image_id"]: r for r in json.load(open(path, encoding="utf-8"))["records"] if r["image_id"] in test}

    def agg(rs):
        rs = list(rs.values())
        return {"n": len(rs), "score_100": round(10 * st.mean(r["score_10"] for r in rs), 2),
                **{k: round(10 * st.mean(r["scores"][k] for r in rs), 2) for k in KEYS}}

    def paired(a, b):
        ids = sorted(a.keys() & b.keys())
        x, y = [a[i]["score_10"] for i in ids], [b[i]["score_10"] for i in ids]
        d = [p - q for p, q in zip(x, y)]
        return {"n": len(ids), "diff_100": round(10 * st.mean(d), 2), "wins": sum(v > 0 for v in d),
                "losses": sum(v < 0 for v in d), "wilcoxon_p": float(wilcoxon(x, y).pvalue) if any(d) else 1.0}

    out = {}
    for arm in ("en", "si"):
        mix = recs(f"{args.dir}/judged_{arm}.json")
        zero = recs(f"judged_{arm}.json")
        vqa = recs(f"ft/judged_{arm}_ft.json")
        gen = [r for r in json.load(open(f"{args.dir}/{arm}_generated.json", encoding="utf-8"))["records"]
               if r["image_id"] in test]
        out[arm] = {"instruct_mix": agg(mix), "zero_shot": agg(zero), "vqa_only_ft": agg(vqa),
                    "mix_vs_zero_shot": paired(mix, zero), "mix_vs_vqa_only_ft": paired(mix, vqa),
                    "mean_generated_tokens": round(st.mean(r["generated_tokens"] for r in gen), 1),
                    "stop_counts": {s: sum(r["stop"] == s for r in gen) for s in ["eos", "stop_string", "max_new_tokens"]}}
    json.dump(out, open(f"{args.dir}/summary.json", "w"), indent=1)
    for arm, v in out.items():
        print(f"{arm}: mix {v['instruct_mix']['score_100']} | 0-shot {v['zero_shot']['score_100']} | "
              f"VQA-only ft {v['vqa_only_ft']['score_100']} | vs ft {v['mix_vs_vqa_only_ft']}")


if __name__ == "__main__":
    main()
