#!/usr/bin/env python3
"""Fine-tuned vs base SinLlama on the 500 VQA test images -> ft/summary_ft.json.

Baselines are the existing 0-shot (judged_<arm>.json) and 5-shot (5shot/judged_<arm>_5shot.json)
runs restricted to the same test image_ids. Those used the OLD prompt wording ("image
description" / "රූප විස්තරය"); the fine-tuned run uses the current one. Paired Wilcoxon
signed-rank on the per-answer score /10.
"""
import json, statistics as st

from scipy.stats import wilcoxon

KEYS = ["question_understanding", "correctness", "completeness", "coherence"]
test = set(json.load(open("ft/test_ids.json")))


def recs(path):
    return {r["image_id"]: r for r in json.load(open(path, encoding="utf-8"))["records"]
            if r["image_id"] in test}


def agg(rs):
    rs = list(rs)
    return {"n": len(rs), "score_100": round(10 * st.mean(r["score_10"] for r in rs), 2),
            **{k: round(10 * st.mean(r["scores"][k] for r in rs), 2) for k in KEYS}}


def paired(a, b):
    ids = sorted(a.keys() & b.keys())
    x, y = [a[i]["score_10"] for i in ids], [b[i]["score_10"] for i in ids]
    d = [p - q for p, q in zip(x, y)]
    return {"n": len(ids), "diff_100": round(10 * st.mean(d), 2),
            "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d), "ties": sum(v == 0 for v in d),
            "wilcoxon_p": float(wilcoxon(x, y).pvalue) if any(d) else 1.0}


def gen_stats(path):
    rs = [r for r in json.load(open(path, encoding="utf-8"))["records"] if r["image_id"] in test]
    return {"n": len(rs), "mean_generated_tokens": round(st.mean(r["generated_tokens"] for r in rs), 1),
            "stop_counts": {s: sum(r["stop"] == s for r in rs) for s in ["eos", "stop_string", "max_new_tokens"]}}


s = {"model": "SinLlama_wiki_uc_gen_mix + QLoRA adapter per language (ft_vqa.py)",
     "split": {k: len(v) for k, v in json.load(open("ft/split.json")).items() if k != "seed"},
     "train_dropped_looping_answer": json.load(open("ft/split.json"))["train_dropped_looping_answer"],
     "judge": "Qwen3.5-9B-Q4_K_M (llama.cpp), same rubric as 0-shot (judge_criteria.json)",
     "baseline_note": "0-shot / 5-shot baselines used the old prompt wording; fine-tuned uses the current one"}
for arm in ["en", "si"]:
    ft, z = recs(f"ft/judged_{arm}_ft.json"), recs(f"judged_{arm}.json")
    f5 = recs(f"5shot/judged_{arm}_5shot.json")
    gen = json.load(open(f"ft/ft_{arm}_generated.json", encoding="utf-8"))
    s[arm] = {"finetuned": agg(ft.values()), "zero_shot": agg(z.values()), "five_shot": agg(f5.values()),
              "ft_vs_zero_shot": paired(ft, z), "ft_vs_five_shot": paired(ft, f5),
              "generation_ft": gen_stats(f"ft/ft_{arm}_generated.json"),
              "generation_zero_shot": gen_stats(f"sinllama_{arm}_generated.json"),
              "generation_meta": gen["meta"],
              "training": {k: v for k, v in json.load(open(f"ft/{arm}/train_state.json")).items()
                           if k != "log_history"}}
s["si"]["translation_meta"] = json.load(open("ft/ft_si_translated_en.json", encoding="utf-8"))["meta"]
json.dump(s, open("ft/summary_ft.json", "w"), ensure_ascii=False, indent=1)
for arm in ["en", "si"]:
    a = s[arm]
    print(f"{arm}: ft {a['finetuned']['score_100']} | 0-shot {a['zero_shot']['score_100']} "
          f"(diff {a['ft_vs_zero_shot']['diff_100']}, p={a['ft_vs_zero_shot']['wilcoxon_p']:.2g}) | "
          f"5-shot {a['five_shot']['score_100']} (n={a['five_shot']['n']}, diff {a['ft_vs_five_shot']['diff_100']}, "
          f"p={a['ft_vs_five_shot']['wilcoxon_p']:.2g})")
    print("   ft criteria", {k: a["finetuned"][k] for k in KEYS})
    print("   tokens ft", a["generation_ft"]["mean_generated_tokens"], "0-shot",
          a["generation_zero_shot"]["mean_generated_tokens"], "| ft stops", a["generation_ft"]["stop_counts"])
