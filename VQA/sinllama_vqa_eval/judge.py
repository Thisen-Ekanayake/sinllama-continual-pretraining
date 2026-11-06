#!/usr/bin/env python3
"""LLM-as-judge for the SinLlama VQA answers (Qwen3.5-9B Q4_K_M on llama-server, thinking off).

The judge sees, all in English: the image description, the question, the dataset's reference
answer and the candidate answer. It scores four language-comprehension criteria 0-10;
the per-answer score /10 is a fixed weighted mean of them (computed here, not by the judge),
and the dataset score /100 is the mean per-answer score x 10.

  en arm: candidate = SinLlama's English answer           (sinllama_en_generated.json)
  si arm: candidate = NLLB English translation of SinLlama's Sinhala answer
                                                          (sinllama_si_translated_en.json)

  python judge.py --arm en --out judged_en.json
  python judge.py --arm si --out judged_si.json
Resumable through <out>.jsonl. Greedy (temperature 0), JSON-schema-constrained output.
"""
import argparse, json, os, statistics, time
from concurrent.futures import ThreadPoolExecutor

import requests

CRITERIA = [
    {"key": "question_understanding", "weight": 0.25,
     "name": "Question understanding",
     "description": "Does the answer show that the question was understood? It must address the specific "
                    "thing asked (the what / why / how / which, and every part of a multi-part question), "
                    "not a different question, a restatement of the question, or generic text."},
    {"key": "correctness", "weight": 0.35,
     "name": "Correctness and faithfulness",
     "description": "Are the facts and reasoning correct and consistent with the image description and the "
                    "reference answer? Penalise contradictions and details invented beyond the description. "
                    "The reference is one good answer, not the only one: a different conclusion that the "
                    "description clearly supports can still score high."},
    {"key": "completeness", "weight": 0.25,
     "name": "Completeness",
     "description": "Does the answer cover the key points of the reference answer, including the supporting "
                    "evidence or explanation when the question asks for it?"},
    {"key": "coherence", "weight": 0.15,
     "name": "Coherence and clarity",
     "description": "Is the answer logically organised, clear and easy to follow, without rambling, "
                    "repetition or self-contradiction?"},
]
SCALE = ("Scale for every criterion: 10 = fully meets it; 7-9 = minor issues; 4-6 = partly meets it; "
         "1-3 = mostly fails; 0 = missing, irrelevant or empty.")
MT_NOTE = ("The student answer was written in Sinhala and machine-translated into English. Ignore "
           "translation artifacts (awkward wording, odd word choices, grammar slips) unless they make the "
           "meaning impossible to recover; judge the content the original answer expressed.")

PROMPT = """You are an expert examiner grading a student's answer to a reading-comprehension question about an image. You cannot see the image; its description below is the ground truth for what the image shows.

Image description:
{description}

Question:
{question}

Reference answer:
{reference}

Student answer:
{candidate}

Grade the student answer on these criteria, each an integer from 0 to 10:
{criteria}
{scale}
{mt_note}
First write a one- or two-sentence rationale (at most 50 words), then give the four scores. Output JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"},
                   **{c["key"]: {"type": "integer", "minimum": 0, "maximum": 10} for c in CRITERIA}},
    "required": ["rationale"] + [c["key"] for c in CRITERIA],
    "additionalProperties": False,
}


def build_prompt(arm, desc, q, ref, cand):
    crit = "\n".join(f"{i}. {c['key']} ({c['name']}): {c['description']}" for i, c in enumerate(CRITERIA, 1))
    return PROMPT.format(description=desc, question=q, reference=ref, candidate=cand, criteria=crit,
                         scale=SCALE, mt_note=(MT_NOTE + "\n") if arm == "si" else "")


def overall(s):
    return round(sum(c["weight"] * s[c["key"]] for c in CRITERIA), 3)


def call(server, prompt, retries=4):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "temperature": 0.0, "top_k": 1, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "grade", "schema": SCHEMA}}}
    err = None
    for a in range(retries):
        try:
            r = requests.post(server + "/v1/chat/completions", json=body, timeout=300)
            r.raise_for_status()
            j = json.loads(r.json()["choices"][0]["message"]["content"])
            if all(isinstance(j.get(c["key"]), int) and 0 <= j[c["key"]] <= 10 for c in CRITERIA):
                return j, None
            err = f"bad scores: {j}"
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        time.sleep(2 * (a + 1))
    return None, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["en", "si"])
    ap.add_argument("--out")
    ap.add_argument("--dump-criteria", help="write the rubric + judge prompt/settings to this JSON and exit")
    ap.add_argument("--english", default="../vqa_english.json")
    ap.add_argument("--gen-en", default="sinllama_en_generated.json")
    ap.add_argument("--trans-si", default="sinllama_si_translated_en.json")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.dump_criteria:
        json.dump({"judge_model": "unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf (llama.cpp llama-server, thinking off)",
                   "decoding": {"temperature": 0.0, "top_k": 1, "seed": 0, "max_tokens": 400,
                                "output": "JSON schema constrained"},
                   "criteria": CRITERIA, "scale": SCALE,
                   "per_answer_score_10": "sum(weight * criterion score), weights " +
                                          " / ".join(f"{c['key']} {c['weight']}" for c in CRITERIA),
                   "dataset_score_100": "mean per-answer score_10 x 10; empty answers score 0 without a judge call",
                   "arms": {"en": "candidate = SinLlama English answer to the English prompt",
                            "si": "candidate = NLLB-200 3.3B English translation of SinLlama's Sinhala answer "
                                  "to the Sinhala prompt; judge told to ignore MT artifacts"},
                   "judge_inputs": "English image description, English question, dataset English reference answer, candidate",
                   "mt_note_si_arm": MT_NOTE, "prompt_template": PROMPT, "schema": SCHEMA},
                  open(args.dump_criteria, "w"), indent=1)
        return
    if not (args.arm and args.out):
        ap.error("--arm and --out are required")

    eng = {r["image_id"]: r for r in json.load(open(args.english, encoding="utf-8"))}
    items = []
    if args.arm == "en":
        for g in json.load(open(args.gen_en, encoding="utf-8"))["records"]:
            items.append({"image_id": g["image_id"], "candidate": g["generated_answer"]})
    else:
        for t in json.load(open(args.trans_si, encoding="utf-8"))["records"]:
            items.append({"image_id": t["image_id"], "candidate": t["translated_answer_en"],
                          "generated_answer_si": t["generated_answer_si"]})
    if args.limit:
        items = items[:args.limit]

    jsonl = os.path.splitext(args.out)[0] + ".jsonl"
    done = {}
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                if j.get("scores"):
                    done[j["image_id"]] = j
    todo = [it for it in items if it["image_id"] not in done]
    print(f"{args.arm}: {len(items)} items, {len(done)} done, {len(todo)} to judge", flush=True)

    def work(it):
        e = eng[it["image_id"]]
        rec = {"image_id": it["image_id"], "description": e["description"], "question": e["question"],
               "reference_answer": e["answer"]}
        if args.arm == "si":
            rec["generated_answer_si"] = it["generated_answer_si"]
            rec["translated_answer_en"] = it["candidate"]
        else:
            rec["generated_answer_en"] = it["candidate"]
        if not it["candidate"].strip():           # nothing to grade: 0 on every criterion
            rec["scores"] = {c["key"]: 0 for c in CRITERIA}
            rec["rationale"] = "Empty answer."
        else:
            j, err = call(args.server, build_prompt(args.arm, e["description"], e["question"],
                                                    e["answer"], it["candidate"]))
            if j is None:
                rec["scores"], rec["error"] = None, err
                return rec
            rec["scores"] = {c["key"]: j[c["key"]] for c in CRITERIA}
            rec["rationale"] = j["rationale"].strip()
        rec["score_10"] = overall(rec["scores"])
        return rec

    t0, n = time.time(), 0
    with open(jsonl, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for rec in ex.map(work, todo):
            n += 1
            if rec["scores"]:
                done[rec["image_id"]] = rec
            else:
                print(f"  FAILED {rec['image_id']}: {rec['error']}", flush=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{n}/{len(todo)}  {el / 60:.1f} min  ETA {el / n * (len(todo) - n) / 60:.1f} min",
                      flush=True)

    recs = [done[it["image_id"]] for it in items if it["image_id"] in done]
    missing = len(items) - len(recs)
    summary = {
        "arm": args.arm, "n_scored": len(recs), "n_failed": missing,
        "score_100": round(10 * statistics.mean(r["score_10"] for r in recs), 2),
        "criteria_100": {c["key"]: round(10 * statistics.mean(r["scores"][c["key"]] for r in recs), 2)
                         for c in CRITERIA},
        "score_10_distribution": {str(b): sum(int(r["score_10"]) == b for r in recs) for b in range(11)},
        "empty_answers": sum(r.get("rationale") == "Empty answer." for r in recs),
    }
    json.dump({"summary": summary, "records": recs}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    if missing:
        raise SystemExit(f"{missing} items failed; rerun to retry them")


if __name__ == "__main__":
    main()
