#!/usr/bin/env python3
"""Score generated answers with Gemini as an LLM judge.

    python inference/judge_gemini.py                                   # default file + model
    python inference/judge_gemini.py --file inference/100_prompts_SinLlama_wiki_clean.json
    python inference/judge_gemini.py --model gemini-3.5-flash --overwrite

Reads a JSON array of {id, category, prompt, generated_answer, ...} and adds a
`judge` object to every row:

    instruction_following  1-10  does the answer do what the prompt asks, in the
                                 form it asks for (length, format, count, task type)
    relevance              1-10  does the content stay on the prompt's topic
    conciseness            1-10  deductions for content the prompt did not ask for
                                 (tangents, padding, repetition, chat artifacts)
    overall                1-10  the judge's holistic score
    reasoning              str   short justification (English)

The file is rewritten after every row, and rows that already carry a `judge`
from the same model are skipped, so a run cut short by rate limits resumes
where it stopped. The API key is read from GEMINI_API_KEY (environment or the
repo's .env).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

JUDGE_PROMPT = """You are a strict evaluator of Sinhala-language answers produced by a language model.
You are given a user PROMPT (in Sinhala) and the model's ANSWER. Judge only the ANSWER.

Score each criterion as an integer from 1 (very poor) to 10 (excellent):

1. instruction_following -- Does the answer do exactly what the prompt asks, in the
   form it asks for? Check the requested task type (article, explanation, list,
   summary, rewrite, calculation...), any requested length ("short", "brief",
   "2 paragraphs", "5 points"), number of items, and structure. An answer that
   ignores a requested length or format must lose marks here.

2. relevance -- Is the content about the prompt's actual topic and does it answer
   the question asked? Penalise drift to related-but-different subjects.

3. conciseness -- Deduct marks for anything unnecessary: content the prompt did not
   ask for, tangents, padding, repeated sentences or lists, invented extra
   questions, references/citations/image captions/wiki-markup leftovers, chat-template
   artifacts, or text that trails off mid-sentence. A focused answer scores high;
   an answer that is mostly unnecessary material scores low.

overall -- Your holistic 1-10 score for the answer as a response to this prompt,
weighing the three criteria above. Unnecessary content must pull the overall down
even if the useful part is good.

Do not reward length. Do not judge factual accuracy beyond what is needed to decide
relevance. Write `reasoning` in English, 2-4 sentences, naming the main problems.

PROMPT:
{prompt}

ANSWER:
{answer}
"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "instruction_following": {"type": "INTEGER"},
        "relevance": {"type": "INTEGER"},
        "conciseness": {"type": "INTEGER"},
        "overall": {"type": "INTEGER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["instruction_following", "relevance", "conciseness", "overall", "reasoning"],
    "propertyOrdering": ["reasoning", "instruction_following", "relevance", "conciseness", "overall"],
}
SCORE_KEYS = ("instruction_following", "relevance", "conciseness", "overall")


def load_key():
    key = os.environ.get("GEMINI_API_KEY")
    env = REPO / ".env"
    if not key and env.is_file():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (environment or .env)")
    return key


def retry_delay(resp, attempt):
    """Seconds to wait after a 429/5xx: the server's RetryInfo if given, else backoff."""
    try:
        for d in resp.json()["error"].get("details", []):
            if d.get("@type", "").endswith("RetryInfo"):
                return float(d["retryDelay"].rstrip("s")) + 1
    except (ValueError, KeyError):
        pass
    return min(5 * 2 ** attempt, 120)


def judge(session, key, model, row, max_retries):
    body = {
        "contents": [{"role": "user", "parts": [{"text": JUDGE_PROMPT.format(
            prompt=row["prompt"], answer=row["generated_answer"])}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
        },
    }
    for attempt in range(max_retries):
        resp = session.post(API.format(model=model), params={"key": key}, json=body, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            try:
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
                verdict = json.loads(text)
                if all(isinstance(verdict.get(k), int) and 1 <= verdict[k] <= 10 for k in SCORE_KEYS):
                    return verdict
                print(f"  [{row['id']}] out-of-range scores, retrying: {verdict}", file=sys.stderr)
            except (KeyError, IndexError, ValueError) as exc:
                print(f"  [{row['id']}] unparseable response ({exc}), retrying", file=sys.stderr)
            time.sleep(2)
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = retry_delay(resp, attempt)
            print(f"  [{row['id']}] HTTP {resp.status_code}, waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        raise SystemExit(f"HTTP {resp.status_code}: {resp.text[:500]}")
    raise RuntimeError(f"gave up on id {row['id']} after {max_retries} attempts")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="inference/100_prompts_SinLlama_wiki_clean.json")
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--min-interval", type=float, default=4.0,
                   help="minimum seconds between requests (free-tier rate limit)")
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--overwrite", action="store_true", help="re-judge rows that already have a score")
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = REPO / path
    rows = json.loads(path.read_text(encoding="utf-8"))
    key = load_key()
    session = requests.Session()

    todo = [r for r in rows if args.overwrite or r.get("judge", {}).get("model") != args.model]
    print(f"{len(todo)}/{len(rows)} rows to judge with {args.model}", file=sys.stderr)
    last = 0.0
    for row in todo:
        time.sleep(max(0.0, args.min_interval - (time.monotonic() - last)))
        last = time.monotonic()
        verdict = judge(session, key, args.model, row, args.max_retries)
        row["judge"] = {"model": args.model, **{k: verdict[k] for k in SCORE_KEYS},
                        "reasoning": verdict["reasoning"]}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        print(f"  [{row['id']}] overall={verdict['overall']} "
              f"(if={verdict['instruction_following']} rel={verdict['relevance']} "
              f"conc={verdict['conciseness']})", file=sys.stderr)

    judged = [r["judge"] for r in rows if "judge" in r]
    if judged:
        means = {k: sum(j[k] for j in judged) / len(judged) for k in SCORE_KEYS}
        print(f"{len(judged)} judged -- mean " + ", ".join(f"{k}={v:.2f}" for k, v in means.items()),
              file=sys.stderr)


if __name__ == "__main__":
    main()
