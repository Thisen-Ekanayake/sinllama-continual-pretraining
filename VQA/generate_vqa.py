#!/usr/bin/env python3
"""Build a VQA dataset from COCO images with Qwen3.5-9B (vision) served by llama-server.

For every image, two calls (both also given the image's 5 human-written COCO
captions as grounding, to cut down on misread details):
  1. image -> detailed description
  2. image + description -> one high-school level (age 17-18) question and its answer
     (JSON-schema constrained output)

Progress is appended to <out>.jsonl (with generation metadata) so a run can be
resumed; at the end the records are written to <out> as a JSON array with the
fields image_id, description, question, answer.

Usage (on the lab box, with start_server.sh running):
  python3 generate_vqa.py --limit 10 --out vqa_sample10.json
  python3 generate_vqa.py --out vqa_coco_val2017.json
"""
import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DESCRIBE_PROMPT = (
    "Describe this image in detail. Cover the main objects, any people or animals "
    "and what they are doing, the setting, colours, the spatial layout (what is "
    "where), and any visible text. Describe only what is actually visible; do not "
    "invent details or guess at things you cannot see. Write it as one or two "
    "plain paragraphs, without headings or bullet points."
)

# Human-written COCO captions, given as grounding so the model corrects its own
# misreadings (object counts, who is doing what). They are short, so the model
# still has to look at the image for detail.
CAPTIONS_BLOCK = (
    "\n\nFor reference, here are short captions of this image written by human "
    "annotators. They are reliable but brief: use them to check your reading of the "
    "image (what the objects are, how many people, what they are doing), and if "
    "they disagree with your impression, trust the captions. Do not mention the "
    "captions or the annotators in your answer.\n{captions}"
)

QA_PROMPT = """Here is a description of the image:

\"\"\"{description}\"\"\"

{captions_block}Using the image and this description, write ONE question for high-school students aged 17-18, and its answer.

Requirements for the question:
- It must be answerable from what is shown in the image (and stated in the description), not from outside facts about this specific scene.
- It should need some reasoning, not just spotting one object: for example inferring the purpose of something, cause and effect, comparing or relating several objects, estimating quantities or sizes, explaining why something is arranged the way it is, or applying everyday science, safety, or social-studies knowledge to the scene.
- Avoid yes/no questions and questions whose answer is a single word read straight off the description.
- Clear, self-contained, one sentence.

Requirements for the answer:
- Correct and consistent with the image.
- 1-3 sentences: give the answer, then briefly justify it with visible evidence.

Return JSON with the keys "question" and "answer"."""

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 10},
        "answer": {"type": "string", "minLength": 2},
    },
    "required": ["question", "answer"],
    "additionalProperties": False,
}

# Qwen3 non-thinking recommended sampling (temp 0.7, top_p 0.8, top_k 20, min_p 0).
# presence_penalty is kept low for the description (curbs repetition in long
# text) and off for Q&A, where the answer should reuse the description's words.
DESC_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                     presence_penalty=0.5, repeat_penalty=1.05)
QA_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                   presence_penalty=0.0, repeat_penalty=1.0)


def image_data_url(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def chat(server, content, max_tokens, sampling, seed, response_format=None, timeout=300):
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
        **sampling,
    }
    if response_format:
        body["response_format"] = response_format
    r = requests.post(server + "/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    return ch["message"]["content"].strip(), ch["finish_reason"], j.get("usage", {})


def load_captions(path):
    with open(path, encoding="utf-8") as f:
        ann = json.load(f)["annotations"]
    caps = {}
    for a in ann:
        caps.setdefault(a["image_id"], []).append(" ".join(a["caption"].split()))
    return caps


def process(path, args, captions):
    image_id = int(os.path.splitext(os.path.basename(path))[0])
    img = {"type": "image_url", "image_url": {"url": image_data_url(path)}}
    meta = {"file_name": os.path.basename(path)}
    caps = captions.get(image_id, [])
    cap_list = "\n".join(f"- {c}" for c in caps)
    describe_prompt = DESCRIBE_PROMPT + (CAPTIONS_BLOCK.format(captions=cap_list) if caps else "")
    qa_caps = (f"Human-written reference captions of the image (reliable but brief):\n{cap_list}\n\n"
               if caps else "")
    meta["n_captions"] = len(caps)

    # 1. description; retry if it was cut off by max_tokens
    for attempt in range(args.retries):
        desc, fin, usage = chat(args.server, [img, {"type": "text", "text": describe_prompt}],
                                args.max_tokens, DESC_SAMPLING, args.seed + attempt)
        if fin == "stop" and desc:
            break
    meta.update(desc_finish=fin, desc_tokens=usage.get("completion_tokens"),
                prompt_tokens=usage.get("prompt_tokens"), desc_attempts=attempt + 1)

    # 2. question + answer (JSON-schema constrained)
    rf = {"type": "json_schema", "json_schema": {"name": "vqa", "schema": QA_SCHEMA}}
    qa, err = None, None
    for attempt in range(args.retries):
        text, fin, usage = chat(args.server,
                                [img, {"type": "text", "text": QA_PROMPT.format(description=desc, captions_block=qa_caps)}],
                                args.max_tokens, QA_SAMPLING, args.seed + 100 + attempt, rf)
        try:
            cand = json.loads(text)
            if fin == "stop" and cand.get("question", "").strip() and cand.get("answer", "").strip():
                qa = cand
                break
            err = f"finish={fin}"
        except json.JSONDecodeError as e:
            err = f"json: {e}"
    meta.update(qa_finish=fin, qa_tokens=usage.get("completion_tokens"), qa_attempts=attempt + 1)
    if qa is None:
        meta["error"] = err
        qa = {"question": "", "answer": ""}

    return {
        "image_id": image_id,
        "description": desc,
        "question": qa["question"].strip(),
        "answer": qa["answer"].strip(),
        "_meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=os.path.expanduser("~/coco/val2017"))
    ap.add_argument("--captions", default=os.path.expanduser("~/coco/annotations/captions_val2017.json"),
                    help="COCO captions file used as grounding ('' to disable)")
    ap.add_argument("--out", default="vqa_coco_val2017.json")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--start", type=int, default=0, help="index into the sorted image list")
    ap.add_argument("--limit", type=int, default=None, help="number of images (default: all)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (= server slots)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    files = files[args.start:] if args.limit is None else files[args.start:args.start + args.limit]
    paths = [os.path.join(args.images_dir, f) for f in files]

    captions = load_captions(args.captions) if args.captions else {}
    print(f"grounding captions: {'off' if not captions else f'{len(captions)} images'}", flush=True)

    ckpt = os.path.splitext(args.out)[0] + ".jsonl"
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "error" not in rec["_meta"]:
                        done[rec["image_id"]] = rec
    todo = [p for p in paths if int(os.path.splitext(os.path.basename(p))[0]) not in done]
    print(f"{len(paths)} images, {len(done)} already done, {len(todo)} to generate", flush=True)

    lock = threading.Lock()
    t0, n_ok, n_err = time.time(), 0, 0
    with open(ckpt, "a", encoding="utf-8") as fout, ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(process, p, args, captions): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # server/network error: skip, a rerun picks it up
                n_err += 1
                print(f"[error] {futs[fut]}: {e}", file=sys.stderr, flush=True)
                continue
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            if "error" in rec["_meta"]:
                n_err += 1
            else:
                n_ok += 1
                done[rec["image_id"]] = rec
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"{i}/{len(todo)} ok={n_ok} err={n_err} {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)

    wanted = [int(os.path.splitext(f)[0]) for f in files]
    records = [{k: done[i][k] for k in ("image_id", "description", "question", "answer")}
               for i in wanted if i in done]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(records)}/{len(wanted)} records to {args.out}"
          + (f" ({len(wanted) - len(records)} missing, rerun to retry)" if len(records) < len(wanted) else ""))


if __name__ == "__main__":
    main()
