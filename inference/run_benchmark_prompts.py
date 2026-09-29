#!/usr/bin/env python3
"""Run the Sinhala benchmark prompt set against the models in inference/models.yml.

    python inference/run_benchmark_prompts.py                        # every model on disk
    python inference/run_benchmark_prompts.py --model SinLlama_wiki
    python inference/run_benchmark_prompts.py --preset greedy --out-dir runs/greedy

Prompts come from inference/sinhala_benchmark_prompts.json (a JSON array of
{id, category, prompt, target_ability, what_to_check}). Model registry, prompt
templates and generation presets are the same files run_inference.py uses, and
the template is picked per model the same way.

One JSON file per model is written to --out-dir/<model>.json, rewritten after
every prompt, so a sweep that dies mid-model keeps what it already generated.
Models whose output file already covers every prompt are skipped unless
--overwrite is given, so re-running after a crash resumes where it stopped.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_inference import find_template, load_models, load_templates, render, select_targets


def load_benchmark(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        row["id"] = str(row["id"])
    return rows


def write_json(path, payload):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def is_complete(path, prompts):
    if not path.is_file():
        return False
    try:
        done = {r["id"] for r in json.loads(path.read_text(encoding="utf-8"))["results"]}
    except (json.JSONDecodeError, KeyError):
        return False
    return all(row["id"] in done for row in prompts)


def run_model(model_name, model_path, tmpl, prompts, gen_kwargs, args, out_file):
    print(f"loading {model_name} from {model_path} (bf16, sdpa, template={tmpl['name']})", file=sys.stderr)
    # Same per-model reseed as run_inference.py so sampled runs stay comparable.
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    eos_id = tokenizer.convert_tokens_to_ids(tmpl["eos"])

    payload = {
        "model": model_name,
        "model_path": str(model_path),
        "template": tmpl["name"],
        "preset": args.preset,
        "generation_kwargs": gen_kwargs,
        "seed": args.seed,
        "results": [],
    }
    try:
        for row in prompts:
            rendered = render(tmpl, row)
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=tmpl.get("add_bos", True),
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            n_in = inputs["input_ids"].shape[-1]

            start = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **gen_kwargs,
                    eos_token_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.perf_counter() - start
            new_ids = output_ids[0][n_in:]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            payload["results"].append({
                "id": row["id"],
                "category": row.get("category"),
                "prompt": row["prompt"],
                "target_ability": row.get("target_ability"),
                "what_to_check": row.get("what_to_check", []),
                "generated": generated,
                "input_tokens": n_in,
                "output_tokens": len(new_ids),
                "hit_eos": bool(len(new_ids) and new_ids[-1].item() == eos_id),
                "seconds": round(elapsed, 2),
            })
            write_json(out_file, payload)
            print(f"  [{model_name}] {row['id']}: {len(new_ids)} tokens in {elapsed:.1f}s", file=sys.stderr)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return payload


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        default=None,
        help="name from models.yml, or a literal model directory; "
             "omit to run every model in models.yml that is on disk",
    )
    p.add_argument("--models-file", default="inference/models.yml")
    p.add_argument("--chat-template-file", default="inference/chat_template.jsonl")
    p.add_argument("--prompts", default="inference/sinhala_benchmark_prompts.json")
    p.add_argument("--hyperparameters-file", default="inference/hyperparameters.yml")
    p.add_argument("--preset", default="default")
    p.add_argument("--max-new-tokens", type=int, default=None, help="override the preset's max_new_tokens")
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="inference/benchmark_outputs")
    p.add_argument("--overwrite", action="store_true", help="re-run models whose output is already complete")
    args = p.parse_args()

    models = load_models(args.models_file)
    templates = load_templates(args.chat_template_file)
    with open(args.hyperparameters_file, encoding="utf-8") as f:
        presets = yaml.safe_load(f)["presets"]
    if args.preset not in presets:
        raise SystemExit(
            f"--preset {args.preset!r} not in {args.hyperparameters_file} (have: {list(presets)})"
        )
    gen_kwargs = dict(presets[args.preset])
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    targets = select_targets(args, models)
    prompts = load_benchmark(args.prompts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{len(prompts)} prompt(s) x {len(targets)} model(s): {', '.join(n for n, _ in targets)}",
        file=sys.stderr,
    )

    failed = []
    for model_name, model_path in targets:
        out_file = out_dir / f"{model_name}.json"
        if not args.overwrite and is_complete(out_file, prompts):
            print(f"skipping {model_name}: {out_file} already complete", file=sys.stderr)
            continue
        tmpl = find_template(model_name, templates)
        try:
            run_model(model_name, model_path, tmpl, prompts, gen_kwargs, args, out_file)
        except Exception as exc:  # one bad checkpoint must not kill the sweep
            if len(targets) == 1:
                raise
            print(f"!! {model_name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(model_name)
            continue
        print(f"  {model_name} -> {out_file}", file=sys.stderr)

    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
