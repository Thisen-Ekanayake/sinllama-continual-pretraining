#!/usr/bin/env python3
"""Ad-hoc generation-quality inference for any model in inference/models.yml.

    python inference/run_inference.py --model SinLlama_uc_instruct_cleaned
    python inference/run_inference.py --model SinLlama_v02 --prompt "..." --preset creative
    python inference/run_inference.py --out results.txt   # every model in models.yml

With no --model, every model listed in models.yml is run in turn, one after the
other on the same GPU. Entries whose checkpoint is not on disk are skipped with
a note instead of aborting the sweep, so a partially-synced set of checkpoints
still produces output for the ones that are there.

Always loads bf16 with sdpa attention -- no quantization. The prompt template
(raw / uc_chat / alpaca) is picked automatically per model from
inference/chat_template.jsonl; see that file for what each one renders and why.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {m["name"]: m["path"] for m in cfg["models"]}


def load_templates(path):
    templates = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                templates.append(json.loads(line))
    return templates


def is_present(model_path):
    """A registry entry counts as present only when its directory holds a
    config.json -- an absent or half-synced checkpoint dir is 'not there'."""
    path = Path(model_path)
    return path.is_dir() and (path / "config.json").is_file()


def resolve_model(model_arg, models):
    if model_arg in models:
        return model_arg, models[model_arg]
    path = Path(model_arg)
    if path.is_dir():
        return path.name, str(path)
    raise SystemExit(
        f"--model {model_arg!r} is neither a name in models.yml nor an existing directory"
    )


def select_targets(args, models):
    """(name, path) pairs to run: the one --model asked for, or every entry in
    models.yml that actually has a checkpoint on disk."""
    if args.model is not None:
        name, path = resolve_model(args.model, models)
        if not is_present(path):
            raise SystemExit(f"--model {args.model!r} resolves to {path}, which has no config.json")
        return [(name, path)]

    targets = []
    for name, path in models.items():
        if is_present(path):
            targets.append((name, path))
        else:
            print(f"skipping {name}: no checkpoint at {path}", file=sys.stderr)
    if not targets:
        raise SystemExit(f"no model in {args.models_file} has a checkpoint on disk")
    return targets


def find_template(model_name, templates):
    for tmpl in templates:
        if any(s in model_name for s in tmpl["models"]):
            return tmpl
    for tmpl in templates:
        if tmpl["name"] == "raw":
            return tmpl
    raise SystemExit(f"no template matched {model_name!r} and no 'raw' fallback in the template file")


def load_prompts(args):
    if args.prompt is not None:
        row = {"id": "cli", "prompt": args.prompt}
        if args.system:
            row["system"] = args.system
        return [row]
    rows = []
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.system:
        for row in rows:
            row.setdefault("system", args.system)
    return rows


def render(tmpl, row):
    """Render one prompt row into the exact string handed to the tokenizer,
    ending right where the model should start generating."""
    fmt = tmpl["format"]
    prompt = row["prompt"]

    if fmt == "raw":
        return prompt

    if fmt == "chat":
        system = row.get("system", tmpl.get("system_prompt", ""))
        sep = tmpl["turn_separator"]
        parts = []
        if system:
            parts.append(tmpl["system_prefix"] + system)
        parts.append(tmpl["user_prefix"] + prompt)
        return sep.join(parts) + sep + tmpl["assistant_prefix"]

    if fmt == "alpaca":
        # Ad-hoc prompts have no separate "input" context -- the whole query is
        # the instruction, same shape as a plain Alpaca question with no input.
        return tmpl["template"].format(instruction=prompt, input="", response="")

    raise ValueError(f"unknown template format: {fmt!r}")


def run_model(model_name, model_path, tmpl, prompts, gen_kwargs, args, echo):
    """Generate every prompt with one model and return the rendered blocks.

    The model is dropped and the HIP cache emptied before returning, so a sweep
    over models.yml holds only one checkpoint in VRAM at a time.
    """
    print(f"loading {model_name} from {model_path} (bf16, sdpa, template={tmpl['name']})", file=sys.stderr)
    # Reseed per model so every model sees the same sampling stream and the
    # generations stay comparable across a sweep.
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

    blocks = []
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

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **gen_kwargs,
                    eos_token_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()

            block = (
                f"=== {row.get('id', '?')} ===\n"
                f"model: {model_name}  template: {tmpl['name']}  preset: {args.preset}\n"
                f"--- prompt ---\n{row['prompt']}\n"
                f"--- generated ---\n{generated}\n"
            )
            blocks.append(block)
            if echo:
                print(block)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return blocks


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
    p.add_argument("--prompts", default="inference/prompts.jsonl")
    p.add_argument("--prompt", default=None, help="single ad-hoc prompt; overrides --prompts")
    p.add_argument("--system", default=None, help="system-prompt override (chat-format models only)")
    p.add_argument("--hyperparameters-file", default="inference/hyperparameters.yml")
    p.add_argument("--preset", default="default")
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="write output here (.txt) instead of the terminal")
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

    targets = select_targets(args, models)
    prompts = load_prompts(args)

    out_path = Path(args.out) if args.out else None
    if out_path:
        # Truncate up front, then append per model, so a sweep that dies on the
        # last checkpoint still leaves the earlier generations on disk.
        out_path.write_text("", encoding="utf-8")

    if len(targets) > 1:
        print(
            f"running {len(targets)} model(s): {', '.join(n for n, _ in targets)}",
            file=sys.stderr,
        )

    total = 0
    failed = []
    for model_name, model_path in targets:
        tmpl = find_template(model_name, templates)
        try:
            blocks = run_model(
                model_name, model_path, tmpl, prompts, gen_kwargs, args, echo=out_path is None
            )
        except Exception as exc:  # one bad checkpoint must not kill the sweep
            if len(targets) == 1:
                raise
            print(f"!! {model_name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(model_name)
            continue
        total += len(blocks)
        if out_path:
            with out_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(blocks) + "\n")
            print(f"  {model_name}: {len(blocks)} generation(s) -> {out_path}", file=sys.stderr)

    if out_path:
        print(f"wrote {total} generation(s) to {out_path}", file=sys.stderr)
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
