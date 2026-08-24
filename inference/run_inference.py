#!/usr/bin/env python3
"""Ad-hoc generation-quality inference for any model in inference/models.yml.

    python inference/run_inference.py --model SinLlama_uc_instruct_cleaned
    python inference/run_inference.py --model SinLlama_v02 --prompt "..." --preset creative
    python inference/run_inference.py --model SinLlama_Bactrianx_Instruct --out results.txt

Always loads bf16 with sdpa attention -- no quantization. The prompt template
(raw / uc_chat / alpaca) is picked automatically per model from
inference/chat_template.jsonl; see that file for what each one renders and why.
"""
import argparse
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


def resolve_model(model_arg, models):
    if model_arg in models:
        return model_arg, models[model_arg]
    path = Path(model_arg)
    if path.is_dir():
        return path.name, str(path)
    raise SystemExit(
        f"--model {model_arg!r} is neither a name in models.yml nor an existing directory"
    )


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


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="name from models.yml, or a literal model directory")
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

    model_name, model_path = resolve_model(args.model, models)
    tmpl = find_template(model_name, templates)
    prompts = load_prompts(args)

    print(f"loading {model_name} from {model_path} (bf16, sdpa, template={tmpl['name']})", file=sys.stderr)
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

    out_blocks = []
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
        out_blocks.append(block)
        if args.out is None:
            print(block)

    if args.out:
        Path(args.out).write_text("\n".join(out_blocks), encoding="utf-8")
        print(f"wrote {len(prompts)} generation(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
