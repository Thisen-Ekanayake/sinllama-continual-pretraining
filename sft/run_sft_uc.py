#!/usr/bin/env python
# coding=utf-8
"""LoRA instruction SFT of SinLlama on UltraChat-Sinhala (multi-turn, sft split).

Everything -- paths, template, LoRA recipe, schedule, early stopping -- comes
from sft/config.yaml:

    python sft/run_sft_uc.py --config sft/config.yaml
    python sft/run_sft_uc.py --config sft/config.yaml --set train.learning_rate=5e-5
    python sft/run_sft_uc.py --config sft/config.yaml --preview 3   # no GPU needed

Forked from run_clm_sft_mmlu_with_peft.py (itself from the upstream
Chinese-LLaMA-Alpaca-3 SFT harness). Differences that matter:

* YAML config instead of ~40 command-line flags.
* Multi-turn chat rendering with offset-based label masking, and a template of
  ordinary trained tokens -- see build_uc_dataset.ChatTemplate.
* `assert_template_tokens_usable`: refuses to train a template whose markers
  are dead weights in this checkpoint (the Llama-3 role tokens are), unless the
  embeddings are being trained.
* Early stopping on eval_loss with load_best_model_at_end.
* The template is written into the run's tokenizer as a `chat_template`, so
  eval and inference reproduce the training format exactly.
* No bitsandbytes path at all: ROCm/MI300X runs bf16.
"""

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import torch
import transformers
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    GenerationConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_uc_dataset import IGNORE_INDEX, ChatTemplate, build_uc_dataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

# Resuming a checkpoint calls torch.load(rng_state.pth, weights_only=True)
# (transformers/trainer.py:3130). torch >= 2.6 refuses the numpy RNG state
# inside it, so every resume dies with an UnpicklingError. Allow-list exactly
# the numpy types HF writes into that file.
try:
    import numpy as _np
    from numpy.core.multiarray import _reconstruct as _np_reconstruct

    _safe = [_np_reconstruct, _np.ndarray, _np.dtype]
    _safe += [getattr(_np.dtypes, n) for n in ("UInt32DType", "Int64DType", "Float64DType")
              if hasattr(_np.dtypes, n)]
    torch.serialization.add_safe_globals(_safe)
except Exception as _e:  # pragma: no cover - best effort, resume still works without it
    logger.debug(f"could not allow-list numpy RNG globals: {_e}")


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Base model directory (e.g. models/SinLlama_v02)."})
    tokenizer_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default="bfloat16")
    attn_implementation: str = field(default="sdpa", metadata={"help": "sdpa | eager | flash_attention_2"})
    low_cpu_mem_usage: bool = field(default=True)


@dataclass
class DataTrainingArguments:
    train_file: str = field(metadata={"help": "UltraChat-Sinhala train parquet."})
    eval_file: Optional[str] = field(default=None)
    eval_samples: Optional[int] = field(default=None, metadata={"help": "Random slice of eval_file to score."})
    eval_seed: int = field(default=42)
    max_seq_length: int = field(default=2048)
    data_cache_dir: Optional[str] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable: str = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank: int = field(default=64)
    lora_alpha: float = field(default=128.0)
    lora_dropout: float = field(default=0.05)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    early_stopping_patience: int = field(default=3)
    early_stopping_threshold: float = field(default=0.0)
    compute_token_accuracy: bool = field(default=True)


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------


def _set_dotted(cfg: Dict[str, Any], dotted: str, raw_value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"--set {dotted}: no config section '{k}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"--set {dotted}: no config key '{keys[-1]}' in section '{'.'.join(keys[:-1])}'")
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str, overrides: List[str]) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set expects key=value, got: {override}")
        key, value = override.split("=", 1)
        _set_dotted(cfg, key.strip(), value.strip())
    return cfg


def resolve_path(path: Optional[str]) -> Optional[str]:
    """Config paths are relative to the repo root, not the caller's cwd."""
    if path is None:
        return None
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else REPO_ROOT / p)


def build_arguments(cfg: Dict[str, Any]):
    model_cfg = dict(cfg["model"])
    data_cfg = dict(cfg["data"])
    train_cfg = dict(cfg["train"])
    lora_cfg = dict(cfg.get("lora") or {})
    es_cfg = dict(cfg.get("early_stopping") or {})
    template = ChatTemplate.from_config(cfg.get("template"))

    flat: Dict[str, Any] = {
        "model_name_or_path": resolve_path(model_cfg["path"]),
        "tokenizer_name_or_path": resolve_path(model_cfg.get("tokenizer_path") or model_cfg["path"]),
        "torch_dtype": model_cfg.get("torch_dtype", "bfloat16"),
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        "low_cpu_mem_usage": model_cfg.get("low_cpu_mem_usage", True),
        "train_file": resolve_path(data_cfg["train_file"]),
        "eval_file": resolve_path(data_cfg.get("eval_file")),
        "eval_samples": data_cfg.get("eval_samples"),
        "eval_seed": data_cfg.get("eval_seed", 42),
        "max_seq_length": data_cfg.get("max_seq_length", 2048),
        "data_cache_dir": resolve_path(data_cfg.get("cache_dir")),
        "preprocessing_num_workers": data_cfg.get("preprocessing_num_workers"),
        "max_train_samples": data_cfg.get("max_train_samples"),
        "max_eval_samples": data_cfg.get("max_eval_samples"),
        "overwrite_cache": data_cfg.get("overwrite_cache", False),
    }

    # LoRA
    flat["trainable"] = ",".join(lora_cfg.get("target_modules", []))
    flat["lora_rank"] = lora_cfg.get("rank", 64)
    flat["lora_alpha"] = float(lora_cfg.get("alpha", 128))
    flat["lora_dropout"] = lora_cfg.get("dropout", 0.05)
    modules_to_save = lora_cfg.get("modules_to_save") or []
    flat["modules_to_save"] = ",".join(modules_to_save) if modules_to_save else None
    flat["peft_path"] = resolve_path(lora_cfg.get("peft_path"))

    # Trainer: the `train` section uses TrainingArguments field names verbatim.
    train_cfg.pop("wandb_project", None)
    train_cfg["output_dir"] = resolve_path(train_cfg["output_dir"])
    flat.update(train_cfg)
    flat.setdefault("logging_strategy", "steps")
    flat.setdefault("logging_first_step", True)
    flat["do_train"] = True
    flat["do_eval"] = flat.get("eval_file") is not None
    flat["ddp_find_unused_parameters"] = False

    # Early stopping needs eval + checkpointing on the same cadence.
    if flat["do_eval"]:
        metric = es_cfg.get("metric", "eval_loss")
        flat["eval_strategy"] = "steps"
        flat["save_strategy"] = "steps"
        flat["load_best_model_at_end"] = True
        flat["metric_for_best_model"] = metric
        flat["greater_is_better"] = not metric.endswith("loss")
        flat["early_stopping_patience"] = es_cfg.get("patience", 3)
        flat["early_stopping_threshold"] = float(es_cfg.get("threshold", 0.0))
        if flat["save_steps"] % flat["eval_steps"] != 0:
            raise ValueError(
                f"train.save_steps ({flat['save_steps']}) must be a multiple of "
                f"train.eval_steps ({flat['eval_steps']}) for load_best_model_at_end"
            )
    else:
        flat["eval_strategy"] = "no"

    es_enabled = bool(es_cfg.get("enabled", True)) and flat["do_eval"]

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(flat, allow_extra_keys=False)
    return model_args, data_args, training_args, template, es_enabled


# --------------------------------------------------------------------------
# Template preflight
# --------------------------------------------------------------------------

# Reserved Llama-3 specials that no sane template emits; used as a reference
# for "this row was never trained".
_UNTRAINED_PROBE_IDS = [128002, 128003, 128004, 128005]


def _read_vocab_rows(model_dir: str, ids: List[int]) -> Optional[Dict[str, torch.Tensor]]:
    """Read embed_tokens / lm_head rows for `ids` straight from the shards.

    Lets the preflight run before (and without) loading the model, so a bad
    template fails in seconds rather than after a multi-minute load.
    """
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover
        return None

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    elif os.path.exists(single_path):
        weight_map = None
    else:
        return None

    names = {"embed": "model.embed_tokens.weight", "head": "lm_head.weight"}
    out: Dict[str, torch.Tensor] = {}
    for key, name in names.items():
        if weight_map is None:
            shard = single_path
        elif name in weight_map:
            shard = os.path.join(model_dir, weight_map[name])
        else:
            continue  # tied lm_head, or an unexpected layout
        with safe_open(shard, framework="pt") as f:
            if name not in f.keys():
                continue
            sliced = f.get_slice(name)
            out[key] = torch.stack([sliced[i : i + 1][0].float() for i in ids])
    return out or None


def assert_template_tokens_usable(
    model_dir: str,
    tokenizer,
    template: ChatTemplate,
    embeddings_trainable: bool,
) -> None:
    """Refuse to train a template made of untrained tokens.

    SinLlama inherits Llama-3's reserved specials as dead weights: embed rows
    128002-128255 are zero vectors (L2 ~1.7e-21, mutually indistinguishable on
    input) and their lm_head rows are cosine 1.000000 with each other, so a
    frozen model can neither read `<|start_header_id|>` nor ever emit
    `<|eot_id|>` over the 253 identical rows beside it. Using the Llama-3 chat
    template here would silently train a model that never stops.
    """
    ids = sorted({
        i
        for text in template.marker_texts()
        for i in tokenizer(text, add_special_tokens=False)["input_ids"]
    })
    if template.add_bos and tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    probe_ids = [i for i in _UNTRAINED_PROBE_IDS if i < len(tokenizer) and i not in ids]

    rows = _read_vocab_rows(model_dir, ids + probe_ids)
    if rows is None:
        logger.warning("could not read embedding shards from %s -- skipping template preflight", model_dir)
        return

    n = len(ids)
    problems: List[str] = []

    embed = rows.get("embed")
    if embed is not None:
        norms = embed[:n].norm(dim=-1)
        for tok_id, norm in zip(ids, norms.tolist()):
            if norm < 1e-3:
                problems.append(
                    f"  input embedding of {tokenizer.convert_ids_to_tokens(tok_id)!r} "
                    f"(id {tok_id}) has L2 {norm:.2e} -- an untrained/zero row"
                )

    head = rows.get("head")
    if head is not None and probe_ids:
        probes = head[n:]
        for tok_id, row in zip(ids, head[:n]):
            cos = torch.nn.functional.cosine_similarity(row.unsqueeze(0), probes, dim=-1).max().item()
            if cos > 0.999:
                problems.append(
                    f"  lm_head row of {tokenizer.convert_ids_to_tokens(tok_id)!r} (id {tok_id}) is "
                    f"cosine {cos:.6f} with the untrained reserved block -- the model could never "
                    f"single it out"
                )

    if not problems:
        logger.info("template preflight: all %d template token ids are trained tokens", len(ids))
        return

    message = (
        "template uses tokens this checkpoint never trained:\n"
        + "\n".join(problems)
        + "\n\nFix by building the template from ordinary tokens (the default `### User:` / "
        "`### Assistant:` markers with `<|end_of_text|>` as the terminator), or by adding "
        "`embed_tokens` and `lm_head` to lora.modules_to_save so those rows can be learned."
    )
    if embeddings_trainable:
        logger.warning("%s\n(continuing: embeddings are in modules_to_save)", message)
    else:
        raise ValueError(message)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def compute_metrics(eval_preds):
    """Next-token accuracy over supervised (assistant) positions only."""
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}
    return {"token_accuracy": float((preds[mask] == labels[mask]).mean())}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------


def run_preview(model_args, data_args, template: ChatTemplate, n: int, embeddings_trainable: bool) -> None:
    """Render a few examples with the supervised span marked, then exit."""
    import pyarrow.parquet as pq

    tokenizer = load_tokenizer(model_args, template)
    assert_template_tokens_usable(
        model_args.model_name_or_path, tokenizer, template, embeddings_trainable
    )

    print("\n=== chat_template (jinja) ===")
    print(template.to_jinja())

    pf = pq.ParquetFile(data_args.train_file)
    batch = next(pf.iter_batches(batch_size=n, columns=["messages"]))
    for k, messages in enumerate(batch.column("messages").to_pylist()):
        text, spans = template.render(messages, bos_token=tokenizer.bos_token or "")
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]

        # Mark the supervised regions the same way the tokenizer sees them.
        marked, cursor = [], 0
        for start, end in spans:
            marked.append(text[cursor:start])
            marked.append("\033[32m" + text[start:end] + "\033[0m")
            cursor = end
        marked.append(text[cursor:])

        n_sup = 0
        si = 0
        for j, (start, end) in enumerate(offsets):
            while si < len(spans) and start >= spans[si][1]:
                si += 1
            if si < len(spans) and start >= spans[si][0] and end <= spans[si][1]:
                n_sup += 1

        print(f"\n=== example {k + 1}: {len(messages)} turns, {len(ids)} tokens, "
              f"{n_sup} supervised ({100 * n_sup / len(ids):.1f}%), "
              f"{'TRUNCATED at a turn boundary' if len(ids) > data_args.max_seq_length else 'fits whole'} "
              f"(max_seq_length {data_args.max_seq_length}) ===")
        print("(green = loss is computed here)\n")
        body = "".join(marked)
        print(body if len(body) < 4000 else body[:2000] + "\n  [... elided ...]\n" + body[-2000:])

    # Round-trip: apply_chat_template must reproduce render() byte for byte.
    messages = batch.column("messages").to_pylist()[0]
    tokenizer.chat_template = template.to_jinja()
    via_template = tokenizer.apply_chat_template(messages, tokenize=False)
    direct, _ = template.render(messages, bos_token=tokenizer.bos_token or "")
    print(f"\napply_chat_template matches the training rendering: {via_template == direct}")
    if via_template != direct:
        raise SystemExit("chat_template and render() disagree -- inference would not match training")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def load_tokenizer(model_args: ModelArguments, template: ChatTemplate):
    # use_fast is mandatory: character offsets drive the label mask, and this
    # tokenizer ships only tokenizer.json (with use_fast=False transformers
    # 4.51.3 silently returns False instead of raising).
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        # A dedicated pad id (config.json's pad_token_id), never the EOS the
        # model is being taught to emit.
        pad_id = 128255 if len(tokenizer) > 128255 else tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)
    tokenizer.chat_template = template.to_jinja()
    return tokenizer


def main():
    import argparse

    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--config", default=str(REPO_ROOT / "sft" / "config.yaml"))
    cli.add_argument("--set", dest="overrides", action="append", default=[],
                     metavar="KEY=VALUE", help="override a config value, e.g. --set train.learning_rate=5e-5")
    cli.add_argument("--preview", type=int, default=0,
                     help="render N examples with the loss mask marked and exit (no GPU)")
    args = cli.parse_args()

    cfg = load_config(args.config, args.overrides)
    model_args, data_args, training_args, template, es_enabled = build_arguments(cfg)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    embeddings_trainable = bool(training_args.modules_to_save)

    if args.preview:
        run_preview(model_args, data_args, template, args.preview, embeddings_trainable)
        return

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}, "
        f"bf16: {training_args.bf16}"
    )
    logger.info(f"config: {args.config}" + (f" (overrides: {args.overrides})" if args.overrides else ""))

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"resuming from checkpoint {last_checkpoint}")

    set_seed(training_args.seed)

    tokenizer = load_tokenizer(model_args, template)
    assert_template_tokens_usable(
        model_args.model_name_or_path, tokenizer, template, embeddings_trainable
    )

    # ---- data ------------------------------------------------------------
    with training_args.main_process_first(desc="tokenizing train split"):
        train_dataset = build_uc_dataset(
            data_file=data_args.train_file,
            tokenizer=tokenizer,
            template=template,
            max_seq_length=data_args.max_seq_length,
            cache_dir=data_args.data_cache_dir,
            preprocessing_num_workers=data_args.preprocessing_num_workers,
            max_samples=data_args.max_train_samples,
            overwrite_cache=data_args.overwrite_cache,
        )
    logger.info(f"num train samples: {len(train_dataset)}")
    logger.info("training example:\n%s", tokenizer.decode(train_dataset[0]["input_ids"]))

    eval_dataset = None
    if training_args.do_eval:
        with training_args.main_process_first(desc="tokenizing eval split"):
            eval_dataset = build_uc_dataset(
                data_file=data_args.eval_file,
                tokenizer=tokenizer,
                template=template,
                max_seq_length=data_args.max_seq_length,
                cache_dir=data_args.data_cache_dir,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                max_samples=data_args.max_eval_samples,
                subsample=data_args.eval_samples,
                subsample_seed=data_args.eval_seed,
                overwrite_cache=data_args.overwrite_cache,
            )
        logger.info(f"num eval samples: {len(eval_dataset)}")

    # ---- model -----------------------------------------------------------
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        attn_implementation=model_args.attn_implementation,
    )
    model.config.use_cache = False

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"model vocab size: {model_vocab_size}, len(tokenizer): {len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"resizing model vocab to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.gradient_checkpointing:
        # bf16 LoRA + gradient checkpointing: the frozen base produces no input
        # grads, so checkpointed activations would have nothing to backprop to.
        model.enable_input_require_grads()

    if training_args.peft_path:
        logger.info(f"loading existing adapter from {training_args.peft_path}")
        model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
    else:
        modules_to_save = training_args.modules_to_save.split(",") if training_args.modules_to_save else None
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=training_args.trainable.split(","),
            inference_mode=False,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            modules_to_save=modules_to_save,
        )
        logger.info(f"LoRA: r={peft_config.r} alpha={peft_config.lora_alpha} "
                    f"targets={peft_config.target_modules} modules_to_save={modules_to_save}")
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- trainer ---------------------------------------------------------
    callbacks = []
    if es_enabled:
        logger.info(
            f"early stopping on {training_args.metric_for_best_model} "
            f"(patience {training_args.early_stopping_patience} evals x {training_args.eval_steps} steps)"
        )
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold,
        ))

    use_accuracy = (
        training_args.do_eval and training_args.compute_token_accuracy and not is_torch_xla_available()
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        compute_metrics=compute_metrics if use_accuracy else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if use_accuracy else None,
        callbacks=callbacks,
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # The merged model should generate with the template's terminator.
    eos_id = tokenizer(template.eos, add_special_tokens=False)["input_ids"][-1]
    GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        max_length=config.max_position_embeddings,
    ).save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        logger.info("*** Evaluate (best checkpoint) ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        state = trainer.state
        logger.info(f"best checkpoint: {state.best_model_checkpoint} "
                    f"({training_args.metric_for_best_model}={state.best_metric})")


if __name__ == "__main__":
    main()
