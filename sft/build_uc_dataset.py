"""Tokenize UltraChat-Sinhala multi-turn dialogues for instruction SFT.

Input is the released parquet (schema `{prompt, prompt_id, messages:[{role,
content}]}`); only `messages` is used, since `prompt` merely duplicates
`messages[0].content`. A full scan of `train_sft.parquet` (207,831 dialogues)
confirms every dialogue is strictly user/assistant alternating, user-first,
assistant-last, with no system turns and 4-14 turns.

Rendering and masking
---------------------
A dialogue becomes one training example. The whole conversation is rendered to
a single string by `ChatTemplate`, then tokenized **in one shot** with the fast
tokenizer's character offsets; a token is supervised iff its character span
falls inside an assistant turn's content or its terminating EOS. Tokenizing the
full string (rather than turn by turn, as the old JSONL loader did) means the
token sequence seen in training is byte-for-byte what `apply_chat_template`
produces at inference, with no BPE boundary drift between the two.

Loss therefore covers assistant content + EOS across *all* assistant turns
(78.7% of tokens); BOS, every marker, and every user turn are IGNORE_INDEX.

Truncation
----------
Dialogues longer than `max_seq_length` are cut at a **turn boundary**: the last
assistant turn that fits whole, so a sequence never ends mid-answer. Only when
the very first assistant turn overflows on its own is the content cut
mid-stream, and then its EOS falls outside the window -- deliberately, so the
model is never taught to stop mid-sentence. Dialogues left with no supervised
token are dropped.

The template used by a model is part of that model, so anything that changes
the token stream -- template fields, tokenizer, sequence length -- goes into
the cache fingerprint.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import datasets
import transformers

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


@dataclass
class ChatTemplate:
    """Plain-text chat template built from ordinary, CPT-trained tokens.

    Deliberately *not* the Llama-3 chat template: in SinLlama the Llama-3 role
    tokens (embed rows 128002-128255) are zero vectors with mutually identical
    lm_head rows, so with frozen embeddings the model can neither read them nor
    ever emit `<|eot_id|>`. See `assert_template_tokens_usable` in
    run_sft_uc.py, which refuses to train such a template.
    """

    user_prefix: str = "### User:\n"
    assistant_prefix: str = "### Assistant:\n"
    turn_separator: str = "\n\n"
    eos: str = "<|end_of_text|>"
    system_prompt: str = ""
    system_prefix: str = "### System:\n"
    add_bos: bool = True

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "ChatTemplate":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = dict(cfg or {})
        unknown = set(cfg) - fields
        if unknown:
            raise ValueError(f"unknown template keys in config: {sorted(unknown)}")
        return cls(**cfg)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    # -- rendering ---------------------------------------------------------

    def _chunks(self, messages: Sequence[Dict[str, str]]) -> List[Tuple[str, str, str, bool]]:
        """(prefix, content, suffix, supervised) per turn, in order."""
        out: List[Tuple[str, str, str, bool]] = []
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            out.append((self.system_prefix, msgs[0]["content"], "", False))
            msgs = msgs[1:]
        elif self.system_prompt:
            out.append((self.system_prefix, self.system_prompt, "", False))
        for m in msgs:
            if m["role"] == "assistant":
                out.append((self.assistant_prefix, m["content"], self.eos, True))
            else:
                out.append((self.user_prefix, m["content"], "", False))
        return out

    def render(
        self,
        messages: Sequence[Dict[str, str]],
        bos_token: str = "",
        add_generation_prompt: bool = False,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Render a dialogue; return the text and the supervised char spans."""
        parts: List[str] = []
        spans: List[Tuple[int, int]] = []
        pos = 0

        def add(s: str) -> None:
            nonlocal pos
            if s:
                parts.append(s)
                pos += len(s)

        if self.add_bos and bos_token:
            add(bos_token)
        for i, (prefix, content, suffix, supervised) in enumerate(self._chunks(messages)):
            if i:
                add(self.turn_separator)
            add(prefix)
            start = pos
            add(content)
            add(suffix)
            if supervised:
                spans.append((start, pos))
        if add_generation_prompt:
            add(self.turn_separator)
            add(self.assistant_prefix)
        return "".join(parts), spans

    def to_jinja(self) -> str:
        """The same rendering as a chat_template, for the saved tokenizer.

        Written into the run's tokenizer_config.json so that
        `apply_chat_template` at eval/inference time reproduces training
        exactly -- one source of truth for the format.
        """
        q = json.dumps  # JSON string escaping is valid Jinja string escaping here
        sep, user, asst, sys_p, eos = (
            q(self.turn_separator), q(self.user_prefix), q(self.assistant_prefix),
            q(self.system_prefix), q(self.eos),
        )
        default_system = ""
        if self.system_prompt:
            default_system = (
                "{%- if messages[0]['role'] != 'system' %}"
                f"{{{{ {sys_p} + {q(self.system_prompt)} + {sep} }}}}"
                "{%- endif %}"
            )
        bos = "{{- bos_token }}" if self.add_bos else ""
        return (
            bos
            + default_system
            + "{%- for message in messages %}"
            + "{%- if not loop.first %}" + f"{{{{ {sep} }}}}" + "{%- endif %}"
            + "{%- if message['role'] == 'system' %}"
            + f"{{{{ {sys_p} + message['content'] }}}}"
            + "{%- elif message['role'] == 'user' %}"
            + f"{{{{ {user} + message['content'] }}}}"
            + "{%- else %}"
            + f"{{{{ {asst} + message['content'] + {eos} }}}}"
            + "{%- endif %}"
            + "{%- endfor %}"
            + "{%- if add_generation_prompt %}"
            + f"{{{{ {sep} + {asst} }}}}"
            + "{%- endif %}"
        )

    def marker_texts(self) -> List[str]:
        """Every literal the template can emit (used by the preflight check)."""
        out = [self.user_prefix, self.assistant_prefix, self.turn_separator, self.eos]
        if self.system_prompt:
            out.append(self.system_prefix)
        return [t for t in out if t]


# -- tokenization -----------------------------------------------------------


def _encode_batch(
    examples: Dict[str, List[Any]],
    tokenizer: transformers.PreTrainedTokenizerFast,
    template: ChatTemplate,
    max_seq_length: int,
) -> Dict[str, List[Any]]:
    texts: List[str] = []
    spans_per_example: List[List[Tuple[int, int]]] = []
    for messages in examples["messages"]:
        text, spans = template.render(messages, bos_token=tokenizer.bos_token or "")
        texts.append(text)
        spans_per_example.append(spans)

    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)

    all_input_ids, all_labels, all_truncated = [], [], []
    all_n_tokens, all_n_supervised = [], []
    for ids, offsets, spans in zip(enc["input_ids"], enc["offset_mapping"], spans_per_example):
        labels = [IGNORE_INDEX] * len(ids)
        turn_end_tokens: List[int] = []  # token count at the end of each assistant turn
        si = 0
        for j, (start, end) in enumerate(offsets):
            while si < len(spans) and start >= spans[si][1]:
                turn_end_tokens.append(j)
                si += 1
            if si < len(spans) and start >= spans[si][0] and end <= spans[si][1]:
                labels[j] = ids[j]
        turn_end_tokens.extend([len(ids)] * (len(spans) - si))

        truncated = len(ids) > max_seq_length
        if truncated:
            fits = [t for t in turn_end_tokens if t <= max_seq_length]
            cut = max(fits) if fits else max_seq_length
            ids, labels = ids[:cut], labels[:cut]

        all_input_ids.append(ids)
        all_labels.append(labels)
        all_truncated.append(truncated)
        # Counters, so the caller can report corpus stats without materializing
        # every sequence back into Python.
        all_n_tokens.append(len(ids))
        all_n_supervised.append(sum(1 for l in labels if l != IGNORE_INDEX))

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "truncated": all_truncated,
        "n_tokens": all_n_tokens,
        "n_supervised": all_n_supervised,
    }


def build_uc_dataset(
    data_file: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    template: ChatTemplate,
    max_seq_length: int,
    cache_dir: Optional[str] = None,
    preprocessing_num_workers: Optional[int] = None,
    max_samples: Optional[int] = None,
    subsample: Optional[int] = None,
    subsample_seed: int = 42,
    overwrite_cache: bool = False,
) -> datasets.Dataset:
    """Load one UltraChat-Sinhala parquet split and tokenize it for SFT.

    `subsample` draws a fixed random slice (used to keep the eval split cheap);
    `max_samples` takes a deterministic head (used for smoke runs). Both are
    applied before tokenization and both are part of the cache key.
    """
    if not tokenizer.is_fast:
        raise ValueError(
            "a fast tokenizer is required (character offsets drive the label mask); "
            "load with AutoTokenizer.from_pretrained(..., use_fast=True)"
        )

    stem = os.path.basename(data_file).split(".")[0]
    key = f"{stem}_{template.fingerprint()}_len{max_seq_length}"
    if subsample:
        key += f"_sub{subsample}s{subsample_seed}"
    if max_samples:
        key += f"_head{max_samples}"
    cache_root = cache_dir or os.path.join(os.path.dirname(data_file), "sft_cache")
    cache_path = os.path.join(cache_root, key)

    if os.path.isdir(cache_path) and not overwrite_cache:
        logger.info(f"loading tokenized dataset from cache: {cache_path}")
        processed = datasets.load_from_disk(cache_path)
        processed.set_format("torch")
        return processed

    logger.info(f"tokenizing {data_file} (template {template.fingerprint()}, max_seq_length {max_seq_length})")
    raw = datasets.load_dataset(
        "parquet",
        data_files={"data": data_file},
        cache_dir=os.path.join(cache_root, "raw"),
    )["data"]
    n_raw = len(raw)
    if subsample and subsample < len(raw):
        raw = raw.shuffle(seed=subsample_seed).select(range(subsample))
    if max_samples and max_samples < len(raw):
        raw = raw.select(range(max_samples))

    tokenized = raw.map(
        _encode_batch,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=raw.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "template": template,
            "max_seq_length": max_seq_length,
        },
        desc="rendering + tokenizing dialogues",
    )
    n_tokenized = len(tokenized)
    kept = tokenized.filter(
        lambda batch: [n > 0 for n in batch["n_supervised"]],
        batched=True,
        num_proc=preprocessing_num_workers,
        desc="dropping dialogues with no supervised token",
    )

    # datasets >= 4 returns a lazy `Column` from ds[name]; 2.x returns a list.
    # Both iterate, so sum() works on either without materializing an array —
    # and these are the three int/bool counter columns, never the sequences.
    n_truncated = sum(1 for t in kept["truncated"] if t)
    total = sum(kept["n_tokens"])
    supervised = sum(kept["n_supervised"])
    logger.info(
        f"{stem}: {n_raw} dialogues in file -> {n_tokenized} used -> {len(kept)} kept "
        f"({n_tokenized - len(kept)} had no supervised token); "
        f"{n_truncated} truncated to a turn boundary ({100 * n_truncated / max(n_tokenized, 1):.1f}%); "
        f"{total / 1e6:.1f}M tokens, {100 * supervised / max(total, 1):.1f}% supervised; "
        f"mean length {total / max(len(kept), 1):.0f}"
    )

    processed = kept.remove_columns(["truncated", "n_tokens", "n_supervised"])
    os.makedirs(cache_root, exist_ok=True)
    processed.save_to_disk(cache_path)
    logger.info(f"cached tokenized dataset at {cache_path}")
    processed.set_format("torch")
    return processed
