# Stage-2 SFT: a bilingual UltraChat `gen` mix

**Date:** 2026-08-22 · **Base:** `SinLlama_uc_instruct_cleaned` · **Code:** `gen/`
**Data:** `gs://sinllama_cpt/UltraChat_gen_bilingual/`

## Why

Stage 1 (`sft/`) trained `SinLlama_v02` on the cleaned Sinhala `sft` split. It produced a
model that follows instructions and stops — and that answers **English prompts in Sinhala**:

> **What is the capital of Sri Lanka?** → ශ්‍රී ලංකාවේ අගනුවර කොළඹ.

Correct, wrong language. Only 1.1% of assistant turns in that corpus were majority-Latin, so
the model learned "reply in Sinhala" as an unconditional rule
(`docs/uc-instruct-evaluation.md` §5). Any generative English evaluation is dead on arrival
until that is undone.

This stage trains on the **`gen`** split — 255,974 dialogues whose `prompt_id`s are disjoint
from the `sft` split already seen — and swaps a quarter of it to the original English, so the
model gets Sinhala instruction data it has not seen *and* an English replay stream in the same
pass.

## The mix

Take the `prompt_id` intersection of the Sinhala and English `gen` files, sort it, and assign
`idx % 4 == 0` to English, the rest to Sinhala. Exactly 25.0% English, fully deterministic,
and **each `prompt_id` lands in exactly one language** — no dialogue is ever trained on twice
in translation.

The rule keys on `prompt_id`, not row index, and that is load-bearing: only **1 of the 28,300**
shared ids sits at the same position in both files. Splitting on file order would have given an
overlap of whatever the two orderings happened to share, which is exactly the thing the design
is trying to avoid.

## Four things about `gen` that the `sft` pipeline does not handle

Measured against the real files, not assumed.

1. **English `train_gen` is not in the repo.** `UltraChat_Sinhala/english/` has only `test_gen`
   and `test_sft` (and the `.parquet` and `.jsonl` there are the *same* data in two formats,
   not two sets). The Sinhala corpus was translated from `HuggingFaceH4/ultrachat_200k` but
   only the Sinhala side was kept. `gen/fetch_data.sh` pulls the 3 English `train_gen` shards
   from HF on the pod. The committed English `test_gen` is enough to exercise every code path
   locally, which is why the whole builder is testable without a GPU or a download.

2. **`gen` dialogues end on a *user* turn.** Turn counts are odd (3/5/7/9/11/13) and 100% end
   user-side — that is what makes them the *generation* splits. The final user turn has no
   reference answer, so it is dropped; keeping it would trail every example with ~10-20%
   unsupervised tokens. After the drop a `gen` dialogue is structurally identical to an `sft`
   one, and the mean turn count lands at 4.3.

3. **`clean_ultrachat.py` would have discarded the entire corpus.** Its `repair()` returns
   `not_assistant_last` for anything not ending assistant-side — before step 2, that is every
   single `gen` dialogue. The trailing-turn drop has to happen first. Cleaning is genuinely
   needed: the Sinhala `gen` split carries the same JW300 artifacts as `sft` (30.2% of turns
   modified; on the 28,300-dialogue eval pass, `injected_list_word` 77,779 hits,
   `jw300_caption` 37,843).

4. **English must not go through the Sinhala cleaner.** The `RULES` are Sinhala regexes and
   would no-op, but `TIDY`'s `[ \t]{2,} → " "` collapses indentation and would corrupt English
   code blocks and markdown. English gets whitespace-strip and structural repair only.

## Configuration

| | stage 1 (`sft/`) | stage 2 (`gen/`) | why |
|---|---|---|---|
| base | `SinLlama_v02` | `SinLlama_uc_instruct_cleaned` | continues from stage 1 |
| adapter | fresh r64 | fresh r64 | each stage independently attributable |
| data | 205,973 SI `sft` | ~192k SI + ~64k EN `gen` | disjoint ids + English replay |
| epochs | 3 | 2 | already instruction-tuned |
| LR | 1e-4 | 5e-5 | same |
| early stop | `eval_loss` | `eval_si_loss` | see below |
| template | `### User:` / `<\|end_of_text\|>` | **identical** | fingerprint `2be2f6eb46` both |

The template is not re-chosen. `SinLlama_uc_instruct_cleaned` already ships exactly this as its
tokenizer's `chat_template`; changing it would strand everything stage 1 taught.

**Early stopping watches `eval_si_loss`, not English and not a mixed loss.** English starts
from a near-zero baseline — the model has effectively never seen an English response — so its
loss falls monotonically and says nothing about when to stop. The risk this run *introduces* is
the other direction: Sinhala degrading as English competes for LoRA capacity. `eval_si_loss`
detects that; `eval_en_loss` is the recovery curve to watch.

This needed one additive change to `sft/run_sft_uc.py`: `data.eval_file` may now be a mapping
`{name: path}`, which Trainer scores separately and prefixes per set. A string still means one
eval set and one `eval_loss`, so `sft/config.yaml` is untouched.

## Running it

```bash
bash gen/fetch_data.sh                  # HF English + GCS Sinhala + verify
python gen/build_mixed_gen.py           # -> data/gen_mixed/{train_gen_mixed,eval_gen_si,eval_gen_en}.parquet
SMOKE=1 bash gen/run_gen_sft.sh         # 2k dialogues, eval every 20 steps
bash gen/run_gen_sft.sh                 # the real run
bash gen/upload_gcs.sh post-train       # merge + publish
```

`gen/run_gen_sft.sh` does all of it in sequence; the individual steps are for when something
needs re-running. Training is `sft/run_sft_uc.py` **unforked** — it is config-driven and
dataset-agnostic, so this stage needs no copy of it, only `gen/config.yaml`.

Expected cost: ~256k dialogues × ~868 tokens ≈ 222M tokens/epoch, effective batch 64 → ~4,000
steps/epoch. At the stage-1 measured 7.9 s/step that is ~8-9 h per epoch, ~17 h for the 2-epoch
cap before early stopping.

## Verified before the run (CPU only)

- assignment is exactly 25.0% English, and **zero** `prompt_id`s land in both languages
- **zero** `prompt_id` overlap between the train mix and the eval slices
- every emitted dialogue is user-first, assistant-last, strictly alternating, even-length
- the cleaner fires on Sinhala (77,779 `injected_list_word` hits) and **not once** on English
- 73.5% / 72.5% of tokens supervised (SI / EN); 2.8% / 0.5% truncated at a turn boundary
- the last supervised token is `<|end_of_text|>` in 50/50 sampled examples, both languages
- `apply_chat_template` reproduces the training rendering byte-for-byte
- the dead-token preflight still rejects `<|eot_id|>` as a terminator
- `sft/config.yaml` still parses and previews identically after the `eval_file` change

## Open items

- **Not yet run.** No GPU existed when this was written; nothing below the CPU checks above has
  been executed. The pass/fail is generative, not a loss curve: ask the merged model *"What is
  the capital of Sri Lanka?"* and see whether it answers in English and stops.
- **`TIDY` may damage code blocks in Sinhala turns** (`[ \t]{2,} → " "` inside fenced code).
  Pre-existing — it shaped `uc_instruct_cleaned` too — so changing it here would confound this
  stage against the last one. Worth a separate look.
- **`pad_token` is `<|end_of_text|>`** in the merged tokenizer, i.e. identical to the EOS,
  despite `sft/config.yaml`'s note about pinning it to 128255 (`load_tokenizer` only sets a pad
  when one is absent). Benign — `DataCollatorForSeq2Seq` pads labels with `-100` and the
  attention mask zeroes those positions — but it is not what the comment claims.
- **No `chat_models` path in `benchmark/main.yml`**, so neither this model nor stage 1 can be
  scored in its own template. Same gap flagged after stage 1; it is a benchmark-side task.
- **No stage-1 replay.** If `eval_si_loss` rises materially, mixing a slice of
  `train_sft_clean.parquet` back into the mix is the first thing to try.
