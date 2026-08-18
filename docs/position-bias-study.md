# Answer-position bias in the SinLlama MMLU evaluations

**Date:** 2026-08-18 · **Models:** `SinLlama_v02`, `SinLlama_uc_instruct_cleaned`
**Results:** `results_latest/MMLU_Debiased.pdf`, `benchmark/results/debias_2026-08-18/`,
`gs://sinllama_cpt/debias_20260818/`

## Why we did this

`MMLU_Benchmark.pdf` reported that the UltraChat chat SFT costs **−3.07 pp** of
SinhalaMMLU accuracy (p < 0.0001). MMLU is scored by taking the model's most likely
answer *digit*, which mixes two different things: whether the model knows the answer,
and which option *slot* it likes regardless of the question.

That confound was measurable rather than theoretical. Against a near-uniform gold
distribution (22/24/25/22%), `SinLlama_v02` answered option 3 on 30% of items and
`SinLlama_uc_instruct_cleaned` answered option 1 on 29%. Splitting accuracy by which
option was *correct* made it starker still:

| answer is | v02 | uc_cleaned | diff |
|---|---|---|---|
| option 1 | 36.88% | 46.26% | **+9.38** |
| option 2 | 42.88% | 37.27% | −5.61 |
| option 3 | 49.32% | 40.69% | −8.63 |
| option 4 | 43.60% | 39.38% | −4.22 |
| option 5 | 23.31% | 15.54% | −7.77 |

uc_cleaned was far better on exactly the option it over-predicts and worse on every
other one — the signature of a shifted prior, not of uniformly lost knowledge. So the
headline −3.07 pp might have been a scoring artefact. This study settled it.

## What we ran

Three arms, same models, same items, same run:

| arm | method | cost | assumption |
|---|---|---|---|
| **raw** | standard argmax over option logits | — | (the status quo) |
| **calibrated** | contextual calibration (Zhao et al. 2021): subtract the prior measured from an all-`N/A` content-free prompt | 1 extra pass | the prior is content-independent |
| **permuted** | full cyclic permutation: score each item *n* times with options rotated so every answer text occupies every slot, then average log-probs | *n*× inference | none |

The raw arm is the permutation arm's rotation-0 pass, so it is a **same-run** baseline.
That matters: `SinLlama_cpt` differs by 6–7 pp between two runs of the same model
because of an sdpa/eager attention bug, so cross-run baselines are not trustworthy here.

Two independent methods were run deliberately — agreement would have validated the
cheap one, and disagreement is itself diagnostic.

## Headline result

**The gap is real knowledge loss.** It survives rigorous debiasing.

### SinhalaMMLU (6878 items)

| arm | uc_cleaned | v02 | gap | McNemar |
|---|---|---|---|---|
| raw | 38.78% | 42.03% | +3.26 pp | p=5.4e-08 |
| **permuted** | **40.94%** | **43.85%** | **+2.91 pp** | p=2.0e-07 |
| calibrated | 36.04% | 38.59% | +2.54 pp | p=4.4e-06 |

Position bias accounts for only 0.35 pp of the 3.26 pp raw gap — about **11%**.

### English MMLU (14042 items)

| arm | uc_cleaned | v02 | gap | McNemar |
|---|---|---|---|---|
| raw | 50.36% | 51.20% | +0.84 pp | p=0.050 |
| **permuted** | **53.15%** | **54.50%** | **+1.35 pp** | p=0.00034 |
| calibrated | 49.38% | 51.03% | +1.65 pp | p=2.2e-05 |

## Secondary findings

1. **Both models are underrated by standard scoring.** Raw understates v02 by 1.8 pp
   (Sinhala) / 3.3 pp (English) and uc_cleaned by 2.2 / 2.8 pp. **Reported English
   catastrophic forgetting is ~3 pp too severe** — v02's real English MMLU is 54.50%,
   not the published 51.18%.
2. **Position bias hurts English more than Sinhala** for this line. v02's English
   C-preference was the largest distortion found: TVD 18.5 pp from gold, falling to
   **2.0 pp** after permutation.
3. **Contextual calibration failed and its numbers are discarded.** It made bias worse
   (SinhalaMMLU TVD ~7 → ~34 pp), cost ~3.4 pp accuracy, and agreed with permutation on
   only 45.8%/49.3% of Sinhala items. The all-`N/A` prompt is far out of distribution
   for a 3-shot Sinhala MMLU context, so it does not measure the prior the model
   actually applies. **Use permutation, not calibration, on these models.**

## A prediction that was wrong

Before running this, the accuracy-by-gold-option table above was read as evidence that
the −3.07 pp gap was *largely positional*. It was not: the prior shift is real, but its
net effect is ~11% of the gap, because the gain on option-1 items and the losses
elsewhere nearly cancel. **A visible prior shift does not tell you its magnitude** —
only the debiasing does. The study was worth running precisely because it overturned
the expectation.

The result also strengthens the paper's claim rather than weakening it: "chat SFT costs
~3 pp of Sinhala knowledge" has now been stress-tested against the obvious confound and
held.

## Code

| file | purpose |
|---|---|
| `benchmark/mmlu/permute_eval.py` | cyclic-permutation scoring, both languages |
| `benchmark/mmlu/compare_debias.py` | arm comparison, paired McNemar, TVD, method agreement |
| `benchmark/mmlu/run_debias.sh` | driver for all arms + GCS upload |
| `benchmark/mmlu/setup_debias_pod.sh` | fetch models/datasets from `gs://sinllama_cpt` |
| `benchmark/mmlu/evaluate_sinhala_mmlu.py` | gained `--calibrate` (English already had it) |

Reproduce:

```bash
bash benchmark/mmlu/setup_debias_pod.sh
DRY_RUN=1 bash benchmark/mmlu/run_debias.sh   # inspect the plan
bash benchmark/mmlu/run_debias.sh
```

Runtime on one MI300X: ~13 min per model per language for permutation
(30,039 forward passes on Sinhala, 56,168 on English), ~3 min for calibration.

## Bugs found and fixed along the way

These were all silent-failure classes — wrong numbers, no error:

- **sdpa vs eager (`4e9a013`).** Both evaluators *requested* `eager`, but a request is
  not a guarantee. On this ROCm stack SDPA mishandles left-padding masks in batched
  inference and collapses predictions onto one option without raising. This already cost
  `SinLlama_cpt` 6–7 pp in a published table, its only symptom being a near-absent
  option B (543/14042 against a 24.7% gold rate). Now asserted at model load; the run
  refuses to produce numbers otherwise.
- **Digit-token collision.** `tok.encode(str(d))[0]` returns the *same* id for every
  digit under a SentencePiece tokenizer (the shared leading-space token), making all
  option scores identical. Fine for SinLlama's Llama-3 BPE, but unguarded — found by
  testing with a Llama-2 tokenizer. Now aborts with a clear message; same guard on the
  English letters.
- **Attention-memory batching (`e175116`).** Eager attention allocates `[B, heads, S, S]`,
  so peak memory grows with B×S², not token count. Batching by count alone is unsafe;
  batches are now bounded by a B×S² budget.
- **gsutil discovery (`ce04a25`).** The Cloud SDK has been in a different place on every
  pod and is never on a non-interactive PATH; both scripts now search several locations.

## Known open items

- The first SinhalaMMLU permutation attempt died with **SIGABRT and no Python
  traceback** ~714 s in. The rerun passed the same point with no change that plausibly
  mattered (longest prompt is 842 tokens, batches still 16), so it looks **transient
  rather than fixed**. The completed numbers are sound; recurrence is possible.
- Only `v02` and `uc_instruct_cleaned` were debiased. `llama-3-8b`, `SinLlama_v01`,
  `SinLlama_cpt` and `Bactrianx` still carry raw-only numbers, so cross-model comparisons
  against the debiased pair are not like-for-like. `llama-3-8b` was already near-perfectly
  calibrated (TVD 1.4 pp) and would gain little; `SinLlama_cpt` and `Bactrianx` were the
  *most* biased (33.5 / 38.6 pp) and would gain most.
- The `SinLlama_cpt` row in `MMLU_Benchmark.pdf` is still the sdpa-era figure
  (35.65 / 40.13) rather than the eager re-eval (41.33 / 47.27). cpt is out of scope for
  the paper but is still shown to the supervisor, so it should be corrected there.
