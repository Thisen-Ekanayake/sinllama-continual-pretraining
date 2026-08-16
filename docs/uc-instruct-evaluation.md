# SinLlama_uc_instruct: what the training produced

Evaluation record for the UltraChat-Sinhala SFT, 2026-08-16.
`models/SinLlama_uc_instruct` = `SinLlama_v02` + the LoRA from `sft/`, merged.

**Verdict: the pipeline worked and the template bet paid off. The model's
remaining defects are inherited from the corpus, not produced by the run.**

| | |
|---|---|
| stops at `<|end_of_text|>` | 11 / 12 prompts, **zero** `### User:` leakage |
| P(EOS) at real turn ends | mean **0.847**, median 0.957, rank-1 at 95% of turns |
| best `eval_loss` | **1.5403** (step 6300), token acc 0.6400, ppl 4.68 |
| wall clock | 16.6 h, stopped by early stopping at epoch 2.22 of 3 |
| blocking defect | ~99% of list items in the training data are corrupted |

---

## 1. The run

Early stopping fired at step 7,200 — three evals past the step-6300 minimum,
exactly as configured (`patience: 3` on `eval_loss`).

```
step   300  eval_loss 1.9464  acc 0.5774
step  3000  eval_loss 1.6141  acc 0.6278
step  6000  eval_loss 1.5451  acc 0.6393
step  6300  eval_loss 1.5403  acc 0.6400   <- best
step  6600  eval_loss 1.5528  acc 0.6394
step  6900  eval_loss 1.5536  acc 0.6398
step  7200  eval_loss 1.5523  acc 0.6401   <- stopped
```

24 evals, monotone descent to 6300, then a clean plateau. Not a premature stop.

| | |
|---|---|
| `train_runtime` | 59,606 s (16.6 h) |
| steps | 7,200 of a 9,732 cap (epoch 2.219) |
| step time | 8.28 s incl. eval; ~8.02 s net of the 24 × 77 s evals |
| `train_samples` | 207,648 (183 of 207,831 dropped by the builder) |
| `total_flos` | 2.446e19 |
| `train_loss` | 1.5775 |
| adapter | 661 MB |

Step time landed within 1.5% of the 7.90 s/step projected in
[mi300x-performance.md](mi300x-performance.md), so that tuning work held up
under a real 16-hour run.

Note `eval_results.json` reports 1.5436 for the final re-evaluation of the
loaded best model, against the 1.5403 recorded at step 6300. The 0.2% gap is
not investigated; most likely batch composition or bf16 reduction order. It does
not affect checkpoint selection, which used the step-6300 value.

## 2. Merge verification

`sft/runs/sft_uc_lora/early_stop/adapter_model.safetensors` is sha256-identical
to `checkpoint-6300/`, confirming `load_best_model_at_end` reloaded the best
weights before the final save. The merge was then checked numerically rather
than assumed: reconstruct `B @ A × (alpha/r) = B @ A × 2.0` from the adapter and
compare against `merged − v02`.

| tensor | ‖ΔW‖/‖W‖ | vs adapter |
|---|---|---|
| `layers.0.self_attn.q_proj` | 22.32% | matches |
| `layers.15.self_attn.v_proj` | 10.86% | matches |
| `layers.31.mlp.down_proj` | 6.61% | matches |
| `embed_tokens` | 0 | bit-identical (frozen) |
| `lm_head` | 0 | bit-identical (frozen) |

The residual on `down_proj` is 2.5%, which is bf16 storage rounding, not a
mismatch: the merged model is saved in bf16, so recovering ΔW as a difference of
two bf16 tensors carries relative error `≈ 2^-9 · ‖W‖/‖ΔW‖` — 2.9% predicted at
`down_proj`'s 6.61%, 0.9% at `q_proj`'s 22.3%. Both observed values sit right on
that curve.

The frozen embeddings matter: they are what the whole template decision rested
on, and they are provably untouched.

ΔW is front-loaded (22% at layer 0, 7% at layer 31). Superficially that is the
same shape as the over-trained bactrianx model, but the cause is different —
207k dialogues over 2.2 epochs at LR 1e-4, against bactrianx's 7,238 examples
re-run 4 times at 2e-4.

## 3. The template bet, settled

The run used `### User:` / `### Assistant:` + `<|end_of_text|>` instead of the
Llama-3 chat template, because in SinLlama the Llama-3 role tokens are dead
weights: `embed_tokens` rows 128002–128255 have L2 ≈ 1.7e-21 and their `lm_head`
rows are mutually cosine-1.000000. With frozen embeddings a model trained on
that template could neither read its role markers nor learn to stop.

That prediction is now confirmed in both directions.

**Free generation** — 12 prompts (6 Sinhala, 3 English, 1 multi-turn, 2 sampled),
greedy unless noted:

- **11 / 12 terminated at `<|end_of_text|>`.** The exception hit the 256-token
  cap.
- **Zero** occurrences of `### User:`, `### Assistant:`, or `### System:` in any
  output. The model never fabricates the next turn.
- Multi-turn follow-up works: given its own prior answer listing tea, rubber and
  coconut, it correctly picked tea and justified it.

**Teacher-forced**, the diagnostic that exposed bactrianx — P(EOS) at the true
end of 57 real assistant turns from `test_sft`:

```
P(EOS)  mean 0.8472   median 0.9569   min 9.10e-02
rank    median 1      worst 4         top-1 at 95% of turn ends
```

Against bactrianx: P(EOS) = 8e-8, rank 16,825 of 139,336. The failure mode that
made the previous instruct model unusable is absent.

## 4. Defect 1 — the corpus is corrupted, and the model learned it faithfully

Asked to describe the water cycle, the model emitted `1. සෞඛ්‍යය වාෂ්පීකරණය`
("1. health evaporation"), numbered items 5 and 6 as `පහයි.` / `හයයි.`, and
elsewhere `[25වන පිටුවේ පින්තූරය]` — "[Picture on page 25]".

None of that is a training bug. All of it is in `UltraChat_Sinhala`. Measured
over a 25,484-turn sample of `train_sft.parquet`:

| artifact | rate |
|---|---|
| numbered list items beginning with an injected junk word | **17,859 / 18,008 = 99.2%** |
| assistant turns containing `සෞඛ්‍යය` ("health") | 38.5% |
| assistant turns containing `මෘදු` ("soft") | 39.6% |
| assistant turns containing `[Nවන පිටුවේ පින්තූරය]` | 33.2% |
| assistant turns that are majority Latin script | 1.10% |

Injected words, by frequency: `සෞඛ්‍යය` 22,743 · `මෘදු` 9,065 · `සීමාව` 2,984 ·
`සයිටම්` 1,704 · `මයික්` 329 · `මිනීමැරුම්` 115. List numbers 5 and 6 are
rendered as the Sinhala sentences `පහයි.` / `හයයි.` ("it is five" / "it is six")
instead of numerals.

A verbatim example from the training data:

```
1. සෞඛ්‍යය ඔබේ Shopify ගිණුමට ලොග් වී ඔබේ ඔන්ලයින් වෙළඳසැලට යන්න.
2. සෞඛ්‍යය ඔබ භාවිතා කරන කොටස් පදනම් කරගත් තේමාව සඳහා ...
3. මෘදු දෙවන රූපයේ සැරිසරන ලක්ෂණය සක්‍රිය කිරීමට ...
4. සෞඛ්‍යය අංශය විවෘතව ඇති විට, ...
පහයි. පෙනෙන සැකසුම් පැනලයේ, ...
හයයි. ලබා ගත හැකි නම්, ...
7. සයිටම් වෙනස්කම් සුරකින්න සහ පෙරදසුන බලන්න.
```

"Health", "soft", and "[Picture on page N]" together are the signature of the
**JW300 / Watchtower parallel corpus**, which is a standard ingredient in Sinhala
MT training sets. The translation system used to build UltraChat-Sinhala was
contaminated with it and leaks its boilerplate into unrelated technical text.

This is the same family of problem as the Sinhala HellaSwag translation, which
was excluded from the benchmark suite for the same reason.

**`test_sft` is contaminated at the same rates** — 38.1% / 39.1% / 33.5%. So
`eval_loss = 1.5403` is partly measuring how well the model predicts
`සෞඛ්‍යය` after `1.`, and is not a clean quality signal. Treat it as a training
diagnostic, not a measure of the model.

## 5. Defect 2 — it answers English in Sinhala

All three English prompts got fluent Sinhala answers:

> **What is the capital of Sri Lanka?** → ශ්‍රී ලංකාවේ අගනුවර කොළඹ.

Correct, but in the wrong language. Only 1.1% of assistant turns in the corpus
are majority-Latin, so the model has effectively never seen an English response
and has learned "reply in Sinhala" as an unconditional rule. Expected given the
data; it will penalise any generative English evaluation.

## 6. Defect 3 — repetition loops past ~200 tokens

Long generations degenerate:
`ජලයේ ඇති ජලයේ ඇති ජලයේ ඇති ...`. Short answers (< ~150 tokens) are clean. Not
yet established whether this is learned from repetition in the corpus or is a
decoding-parameter problem — the artifact audit covers the first, and a
`repetition_penalty` / `no_repeat_ngram_size` sweep covers the second.

## 7. What has not been measured

- **No benchmark scores.** `SinLlama_uc_instruct` is not in
  `benchmark/main.yml`. Adding it to `models:` and scoring it `raw` gives an
  apples-to-apples read against v02 on MMLU-Sinhala, Global-PIQA and
  HellaSwag-EN. Scoring it in its own chat template needs the `chat_models`
  path that does not exist yet.
- **No instruction-following evaluation** — no win-rate against v02, no
  judged quality set.
- The corpus artifacts should be cleaned and the run repeated before any of
  the above is worth trusting, since they contaminate train and test alike.

## 8. Reproducing

On the pod, from the repo root, with `venv-rocm7` active:

```bash
python sft/verify_merge.py      # adapter -> merged weight provenance
python sft/test_generate.py     # generation, stopping, EOS health
```

`test_generate.py` takes an optional model dir, so it also runs against a
checkpoint's merge or against `models/SinLlama_v02` for a base-model contrast.
