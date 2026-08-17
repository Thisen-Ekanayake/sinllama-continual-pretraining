# SinLlama benchmark report

Six checkpoints scored 2026-08-17 on Sinhala/English knowledge, commonsense
reasoning, and three downstream Sinhala classification tasks. Published as an
artifact; this file is the version-controlled copy.

Tables below are generated — regenerate with:

```bash
python benchmark/collect_report.py --full
```

## Headline

| | |
|---|---|
| Sinhala MMLU, v02 over Llama-3-8B | **+9.61** (42.09 vs 32.48) |
| English MMLU, cost of adaptation | **−14.54** (51.00 vs 65.54) |
| Sentiment finetuned, best SinLlama over base | **+8.43** (79.91 vs 71.48) |

## Zero-shot / few-shot knowledge and reasoning

Accuracy %, all raw-prompt scored, higher is better.

| model | MMLU-Si | MMLU-En | PIQA-Si-p | PIQA-En-p | PIQA-Si-n | PIQA-En-n |
|---|---|---|---|---|---|---|
| Llama-3-8B | 32.48 | **65.54** | 31.58 | **58.95** | 57.61 | **68.48** |
| SinLlama_v01 | 37.85 | 48.38 | **47.37** | 49.47 | 55.43 | 50.00 |
| SinLlama_cpt | 35.72 | 39.39 | 35.79 | 41.05 | 59.78 | 48.91 |
| SinLlama_v02 | **42.09** | 51.00 | 47.37 | 51.58 | **61.96** | 55.43 |
| Bactrianx-Instruct | 27.71 | 32.67 | 31.58 | 27.37 | 52.17 | 52.17 |
| uc_instruct_cleaned | 39.02 | 50.35 | 47.37 | 48.42 | 54.35 | 50.00 |

Items: `MMLU-Si` n=6878, `MMLU-En` n=14042, `PIQA-Si-p` n=95, `PIQA-En-p` n=95, `PIQA-Si-n` n=92, `PIQA-En-n` n=92

## Downstream, after per-model LoRA finetuning

Test accuracy %.

| model | News | Sentiment | Writing |
|---|---|---|---|
| Llama-3-8B | 87.69 | 71.48 | 92.57 |
| SinLlama_v01 | 91.08 | 79.58 | 98.32 |
| SinLlama_cpt | **93.85** | **79.91** | 98.32 |
| SinLlama_v02 | 92.31 | 79.69 | 97.92 |
| Bactrianx-Instruct | 91.69 | 77.03 | 97.44 |
| uc_instruct_cleaned | 91.69 | 78.91 | **98.48** |

Macro-F1 %.

| model | News | Sentiment | Writing |
|---|---|---|---|
| Llama-3-8B | 86.63 | 71.42 | 91.40 |
| SinLlama_v01 | 89.94 | 79.66 | 98.20 |
| SinLlama_cpt | **93.25** | **79.92** | 98.16 |
| SinLlama_v02 | 91.47 | 79.61 | 97.66 |
| Bactrianx-Instruct | 90.81 | 76.95 | 97.06 |
| uc_instruct_cleaned | 91.08 | 78.76 | **98.33** |
## Findings

1. **Sinhala adaptation is real; English pays for it.** v02 gains 9.6pp of
   Sinhala MMLU over the base model on 6,878 items and gives up 14.5pp of
   English MMLU.
2. **Finetuning flattens the differences between checkpoints.** All five
   Sinhala-adapted models land within 2.8pp on news, 2.9pp on sentiment and
   1.0pp on writing, against 4–8pp gaps to the base model. The zero-shot
   spread does not carry through to a finetuned deployment.
3. **Zero-shot collapse is not representation damage.** Bactrianx-Instruct is
   near chance zero-shot (27.71 Sinhala MMLU; *worse* at 25.14 when scored in
   its own Alpaca template, so it is not a scoring artifact) yet reaches 91.69
   on news after finetuning.
4. **Chat SFT costs ~3pp of Sinhala knowledge and little else.**
   uc_instruct_cleaned is −3.07 vs v02 on Sinhala MMLU (paired McNemar,
   p < 0.0001), statistically unchanged in English (−0.65, p = 0.13), and
   within 1pp of v02 downstream while taking the best writing-style score.
5. **The prompt template does not matter.** Scoring the chat model in its own
   `### User:`/`### Assistant:` template moved Sinhala MMLU by −0.42 and
   English by −0.39, neither significant. Raw scoring is not penalising the
   instruct models.
6. **Corpus cleaning bought +1.15pp English MMLU** (p = 0.0006) and nothing
   measurable in Sinhala. Its real payoff is generation quality — see
   [ultrachat-cleaning.md](ultrachat-cleaning.md).

## Caveats

- **The four PIQA columns carry no signal.** At n = 92–95, one item is 1.1pp
  and SE near 50% is ~5.2pp. Re-running the *same* model moved PIQA-En-∥ by
  4.21pp. This also retires the earlier "v02 regresses ~3pp on Global-PIQA"
  claim — that was three items.
- **Two snapshots, one control.** Everything except uc_instruct_cleaned comes
  from the 2026-08-01 run. v02 appears in both: MMLU reproduces to ±0.19pp, so
  MMLU is safe to read across rows; PIQA does not reproduce.
- **Uniform raw prompting** across all models, which is what makes the columns
  comparable.
- **Model names differ between snapshots** — `SinLlama_cpt_merged` in MC runs
  vs `SinLlama_cpt` downstream, Bactrianx spelled three ways. Joined by an
  explicit alias map in `benchmark/collect_report.py`.
- **Not measured:** HellaSwag (the bucket path holds results, not the dataset,
  and it is absent everywhere), and any instruction-following or
  generation-quality evaluation — the most important axis for a chat model.

## Method

Single MI300X, bf16, sdpa. MMLU-Si 3-shot / MMLU-En 5-shot, answer scored as the
next-token digit or letter. Downstream: per-model LoRA r=16 α=32 on seven
projections, max_seq 512, lr 2e-4, seed 42, 1.5/1.0/0.4 epochs for
news/sentiment/writing. Significance by paired McNemar with continuity
correction on item-aligned predictions.

Sources: `benchmark/results/prior_2026-08-01/`,
`benchmark/results/uc_instruct_2026-08-17/`.
