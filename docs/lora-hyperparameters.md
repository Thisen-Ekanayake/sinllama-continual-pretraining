# LoRA hyperparameters for the textbook SFT stage

Which LoRA settings to use for instruction-tuning the 8B SinLlama on the
textbook dataset (`textbook_sft/config.yaml`), based on a full read of
Zhang et al., *When Scaling Meets LLM Finetuning: The Effect of Data, Model and
Finetuning Method* (ICLR 2024, arXiv:2402.17193, `lora_size.pdf` in the repo
root). Each recommendation is then checked against the paper's own fitted
numbers, three independent LoRA studies, this project's W&B runs, and this
project's trained adapters. Written 2026-10-22.

## Recommendation

Two values change. Everything else in `textbook_sft/config.yaml` stays as it is.

| setting | current config | recommended | main basis |
|---|---|---|---|
| method | LoRA | **LoRA** | paper §5: LoRA suits tens of thousands of examples, and parameter-efficient tuning keeps zero-/few-shot ability better than full fine-tuning. Full fine-tuning an 8B model also does not fit a 32 GB GPU |
| target modules | q, k, v, o, gate, up, down | **unchanged** | paper applies LoRA to attention and feed-forward. QLoRA, Biderman et al. and *LoRA Without Regret* all find attention-only LoRA underperforms |
| rank `r` | 8 | **8** (16 is equally fine; 64+ gains nothing) | paper: rank 4→128 barely changes loss (\|α_t\| ≤ 0.0044). Your r8 adapters put 90% of each module's update into 3–4 of their 8 directions |
| `alpha` | 16 | **16** (α = 2r) | not covered by the paper. Same α/r as every earlier stage |
| dropout | 0.05 | **0.05** | not covered by the paper |
| learning rate | 5.0e-5 | **2.0e-4** (fall back to 1.0e-4; see [§5](#5-what-is-still-unverified)) | paper: the grid-searched LoRA LR is 10× the full-fine-tuning LR. The same ratio appears in two independent studies. The r8 runs at 5e-5 converged slowly and showed no forgetting |
| effective batch | 64 | **64** | paper used 16 for LoRA vs 128 for full fine-tuning; 32 is a reasonable variant ([§3](#3-the-recommendation-setting-by-setting)) |
| `num_train_epochs` (cap) | 6 | **3** | wiki v2 peaked at epoch 2.8 at a 4× lower LR, then overfit. With a 3-epoch cap the cosine schedule actually decays before early stopping fires |
| schedule, warmup, weight decay, grad clip | cosine, 3%, 0, 1.0 | **unchanged** | paper also used a cosine schedule |
| early stopping | `eval_textbook_loss`, patience 3 × 100 steps | **unchanged** | paper also keeps the best dev-loss checkpoint |
| embeddings | frozen | **frozen** | the paper counts only the LoRA matrices. Your template uses already-trained tokens |

```yaml
# textbook_sft/config.yaml -- the only edits
train:
  # A cap, not the plan: early stopping (below) ends the run. ~435 steps/epoch.
  num_train_epochs: 3        # was 6
  learning_rate: 2.0e-4      # was 5.0e-5 -- keep the ".0": "2e-4" is parsed as a string
```

The existing comment `~283 steps/epoch` was copied from wiki v2. This mix has
27,850 rows, so at batch 64 an epoch is **435 steps**.

## 1. The dataset

The "~5M tokens" is the raw corpus. The training mix is about 2.6 times larger,
and 82% of it is UltraChat replay. Measured with the SinLlama tokenizer and the
training chat template (`sft/build_uc_dataset.ChatTemplate`):

| | rows | tokens | supervised tokens | median length |
|---|---|---|---|---|
| raw textbook Markdown (113 books) | — | 5,410,378 | — | — |
| textbook SFT rows | 13,925 | 2,551,843 | 2,261,846 (88.6%) | 132 |
| UltraChat replay rows | 13,925 | 11,770,298 | 8,572,470 (72.8%) | 759 |
| **train_mix.parquet** | **27,850** | **14,322,141** | **10,834,316** | |
| *for comparison: wiki v2 mix* | *18,092* | *10,005,674* | *7,765,569* | |

- The mix is 50/50 by rows but **18/82 by tokens** (21/79 supervised). Wiki v2
  had almost the same split (23/77), so its training curves are a good guide
  for this run.
- The SFT rows carry 2.26M supervised textbook tokens, **42% of the raw
  corpus**. The rest was dropped as noise, exercises or duplicates.
- In the paper's units, 27,850 examples fall inside the range it studied for
  LoRA (8K–100K examples). The findings below are therefore not
  extrapolations in data size.

## 2. What the paper establishes

Setup: Google's bilingual decoder-only LLMs, 1B–16B (multi-query attention,
UL2 objective, Adafactor optimizer), fine-tuned on WMT14 En-De, WMT19 En-Zh
and multilingual summarization (MLSum). Their 8B has 32 layers and model
dimension 4096, close to Llama-3-8B. They compare full fine-tuning (FMT),
prompt tuning and LoRA, with LoRA on attention and feed-forward layers at
**r = 4** by default (p. 3).

Findings relevant to LoRA, in paper order:

1. **Fine-tuning follows a multiplicative power law**
   L̂(X, D_f) = A · X^−α · D_f^−β + E, where X is model size, pretraining data
   or LoRA rank, and D_f is the number of fine-tuning examples (Eq. 1).
2. **Model size matters more than pretraining data** (α_m > α_p, p. 6).
3. **LoRA rank barely matters.** "Scaling PET parameters is ineffective,
   delivering limited gains for both LoRA and Prompt … |α_t| ≪ 1e−2, and even
   results in inverse scaling in some settings, e.g. LoRA on En-De." Ranks
   tested: 4, 8, 16, 32, 48, 64, 128. The fitted LoRA exponents are
   **−0.0017 / 0.0044 / 0.00022** (Table 4), and "LoRA is more stable and
   reliable" than prompt tuning (p. 7, Fig. 4). *Measured on the 1B model
   only.*
4. **Data matters more than rank.** LoRA's data exponent β is 0.081 / 0.025 /
   0.030, an order of magnitude above its rank exponent. Full fine-tuning is
   hungrier for data (β 0.15 / 0.14 / 0.087, p. 7).
5. **The best method depends on data size.** "When only few thousands of
   finetuning examples are available, PET should be considered first … With
   sightly larger datasets, LoRA would be preferred due to its stability and
   slightly better finetuning data scalability. For million-scale datasets,
   FMT would be good" (p. 8). The crossover where full fine-tuning overtakes
   LoRA at ~8B ranges from about 4K examples (MLSum) to beyond hundreds of
   millions (En-Zh) (Fig. 5).
6. **LoRA keeps general ability.** Prompt and LoRA keep zero-shot ability
   better than full fine-tuning, "particularly when the base LLM is large"
   (p. 9). Full fine-tuning erodes few-shot ability while PET "retains most of"
   it (p. 18).
7. **Optimization recipe** (appendix, p. 15). Loss on the target only;
   Adafactor with cosine decay. LR grid-searched on the 1B model:
   **LoRA 1e-2 vs full fine-tuning 1e-3**. Batch **16 for LoRA/prompt vs 128
   for full fine-tuning**. Training ran up to 200K steps or 100 epochs, and the
   **best dev-loss checkpoint** was kept.

Limits the authors state (p. 9): only closed generation tasks (translation and
summarization), "imperfection of the optimization and evaluation for Prompt
and LoRA in some setups", and no open-ended generation, which they leave as
future work. The paper says nothing about α, dropout, warmup, weight decay or
AdamW learning rates. Its absolute LRs are for Adafactor and do not transfer;
only the LoRA-to-full-FT ratio does.

Reading note: the paper's "PPL" axes are log-perplexity (nats per token). Its
Fig. 10 shows "Valid PPL" values below 1, which a true perplexity cannot take.
The differences quoted below are therefore in the same units as our
`eval_*_loss`.

## 3. The recommendation, setting by setting

**Method: LoRA.** At 27,850 examples (13,925 of them textbook), the paper's
rule of thumb points to LoRA. Its generalization results (finding 6) matter
here because this stage must not erase UltraChat instruction-following, which
is the reason replay is in the mix. Full fine-tuning of 8B with AdamW also
does not fit the 32 GB RTX PRO 4500.

**Where: all seven projections.** The paper puts LoRA on attention and FFN. All
three independent studies in [§4.2](#42-independent-studies) find that the MLP
projections carry most of the gain.

**Rank: 8.** The paper's rank exponent is ~0, and rank is the one setting
where more is clearly not better. r8 on all seven projections of this model is
20.97M parameters, 0.26% of the model. That is the same order as the paper's
default (r = 4 = 0.19% of its 1B). r16 is equally defensible as cheap headroom
for the knowledge-heavy textbook half (+21M parameters, about 0.3 GB more
optimizer state). r64/r128 add memory and nothing else.

**Learning rate: 2.0e-4.** The paper's grid search found LoRA wants 10× the
full-fine-tuning LR. *LoRA Without Regret* measured the same factor (9.8×) for
Llama-3 and Qwen3 with AdamW, and Biderman et al. say "one order of magnitude
higher than full finetuning, often ranging between 5e−5 and 5e−4".

PEFT multiplies BA by α/r, and Adam makes each step roughly LR-sized whatever
the gradient scale. So at a fixed rank, the per-step change in the weights
scales with α·LR. *LoRA Without Regret* finds that at a fixed α the best LR is
about the same for every rank. Together these make **α·LR** a fair
first-order yardstick across runs. On that scale the current setting sits 4×
below the published recipes:

| run | r | α | LR | α·LR |
|---|---|---|---|---|
| wiki, wiki_clean, wiki v2, **textbook (current)** | 8 | 16 | 5e-5 | 8.0e-4 |
| **recommended** | 8 | 16 | 2e-4 | **3.2e-3** |
| QLoRA Guanaco-7B (published script) | 64 | 16 | 2e-4 | 3.2e-3 |
| Biderman et al., math IFT, Llama-2-7B | 16 | 32 | 1e-4 | 3.2e-3 |
| Biderman et al., code IFT, Llama-2-7B | 16 | 32 | 2e-4 | 6.4e-3 |
| this project, gen stage → UC_Gen | 64 | 128 | 5e-5 | 6.4e-3 |
| this project, UC stage 1 → uc_instruct_cleaned | 64 | 128 | 1e-4 | 1.28e-2 |

Two cautions for the 2.0e-4 step. If you change the rank, either keep α fixed
(*LoRA Without Regret* fixes α = 32 and finds the best LR about the same at
every rank) or keep α = 2r and re-tune the LR. And write `2.0e-4`, not `2e-4`
([§6](#6-side-findings)).

**Batch: 64.** The paper chose 16 for LoRA against 128 for full fine-tuning,
and *LoRA Without Regret* finds LoRA "less tolerant of large batch sizes".
The paper's examples are short, though: 16 of them is roughly 1K
(translation) to 9K (summarization) tokens per step, versus about 33K here.
The published 1e-4–2e-4 LoRA rates were used at batch 16 (QLoRA) and at
192–768 (Biderman et al.), so 64 sits inside that range. Keeping 64 also keeps
the W&B history below comparable. Batch 32 (`gradient_accumulation_steps: 8`)
is the variant to try if you want to test this; it does not change memory.

**Epochs and stopping.** The paper trained long and kept the best dev-loss
checkpoint, which is what early stopping plus `load_best_model_at_end`
already do. Lower the cap from 6 to 3: wiki v2 peaked at epoch 2.8 at a 4×
lower LR, and its 6-epoch cosine was still at about half the peak LR when the
run was stopped.

**Expectation, not a hyperparameter.** The paper finds that LoRA leans on what
the base model already knows (α_m and α_p "clearly larger" than β), and
Biderman et al. report that
"LoRA learns less". This stage will mostly teach textbook register and
structure and reinforce facts the model has partly seen. It will not reliably
inject large amounts of new fact. Per the paper, more clean textbook rows would
move the loss more than any rank change. Only 42% of the raw corpus made it
into rows, and the scanned books are not yet extracted.

## 4. Validation

### 4.1 The paper's own numbers, re-derived

Plugging Table 4's LoRA fits into Eq. 1 reproduces Fig. 4's curves within
about 0.015, the error expected from Table 4 rounding A to two significant
figures. The same fitted law then puts the rank and data effects side by side:

| task | r8 → r64 | 50K → 100K examples (at r8) | an 8× rank increase is worth |
|---|---|---|---|
| WMT14 En-De | **+0.0020** (worse) | −0.0319 | 0.96× the data |
| WMT19 En-Zh | −0.0081 | −0.0156 | 1.44× the data |
| MLSum | −0.0008 | −0.0357 | 1.02× the data |

Loss in nats per token, at 100K examples. Doubling the data helps about 2×
(En-Zh) to 45× (MLSum) more than going from rank 8 to 64, and on En-De the
extra rank hurts. Every rank effect is at or below the eval-to-eval jitter of
our own runs (about ±0.01 on the wiki_clean plateau).

### 4.2 Independent studies

| claim | Zhang et al. 2024 (this paper) | *LoRA Without Regret* (Thinking Machines, 2025) | QLoRA (Dettmers et al., 2023) | *LoRA Learns Less and Forgets Less* (Biderman et al., TMLR 2024) |
|---|---|---|---|---|
| rank barely matters at this data scale | yes (r 4–128, 1B) | yes for "small-to-medium-sized instruction-tuning" sets. Rank matters only once data "exceed LoRA capacity" | "LoRA r is unrelated to final performance if LoRA is used on all layers" (LLaMA-7B, Alpaca) | **partly no**: for code IFT (73M tokens) "ranks 16−64 tend not to suffice", so they suggest r = 256. For math (103M tokens) "r=64 suffices to approach full finetuning". Both datasets are 5–10× ours |
| use all layers, especially MLP | attention + FFN | "attention-only LoRA significantly underperforms MLP-only LoRA" | "LoRA on all linear transformer block layers are required to match full finetuning" | "targeting just 'Attention' underperforms both 'MLP' and 'All'" |
| LoRA LR ≈ 10× full-FT LR | 1e-2 vs 1e-3 (Adafactor) | "lower by a factor of 10" for full FT; fitted multiplier 9.8 | 2e-4 for 7B (r64, α16) | "one order of magnitude higher"; LoRA 1e-4 vs full FT 1e-5 on math |
| smaller batches for LoRA | 16 vs 128 | "LoRA is less tolerant of large batch sizes"; more rank does not fix it | batch 16 for 7B | not tested; IFT used 192 (code) and 768 (math) |
| LoRA preserves the base model | zero-/few-shot retained | — | — | "LoRA better maintains the base model's performance on tasks outside the target domain" |

The one disagreement, Biderman et al. on code, involves datasets 5–10 times
larger than ours that teach a genuinely new skill. Their own advice is the
safe reading here: α = 2r, and "sweeping over learning rates between
[1e−5, 5e−4], picking the highest value that enables stable training".

A rough capacity check, using the ~2 bits per parameter that *LoRA Without
Regret* cites from prior work: r8 holds about 42M bits. All 2.26M supervised
textbook tokens at 2.5–3 nats each (the wiki runs started near 2.9) come to
8–10M bits, and most of that is irreducible. That ceiling sits 4–5× below r8's
capacity. The replay half is text the model has already been trained on.

### 4.3 This project's runs (W&B `sinllama-sft-uc`)

| run | setting | eval loss over training | what it shows |
|---|---|---|---|
| `04dambcz` wiki | r8, 5e-5, 3-epoch cosine | 2.1685 → 2.0765 (ep 0.9) → **2.0193 (ep 3.0, still falling)** | slow at 5e-5 |
| `4uhngcpa` wiki_clean | r8, 5e-5, 10-epoch cosine | 3.0038 → 2.8728 (ep 0.9) → 2.8104 (ep 3.0) → **best 2.7998 at ep 6.4** | best checkpoint needed 6 epochs |
| `qa5s0fwi` wiki v2 | r8, 5e-5, 6-epoch cosine, 50/50 replay | wiki 2.9128 → **2.7849 (ep 2.8)** → 2.8309 (ep 3.2); uc 1.6013 → 1.5785 → 1.5906 | overfits once epoch 4 starts; **no forgetting** of UltraChat at this LR |
| `e3yh3pml` UC_Gen | r64, α128, 5e-5 (α·LR 8× higher) | si 1.5815 → best 1.5704 → 1.5763 | the larger effective step did no harm to Sinhala |

The 5e-5 / α16 setting is safe but slow. It takes about three epochs to
converge and shows no sign of instability or forgetting. That leaves room to
raise the LR, which is consistent with the literature, though it does not by
itself prove 2e-4 is better ([§5](#5-what-is-still-unverified)). The wiki v2
pod took 6.0 h for 900 steps (RTX PRO 4500, evals included). The textbook mix
carries a similar ~33K tokens per step, so expect about 2.5–3 h per epoch.

### 4.4 How much of its rank each adapter uses

ΔW = (α/r)·BA for every adapted module of the trained adapters (from GCS and
`models/`). The singular values come from the r × r core of two thin QR
decompositions. Medians over 224 modules:

| adapter | data | effective rank | directions holding 90% of ‖ΔW‖² | smallest/largest singular value |
|---|---|---|---|---|
| wiki v2, r8 | 10.0M-token wiki + replay mix | 6.4 / 8 | **4** (gate_proj: **1**) | 0.16 |
| wiki_clean, r8 | ~10.6K wiki articles | 5.9 / 8 | **3** (gate_proj: **1**) | 0.13 |
| *v01 CPT adapter, r64 (contrast)* | *304M tokens × 4 epochs* | *62.4 / 64* | *52* | *0.37* |

The two SFT adapters trained on data like ours leave much of their r8 unused:
90% of each module's update sits in 3–4 directions. Only the long
continued-pretraining run fills all 64. This supports, but does not prove,
"r8 is not the bottleneck". A decaying spectrum can also reflect a low LR, so
the r8-vs-r16 run in §5 is the direct test.

## 5. What is still unverified

The paper settles method, layers and rank for this setting. The exact LR is the
one recommendation that is extrapolated: from the paper's ratio, the published
7B recipes and our slow-but-stable 5e-5 runs, not measured on this model and
data. Two cheap ways to close it, on the pod as usual:

1. **Single run with guard rails.** Train at `2.0e-4` with the config above.
   Fall back to `1.0e-4` if `eval_uc_loss` rises more than ~0.02 above its
   first evaluation, or if training loss or grad-norm spikes. The starting
   model, wiki v2's checkpoint-800, scored 1.582 on the identical 500 replay
   rows.
2. **LR sweep**, one epoch each (~2.7 h per run). Keep the highest LR that is
   stable and lowest on `eval_textbook_loss` without raising `eval_uc_loss`.

```bash
for LR in 1.0e-4 2.0e-4 4.0e-4; do
  python sft/run_sft_uc.py --config textbook_sft/config.yaml \
    --set train.learning_rate=$LR --set train.num_train_epochs=1 \
    --set train.output_dir=/workspace/sinllama_runs/textbook_lr_$LR \
    --set train.run_name=sinllama-textbook-lr-$LR
done
# Optional rank check at the chosen LR: hold alpha fixed so the LR transfers.
#   --set lora.rank=16 --set lora.alpha=16
# The paper predicts a difference within ~0.01 of r8.
```

## 6. Side findings

- **`--set` with a bare exponent breaks training.** PyYAML reads `2e-4` and
  `5e-5` as strings (`2.0e-4` is a float), so
  `--set train.learning_rate=2e-4` reaches AdamW as `'2e-4'`. It then fails
  with `TypeError: '<=' not supported between instances of 'float' and 'str'`,
  after the model has loaded. The usage examples in `sft/run_sft_uc.py`,
  `sft/config.yaml` and `gen/config.yaml` use exactly that form. Write
  `2.0e-4`.
- `textbook_sft/config.yaml` says `~283 steps/epoch`; it is 435 for this mix.

## Sources

- B. Zhang, Z. Liu, C. Cherry, O. Firat. *When Scaling Meets LLM Finetuning:
  The Effect of Data, Model and Finetuning Method.* ICLR 2024,
  [arXiv:2402.17193](https://arxiv.org/abs/2402.17193) (`lora_size.pdf`).
- J. Schulman and Thinking Machines Lab. *LoRA Without Regret*, 2025,
  <https://thinkingmachines.ai/blog/lora/>.
- T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. *QLoRA: Efficient
  Finetuning of Quantized LLMs*, 2023,
  [arXiv:2305.14314](https://arxiv.org/abs/2305.14314); Guanaco-7B recipe from
  [`scripts/finetune_guanaco_7b.sh`](https://github.com/artidoro/qlora/blob/main/scripts/finetune_guanaco_7b.sh).
- D. Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR 2024,
  [arXiv:2405.09673](https://arxiv.org/abs/2405.09673).
- This repo: `textbook_sft/`, `wikipedia_sft/config_v2.yaml`,
  `sft/run_sft_uc.py`, `docs/research-history.md`. W&B runs `04dambcz`,
  `4uhngcpa`, `qa5s0fwi`, `e3yh3pml`. Adapters
  `gs://sinllama_cpt/SinLlama_wiki_uc_gen_mix_adapter/` and
  `gs://sinllama_cpt/sinllama_wiki_cleaned_adapters/wiki_sft_lora_clean/`.
