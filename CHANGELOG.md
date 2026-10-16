# Changelog

All notable changes to this repository, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The repo has no version
tags, so each entry is a dated project phase; the phase numbers match §4 of
[`docs/research-history.md`](docs/research-history.md), which has the full
narrative, model cards and evidence. Short hashes point to the commit that made
the change. Model names refer to the merged weights in `gs://sinllama_cpt`.

## [Unreleased]

### Added
- `inference/judge_gemini.py`: an LLM judge (Gemini) that scores generated answers 1–10 on
  instruction following, relevance, conciseness and overall quality. It resumes after rate limits.
- A 100-prompt LLM-judge evaluation (2026-10-15 – 10-16). Each model answers
  `inference/100_prompts.json` once (`default` preset, seed 42, run until EOS), and
  `gemini-3.8-flash` scores the answers with the `judge_gemini.py` rubric. The prompts, answers
  and scores are gitignored data under `inference/`. Overall scores out of 10:
  - `SinLlama_wiki_clean`: 3.61, conciseness 3.45. It answers every prompt with a full
    encyclopedia article (median 2,240 characters; some needed 4,096 tokens to finish), because
    its stage-3 data paired every "tell me about X" question, including "briefly", with the
    whole article.
  - Its parent `SinLlama_UC_Gen_Model`: 5.11. It wins 64 of the 100 prompts and loses 19.
- The Wikipedia SFT v2 pipeline: `wikipedia_sft/build_wiki_mix_dataset.py` and
  `wikipedia_sft/config_v2.yaml`. It keeps the v1 articles and cleaner, but:
  - **Answer length matches the question:**
    - the lead section (capped at 1,500 characters) for "what is / tell me about X";
    - the first 2–3 sentences for "briefly";
    - the full article, capped at 2,000 characters on a paragraph boundary, for "in detail".
    The split is 50 / 25 / 25 %, chosen deterministically per article.
  - **Weak articles are dropped:** non-Sinhala, mostly lists or tables, a lead under 150
    characters, or run-on text. That keeps 9,546 of 10,917.
  - **Captions and `[ සංස්කරණය ]` edit links are stripped.**
  - **The mix is 50 % Sinhala UltraChat `gen` replay**, cleaned with `gen/build_mixed_gen.py`'s
    `prepare()`, giving 18,092 training rows.
  - **Training:** separate `eval_wiki_loss` and `eval_uc_loss`. Early stopping on
    `eval_wiki_loss` (patience 3 × 100 steps, 6-epoch cap), gradient checkpointing on for
    32–48 GB GPUs, checkpoints on the RunPod network volume, W&B logging.
- The **`SinLlama_wiki_uc_gen_mix`** model (`gs://sinllama_cpt/SinLlama_wiki_uc_gen_mix/`; the
  step-800 checkpoint is in `SinLlama_wiki_uc_gen_mix_adapter/`):
  - **Training:** a fresh LoRA r8 on `SinLlama_UC_Gen_Model`, using the v2 pipeline, on an RTX PRO
    4500 (32 GB; peak 29.1 GB at batch 4 × 2,048). W&B run `qa5s0fwi`.
  - **Stopping point:** the run was stopped after epoch 3, at step 900, because step 900 was
    worse on both eval sets. The best checkpoint is step 800 (epoch 2.84):
    `eval_wiki_loss` 2.785, `eval_uc_loss` 1.582. The merge passes `sft/verify_merge.py`.
  - **100-prompt judge:** overall 5.44 (UC_Gen 5.11, wiki_clean 3.61); instruction following
    5.96, relevance 6.01, conciseness 6.22.
    - Median answer is 954 characters, and every answer ends within 1,024 tokens.
    - Against `SinLlama_wiki_clean` it scores +1.83 overall, winning 67 prompts and losing 15.
    - Against UC_Gen it scores +0.33 overall, which is within noise (95 % CI −0.05 to +0.73).
      Only conciseness clears zero (+0.41).
    - Gains: Wikipedia-style prompts 4.40 vs 3.60, Instruction Following prompts 7.40 vs 4.20.
    - Losses: Creative Writing 5.14 vs 6.29.
    - Factual errors are unchanged: 37 vs 38 judge mentions.
- `benchmark/mmlu/generate_sinhala_mmlu.py`: a free-text SinhalaMMLU generation study, in chat
  (zero-shot) or 3-shot format, with the logit-scored answer joined in. First result:
  `SinLlama_wiki_clean` scores 31.3 % written vs 38.1 % logit-scored.
- `docs/research-history.md`: the project history from February to September 2026, with model
  cards, datasets, evaluation, open questions and corrections to earlier write-ups.
- `Wikipedia_Dataset/wikiextractor/`: a vendored copy of WikiExtractor.

### Known issues
These were found while writing `docs/research-history.md` (§2) and have not been fixed in the repo yet:
- `docs/benchmark-report.md` and `inference/models.yml` present `SinLlama_v02` as a base/CPT
  checkpoint. It is v01 plus a short MMLU-format SFT, so part of its MMLU lead is format learning.
- `benchmark/results/prior_2026-08-01/` holds results from a 2026-08-09/10 run, not 08-01.
- The downstream tables report single runs. Repeating a run with the same config and seed moves
  news macro-F1 by up to 3.2 pp, so the gaps between SinLlama variants are noise.
- The `assert_eager` docstrings in `benchmark/mmlu/evaluate_{sinhala,english}_mmlu.py` give
  47.27 / 41.33 for `SinLlama_cpt`. Those are the scores of `SinLlama_cpt_v02_new`.
- The news-category prompt in `benchmark/downstream/finetune_news_category.py` lists the labels
  in a different order from the one the data actually uses. 4 of the 5 classes are mislabelled.
- `sft/config.yaml` still refers to `gs://sinllama-cpt` (hyphen). The bucket is `sinllama_cpt`.
- `results/latex/Downstream_Tasks.tex` copies the SinLlama paper's Table III rows without saying so.
- Global-PIQA still loads models with SDPA attention.
- `evaluate_english_mmlu.py` has no memory cap on batch × sequence².
- With a mapping `data.eval_file` (as in `wikipedia_sft/config_v2.yaml`), `sft/run_sft_uc.py`
  logs the second eval set's metrics (`eval_uc_*`) only to W&B, not to the text log.
- The 100-prompt judge comparison is one sampled answer per prompt from a single judge. The
  v2-vs-UC_Gen gap (+0.33) needs several samples per prompt before it can be confirmed.
- Evaluation gaps: `SinLlama_UC_Gen_Model` has no raw or permuted MMLU score. No model from v02
  onward has a HellaSwag score. The English debias run for `SinLlama_wiki_clean` failed.

## 2026-09-29: Wrap-up runs (phase 13)

### Added
- The `SinLlama_wiki_clean` model, benchmarked on an NVIDIA A40: MMLU-Si 38.11 (40.65 permuted),
  MMLU-En 51.00; downstream accuracy news 92.9 / sentiment 80.0 / writing 98.6 (`34d1031`).
- `SinLlama_wiki_clean` in the inference model registry, and a `run_benchmark_prompts` script for
  the 10-prompt Sinhala generation probe across 10 models (`eb81b70`).

## 2026-08-24 – 08-25: Inference tooling and Wikipedia SFT (phase 12)

### Added
- Inference tooling for ad-hoc generation checks:
  - `inference/models.yml`, the model registry (`3d47f06`);
  - `chat_template.jsonl`, per-model prompt templates (`729a05c`);
  - `hyperparameters.yml`, generation presets (`4f7015e`);
  - `prompts.jsonl`, seed prompts (`fed796f`);
  - the `run_inference.py` + `run.sh` bf16 runner (`3765656`), which runs every model when
    `--model` is omitted (`d42ad63`).
- Wikipedia preparation scripts: article categorisation, word counts and tokenisation (`4f36e24`).
- The stage-3 pipeline in `wikipedia_sft/`: fetch the dump (`6982b36`), build the dataset
  (`1889159`), config (LoRA r8, `9c39bb3`), orchestration (`c3e41b4`) and GCS upload (`399370b`).
  It produced the **`SinLlama_wiki`** model, trained from `SinLlama_UC_Gen_Model`.
- `SinLlama_wiki` registered in the benchmark and inference registries (`20bcf78`).
- `requirements_amd.txt` for the ROCm stack (`5e40a5e`, tracked in `b4cc976`).

### Changed
- `wikipedia_sft` strips the extraction artifacts from the corpus: first the link markup
  (`fd78962`), then all of them (HTML and MediaWiki tags, entities, templates, footnotes, URLs).
  The rerun has a 10-epoch cap (`f2b9731`) and produced **`SinLlama_wiki_clean`**.

### Fixed
- Removed WikiExtractor flags that do not exist from `fetch_wikipedia_dump.sh` (`3786a80`).
- Pinned the ROCm wheel index in `requirements_amd.txt` to rocm6.2 (`fea1bfd`).

### Removed
- `hdbscan` from `requirements.txt` (`537e75a`).

## 2026-08-22: Gen stage (phase 11)

### Added
- The bilingual stage-2 SFT pipeline in `gen/`: mix builder (75 % Sinhala / 25 % English
  UltraChat `gen` dialogues), config, data fetch, runner and upload scripts
  (`b02f134`…`6461121`), documented in `docs/gen-stage.md` (`7d0e151`).
  It produced the **`SinLlama_UC_Gen_Model`** model, the first one that answers English in English.
- Separate eval losses per language: `data.eval_file` can be a mapping (`7c28b88`).
- A script for the distribution of predicted MMLU answers (`03bfcbf`, `9c0b126`).

### Changed
- `sft/upload_gcs.sh` artifact names can be set through environment variables (`0e8f277`).

### Fixed
- The GCS bucket name typo in `benchmark/main.yml`. Benchmark uploads had been going nowhere since
  Aug 6 (`1ac80c3`).

## 2026-08-18 – 08-22: Position-bias study (phase 10)

### Added
- Contextual calibration (`41aa1fe`) and cyclic-permutation scoring (`dcd1783`) for MMLU, an
  arm-comparison script (`00c6de7`), the study runner (`801fd52`, `4ef2972`) and VRAM-gated
  parallel dispatch (`1f57340`).
- `docs/position-bias-study.md` (`b646efc`), extended to all six models (`8471099`). The study
  adopted permutation and discarded calibration, which made the bias worse.
- An assertion that MMLU models load with eager attention, and a guard against digit-token
  collisions (`4e9a013`).

### Fixed
- Permutation batches are bounded by attention memory (`e175116`).
- The position-bias doc's claim about the `SinLlama_cpt` figures was retracted (`8471099`).

## 2026-08-15 – 08-17: Chat SFT, stage 1 (phase 9)

### Added
- The UltraChat-Sinhala SFT pipeline: config, multi-turn dataset builder, trainer with early
  stopping, runner and GCS fetch (`5e9b82d`…`36d7604`). It uses a `### User:` / `### Assistant:`
  template, because the Llama-3 chat-template role tokens are untrained in SinLlama.
  It produced **`SinLlama_uc_instruct`**.
- MI300X performance work: a micro-benchmark script, hipBLASLt and TunableOp, and Liger fused
  cross-entropy on by default (`b6b2d97`, `0320029`, `3935280`). Recorded in
  `docs/mi300x-performance.md` (`c79926c`); step time fell from 13.5 to 7.9 s.
- `docs/uc-instruct-evaluation.md` (`cd78e79`), plus merge-verification and generation
  acceptance tests (`269a42d`).
- `sft/clean_ultrachat.py`, which removes the JW300 machine-translation artifacts
  (`b849323`, `8b665cb`), documented in `docs/ultrachat-cleaning.md` (`4826edc`, `95c0446`).
  Training on the cleaned corpus (`1bba54f`) produced **`SinLlama_uc_instruct_cleaned`**.
- Chat-template scoring for MMLU and PIQA (`cfecedc`, `6e9dd04`), a cross-snapshot result
  collector (`bf92959`) and the six-model `docs/benchmark-report.md` (`bf4a98d`).

### Changed
- `docs/*.md` is tracked through `.gitignore` instead of being force-added (`d2a7c2d`).

### Fixed
- Checkpoint resume (numpy RNG globals allow-listed) and ROCm allocator fragmentation (`8d98909`).
- Dataset statistics with `datasets>=4` Column objects (`c32f02f`).
- The PIQA `ModuleNotFoundError` (`e0b79cf`).

### Removed
- The superseded UltraChat SFT scripts (`ba544d3`).

## 2026-08-05 – 08-06: SinLlama_v02 and the unified benchmark runner (phase 8)

### Added
- MMLU-Sinhala `auxiliary_train` SFT: data cleaning, a validation proxy and LoRA SFT scripts
  (`0463d64`…`823838a`). This produced **`SinLlama_v02`** (v01 + about 250 SFT steps). Its
  adapter was not saved, so it cannot be rebuilt.
- `benchmark/benchmark.sh` with a YAML config system for every benchmark task (`8a173c3`).

### Changed
- The task scripts moved under `benchmark/` (`2aabdc7`).

### Fixed
- Downstream evaluation uses eager attention instead of SDPA, which had produced unparseable
  batched generations (`a4752a8`).

## 2026-07-21 – 07-24: Global-PIQA (phase 7)

### Added
- Global-PIQA evaluation, parallel and non-parallel, in Sinhala and English (`e83658a`…`5832e3a`),
  with result combination, SDPA-vs-eager comparison and charts (`fcaa3f7`…`cfc9da4`).

### Changed
- Settled on 8-shot evaluation after comparing zero-shot and few-shot (`cf6f321`, `431d9ed`).
  Sinhala prompt templates and cue markers were revised (`b70ef2d`, `a66a302`). The set later
  proved too small (n = 92–95) to rank models.

## 2026-07-03 – 07-08: Benchmarks and weights analysis (phase 6)

### Added
- SinhalaMMLU (`7785dae`), English MMLU (`d212cda`) and HellaSwag (`df1e558`) evaluation, with
  parallel runners and GCS upload; a radar chart of MMLU results (`9366498`).
- The merged Bactrian-X instruct model, with checks on bf16 and embedding size (`a25f83e`).
- Weight, attention, embedding and MLP analysis notebooks across Llama-3-8B → v01 → {cpt,
  Bactrian-X} (`60159f9`…`bce847a`). They first showed the untrained special-token embeddings.

### Fixed
- SDPA with left padding on ROCm collapsed batched MMLU answers onto option A. Switched to eager
  attention (`249bb13`). Results produced before this fix are invalid.
- An out-of-memory error that left attention and MLP notebook output incomplete (`c8d2d52`).

## 2026-06-12 – 06-22: Corpus analysis and June CPT (phases 4–5)

### Added
- Corpus analysis: text extraction, HelaBERT embeddings and vocabulary / token-shift notebooks
  (`db6d97b`, `13abe2c`, `cd7e41b`). It found that the CPT corpus is close to a lexical subset of
  v01's own pretraining corpus.
- Chinese-LLaMA-style LoRA CPT (`cpt/run_clm_pt_with_peft.py`, `cpt/run_pt.sh`;
  `083ceab`…`7cde36e`) and a low-memory LoRA merge (`3b2a46b`). This produced **`SinLlama_cpt`**,
  the only CPT model that survives.
- Downstream LoRA fine-tuning for news category, sentiment and writing style, using the SinLlama
  paper's prompts (`e7186b2`…`7599929`), plus a health check for merged models (`e70b37f`).
- Scripts to check classification datasets for leakage and clean them (`35fb19b`).

## 2026-04-13 – 04-18: Two-stage CPT on MI300X (phase 3)

### Added
- A two-stage CPT script for the AMD MI300X: stage 1 trains only the embeddings and LM head,
  stage 2 trains LoRA plus embeddings (`563e59a`, `daf3725`). Stage 2 crashed at epoch 2.11 and no
  artifacts were kept.
- `LICENSE` and `README.md` (`b700a88`, `e2baab5`).

## 2026-03-06 – 03-14: Hyperparameter search and classification (phase 2)

### Added
- Ray Tune + Optuna hyperparameter search for classification (`e61c057`, `eecb3a7`).
- KL-divergence between the corpus and the task data (`e02dc36`); Unicode cleaning, language
  detection and train/val/test splits (`b946912`…`4dcb868`).
- A sliding-window classifier (`61400f6`). Its runs went NaN and it was abandoned.

### Fixed
- ROCm out-of-memory and tf32 crashes in hyperparameter-search training (`e45ab8d`, `56c9334`).
- A Sinhala ZWJ (U+200D) being stripped as a hidden character (`4dcb868`).

### Known issues
- The search metric read the final padding token instead of the label, so every "best" config
  reported 100 % accuracy. This was never fixed; the approach was abandoned.

## 2026-02-16 – 02-28: First CPT (phase 1)

### Added
- LoRA continual pretraining with W&B logging (`311c88e`, `dd42638`); perplexity and token-count
  scripts (`f0bb963`, `523f184`); a generation comparison of the base and CPT models (`9dc29b6`).
- The first downstream classification: preprocessing per category, LoRA fine-tuning, adapter
  merging and an evaluation script with a confusion matrix (`6684d7d`…`55399f0`).

### Removed
- Unsloth, which was tried and dropped (`199aad9`, `e540b83`).

Stage 2 finished (held-out perplexity fell from 63.2 to 58.3), but its weights never reached GCS.

## 2026-02-06 – 02-13: Orientation (phase 0)

### Added
- Merged Llama-3-8B with the `polyglots/SinLlama_v01` adapter in bf16 (`95c7854`). Every later
  model is built on this merge.
- Inspection of the merged model: weight shapes, perplexity, weight and activation
  distributions, column importance, magnitude pruning (`0984fb1`…`8e6dc46`). No perplexity or
  pruning results were recorded.
- An OASST2 → Sinhala instruction set (clustering, GPT-4o-mini cluster scoring, translation) and
  QLoRA fine-tuning scripts for an 8 GB GPU (`a787330`…`fff0c6a`). They were superseded by UltraChat.

[Unreleased]: https://github.com/Thisen-Ekanayake/sinllama-continual-pretraining/compare/eb81b70...HEAD
