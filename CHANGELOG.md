# Changelog

All notable changes to this repository, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The repo has no version
tags, so each entry is a dated project phase; the phase numbers match §4 of
[`docs/research-history.md`](docs/research-history.md), which has the full
narrative, model cards and evidence; phases 14 onwards are not in it yet. Short
hashes point to the commit that made the change. Model names refer to the merged
weights in `gs://sinllama_cpt`. That bucket was deleted with its GCP project on
2026-11-02; the same files are in Google Drive under `sinllama_cpt/`, where every
later model is stored too.

## [Unreleased]

### Known issues
Not fixed in the repo yet. The first group comes from the phase-19 sweep; the rest were found while
writing `docs/research-history.md` (§2).
- From the `SinLlama_instruct_mix` sweep:
  - **The lab-box downstream probes saved no per-item predictions**, so the differences from the
    parent are item counts with no paired test.
  - **167 English MMLU items were cut to 768 tokens** for both models. The 16 GB loader
    (`tb_eval/mmlu16_fixed.py`) has to do this to fit them in memory.
  - **The first English MMLU run is invalid.** 1,813 items failed on a cascade of out-of-memory
    errors. It is kept as `EnglishMMLU_results_permuted_INVALID_oom`.
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
- Evaluation gaps (HellaSwag is out of scope):
  - **No permuted English MMLU** for `SinLlama_UC_Gen_Model` or `SinLlama_wiki_clean`. Their
    permuted Sinhala arms were scored on the lab box in phase 16. `SinLlama_wiki_uc_gen_mix` has
    both languages (phases 16 and 19).
  - **No 100-prompt judge** for `uc_instruct_cleaned`, `wiki`, `Bactrianx_Instruct`,
    `textbook_mix` or `instruct_mix`.
  - **`uc_instruct` (raw corpus)** has MMLU and PIQA data but no downstream probes or judge, and
    is not in the reports.

## 2026-11-19 – 11-24: Bilingual instruction mix (phase 19)

### Added
- The **`SinLlama_instruct_mix`** model, the first SinLlama model that answers
  questions about images. It was trained and benchmarked on the 16 GB RTX 4070 Ti SUPER lab machine.
  - **Data** (`instruct_sft/build_instruct_mix.py`, `7a86e5f`): 355,259 training rows and 147.8M tokens,
    exactly 75 % Sinhala / 25 % English by tokens, with a 1,600-row validation set split by source.
    Sources:
    - COCO train2017 VQA: the first 100k of its 118,287 images (99,693 training rows per
      language), Qwen English answers plus their NLLB Sinhala translations, with the looping
      rows repaired;
    - UltraChat in Sinhala and English;
    - Aya Dataset and Aya Collection, with the NLLB ZWJ spelling artifacts repaired;
    - LKValues.
  - **Training** (`instruct_sft/train_instruct.py`, `run_instruct.sh`, `80944ec`):
    - QLoRA r64 / α16 on all 7 projections of `SinLlama_wiki_uc_gen_mix`: 1 epoch, 10,069 steps,
      LR 2e-4 cosine, loss on the answers only.
    - It survived two lab-box power losses through an `@reboot` resume hook and checkpoints every
      100 steps; the GPU is now capped at 220 W.
    - Early stopping (patience 4) never triggered. The best eval loss was 1.418, at step 10,000.
  - **Weights** (Google Drive, not GCS):
    - merged bf16 model in `gdrive:sinllama_cpt/SinLlama_instruct_mix/`;
    - adapter in `gdrive:sinllama_cpt/vqa-early-stopping/best_step10000_no_early_stop/`.
  - **Benchmarks** (`instruct_sft/bench/`, `58b1701`; lab-box evaluators in `tb_eval/`, `327f653`):
    - Every number is against its parent `SinLlama_wiki_uc_gen_mix`, re-scored on the same machine,
      with paired tests.
    - **COCO VQA** (Qwen judge, out of 100, `ddbe3d3`):
      - English 88.07, against 76.65 zero-shot and 81.89 for the VQA-only fine-tune
        (+6.18 over the fine-tune, p = 2e-15).
      - Sinhala 60.68, against 37.11 and 51.41 (+9.27 over the fine-tune, p = 1e-10).
    - **SinhalaMMLU:** 37.07 % raw / 40.32 % permuted, which is −1.57 pp (p = 7e-4) and
      −1.12 pp (p = 0.008).
    - **English MMLU:** 50.31 % raw / 53.60 % permuted, which is −1.84 pp (p = 1.2e-9) and
      −1.37 pp (p = 1.5e-7). The raw score has more option-A bias (TVD to gold 14.2 vs 7.2 pp).
    - **Open-book SinhalaMMLU:** 37.16 % closed book and 46.12 % with bge-m3 top-3 retrieval,
      about −1.5 pp in both settings.
    - **Belebele:** Sinhala with the passage 71.1 % permuted, −2.6 pp (p = 0.013). The other three
      splits are not significant.
    - **XL-Sum and M2DS:** ROUGE and BERTScore level with the parent in Sinhala and English
      (no significant difference).
    - **Downstream probes** (4-bit QLoRA, macro-F1): writing 98.04 / news 91.05 / sentiment 79.94,
      against the parent's 97.66 / 91.37 / 79.46, which is within ±5 test items.
    - **Global-PIQA** (8-shot, raw, 2026-11-24): average of the 4 splits 51.69 %, against 50.32 % for the
      parent re-scored on the same machine (372 of its 374 items match its A40 run). No split
      differs significantly (exact McNemar, p ≥ 0.13).
    - The Gemini 100-prompt judge was not run.
  - **Reports:** the model is added to `results/latex/` and `results/pdf/`:
    - a new section each in `VQA_Benchmark` and `MMLU_Debiased` (Appendices B and C);
    - rows in `Belebele`, `MRL_Benchmarks`, `XLSum_Summarization`, `M2DS_Summarization` and
      `Downstream_Tasks`;
    - a column in `MMLU_Benchmark`;
    - a node in the model tree, a table column and the weights' location in `Model_Comparison`.

    The raw predictions are in `results/data/instruct_mix_lab_4070ti_20261122/`.
  - **`results/latex/Instruct_Mix_Results`**: the model, its parent, `SinLlama_v01` and llama-3-8b
    side by side on every benchmark, with a sources table.
- `tb_eval/mmlu16_fixed.py`: English MMLU on the 16 GB lab box. It runs `permute_eval.py` with the
  memory fixes from `tb_eval/rag/rag_eval.py`:
  - the embedding lookup runs on the CPU;
  - the out-of-memory retry runs outside the `except` block;
  - a single item that still runs out of memory is re-scored at 768 tokens and counted on a
    `FALLBACK` line (`ebc01ad`).
- `instruct_sft/bench/bench_chain2.sh`: the English MMLU rerun for `SinLlama_instruct_mix` and its
  parent with `mmlu16_fixed.py`. It waits for `bench_chain.sh` to finish (`ebc01ad`).
- `instruct_sft/bench/upload_merged.sh`: uploads the merged model to Drive and retries until
  `rclone check` reports 0 differences (`6626e3d`).
- `instruct_sft/bench/piqa_job.sh`: Global-PIQA for the model and its parent on the lab box, through
  `piqa_16gb.py` with the `mrlbenchmarks/config.yml` settings.

### Fixed
- `instruct_sft/on_reboot.sh` checks only the last line of the benchmark-chain logs. The logs keep
  the `CHAIN_DONE` lines of earlier passes, so a reboot would not have restarted an unfinished chain.
  The hook now also restarts `bench_chain2.sh` (`4a98b0b`).
- `instruct_sft/bench/downstream_job.sh` evaluates with batch 2 (`EVAL_BS`). The config's batch 16
  builds 16 × 512 × 139k fp32 logits and ran out of memory at the first evaluation. The batch size
  changes only memory use, not the loss or the predictions (`4a98b0b`).
- `VQA_Benchmark` said the mix used 118k train2017 images and about 30× more VQA data than the
  3,994-row fine-tune. It used the first 100k images, so about 25×.

## 2026-11-04 – 11-22: COCO visual question answering (phase 18)

### Added
- **A COCO val2017 VQA set** (`VQA/`, `30c376b`; documented in `docs/vqa-dataset.md`):
  - **English:** for each of the 5,000 images, Qwen3.5-9B (llama.cpp, grounded by the image's 5 COCO
    captions) writes a description, a question for 17–18-year-old students, and its answer.
  - **Sinhala:** NLLB-200 3.3B translates it sentence by sentence. Three post-processing steps fix
    NLLB's failure modes:
    - `repair_nllb.py`: dropped clauses, loops, and a "picture on page 23" hallucination on about
      14 % of sentences;
    - `postprocess_si.py`: rare Sinhala letters NLLB leaves out, and doubled punctuation;
    - `sinhala_normalize.py`: lost ZWJ conjuncts.

    The final file has 0 page hallucinations, 0 loops and 0 unresolved sentences.
- **A VQA benchmark for SinLlama** (`VQA/sinllama_vqa_eval/`, `1ddb61b`):
  - The model gets the description and the question and writes an answer. Sinhala answers are
    translated back to English with NLLB.
  - Qwen3.5-9B judges each answer against the reference on four criteria, weighted 0.25 / 0.35 /
    0.25 / 0.15, for a score out of 100.
  - `SinLlama_wiki_uc_gen_mix` zero-shot: English 75.97, Sinhala 38.43. 5-shot doubles the answer
    length but does not help: 75.69 / 39.07.
- **Per-language VQA QLoRA fine-tunes** (`ft_vqa.py`, `run_ft.sh`, `2cfab0c`):
  - 3,994 / 500 / 500 split by image; r8; best checkpoint at step 120.
  - On the 500 test images: English 81.89 vs 76.65 zero-shot (p = 2e-8); Sinhala 51.41 vs 37.11
    (p = 6e-17).
  - The adapters are in Drive `sinllama_cpt/SinLlama_wiki_uc_gen_mix_vqa_adapters/`.
  - The fine-tune tables are in `VQA_Benchmark` (`e344e5f`).
- **COCO train2017 VQA** (`VQA/train2017/`, `83abc6f`): the same pipeline in 10k-image chunks on the
  lab box.
  - It was stopped after 100k images, when a kernel update left the machine without its GPU driver
    (2026-11-11).
  - `scan_loops.py` and `fix_loops.py` found 142 looping records. They regenerated or re-translated
    135 and dropped 7.
  - The 100k set is in Drive `VQA_full/` and is the VQA part of the phase-19 instruction mix.
- New NLLB options, all off by default so the val2017 build still reproduces (`140ea58`, `8313e87`):
  - `translate_nllb.py --nopunct-first --logprob`;
  - `postprocess_si.py --fix-split-hal`;
  - `repair_nllb.py --long-loops`, which catches loops of 4–10-word units.

### Known issues
- The judge model, Qwen3.5-9B, also wrote the reference answers, so the judge may favour answers
  like its own.
- The zero-shot and 5-shot VQA runs used the older prompt wording ("image description"). The
  fine-tunes use the current wording.

## 2026-10-30 – 11-04: Summarization benchmarks (phase 17)

### Added
- **XL-Sum** (`xlsum_eval/`, scoring and report scripts in `results/data/xlsum_lab_4070ti_20261030/scripts/`;
  `51abba4`):
  - **Setup:** a QLoRA r16 fine-tune per model on the lab box (best checkpoint by validation loss,
    beam 4).
  - **Data:** the Sinhala split (3,249 / 500 / 500) and a seed-42 English sample of the same size.
  - **Metrics:** ROUGE, and BERTScore with HelaBERT_Large for Sinhala and RoBERTa-large for English.
  - **Sinhala ROUGE-1/2/L:**
    - llama-3-8b 24.89 / 11.48 / 22.08;
    - `SinLlama_v01` 28.54 / 14.74 / 25.35, which beats XL-Sum's mT5 (27.29 / 13.38 / 23.47);
    - `wiki_uc_gen_mix` 28.43 / 14.10 / 25.04;
    - `textbook_mix` 28.26 / 13.94 / 24.76.
  - **English ROUGE-1:** llama-3-8b leads (41.36) and the SinLlama models score 37.0–37.8. That gap
    of about −4 is the English cost of the Sinhala CPT.
  - The SinLlama variants do not differ significantly from one another in either language.
- **M2DS multi-document summarization** (`m2ds_eval/`, `6bf3aef`):
  - **The official splits leak.** 72.6 % of the Sinhala test articles are also in the training set,
    and a model trained on them reaches ROUGE-1 45.57 by memorising them.
  - **`m2ds_dedup.py` removes the overlap.** It drops every training cluster that shares an article
    or a summary sentence with validation or test. That leaves 2,396 / 4,942 training clusters in
    Sinhala and 10,670 / 13,496 in English.
  - **The English scores strip a BBC promo line.** 812 of 1,687 English test references end with it.
  - **Sinhala ROUGE-1:** llama-3-8b 22.06, `SinLlama_v01` 24.46, `wiki_uc_gen_mix` 23.78.
  - **English ROUGE-1:** llama-3-8b 49.79, `SinLlama_v01` 46.39, `wiki_uc_gen_mix` 46.56.
- Reports `results/latex/XLSum_Summarization` and `M2DS_Summarization`.

### Fixed
- XL-Sum decoding no longer blocks repeated n-grams. With a decoder-only model,
  `no_repeat_ngram_size=2` also counts the prompt, so the model could not copy any word pair from
  the article. The first Sinhala runs are invalid, are kept as `*_INVALID`, and were decoded again.

### Changed
- The GCP project, and with it `gs://sinllama_cpt`, was deleted on 2026-11-02 when its credits ran
  out. The lab-box chains now fetch models from the Google Drive backup with `rclone`.

## 2026-10-24 – 10-26: Lab-box evaluation (phase 16)

### Added
- **Evaluators that fit the 16 GB RTX 4070 Ti SUPER lab machine** (`tb_eval/`, committed in `327f653`):
  - `permute_16gb.py`: SinhalaMMLU raw and permuted, with the `lm_head` on the CPU;
  - `belebele/`: Belebele;
  - `rag/`: open-book SinhalaMMLU.

  Every row of `MMLU_Debiased` Appendix B was re-scored on this machine. `SinLlama_wiki` re-scored
  there drifts from its MI300X run (permuted +0.43 pp, p = 0.015), so lab and pod rows are not mixed.
- **Permuted SinhalaMMLU:**
  - `textbook_mix` 38.61 / 41.58 against its parent's 38.67 / 41.44, not significant.
  - `google/gemma-4-E4B-it` 37.74 / 41.55.
  - Gemma with a textbook LoRA 38.31 / 42.35, still not significant against the base.

  So about 41.5 % looks like a capacity ceiling for models this size, not a gap in SinLlama's data.
- **Open-book SinhalaMMLU (`MMLU_Debiased` Appendix C, `RAG_Subject_Breakdown`):**
  - **Setup:** bge-m3 retrieves the top 3 of 33,090 textbook passages for each question.
  - **`wiki_uc_gen_mix`:** 38.59 % → 47.66 % (+9.07 pp, p = 8e-58). The gain comes from grade 6–11
    questions whose subject has a textbook (+15.7 pp). Retrieving with HelaBERT_Large instead of
    bge-m3 gives 39.40 %, not significant.
  - **`textbook_mix`:** 38.72 % → 46.74 %.
- **Belebele** (`results/latex/Belebele`). Permuted accuracy for Sinhala with the passage:
  llama-3-8b 53.3, `SinLlama_v01` 66.7, `wiki_uc_gen_mix` 73.6, `textbook_mix` 72.6.

### Fixed
- `rag_eval.py` retries an out-of-memory error outside the `except` block. Retrying inside it, as
  `permute_eval.py` does, cascaded and left about 4,300 items unscored.
- Coverage is split by subject and by level. The corpus stops at grade 11, so A/L questions count as
  having no textbook even when their subject has one.

## 2026-10-21 – 10-30: School textbooks (phase 15)

### Added
- **The Sinhala textbook corpus** (`textbooks/`, `be1065a`, `2d17bc5`):
  - **Source:** 113 Sinhala-medium government textbooks for grades 6–11, from edupub.gov.lk,
    converted to Markdown (about 18.9M characters).
  - **Extraction:**
    - arts books are rule-based;
    - technical books use rules plus a local Qwen3.5-9B vision model, on maths and science
      fraction pages only.
  - **Not extracted:** the scanned books, which still need OCR.
- **The textbook SFT pipeline** (`textbook_sft/`, `811e82a`, `5f5b83e`, `17aa8bc`):
  - **Data:** 13,180 textbook rows plus 1,988 cleaned UltraChat replay rows, at equal token counts.
  - **Validator:** checks for leakage across sources.
  - **`docs/lora-hyperparameters.md`.**
- **The `SinLlama_textbook_mix` model** (registered in `benchmark/main.yml`, `b6915ba`):
  - **Training:** LoRA r8 on `wiki_uc_gen_mix`, on an A40. It stopped early, with the best
    checkpoint at step 500.
  - **MMLU:** Sinhala 38.63 % (p = 0.70 against the parent) and English 50.63 % (−1.45 pp,
    p = 2.3e-8).
  - **Downstream macro-F1:** writing 97.77 / news 90.78 / sentiment 79.72.
  - **Answer-position bias grew:** Sinhala TVD 12.7 → 15.5 pp.
- **The `SinLlama_textbook_cpt` model** (`textbook_cpt/`, `add7e6a`): plain causal-LM LoRA on the
  textbooks only, LR 1e-4 for 5 epochs.
  - SinhalaMMLU 36.22 %, −2.59 pp against the parent (p = 5.7e-6). Its answers lean to option 3.
  - The English MMLU run and the downstream probes were stopped.
- **Other evaluators:**
  - `benchmark/mmlu/eval_hf_model_on_pod.py` (`46b968b`) scores a Hugging Face Hub model on
    SinhalaMMLU on a remote pod.
  - `eval_sinhalalm_custom.py` (`2795eb7`) scores HelaAI's 512-token SinhalaLM.
  - `inference/100_prompts_q_and_a/gen_question.py` (`6a036a9`) writes the reverse questions for the
    judge.
  - 8-bit loading in `evaluate_sinhala_mmlu.py` and `generate_sinhala_mmlu.py` (`c885141`).

### Known issues
- SinhalaMMLU is built from grade 6–11 term tests on the same curriculum as the textbooks:
  - 31 of 7,044 questions have at least 50 % 6-gram overlap with the corpus;
  - 252 have some overlap.

  At the user's request only the standard benchmark is reported, with no decontaminated subset.

## 2026-10-15 – 10-20: Wikipedia SFT v2 and the LLM judge (phase 14)

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
- `SinLlama_wiki_uc_gen_mix` in `benchmark/main.yml`, scored raw like its siblings. On 2026-10-19 it
  was run through the standard harness on an NVIDIA A40, the same GPU as `SinLlama_wiki_clean`.
  Differences are against `SinLlama_wiki_clean`, with exact McNemar p-values for MMLU:
  - **MMLU:** Sinhala 38.80 % (+0.69 pp, p = 0.14). English 52.08 % (+1.08 pp, p = 5.3e-5), the
    best English score of any SinLlama-line checkpoint.
  - **Global-PIQA:** 4-split average 50.32 % (wiki_clean 50.58 %), within noise.
  - **Downstream macro-F1:** news 91.41 / sentiment 78.88 / writing 97.65, against wiki_clean's
    92.44 / 80.25 / 98.53. That is −2 / −12 / −10 test items; it still beats its parent UC_Gen on
    news by +2.23 F1.
  - Results are in `gs://sinllama_cpt/{mmlu_results_bf16,mmlu_results_english,piqa_*,downstream_*}/`.
  - A debiased-MMLU run was started and cancelled; it has no permuted or calibrated score.
- Raw MMLU for `SinLlama_UC_Gen_Model` (2026-10-19, NVIDIA L4, batch size 8), filling its
  missing-score gap:
  - Sinhala 38.09 %: −0.93 pp against its parent `uc_instruct_cleaned` (p = 0.011).
  - English 51.92 %: +1.57 pp (p = 1.1e-7).
  - Its three stage-3 children against it, net items Sinhala / English:
    - `wiki`: +46 / −42 (not significant).
    - `wiki_clean`: +1 / −129 (English p = 0.0011).
    - `wiki_uc_gen_mix`: +49 (p = 0.047) / +23 (not significant).
- Both runs are added to the reports in `results/latex/` and `results/pdf/` (gitignored), with the
  raw predictions under `results/data/` and entries in `results/data/README.md`:
  - **`SinLlama_wiki_uc_gen_mix`** (marked ◇) is in `MMLU_Benchmark`,
    `MMLU_Answer_Distribution`, `MRL_Benchmarks`, `MRL_Answer_Distribution` and
    `Downstream_Tasks`.
  - **`UC_Gen`** (marked §) is in `MMLU_Benchmark`, `MMLU_Answer_Distribution` and
    `MRL_Answer_Distribution`. The last uses UC_Gen's existing 2026-08-22 PIQA run.
  - **`results/latex/Model_Comparison.tex`**: a one-page table comparing v01, v02, UC_Gen,
    wiki_clean and wiki_uc_gen_mix on MMLU, Global-PIQA and the 100-prompt judge.
- `docs/research-history.html`, a saved copy of the published research-history page.
  `docs/model-tree.html`, a standalone model tree updated with `SinLlama_wiki_uc_gen_mix`. Both
  are gitignored as `*.html`.

### Changed
- Two readings in the benchmark reports were corrected once UC_Gen's raw MMLU existed:
  - **The English MMLU recovery happened in the gen stage, not the Wikipedia stage.** Against
    UC_Gen, `wiki` and `wiki_uc_gen_mix` are flat in English, and only `wiki_clean` loses ground.
  - **The Sinhala first-option answer bias is inherited from UC_Gen, not added by stage-3 SFT.**
    UC_Gen puts 38.2 % of its answers on option 1 (TVD 16.0 pp; 85 and 87 of 92 on non-parallel
    PIQA). Every stage-3 child lowers that bias (TVD 11.6–12.7 pp). Earlier text had compared the
    children with `uc_instruct_cleaned`, two stages back.

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
