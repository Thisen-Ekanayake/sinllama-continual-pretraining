# COCO VQA dataset (English + Sinhala)

5000 COCO val2017 images → description, one question for students aged 17–18, and its answer. Generated in English with Qwen, then translated to Sinhala with NLLB and repaired.

Scripts are in `VQA/`; everything ran on the lab box (`cse_g2@10.8.100.21`, RTX 4070 Ti Super 16 GB) in `~/vqa/`.

## Output

| File | Content |
|---|---|
| `VQA/vqa_english.json` | 5000 records: `image_id`, `description`, `question`, `answer` |
| `VQA/vqa_sinhala.json` | same 5000 records, same order, Sinhala text |

`image_id` is the COCO id (the file `000000000139.jpg` has id 139).

## 1. English generation

- **Model:** Qwen3.5-9B Q4_K_M + `mmproj-F16` vision projector, served by llama.cpp `llama-server` (`start_server.sh`: 4 parallel slots, `--image-min-tokens 1024`).
- **Script:** `generate_vqa.py`, unattended via `run_full.sh`. It makes two calls per image:
  1. image → description
  2. image + description → question + answer (constrained to a JSON schema)
- **Grounding:** both calls also get the image's 5 human COCO captions (`captions_val2017.json`). Without them the model missed details; for example, it didn't notice that the stop sign in image 724 is upside down.
- **Settings:** max tokens 512; thinking off; temperature 0.7, top_p 0.8, top_k 20 (Qwen's recommended values with thinking off).
- **Result:** 5000/5000 in 3 h 16 min, 0 failures.

## 2. Sinhala translation

**Model:** NLLB-200 3.3B (fp16), `eng_Latn → sin_Sinh`, 4 beams. Text was split into sentences: 47,350 unique sentences, each translated separately.

### NLLB problems found

| Problem | Example | Size |
|---|---|---|
| ZWJ dropped from conjuncts (known, from UltraChat) | `ප් රධාන` instead of `ප්‍රධාන` | ~41k sites |
| Hallucinated stock phrase | "No people or animals are present." → `[25වන පිටුවේ පින්තූරය]` ("[picture on page 25]") | ~14% of sentences on the first try |
| Dropped clauses | "Why X, and what does Y suggest?" → only "Why X?" | 594 sentences |
| Repetition loops | `තණකොළ තණකොළ තණකොළ …` | 123 sentences |
| Rare letters never produced; the English word fills the gap | `සං signs ා` → `සංඥා`, `නිහ quiet` → `නිහඬ`, `historical තිහාසික` → `ඓතිහාසික` | ~880 sites |

- The hallucination happens every time, in fp16, bf16 and fp32, with beam search or greedy. It is not a precision bug.
- Letters NLLB never produces: ඓ (not in its vocabulary), ඍ ඖ ඞ ෟ ෲ. Letters it rarely produces: ඥ ඈ ඬ ඵ ඝ ඨ ඤ.

### How it was cleaned

**Step 1: `translate_nllb.py`** (lab box, GPU, ~1.5 h, via `run_translate.sh`)
- Translates each sentence as is.
- If the result is broken, retries without the final full stop, then re-appends it. This fixes the hallucination.
- If still broken, splits the sentence at conjunctions ("but", "while", …) and translates each part.
- Broken means: a hallucinated phrase, a loop, or a translation shorter than 0.55× the source's length (a normal translation is ~0.95×).

**Step 2: `repair_nllb.py`** (lab box, GPU, ~6 min) retranslates the 756 sentences that were still broken:

| Fix | Sentences |
|---|---|
| first letter lowercased | 363 |
| lowercased + repetition blocking | 146 |
| repetition blocking | 90 |
| split into parts (two-part question → two Sinhala questions) | 72 |
| consensus of 24 samples (MBR, chrF) | 85 |

A single random sample changed the meaning about 1 time in 3 (a cat became a dog). That's why samples are used only as a last resort, and only through the consensus pick.

**Step 3: `postprocess_si.py`** (local, CPU, seconds) rebuilds `vqa_sinhala.json` from the two sentence caches:
- ZWJ restoration with the UltraChat lexicon (`sinhala_normalize.py`): 40,696 joins.
- Rare-letter gaps: the English filler is replaced with the rare letter that turns the fragments into an attested word. 882 fixed.
  - The candidate letters are limited to the rare ones; allowing common letters gave false fixes.
  - Without a source-word match, a fix is accepted only if the letter joins both neighbours into one attested word.
- Stem-prefix ZWJ for loanwords and inflections missing from the lexicon: 627 joins (`ෆ්‍රිස්බී` "frisbee", `හයිඩ්‍රන්ට්` "hydrant"). Real word boundaries (`බස් රථය` "bus") are left alone.
- Doubled final punctuation (`.?`) collapsed: 5 cases.

## Final checks (`vqa_sinhala.json`)

- 5000 records, ids and order match the English file, 0 empty fields
- 0 page hallucinations, 0 repetition loops, 0 unresolved sentences
- Sinhala/English length ratio: median 0.92 (description), 0.86 (question), 0.94 (answer); minimum 0.55

## Known limits

- NLLB word-sense errors are not detected, e.g. tennis "court" → `අධිකරණය` (law court).
- A few sampled repairs have word slips, e.g. "skin" → `පීසා` ("pizza") in one fruit description.
- ~380 English words remain: URLs, names, and nouns NLLB left untranslated (`zucchini`, `countertop`).

## Rebuild

The intermediate files were removed locally. The lab box still has them in `~/vqa/`: `vqa_sinhala.cache.jsonl`, `vqa_sinhala.cache.repair.jsonl` and `repair_nllb.py`'s report.

```bash
# lab box (~/tts_env)
./run_translate.sh                                   # step 1 -> vqa_sinhala.cache.jsonl
~/tts_env/bin/python repair_nllb.py --cache vqa_sinhala.cache.jsonl   # step 2
# local: copy the two caches into VQA/, then
python3 postprocess_si.py --en vqa_english.json \
    --cache vqa_sinhala.cache.jsonl vqa_sinhala.cache.repair.jsonl --out vqa_sinhala.json
```

## Note for other Sinhala NLLB datasets

UltraChat, MMLU, PIQA and HellaSwag (Sinhala) used the same model:
- **Page hallucination:** UltraChat-Sinhala has it in 126 of the 200 dialogues of `samples/part_01.first200.sinhala.jsonl`.
- **Rare-letter gaps:** probably present too; search for `සං sign` or `නිහ quiet`.
