# Cleaning UltraChat-Sinhala

Why `sft/clean_ultrachat.py` exists, what it removes, and what it deliberately
leaves alone. Companion to [uc-instruct-evaluation.md](uc-instruct-evaluation.md),
which records what the *uncleaned* corpus did to the trained model.

## 1. The problem

`UltraChat_Sinhala` was machine-translated by a system contaminated with the
**JW300 / Watchtower** parallel corpus — a standard ingredient in Sinhala MT
training sets — and it leaks that corpus's furniture into unrelated text. The
damage is mechanical and positional, which is what makes it removable.

The first trained model reproduced all of it faithfully. Asked to describe the
water cycle it emitted `1. සෞඛ්‍යය වාෂ්පීකරණය` ("1. health evaporation") and
numbered its steps `පහයි.` / `හයයි.`. That is not a training bug — it is the
model correctly learning what it was shown.

A verbatim training example:

```
1. සෞඛ්‍යය ඔබේ Shopify ගිණුමට ලොග් වී ...
2. සෞඛ්‍යය ඔබ භාවිතා කරන කොටස් පදනම් කරගත් තේමාව ...
3. මෘදු දෙවන රූපයේ සැරිසරන ලක්ෂණය ...
4. සෞඛ්‍යය අංශය විවෘතව ඇති විට, ...
පහයි. පෙනෙන සැකසුම් පැනලයේ, ...
හයයි. ලබා ගත හැකි නම්, ...
7. සයිටම් වෙනස්කම් සුරකින්න සහ පෙරදසුන බලන්න.
```

Every numbered item carries a junk word; items 5 and 6 lost their numerals
entirely. **99.2% of numbered list items in the corpus are affected**, and
`test_sft` is contaminated at the same rates as `train_sft`, so `eval_loss` on
the raw corpus partly measures fit-to-artifact.

## 2. The rules

Run `python sft/clean_ultrachat.py --rules` to print these from the source.

| # | rule | what it fixes | scale |
|---|---|---|---|
| 1 | `word_numeral_5` / `word_numeral_6` | `පහයי.` / `හයයි.` at line start → `5.` / `6.` — the translator rendered the numerals as the sentences "it is five" / "it is six" | 221,440 |
| 2 | `injected_list_word` | a junk word wedged between a list marker and its content: `සෞඛ්‍යය` (health), `මෘදු` (soft), `සීමාව`, `සයිටම්`, `මයික්`, `මිනීමැරුම්` | 964,134 |
| 3 | `jw300_caption` | `[Nවන පිටුවේ පින්තූරය]` — "[Picture on page N]" | 537,535 |
| 4 | `mediawiki_edit_link` | `[සංස්කරණය]` — "[edit]", a MediaWiki section-edit link dropped into recipes, travel guides and fiction | 127,156 |
| 5 | `tidy_*` | doubled spaces, space-before-punctuation, 3+ blank lines, trailing whitespace left behind by 1–4 | 641,624 |

Counts are substitutions over the full `train_sft.parquet`.

**Corpus cost: 0.89%.** 207,831 dialogues in, **205,973 kept**, 1,858 dropped and
2,297 truncated; 477,099 of 1,315,386 turns modified (36.3%). `test_sft` behaves
identically: 23,106 → 22,917 kept (0.82% cost), 36.0% of turns modified.

**Ordering is load-bearing.** Word-numerals must run first: a line reading
`පහයි. සෞඛ්‍යය foo` only matches rule 2's `^\d+\.` anchor once rule 1 has
turned it into `5. සෞඛ්‍යය foo`. Rules 3 and 4 are position-independent and
mutually disjoint. Tidy runs last because boilerplate spans cluster in runs of
two or more, so removing them routinely leaves doubled spaces.

Rules apply to **both** user and assistant turns — the artifacts appear in user
turns too (captions in 6.9%, edit-links in 1.6%).

### Dialogue repair

Some turns are *nothing but* boilerplate — `[25වන පිටුවේ පින්තූරය]` and no other
text — so cleaning empties them. Rather than discard the whole conversation, the
script truncates to the longest prefix still ending on an assistant turn.
**54% of affected dialogues survive that way**; the rest are dropped.

## 3. What was deliberately not done

An audit swept the corpus along six independent lenses. Its null results are as
useful as its findings, and several tempting rules were measured and rejected:

- **Unicode normalization** — unnecessary. The corpus is already 100% NFC:
  0 of 41,719 sampled turns change under `normalize("NFC")`.
- **Any ZWJ rule** — dangerous. 100% of 535,574 U+200D occurrences are correctly
  preceded by U+0DCA. Inserting ZWJ into bare `්ර`/`්ය` would corrupt කාර්ය,
  පර්යේෂණ, සූර්ය, දුම්රිය and thousands of other correctly spelled words, where
  those sequences are genuine hal-consonant + consonant.
- **Legacy visual-order vowel signs** (627 occ), **exploded rakaransaya
  clusters** (191), **fraction slashes** (98), **orphan variation selectors**
  (195). All real, all under 0.4% of turns, all requiring codepoint
  transposition. Not worth the risk for the yield.
- **False friends, confirmed legitimate:** `රූප සටහන` (338 turns) means
  "diagram" — UML and ER diagrams, teaching aids. Empty `[]` (973) is
  `String[] args` and Python empty lists inside code fences. Neither is an
  artifact.
- Also measured and absent: `මුරටැඹ` (Watchtower masthead) 0 occurrences,
  publication codes (`w05 3/1` style) 0, subscription/copyright lines 0,
  ZWSP/BOM/NBSP/ZWNJ/soft-hyphen/CR all 0, smart quotes and em-dashes 0.

## 4. What cleaning does not fix

The script repairs mechanical damage. It cannot repair **translation quality**,
and nobody should read a cleaned corpus as a good one:

- **Mistranslated technical terms.** The trained model rendered "gradient
  descent" as `ග්‍රිඩ් රේන්ඩින්` — meaningless. No regex reaches this.
- **Hallucinated facts** carried over from translation.
- **Diffused religious vocabulary** — `යෙහෝවා` woven into narrative prose
  (~22 turns per 74k sampled). Positionally unpredictable, so no rule should
  attempt it.
- **The corpus is 98.9% Sinhala.** Only 1.1% of assistant turns are majority
  Latin, so a model trained on it answers English prompts in Sinhala. That is a
  data-composition problem, not an artifact — fixing it means mixing in English
  instruction data, not cleaning.

## 5. Audit provenance, honestly

The six-lens audit was run as a parallel workflow with an adversarial
false-positive reviewer per proposed rule. **It hit a session limit partway
through: two of six lenses completed and every automated verifier failed.**

So the rule set here is *not* the full audit's output. It is:

- the three rules established by direct measurement before the audit ran;
- plus two rules salvaged from the JW300 lens (the caption superset and the
  `[edit]` family), **each re-verified by hand** against a 76,154-turn sample:
  the caption superset provably subsumes the original rule (0 matches lost,
  +690 gained), the `[edit]` rule removes 99.8% of the bracketed
  `සංස්කරණය` family while leaving all 1,404 running-word uses of the same word
  intact, and both leave the `රූප සටහන` and `[]` false friends untouched;
- plus the null results above, which cost nothing to honour.

Findings from the four lenses that never completed — numbered-list variants
beyond the six known junk words, bullet-list injection, code-fence damage,
repetition loops — remain **unaudited**. The residual scan reports ~0.3% of
numbered list items still carrying an unrecognised junk word, which is roughly
where a completed numbered-lists lens would have paid off. Re-running the audit
is the obvious next improvement.

## 6. Running it

```bash
python sft/clean_ultrachat.py --rules      # print the rule table
python sft/clean_ultrachat.py --dry-run    # measure, write nothing
python sft/clean_ultrachat.py              # -> UltraChat_Sinhala/*_clean.parquet
```

It streams by row group, so it never loads the 700 MB file, and it writes to a
**new filename on purpose**: the tokenized-dataset cache in
`build_uc_dataset.py` is keyed on `{filename_stem}_{template_fingerprint}_len{N}`,
so an in-place edit would silently reuse the stale cache built from dirty text.

After cleaning, point `data.train_file` / `data.eval_file` in `sft/config.yaml`
at the `_clean` files and re-run `sft/run_sft_uc.sh` into a fresh `output_dir`.
