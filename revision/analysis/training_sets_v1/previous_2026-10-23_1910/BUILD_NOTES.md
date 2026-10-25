# Training sets v1 (T6) — build notes, 23 Oct 2026

Built by `python3 analysis/training_sets_v1/build_arms.py` (stdlib only, seed 20261023, refuses to overwrite).
Re-running the script into a scratch copy reproduced every output byte-for-byte. Inputs and their sha256 are in
`build_summary.json` (candidates.json, 22 label files, keyword_sentiment.py, LM dictionary, rubric v2).
No new labelling: final labels = the 1,100 model labels in `analysis/corpus_v2/labels_*.json` with review overrides
applied (0 overrides at build time). `exclude` items are never used.

## Reading of the plan
- `tasks/plan.md` line 40 (T6): "(a) LIME-guided: 300 neutral lexicon-loaded; (b) random-neutral: 300 neutral drawn at
  random from all labelled neutrals, disjoint from (a)'s items where possible; (c) mixed: 100 loaded-neutral +
  100 positive + 100 negative. AC: each 300 rows". Read as: (a) and (b) are **all-neutral** 300-row sets with no
  positives/negatives; (c) is 100/100/100. Each set is the whole fine-tuning corpus for its arm.
- Dom's decision, 23 Oct: (a) and (b) **may share** neutrals (supersedes "disjoint where possible").
- "Loaded" (plan.md line 38, T4 rule): >=1 token from `_POSITIVE_WORDS | _NEGATIVE_WORDS` in
  `investor-sentiment-dashboard/backend/app/models/keyword_sentiment.py` (198 terms), tokenised as in
  `analysis/corpus_rule.py`. As in progress.py assumption A1, random-group items with a hit count as loaded.
- Low-information neutrals **count** (progress.py A2: the plan does not exclude them). Counts without them are reported.
- Fallback (plan.md line 83, Risks): "if <300, relax to >=1 hit from an extended list and report".

## Widening procedure (pre-specified, label-blind order)
1. Source: Loughran-McDonald Master Dictionary 1993-2025 (Notre Dame SRAF, https://sraf.nd.edu/loughranmcdonald-master-dictionary/,
   free for academic use), saved unchanged as `sources/LM_MasterDictionary_1993-2025.csv` (sha256 e2d13286…859d).
   Terms whose `Positive` or `Negative` flag is > 0 (negative = removed), lower-cased: 2,613 not already in the base lexicon.
2. Order: document frequency in the texts of all 1,100 pool items (labels not consulted), descending, ties alphabetical.
   213 terms occur at least once; the other 2,400 cannot load anything and were dropped.
3. Add terms in that order; stop at the first term after which loaded neutrals >= 300.
4. Arm (a) takes base-loaded neutrals, then neutrals loaded by each added term in order (seeded sample of the last
   term's neutrals if it overshoots).

## Result
- Base lexicon: 225 loaded neutrals (58 low-info). All 213 LM terms were added without reaching 300:
  only 14 terms brought new neutrals (21 in total) -> **246 loaded neutrals; arm (a) is 54 short of 300.**
  Terms in `lexicon_extended.txt` (all 213, in order); per-term counts in `widening_log.json`.

| term | LM list | doc freq (1,100) | new loaded neutrals | of which low-info |
|---|---|---|---|---|
| good | positive | 63 | 3 | 1 |
| best | positive | 24 | 2 | 0 |
| breakdown | negative | 19 | 1 | 1 |
| question | negative | 19 | 2 | 2 |
| better | positive | 17 | 3 | 0 |
| lose | negative | 8 | 1 | 0 |
| able | positive | 7 | 2 | 1 |
| ill | negative | 5 | 1 | 0 |
| against | negative | 4 | 1 | 0 |
| highest | positive | 2 | 1 | 0 |
| friendly | positive | 1 | 1 | 1 |
| popular | positive | 1 | 1 | 0 |
| seized | negative | 1 | 1 | 0 |
| unemployed | negative | 1 | 1 | 0 |

- Without low-info neutrals: 167 base-loaded, 182 after all 213 terms (118 short).
- The 1,100 items contain only 304 neutrals; 58 contain no base or LM term (24 low-info). Reaching 300 loaded
  neutrals would require loading 300 of 304 neutrals, i.e. "loaded" would stop meaning anything; more labelled
  data (or a smaller arm) is the only way to a 300-row arm (a). Not done here (no new labelling authorised).

## Arms
| arm | rows | neutral | positive | negative | low-info | channels (reddit/twitter/news) | groups (lexicon_hit/random) |
|---|---|---|---|---|---|---|---|
| a | 246 | 246 | 0 | 0 | 64 | 165/70/11 | 173/73 |
| b | 300 | 300 | 0 | 0 | 87 | 181/99/20 | 171/129 |
| c | 300 | 100 | 100 | 100 | 26 | 147/143/10 | 219/81 |

- (b): seeded sample (seed+2) of 300 of the 304 labelled neutrals. (c): seeded (seed+3) 100 base-loaded neutrals,
  100 of 464 positives, 100 of 297 negatives.
- Overlap: a∩b 243 (of 246), a∩c 100, b∩c 100, a∩b∩c 100 (c's neutrals are all in a and b).
- Checks (asserted in the script and re-verified): no duplicate ids or normalised texts within any arm; no overlap with
  the frozen 338 or the original 359 benchmark ids/texts; labels only positive/negative/neutral; manifest rows align
  with data rows.

## Files
`arm_{a,b,c}.json` ([{text, label}], the format `finetune_cv.py` loads), `arm_{a,b,c}_manifest.json` (ids, item,
label, low_information, channel, ticker, sampling group, base hits, loaded_by, per-channel counts, data sha256),
`lexicon_extended.txt`, `widening_log.json`, `build_summary.json`, `sources/`.

## Caveats for Dom / reviewers
- Arm (a) is 246, not 300, so the plan's "each 300 rows" AC fails; options: label more candidates, cut every arm to 246
  rows, or report (a) at 246.
- (a) and (b) share 243 items, so the LIME-guided vs random-neutral contrast is nearly void with this pool.
- The widening's stopping rule used labels (counting neutrals) but the term order did not.
- 'question', 'able', 'good' etc. are general-English LM terms; with only 21 neutrals added the base lexicon still drives (a).
- Labels are model labels only; Dom's 30-item blind neutral check (T5 AC) has not been done.
