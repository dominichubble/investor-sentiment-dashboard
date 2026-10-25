# Training sets v1 (T6) — rebuild, 23 Oct 2026 ~20:36 (UTC+1)

The 19:10 build (arm (a) 246/300) is kept unchanged in `previous_2026-10-23_1910/` (moved, not deleted; its
`build_arms.py` expects to sit in this folder, so run it from a copy placed here if it is ever re-run). `sources/`
(LM dictionary) stayed in place because both builds read it.

## What changed: more labelled data (Dom chose "label more")
- New draw `analysis/build_corpus_extension.py` -> `analysis/corpus_v2/candidates_ext_1100.json` (+ `sampling_manifest_ext_1100.json`,
  `blind_1100.json` .. `blind_2250.json`). Same code path as the original T4 draw (`corpus_rule.eligible_frame`,
  `read_lexicon`, `sample_candidates`, `db_lookup.connect`, same read-only query), new seed **20261023**, existing 1,100
  candidates removed from the eligible frame by id and normalised text, original 359 + frozen 338 benchmark excluded.
  Size 1,200 at the original 7:4 ratio: 764 lexicon-hit + 436 random. Items c2_1100 .. c2_2299.
- Labelled with the identical method as batches 0150-1050 (codex exec, gpt-6-astra, effort high, read-only sandbox,
  unchanged rubric v2 prompt, 50 per batch, validated import). 24 batches, no retries. Log: `labelling_pipeline/batches/run_log_1100_2299.jsonl`.
- Pool now 2,300 labelled items: positive 979 / neutral 593 / negative 635 / exclude 93; low-information 191.
- `labelling_pipeline/common.py` `load_candidates()` now appends `candidates_ext_*.json` after `candidates.json`.

## Build rules (build_arms.py, seed 20261023)
- Plan reading as before (plan.md line 40): (a) and (b) are all-neutral 300-row sets; (c) 100/100/100.
- "Loaded" = ORIGINAL lexicon rule (plan.md line 38): >=1 token of `_POSITIVE_WORDS | _NEGATIVE_WORDS` (198 terms).
  Random-group items with a hit count (progress.py A1). Low-info neutrals count (A2).
- (a): 423 loaded neutrals exist -> seeded sample of 300 (seed+1). **No widening needed**; the LM procedure was not run
  (`terms_added` empty; `lexicon_extended.txt` = base list only).
- (b): seeded sample (seed+2) from neutrals NOT in (a). Only 293 exist (593 - 300), so all 293 are used plus a seeded
  top-up of 7 from (a) -> overlap a∩b = 7.
- (c): seeded (seed+3): 100 base-loaded neutrals + 100 positives + 100 negatives.

## Result
| arm | rows | neutral | positive | negative | low-info | reddit/twitter/news | lexicon_hit/random group | base-loaded |
|---|---|---|---|---|---|---|---|---|
| a | 300 | 300 | 0 | 0 | 87 | 199/92/9 | 239/61 | 300 |
| b | 300 | 300 | 0 | 0 | 105 | 153/124/23 | 107/193 | 130 |
| c | 300 | 100 | 100 | 100 | 31 | 138/146/16 | 217/83 | (100 neutrals all loaded) |

Overlaps: a∩b 7, a∩c 62, b∩c 40, a∩b∩c 2.
Checks (asserted and re-verified): no duplicate ids/normalised texts within any arm; no overlap with the 359+338
benchmark; manifest rows align with data rows; re-running into a scratch copy reproduced every output byte-for-byte.
All 46 label files: 2,300 ids, each once, ids match blind files, no duplicate texts or DB ids across the pool.

## Caveats
- If low-information neutrals were NOT counted, only 299 non-low-info loaded neutrals exist (1 short).
- (b) is "random neutral" but, because it must avoid (a), 130 of its 300 are lexicon-loaded neutrals that (a) did not
  draw; it is not a random sample of all neutrals. The a-vs-b contrast is 100% vs 43% loaded.
- Labels are model labels only; Dom's 30-item blind neutral check (T5 AC) is still outstanding.
