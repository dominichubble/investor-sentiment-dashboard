# corpus_v2 labelling pipeline (T5 scaffolding)

Stdlib-only Python (run with `python3` or `.venv/bin/python`, both 3.14). Run from `~/Work/dsfe-revision`.
Nothing here edits an existing file: label files are created with mode `x` and set read-only, logs are append-only,
and final labels (model label + overrides) are computed when read.

| File | Purpose |
|---|---|
| `common.py` | paths, pool loading (checks `c2_NNNN` order), benchmark blocklist, label loading, override application |
| `label_batches.py` | next batch / export prompt / import reply / API stub |
| `review.py` | terminal accept/override/skip review; `--blind` 30-item neutral check; `--report` |
| `progress.py` | coverage, class mix by group, overrides, the three T6 arms toward 300, neutral-shortfall estimate |

## 1. Labelling (`label_batches.py`)
Global options go **before** the subcommand: `--batch-size N` (default 50), `--out DIR` (test folder), `--dry-run`.

```
python3 analysis/labelling_pipeline/label_batches.py next              # read-only: next ids, files involved
python3 analysis/labelling_pipeline/label_batches.py export            # blind file (reused if identical) + prompt
#   -> paste analysis/labelling_pipeline/batches/prompt_0150.md into Codex (pick the model + effort first)
#   -> save the model's JSON reply as analysis/labelling_pipeline/batches/inbox_0150.json
python3 analysis/labelling_pipeline/label_batches.py import --start 150 --labeller "OpenAI GPT-6 Astra (gpt-6-astra)"
#   -> validates, then writes NEW analysis/corpus_v2/labels_0150.json (read-only); repeat
```
* Order: pool order (`c2_0000`..`c2_1099`, already shuffled by `corpus_rule.sample_candidates`); the next batch starts at
  the first unlabelled item. With size 50 the existing `blind_NNNN.json` files are reused byte-for-byte, so
  `input_sha256` matches the earlier convention. Other sizes create `blind_NNNN_nK.json` (new files only).
* Validation before saving: reply is JSON (code fences allowed), no duplicate keys, every batch id exactly once, no extra
  ids, label in positive/negative/neutral/exclude, `low_information` boolean and only on neutral (same rule as
  `save_corpus_labels.py`), non-empty rationale, and no batch item overlapping the benchmark by id or normalised text
  (frozen 338 **plus** all 359 original ids/texts, per the plan.md 23 Oct amendment).
* The labels file schema/meta is identical to `labels_0100.json` (`labeller`, `reasoning_effort`, `method`, `rubric`,
  `rubric_sha256`, `date` (UTC), `input_sha256`, `fields_shown`, `blindness`).
* **`--labeller` must name the model actually selected in Codex for that turn.** The default is the Astra string used so
  far, but see the caveat below about `labels_0100.json`.
* Idempotent/resumable: an existing `labels_NNNN.json` is never touched; re-running import says so and exits 0.
* The prompt contains a rule summary plus the full verbatim text of `analysis/LABELLING_RUBRIC_v2.md` (sha256 recorded).

API mode (stub, untested against the live API, disabled by default):
```
OPENAI_API_KEY=... python3 analysis/labelling_pipeline/label_batches.py api --model gpt-6-astra --confirm-api [--reasoning-effort high]
```
Without `--confirm-api`, or without `OPENAI_API_KEY`, it refuses (exit 2). It posts one Responses API request per batch,
stores the raw reply as `batches/api_reply_NNNN.json`, then runs the same validation/import with
`labeller="OpenAI API (<model>)"`. Check that the key can see the model (`/v1/models`) before relying on it.

## 2. Review (`review.py`)
```
python3 analysis/labelling_pipeline/review.py [--label neutral] [--all]      # a=accept, o=override, s=skip, q=quit
python3 analysis/labelling_pipeline/review.py --blind                         # 30 model-neutral items, label hidden
python3 analysis/labelling_pipeline/review.py --report                        # neutral precision of the blind check
```
* Review writes `overrides_log.jsonl` (append-only: timestamp, id, action, old, new, note, reviewer). The latest
  `override` per item wins; `accept` entries only mark items as reviewed.
* `--blind` freezes its sample (30 items, seed 20261023, drawn from model-neutral items) in `blind_check_sample.json` on
  first run; later runs reuse it. Answers go to `blind_check_log.jsonl` and are **not** applied as overrides. Run it
  once enough labelling is done that the sample represents the corpus (it samples from whatever is labelled then).
* Answers are read from stdin, so it can be driven by a piped file.

## 3. Progress (`progress.py`)
`python3 analysis/labelling_pipeline/progress.py` — read-only. Arm rules and assumptions A1-A4 are in `--help`
(plan.md line 40 for T6 composition, line 38 for the lexicon rule, line 83 for the fallback).

## What Dom needs to provide / decide
1. Backend: `codex` (paste prompts into the Codex app) or `api` (needs `OPENAI_API_KEY` in the environment; it was not set
   when this was built) and explicit approval to spend API credit.
2. Model and reasoning effort to record (`--labeller`, `--reasoning-effort`), and to keep them fixed for all batches.
3. Batch size (50 matches the existing files and blind batches).
4. Who does the 30-item blind neutral check (AGENTS.md: Dom does all independent human labels) and when.
5. Decisions on progress.py assumptions A1-A3 (what counts as "loaded"; low-information neutrals in arms; arm overlap).
6. Approval to commit this folder (nothing has been committed).

## Caveat found while building
`labels_0100.json` meta says `OpenAI GPT-6 Astra (gpt-6-astra)` because `save_corpus_labels.py` hard-codes it, but the
Codex session (bd35) turn that produced it on 23 Oct ran with `model=gpt-6.1-sol, effort=high` in its turn context.
Batches 0000/0050 ran under `gpt-6-astra`, effort high. The file has not been changed; Dom should decide how to record it.
The 23 Oct batch was also shown to the model without the channel field although `fields_shown` lists it.
