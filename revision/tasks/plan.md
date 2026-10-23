# DSFE-635 revision: implementation plan

## Context
Reviewers returned Major Revisions on "Diagnosing and mitigating the neutral-class deficit…" (dsfe-635). Deadline is 22 Oct per Stephen, 27 Oct if counted from the journal's 14 days (his reply is pending). Three facts reshape the work:
1. The original 300-sample fine-tuning corpus and the run behind Table 5 are lost, and Table 5 is internally inconsistent (neutral precision 91.8% is impossible at 71.7% recall on 60 items). Everything fine-tuning-related is re-run from scratch.
2. The original 200-item benchmark was not sampled from the corpus. A new benchmark was sampled from the Neon DB (359 items: Reddit 115, news 124, Twitter 120; 183 neutral / 113 positive / 63 negative; labelled by Claude under `analysis/LABELLING_RUBRIC.md`).
3. Dom's blind check of 100 items gave 76.4% agreement, κ = 0.614 (n = 89). Disagreements: 11 of 21 are "LLM neutral, Dom negative" (I under-call negatives, mostly chart calls and price moves). Dom excluded 11 factual-filler items I called neutral. Decisions taken: such filler stays **neutral**, and rubric v2 is verified with a **second blind check of 50 fresh items**.

Other decisions already made: fine-tuning runs **locally on CPU** (Intel i9, 8 cores, 64 GB; switched from Colab on 20 Oct, Colab kept as fallback); label meaning = investor view of the asset discussed; core experiments only, the full LIME study and extra baseline go in the rebuttal (a minimal neutral→positive LIME analysis stays in); Stephen has been emailed about the lost corpus and benchmark inaccuracy.

Intended outcome: a resubmission with freshly run, reproducible results (random-neutral and mixed controls, 5 seeds, paired tests, full confusion matrices, external PhraseBank row), an accurate data description, all wording fixes, and a reviewer-comment → change map for Stephen's letter. The result may be less flattering than Table 5; the plan commits to reporting it either way and reframing the claim if the random-neutral control matches the LIME-guided corpus.

## Dependency graph
```
Rubric v2 ─► relabel 359 ─► Dom check #2 ─► [A] benchmark frozen ─┐
DB candidate pool ─► label ─► corpora (LIME-guided / random / mixed) ─┤
fine-tune harness (finetune_cv.py refactor) ─► local run script ──────┴─► [B] stock+smoke run
   ─► full runs (stock, 3 arms × 5 seeds) ─► analysis (paired stats, tables) ─► [C] results frozen
   ─► LIME minimal ─┐
                    └─► manuscript (data → methods → results → wording) ─► [D] compile + number check
                                                                       ─► response table ─► package
```
Parallel: Dom's check ‖ candidate-pool labelling ‖ harness refactor. The manuscript Data/Methods sections can be drafted while the local runs go.

## Where code goes (reuse first)
- Library code: `backend/app/evaluation/` (existing `stats.py` has `mcnemar_test`, `stratified_bootstrap_ci`, `accuracy`, `macro_f1`, `class_recall`, `summarise_runs`; `metrics.py` has `evaluate_predictions`; `finetune_cv.py`, `phrasebank_eval.py` to be extended, not replaced).
- Data-build scripts: from `analysis/` into `backend/scripts/dsfe_revision/`; outputs into `backend/data/evaluation/v2/`. LIME: reuse `app/explainability/lime_explainer.py` (`LIMEExplainer.explain/explain_batch`) and `backend/scripts/generate_lime_examples.py`.
- Tests: new `backend/tests/evaluation/` (stats and corpus builders need only numpy/scipy; torch paths smoke-tested locally under Python 3.11 / torch 2.2.2 CPU).
- Git: the repo is on `main` with no changes of mine. Before the first repo edit, call `propose_branch` (suggest `dsfe-635-revision`); commit per task, never push or open a PR without asking.

## Phase 1: Freeze the benchmark (fail-fast on label quality)
**T1. Rubric v2 from the 32 disagreements/excludes (S).** Review the 21 disagreements with Dom; write v2 rules (e.g. chart/price-move calls with a clear direction are positive/negative; factual filler = neutral with a `low_information` flag). AC: v2 file lists each rule with an example item; v1 retained. Verify: Dom reads and approves.
**T2. Relabel all 359 under v2, blind to Dom's labels (M).** AC: `benchmark_v2_labels_v2.json` with model name, rubric version, date; κ(v2 vs Dom, same 100 items) reported alongside κ(v1)=0.614, with the caveat that v2 was tuned on those items. Verify: script recomputes both κ from files.
**T3. Dom blind check #2, 50 fresh items (S, Dom ~20 min).** Sample from the 259 items outside the first 100, stratified by channel, labels hidden. AC: unbiased κ on rubric v2 reported. Verify: `analysis/score_check.py` output committed.
**Checkpoint A:** benchmark frozen (SHA-256 of `benchmark_v2_final.json` recorded), both κ values written to the decisions log, Dom signs off. Do not touch labels after this point.

## Phase 2: Corpora (can run in parallel with Phase 1)
**T4. Candidate pool from the DB (M).** Read-only query; exclude all 359 benchmark ids and normalised duplicates; same length/English filters; fixed seed. Draw ~700 lexicon-hit candidates (rule fixed in `corpus_rule.py` *before* labelling: ≥1 token from the repo's `_POSITIVE_WORDS`/`_NEGATIVE_WORDS`) and ~400 unrestricted random candidates. AC: manifest of ids, seeds, filters. Verify: test asserts zero overlap with benchmark ids and texts.
**T5. Label candidates (M, ~1,100 items).** Claude labels in chunks under rubric v2 (open question below: script via API vs in-context). AC: labels + flags saved; Dom blind-checks 30 corpus neutrals (neutral precision reported).
**T6. Build the three training sets (S).** (a) LIME-guided: 300 neutral lexicon-loaded; (b) random-neutral: 300 neutral drawn at random from all labelled neutrals, disjoint from (a)'s items where possible; (c) mixed: 100 loaded-neutral + 100 positive + 100 negative. AC: each 300 rows, disjoint from benchmark, deterministic, manifest with per-channel counts. Verify: `pytest backend/tests/evaluation/test_corpora.py`.
**Checkpoint B1:** corpora frozen and hashed; sizes and source mix reported.

## Phase 3: Harness and a smoke run
**T7. Refactor `finetune_cv.py` into a condition runner (M).** Add: `--arm`, 90/10 stratified train/validation split, model selection on validation (best epoch), `get_linear_schedule` with warm-up (document it), weight decay, device = cpu by default (cuda if present), explicit `max_length=128`, seed, library versions; **save per-item predictions and logits for every (arm, seed)**; stop padding the whole corpus to 512. AC: one JSON per run with config + predictions. Verify: CPU smoke test (8 items, 1 epoch) passes locally in a Python 3.11 venv with `torch==2.2.2` and a pinned older `transformers`/`numpy<2`.
**T8. Local run script `backend/scripts/dsfe_revision/run_all.py` (M).** Runs stock inference on benchmark v2 + keyword baseline, then arms × seeds, one resumable JSON per run (skips finished runs). Needs Dom's OK for two downloads: pinned packages (torch ≈150 MB) and `ProsusAI/finbert` weights (≈440 MB). AC: runs end-to-end at 1 seed × 1 epoch and the first run's wall time is measured to confirm the ~3–5 min/run estimate. Verify: output JSON validates against the run schema. Fallback: Colab if per-run time exceeds ~15 min.
**T9. Stock FinBERT on v2 + refresh baseline tables (S).** Recompute summary, per-class, confusion matrix, McNemar vs keyword baseline on v2 (adapt `scripts/mcnemar_and_per_sample.py`). AC: confusion matrix integers reproduce every reported percentage; neutral errors enumerated (N→pos vs N→neg). Verify: `check_consistency.py` recomputes precision/recall/F1 from counts and asserts they match to the printed rounding.
**Checkpoint B:** stock results on v2 consistent; smoke run works; go/no-go for the full run.

## Phase 4: Full runs and analysis
**T10. Full local run (M, mostly waiting; machine kept awake and plugged in).** stock + 3 arms (LIME-guided, random-neutral, mixed) × seeds {13, 21, 42, 84, 168} = 15 fine-tuning runs (estimated 3–5 min each on CPU; to be measured in T8) + class-weighted/threshold-calibration control on stock FinBERT (cheap, optional if time). AC: 15+ prediction files plus run configs saved.
**T11. `analyze_finetune.py` (M, local, numpy/scipy only).** Per arm: mean ± SD over seeds for accuracy, macro-F1, neutral P/R/F1; full per-class table; pooled confusion matrix; predicted-class distribution before/after; paired McNemar and paired bootstrap CIs for stock vs each arm and LIME-guided vs random-neutral (item-level, on the seed-averaged metric), neutral-subset analysis. AC: emits `results/*.json` and LaTeX table fragments. Verify: unit tests on synthetic predictions with known McNemar/bootstrap answers; reproduces `mcnemar_results.json` on the old data.
**T12. External PhraseBank row (S).** Fix `phrasebank_eval.py` to read the downloaded zip (the HF loading script is no longer supported by `datasets`); run stock + keyword + fine-tuned arms; note FinBERT saw PhraseBank in pre-training. AC: table with McNemar and CIs.
**T13. Minimal LIME error analysis (S–M).** LIME (existing `LIMEExplainer`) on all stock-FinBERT neutral errors on v2, split N→positive / N→negative; aggregate top trigger terms; produce one N→positive example figure; replace correct pos/neg LIME figures. Stability, correct-neutral comparison and masking test stay in the rebuttal unless time allows (stretch S1).
**Checkpoint C:** all numbers frozen under `results/`; decide the narrative (does LIME-guided beat random-neutral?). Dom and Stephen review before manuscript text is finalised.

## Phase 5: Manuscript and response
**T14. Data section.** Benchmark v2 description (frame, strata, period, tickers, sources, class mix, top-up disclosure with 21/60 hit rate, rubric, labeller, κ values); corpus description (rule, labeller, check); accurate description of the original 200 if retained (appendix) or dropped. **T15. Methods.** Full fine-tuning specification table; control arms; statistical protocol. **T16. Results.** Tables generated from `results/` via `\input` (no hand-typed numbers); new figures. **T17. Wording and limitations.** R1-3/R2-1 moderation, abstract "no loss on polarised classes", "generalises beyond FinBERT" → "may", "approximately balanced", merge Figs 1–2, rewrite limitations and reproducibility paragraphs. **T18. Marked-up PDF** (`latexdiff`) and clean PDF. **T19. Response table** (each R1-1..5, R2-1..8, minor 1..5 → change/section or limitation). Each task ≤ 3 files.
**Checkpoint D:** `pdflatex` ×2 clean; `check_numbers.py` confirms every number in the text appears in `results/`; Stephen reviews.

## Phase 6: Package
**T20.** Zip, update `REVISION_TASKS.md`, commit on the branch, send to Stephen for the letter and upload.

## Critical path and schedule (working target: submit Tue 27 Oct; decided 20 Oct)
Journal says "within 14 days" of 13 Oct = Tue 27 Oct. Stephen's email said Tue 22/10, but 22 Oct is a Thursday, so his date is inconsistent; his reply is pending. **First action: Dom checks the due date shown on the resubmission page** (the `aimspress.jams.pub/.../resubmit/...` link in the journal's email) and tells Stephen.

| Day | Work | Dom needed |
|---|---|---|
| Tue 20 (tonight) | T1 done; handoff; start T2 relabel | check #2 done |
| Wed 21 | T2 score + **Checkpoint A**; T4/T5 candidates + labelling; T7 harness; install env | approve Checkpoint A (15 min) |
| Thu 22 | T6 corpora (**B1**); T8 smoke run + T9 stock results (**B**); Data/Methods drafting | label 30 corpus neutrals (15 min); tell Stephen status |
| Fri 23 | T10 full local run (~1-1.5 h); T11 analysis; T12 PhraseBank | none (laptop awake) |
| Sat 24 | T13 minimal LIME; **Checkpoint C**; T14-T15 Data/Methods text | review results (30 min) with Stephen |
| Sun 25 | T16-T17 Results + wording; T18 marked-up/clean PDFs; T19 response table | read draft (1 h), AI-use declaration wording |
| Mon 26 | **Checkpoint D**: compile, number check, Stephen review; fix list | review + sign-off (1 h) |
| Tue 27 | Package and upload (Stephen's letter) | submit |

No spare day: aim for an internal freeze Mon 26 noon. **Cut list if time slips** (in order): T13 beyond the one N->positive example; T12 PhraseBank; mixed arm; 5 -> 3 seeds. If the true deadline is Thu 22, this plan does not fit: ask the editor for an extension rather than cutting the controls.

## Risks
| Risk | Impact | Mitigation |
|---|---|---|
| Random-neutral control matches LIME-guided | High (claim collapses) | Pre-commit to reporting it; reframe as "neutral exposure fixes the deficit; LIME explains why" |
| Few loaded neutrals in the DB (lexicon precision ~35%) | Med | Draw 700 candidates; if <300, relax to ≥1 hit from an extended list and report |
| LLM-label bias (neutral default) | Med | Rubric v2, κ on 50 fresh items, report both κ values |
| CPU runs slower than estimated, or laptop sleeps | Med | Measure in T8; resumable per-run JSONs; keep-awake; Colab fallback |
| Deadline | High | See schedule; ask Stephen for extension if 22 Oct |
| DB secret exposure | Low | `.env` gitignored, read-only queries, never print the URL |

## Open questions
1. Label candidates by in-context chunks (no API key needed) or by a versioned script via the Claude API (reproducible, documented prompt)? Default: in-context, record the model and rubric.
2. Keep the original 200 in the paper as a labelled appendix set, or drop it?
3. Does Dom want the class-weighted/threshold control (reviewer 1's optional ask) or skip it?

## Verification (end to end)
- `./.venv/bin/python -m pytest backend/tests/evaluation` (stats, corpus disjointness/size/determinism, consistency checker).
- Local smoke run produces valid per-item prediction JSONs.
- `check_consistency.py`: every reported precision/recall/F1 is derivable from integer counts.
- `check_numbers.py`: manuscript numbers ⇄ `results/`.
- `cd DSFE_Manuscript && pdflatex main && pdflatex main` with 0 undefined references.

## Deliverables on approval
Copy this plan to `tasks/plan.md` and the task list to `tasks/todo.md` at the working-folder root (neither exists, so nothing is overwritten); the T1–T20 headings become the checklist with their AC and verification lines.

## Execution amendment — 23 Oct
Checkpoint A approved in voice on 21 Oct: evaluate `analysis/benchmark_v2_frozen.json` (338 items), not the historical 359-item file. Continue excluding all original 359 IDs/texts from training. Dom confirmed both human check files were his labels and check #2’s 19 exclusions were unjudgeable. Report reserve sentiment agreement conditional on 31 eligible items; human eligibility decisions inform final membership.

Dom requested frequent local commits and durable multi-agent handoffs. Local branch `dsfe-635-revision` created 23 Oct following explicit proposal via Git (named tool unavailable). Checkpoint snapshots go in repo `revision/`; scripts remain staged in workspace `analysis/` pending migration into the planned backend locations. No pushes or external contact authorised. The original daily schedule is behind; do not silently reduce arms, seeds or controls.
