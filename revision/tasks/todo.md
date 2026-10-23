# DSFE-635 revision: task list
Plan: `tasks/plan.md`. Status updated 23 Oct 2026. Tick a box only when its verification passes.

## Handoff notes (read first)
- Working folder: `/Users/dhubble/Work/dsfe-revision`. Repo: `investor-sentiment-dashboard/` (on `dsfe-635-revision`; local Git branch created 23 Oct following Dom’s explicit request for frequent commits; unavailable `propose_branch` tool replaced with an explicit branch proposal and ordinary Git). Scripts/data so far: `analysis/`. Sampling venv: `.venv` (Python 3.14). Training venv: `.venv311` (Python 3.11); package installation being verified. Decisions log: `investor-sentiment-dashboard/DSFE_Manuscript/REVISION_TASKS.md`.
- Compute: local CPU, Python 3.11 venv with `torch==2.2.2` (last Intel-Mac wheel). Needs Dom's OK to download packages and `ProsusAI/finbert` weights. Use `request_keep_awake` for long runs. Colab is only a fallback.
- DB: Neon URL is in `investor-sentiment-dashboard/.env` (gitignored). Read-only queries only (`analysis/db_lookup.py` sets read-only). Never print or log the URL.
- **Blindness rules:** do NOT open `dom_label_check_2_completed.csv` (in Downloads) or any file with Dom's labels until the rubric-v2 relabel (T2) is written and saved. Check #2's 50 items (`analysis/labels/dom_check2_items.json`, sha256 starts 09264c24) are a reserve set: never use them to tune the rubric. Dom's check #1 file is `~/Downloads/dom_label_check_completed.csv` (kappa v1 = 0.614, n = 89).
- Labelling provenance: v1 labels by Claude Sonnet 5.5 (rubric v1, `analysis/LABELLING_RUBRIC.md`). Any later labeller must be recorded (model, rubric version, date) in the label file.
- Dom does ALL independent human labels himself. Never substitute model labels for them.
- Do not push, open PRs, send emails or message Stephen without Dom's explicit OK. The AI-use declaration in the paper needs Dom's approval of the wording (LLM labelling, AI-assisted code and drafting).
- Stephen has not replied on the exact deadline (22 vs 27 Oct) as of 20 Oct evening; the latest message from Stephen is 20 Oct 07:57 UTC. Re-check before scheduling.
- Decisions resolved 20 Oct: rubric v2 approved (R1 asset direction for position gains; R2 follow stated direction in charts/prices; R3 skeptical net tone = negative; R4 initiatives/favourable policy = positive; R5 mixed PT lists neutral; R6 factual filler neutral + `low_information` flag). Corpus candidates labelled **in-context** (record model/rubric/date). Original 200: **drop from main results**, appendix only if space. Class-weighted control: **skip unless time allows**. **Downloads pre-approved by Dom:** PyPI packages (torch 2.2.2 CPU, numpy<2, older transformers + small deps) and `ProsusAI/finbert` weights from huggingface.co, into a new Python 3.11 venv.
- Note: under R1, Dom's own check labels on position-gain posts (he marked them positive) will disagree with v2 by construction; report such disagreements as such.
- Still open (Dom's): Stephen's deadline reply (22 vs 27 Oct); laptop plugged in/awake for the runs; app permission mode so the next model does not stall on prompts.
- **Provenance resolved (21 Oct):** Dom confirmed in voice that he personally chose the labels in both completed check sheets. Download-origin metadata does not establish label authorship. Treat the checks as Dom’s independent labels. Dom also confirmed that all 19 check #2 exclusions were unjudgeable, not neutral/factual filler. Preserve these human exclusions unchanged. Checkpoint A was explicitly approved later on 21 Oct; see approval receipt below.

## Schedule: target submit Tue 27 Oct (decided 20 Oct)
- [ ] **Dom: check the due date on the journal's resubmission page and tell Stephen** (his "Tue 22/10" is inconsistent: 22 Oct is a Thursday). Day-by-day table in `tasks/plan.md`. Internal freeze Mon 26 Oct noon; cut list in the plan if time slips.

## Done so far
- [x] Repo cloned, `.git` restored; Neon DB reachable read-only
- [x] Benchmark v2 sampled (359 items, 3 channels), labelled under rubric v1
- [x] Dom blind check #1 scored: agreement 76.4%, kappa 0.614 (n=89)
- [x] Email to Stephen sent (lost corpus, inaccurate benchmark description)
- [x] Compute decision: local CPU (Python 3.11, torch 2.2.2), not Colab (20 Oct)

## Phase 1: Freeze the benchmark
- [x] T1 Rubric v2 from the 32 disagreements/excludes. Done 20 Oct: `analysis/LABELLING_RUBRIC_v2.md` (R1-R7, example items, v1 kept); Dom approved all four rule choices.
- [ ] T2 Relabel 359 under v2, blind to Dom's labels. AC: model/rubric/date recorded; kappa v1 and v2 both reported with tuning caveat. Verify: script recomputes both.
  - 20 Oct: relabel saved before any check file was opened: `analysis/benchmark_v2_labels_v2.json` (claude-opus-5-5, rubric v2, read-only, sha256 ec01c024...). 148 neu / 127 pos / 81 neg / 3 exclude (b2_133, b2_168, b2_199: R7 mention list/thread header, Dom to decide); 72 `low_information`. 59 of 359 changed from v1 (neu->pos 24, neu->neg 20, pos->neu 10, neg->neu 2, neu->exc 3).
  - Scorer `analysis/score_kappa.py` reproduces check #1 v1 exactly (κ 0.614, 76.4%, n=89). Check #1: v2 κ 0.877 [0.78, 0.96], 92.1%, n=89 (optimistic: v2 tuned on these items; 2 of 7 disagreements are R1 by construction). Check #2 (reserve): v1 κ 0.805 [0.61, 0.95], v2 κ 0.807 [0.61, 0.95], 87.1%, n=31 only (19 of 50 marked exclude, incl. 5 items v2 calls positive). Outputs: `analysis/scores/check{1,2}_v1_v2.json`.
  - Provenance confirmed by Dom on 21 Oct: these are his own labels. T2/T3 remain unticked pending full verification/commit requirements; Checkpoint A was approved later on 21 Oct.
- [ ] T3 Dom blind check #2 on 50 fresh items. AC: unbiased kappa on rubric v2. Verify: `score_check.py` output committed. (Scored 20 Oct, see T2; provenance confirmed by Dom 21 Oct; Dom confirmed the 19/50 excludes were unjudgeable on 21 Oct; final membership was approved later at Checkpoint A.)
- [x] **Checkpoint A**: Dom explicitly approved the 338-item proposal in voice on 21 Oct; frozen as `analysis/benchmark_v2_frozen.json`, hash and both kappas in `analysis/CHECKPOINT_A_APPROVAL.md`. Repository decisions-log sync and checkpoint commit are being completed on 23 Oct.

## Phase 2: Corpora
- [x] T4 Candidate pool: 1,100 items saved 21 Oct (700 lexicon-hit + 400 unrestricted random); `analysis/corpus_v2/sampling_manifest.json` records query, seed, filters and hashes. Two focused tests passed plus live assertions: 1,100 unique IDs/texts, zero overlap with all original 359 benchmark IDs/texts, deterministic sample. Channel mix: Twitter 588 / Reddit 469 / news 43. Versioned snapshot and task commit being completed on 23 Oct.
- [ ] T5 Label ~1,100 candidates under rubric v2. AC: labels + flags saved; Dom checks 30 neutrals.
- [ ] T6 Build LIME-guided / random-neutral / mixed sets (300 each). AC: disjoint from benchmark, deterministic, manifest. Verify: `pytest backend/tests/evaluation/test_corpora.py`.
- [ ] **Checkpoint B1**: corpora hashed; sizes and source mix reported.

## Phase 3: Harness and smoke run
- [ ] T7 Refactor `finetune_cv.py` into a condition runner (90/10 split, validation model selection, schedule, max_len 128, device auto, per-item predictions saved). Verify: smoke test.
- [ ] T8 Local run script `run_all.py` (resumable). Needs Dom's OK for pinned packages + FinBERT weights downloads. Verify: 1 seed x 1 epoch end-to-end; measure per-run time (fallback: Colab if >15 min/run).
- [ ] T9 Stock FinBERT + keyword on v2; refresh baseline tables. Verify: `check_consistency.py` (all % derivable from integer counts).
- [ ] **Checkpoint B**: stock results consistent; smoke run works; go/no-go for full run.

## Phase 4: Full runs and analysis
- [ ] T10 Full local CPU run: 3 arms x 5 seeds (+ optional class-weighted control); machine awake and plugged in.
- [ ] T11 `analyze_finetune.py`: mean/SD, per-class, confusion matrices, paired McNemar + bootstrap CIs, neutral-subset, LIME-guided vs random-neutral. Verify: unit tests on synthetic data; reproduces old `mcnemar_results.json`.
- [ ] T12 PhraseBank external row (fix loader to read the zip).
- [ ] T13 Minimal LIME: all neutral errors, N->pos / N->neg top terms, one N->pos figure.
- [ ] **Checkpoint C**: results frozen under `results/`; narrative decided; Dom + Stephen review.

## Phase 5: Manuscript and response
- [ ] T14 Data section (benchmark v2, top-up disclosure, corpora, kappas, original-200 status)
- [ ] T15 Methods (fine-tuning spec table, arms, stats protocol)
- [ ] T16 Results (tables generated from `results/`, new figures)
- [ ] T17 Wording + limitations (R1-3, R2-1, abstract, conclusion, "approximately balanced", merge Figs 1-2)
- [ ] T18 Marked-up (latexdiff) and clean PDFs
- [ ] T19 Response table: R1-1..5, R2-1..8, minor 1..5 -> change or limitation
- [ ] **Checkpoint D**: pdflatex x2 clean; `check_numbers.py` passes; Stephen reviews.

## Phase 6: Package
- [ ] T20 Zip, update `REVISION_TASKS.md`, commit on branch, hand to Stephen.

## Open questions
- [x] Label candidates in-context (decided)
- [x] Original 200: drop from main results (decided)
- [x] Class-weighted control: skip unless time allows (decided)
- [ ] Stephen's reply on the exact deadline (22 vs 27 Oct)

### Human exclusion clarification (21 Oct, voice)
Dom confirmed that the 19 check #2 items marked exclude were unjudgeable. This is a human judgement, not an instruction to change model labels or silently remove items from the benchmark. The reported reserve-set sentiment agreement remains conditional on the 31 human-rated classifiable items (v2 kappa 0.807, 87.1% agreement). Final benchmark inclusion/exclusion handling must be explicitly approved at Checkpoint A; do not retune rubric v2 on the reserve items.

### Checkpoint A proposal (21 Oct)
Prepared `analysis/benchmark_v2_proposed.json`, `analysis/benchmark_v2_proposed_manifest.json`, and `analysis/BENCHMARK_CHECKPOINT_A_PROPOSAL.md` at Dom’s request. 21 unique removals yield 338 items; class counts {"neutral": 135, "positive": 122, "negative": 81}. Original files verified unchanged. Historical proposal: superseded by the explicit approval recorded below.

### Approved benchmark (21 Oct)
Use `analysis/benchmark_v2_frozen.json` for evaluation: 338 items (135 neutral / 122 positive / 81 negative), SHA-256 f6be08fced2b15be75f0b594e81c28eedd94b2bd12a6e7a61a669b6708d82008. Keep the original 359 IDs/texts excluded from corpus sampling as planned, including the 21 dropped evaluation items. The reserve set must not be used for rubric tuning. T2/T3 checkboxes remain pending full verification/commit requirements, not further human approval.

### Active handoff — 23 Oct
Dom explicitly requested process documentation and frequent local commits for work across agents. Branch `dsfe-635-revision` created using local Git after an explicit proposal because the named tool is unavailable. No push authorised. Committed recovery snapshots are under repo `revision/`; workspace `tasks/` and `analysis/` remain live files. Start with repo `revision/README.md` for current recovery pointers. T5: 150/1100 explicit model labels saved (`labels_0000`, `labels_0050`, `labels_0100`); the batch starting 150 has been read but NOT labelled/saved. No human corpus-check sheet yet, no training, no corpus freeze.
