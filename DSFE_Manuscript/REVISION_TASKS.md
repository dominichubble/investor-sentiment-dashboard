# dsfe-635 major revision: task list

Paper: *Diagnosing and mitigating the neutral-class deficit in financial sentiment analysis* (DSFE, AIMS Press).
Decision: Major Revisions, 13 Oct 2026. Journal says "within 14 days"; Stephen's emails say 22 Oct (his 14 Oct email said 12 days, i.e. 26 Oct). **Confirm the exact date with Stephen.**
Division of labour (Stephen, 20 Oct): Dom does the corrections, Stephen writes the letter to the Editor. Dom therefore also owes Stephen a reviewer-comment → change map for that letter.

Reviewer comment refs: R1-1..5 = Reviewer 1 main concerns, R2-1..8 = Reviewer 2 major comments, R2-m1..m5 = Reviewer 2 minor comments.

---

## 0. Blockers found in the repo (do first)

- [ ] **The real 300-sample fine-tuning corpus is not in the repo.** Only `backend/data/evaluation/finetune_corpus.EXAMPLE.json` (3 template rows) exists. Locate the original file (laptop, Colab, old branch). Every experiment below depends on it.
- [ ] **The code that produced the original Table 5 result is not in the repo, and neither are the fine-tuned model's predictions.** `finetune_cv.py` is a new script. The original run's confusion matrix and per-sample predictions have to be regenerated.
- [ ] **`finetune_cv.py` does not match the paper's protocol** (paper §3.x: 90/10 train/validation split, fixed seed). The script trains on 100% of the corpus with no validation split, plain AdamW, no LR schedule, `max_length=512`, and only picks cuda or cpu. Decide which one is canonical, make the script match it, and document it (needed for R1-5).
- [ ] **Compute.** Python env here has no torch. Paper says ~6 min per run on one GPU. Plan for roughly 15–20 runs (see §1): use Colab/GPU, or add MPS support for the Mac.
- [ ] **Which LaTeX is the base?** `DSFE_Manuscript/main.tex` (article class) and `aims_submission/main.tex` (aims.cls) differ. The journal says to revise the version from the resubmission link; Stephen's 14 Oct zip may be a third version. Confirm which one to edit and diff it against the repo.

## 1. New experiments (core of the revision)

- [ ] **Random-neutral control (R1-3, R2-2, the most important point).** Fine-tune on 300 randomly sampled neutral financial sentences under identical settings. Pool must be disjoint from the benchmark (e.g. neutral rows of Financial PhraseBank `sentences_allagree`, or neutral text from the project corpus). Three-way comparison: stock vs random-neutral vs LIME-guided.
- [ ] **Mixed / class-balanced fine-tuning control (R1-3, R2-3).** Same 300 neutral + replay of positive/negative examples. Tests the class-prior-shift explanation.
- [ ] **Class-weighted training or threshold calibration on stock FinBERT (R1-3, "if feasible").** A cheap control: no new data, just re-weighting or a neutral-logit offset.
- [ ] **Multi-seed runs (R1-4, R2-6).** At least 5 seeds (13 21 42 84 168 is already the script default) for every condition above. Report mean ± SD for accuracy, macro-F1, neutral recall, neutral F1.
- [ ] **Paired statistics for fine-tuned vs stock (R1-4, R2-5).** McNemar plus paired bootstrap CIs for accuracy, macro-F1, neutral recall, neutral F1, and a neutral-subset-only paired analysis. `stats.py` already has `mcnemar_test` and `stratified_bootstrap_ci`; the bootstrap needs a paired version. Also report item-level disagreement counts (b/c cells).
- [ ] **Full post-fine-tuning confusion matrix, per-class P/R/F1 for all 3 classes, and predicted-class distribution before vs after (R2-3).**
- [ ] **External validation (R1-4, R2-7).** Run `python -m app.evaluation.phrasebank_eval --finbert --config sentences_allagree` and report the results as a new table. Note the PhraseBank must not overlap with whatever corpus is used for the random-neutral control.
- [ ] **One stronger contemporary baseline (R2-8).** e.g. `yiyanghkust/finbert-tone`, a RoBERTa/DeBERTa financial sentiment model, or a zero-shot LLM. Evaluate on the same 200 items with McNemar vs FinBERT.

## 2. LIME / error analysis

- [ ] **Analyse all 28 neutral errors, not hand-picked examples (R1-3, R2-1).** 17 are neutral→positive and 11 neutral→negative, so the "negative risk vocabulary" story covers fewer than half. Run LIME on every one (`backend/scripts/generate_lime_examples.py` is the starting point) and aggregate top trigger terms separately for N→pos and N→neg.
- [ ] **Explanation stability (R1-3).** Repeat LIME with several random seeds / sample counts per item and report rank stability of the top features.
- [ ] **Comparison group (R1-3).** Run the same LIME analysis on the 32 correctly classified neutral items and compare attribution patterns.
- [ ] **Counterfactual / ablation test (R1-3).** Mask or replace the identified trigger terms and measure how often the prediction flips to neutral.
- [ ] **Add a neutral→positive LIME example figure, and swap the correctly-classified pos/neg LIME figures for neutral-error ones (R2-1, R2-m3).**

## 3. Verify and fix existing numbers

- [ ] **Table 5 is internally inconsistent (R2-4).** With 60 neutral items, recall 71.7% = 43/60, but precision 91.8% is not 43/n for any integer n (n=46 gives 93.5%, n=47 gives 91.5%). The F1 (80.5%) was apparently computed from the rounded percentages. Recompute everything from the real confusion matrix and report the counts. Stock row checks out (32/60 = 53.3%, 32/33 = 97.0%).
- [ ] Re-verify every other figure in the abstract, results and conclusion against the regenerated outputs (18.4 pp, 11.7 pp, 4.5 pp, "within 0.5 points" on polarised classes).

## 4. Documentation fixes (no new experiments)

- [ ] **Benchmark provenance (R1-1, R2-7).** `labeled_dataset.py` stores only `text` and `label` for each of the 200 items: no source, date or ticker metadata. Reconstruct and report: how items were chosen (sampled from the 150k+ corpus, or hand-assembled?), counts per source (Reddit / X / news), class distribution per source, collection period, ticker diversity, inclusion/exclusion rules. If the metadata cannot be reconstructed, say so honestly and moderate the "realistic multi-source benchmark" wording. Report per-source performance if it can be recovered.
- [ ] **Fine-tuning corpus provenance (R1-2).** Who assigned the neutral labels; whether verified independently; how candidates were retrieved; how "lexically sentiment-loaded" was defined (word list / criteria); whether that criterion was fixed before or after seeing LIME results. If it was after, say so. Consider releasing the corpus (or IDs) so it is reproducible.
- [ ] **Full fine-tuning spec (R1-5).** All layers vs head only; original head retained; optimiser, loss, regularisation, LR schedule, tokenisation and max length, validation/model-selection procedure, exact seeds, software versions (torch, transformers, CUDA). Put in a table or appendix.

## 5. Wording changes

- [ ] Moderate "single systematic failure mode" and "systematic, repairable mechanism" unless §2 supports it (R1-3, R2-1).
- [ ] Present the Table 5 gain as preliminary until multi-seed and paired results are in (R1-4).
- [ ] Abstract: soften "no loss on the polarised classes" (R2-m4).
- [ ] Conclusion: "generalises beyond FinBERT" → "may generalise beyond FinBERT" (R2-m5).
- [ ] "balanced across classes" → "approximately balanced (70/70/60)" (R2-m1).
- [ ] Merge or trim Figures 1 and 2, which overlap, and use the space for error analysis (R2-m2).
- [ ] Update §3 / §5 / limitations / future-work text that currently says multi-seed, CV and PhraseBank are "provided as code" or "future work", since the results will now be in the paper.

## 6. Submission package

- [ ] Highlight all changes in the revised manuscript (journal asks for visible tracked changes; for LaTeX use `latexdiff` or a coloured `\added{}`/`\changed{}` macro, or compile a marked-up PDF).
- [ ] Clean (unmarked) revised manuscript as well.
- [ ] Point-by-point response table for Stephen: each reviewer comment → what was changed → section/page, or why it could not be done. The journal explicitly allows "impossible to address" items to be explained in the rebuttal.
- [ ] Rebuild the submission zip (see `DSFE_Manuscript/README.md` section 1 for the build) and upload via the resubmission link.

---

## Suggested triage if time is short

1. **Must do:** §0 blockers, random-neutral and mixed controls, 5 seeds, paired tests, full confusion matrices, Table 5 fix, benchmark/corpus/fine-tuning documentation, all wording changes.
2. **Should do:** all-28-errors LIME analysis with stability and ablation, PhraseBank run.
3. **Could defer to the rebuttal as limitations:** extra baseline model (R2-8), per-source performance if metadata is unrecoverable, a larger independent held-out set. Ask Stephen whether to request a short extension for these.

---

## Decisions and findings log (20 Oct 2026)

- Original 300-sample corpus and the original fine-tuning run **cannot be found** (not in git history, not on disk). Table 5 is therefore unverifiable: re-run from scratch and report only new numbers.
- `.git` was not copied when the session moved folders; re-created by `git init` + `fetch origin main` + `reset`. Repo = `origin/main` plus this file.
- Neon DB (read-only): `sentiment_records` = 159,081 rows (Reddit 85,442; news 39,281; Twitter 34,358; Aug 2017 to Aug 2026). Labels in it are FinBERT's own, so they are not ground truth.
- **Original 200-item benchmark is not in the DB** (0 exact matches; best fuzzy overlap 57% on one generic sentence). Dom confirms the items were **LLM-generated**. The paper must say so, and must drop "spanning the three ingestion channels" / "realistic multi-source". The LLM cross-check kappa (0.9925) is weak evidence for LLM-written items and should be described as such.
- Chosen plan: (1) build a **new ~300-item benchmark sampled from the DB**, stratified by channel, source/ticker/date recorded; (2) label with an LLM, with a ~100-item human spot check by Dom, report agreement; (3) build the **300-sample fine-tuning corpus from the DB pool** (neutral, LLM-labelled, lexicon-loaded by a rule fixed before looking at LIME), plus an equal-size random-neutral control from the same pool; keep PhraseBank only for the external-comparison row (note FinBERT saw it in pre-training).
- Open: where to run GPU fine-tuning (Colab / Mac MPS / university GPU).
- Decisions (20 Oct, later): GPU runs on **Google Colab**. Label meaning = **investor view of the ticker** (PhraseBank taxonomy). Class balance = random sample first, then **top up with the keyword lexicon** (FinBERT-independent) to ~60+ per class, reported in the paper. Scope = **core experiments**; extra baseline model and full LIME study go in the rebuttal as limitations. Stephen to be told about the lost corpus and LLM-generated benchmark now.
- Benchmark v2 sampling frame drawn: `analysis/sample_benchmark_v2.py` -> `analysis/benchmark_v2_sample.json` (seed 20261020; 100 each of reddit/news/twitter; 134 tickers; 38 sources; FinBERT labels not read).
- Benchmark v2 built (20 Oct): `analysis/benchmark_v2_final.json`, 359 items = 284 random (after 16 exclusions: 10 news, 5 reddit, 1 twitter; all spam/boilerplate/mention lists) + 16 replacements (1 excluded, 15 kept) + 60 negative-lexicon top-ups (21 of 60 labelled negative, 26 neutral, 12 positive; lexicon precision ~35%) minus 1. Classes: 183 neutral, 113 positive, 63 negative. Channels: reddit 115, news 124, twitter 120. 155 tickers, 39 sources. LLM labeller = Claude (claude-sonnet-5-5), rubric in `analysis/LABELLING_RUBRIC.md`. Random-sample class mix (natural): 145/98/41 neutral/pos/neg. Report random and top-up items separately. News negatives are scarce (13 total).
- Pending: Dom's blind 100-item check (`analysis/dom_label_check.csv`, drawn from the 284 random items only; top-ups and replacements are unchecked) -> compute kappa. Next build: neutral corpus pool + random-neutral control, excluding all benchmark v2 ids.
- Decision (20 Oct): compute switched from Colab to **local CPU** (Intel i9-9980HK, 8 cores, 64 GB; no usable GPU; torch 2.2.2 is the last Intel-Mac wheel, needs Python 3.11, which is installed). Per-run time estimate 3–5 min, to be measured; Colab remains the fallback.
- Decisions (20 Oct): factual-filler items stay **neutral** in rubric v2; second blind check of 50 fresh items agreed; plan approved and saved in `tasks/plan.md` and `tasks/todo.md` (working-folder root).
- Decisions (20 Oct): rubric v2 approved (`analysis/LABELLING_RUBRIC_v2.md`, rules R1-R7); candidates labelled in-context; original 200 dropped from main results; class-weighted control skipped unless time; PyPI + FinBERT-weights downloads pre-approved for the local CPU run. Open: Stephen's deadline reply.

## Execution update — 23 October 2026

Dom confirmed in voice that both human label checks were his own judgements; 19 reserve exclusions were unjudgeable. On 21 Oct he approved Checkpoint A: 338-item frozen benchmark (135 neutral / 122 positive / 81 negative). Approval receipt, hash and score files are preserved in `revision/analysis/`. Reserve kappa v2 is 0.807 on 31 classifiable items, conditional on excluding 19 human-unjudgeable items; v2 was tuned on check #1, not check #2. Membership now uses reserve eligibility decisions.

On 23 Oct Dom explicitly requested durable documentation and frequent local commits for cross-agent work. `revision/README.md` is the compact recovery handoff; `revision/tasks/` and `revision/analysis/` are versioned snapshots of the outer live workspace. The required named branch tool was unavailable; a branch was explicitly proposed and created with local Git before repo edits. No push authorised.
