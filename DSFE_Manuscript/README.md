# DSFE manuscript — Investor sentiment / FinBERT neutral-class study

LaTeX source and reproducibility code for the journal manuscript:

> **Diagnosing and mitigating the neutral-class deficit in financial sentiment
> analysis: An explainability-guided approach on multi-source investor text**

Target venue: *Data Science in Finance and Economics* (DSFE), AIMS Press —
special issue *Machine Learning in Economics and Finance*.

---

## 1. Building the PDF

Requirements: a TeX distribution (MiKTeX / TeX Live) with `newtx`, `natbib`,
`booktabs`, `authblk`, `titlesec`, `hyperref` (all standard).

```bash
cd DSFE_Manuscript
pdflatex main          # pass 1
pdflatex main          # pass 2 (resolves cross-references)
# or simply:  latexmk -pdf main
```

The bibliography is an **inline `thebibliography`** (author–year via `natbib`
bibitem labels), so **no `bibtex` run is required** and the document compiles in
plain `pdflatex`. Verified clean: 17 pp, 0 undefined citations/references, 0
overfull boxes > 20 pt.

### Formatting notes (per the AIMS/DSFE author instructions)
- Times New Roman 12 pt, ~15 pt leading (`newtx` + `setstretch{1.25}`).
- **Author–year citations** (the instructions mandate this; it overrides the
  numbered style seen in some published AIMS PDFs).
- Sentence-case title and headings; three-line (`booktabs`) tables with the
  caption **above**; figure captions **below**; no vertical rules.
- Required end-matter present: *Use of Generative-AI tools declaration*,
  *Acknowledgments* (with funding sentence + APC note), *Conflict of interest*,
  *Data availability*.

### Placeholders to fill before submission
Search `main.tex` for `TO SUPPLY`:
- corresponding-author telephone and full postal address.

See `../GAPS` summary in the chat/handover for the full pre-submission checklist
(DOIs, ISO-4 journal abbreviations, reviewer suggestions, ethics-committee name,
and the AI declaration wording you must confirm).

### Figures
`figures/` contains the four result figures copied from the dissertation
(`metrics_barchart`, `confusion_matrix`, `lime_positive`, `lime_negative`), all
vector PDF. For final submission, AIMS asks for embedded fonts and ≥ vector/300 dpi
— these already satisfy that as vector PDFs.

---

## 2. Reproducibility code (GAPS #7 and #8)

New, runnable scripts were added under `backend/app/evaluation/` to back the
manuscript's statistical claims and to address two reviewer-facing gaps. They
**compute** numbers when run against the real models/data; none are fabricated.

| File | Purpose | Gap |
|------|---------|-----|
| `stats.py` | McNemar test + stratified-bootstrap CIs + metric helpers. Reproduces the recorded `mcnemar_results.json` exactly. | backbone |
| `finetune_cv.py` | Multi-seed (or k-fold) fine-tuning → benchmark eval, reporting mean ± SD and a paired McNemar vs the stock model. | **#7** single-seed |
| `phrasebank_eval.py` | Runs FinBERT + keyword baseline on the public Financial PhraseBank, with McNemar + bootstrap, for external comparability. | **#8** n=200 only |

### Dependencies
The two heavyweight scripts need the project's ML stack plus:
```bash
pip install datasets scikit-learn      # PhraseBank loader + StratifiedKFold
# torch, transformers, numpy, scipy are already in the backend requirements
```

### Running (from the `backend/` directory)

**External Financial PhraseBank comparison (Gap #8):**
```bash
python -m app.evaluation.phrasebank_eval --finbert \
    --config sentences_allagree \
    --output data/evaluation/phrasebank_results.json
```
Produces accuracy / macro-F1 / per-class metrics + bootstrap CIs for both
classifiers and a paired McNemar test — a directly comparable external row.

**Multi-seed cross-validated fine-tuning (Gap #7):**
1. Create the real fine-tuning corpus (see the template
   `data/evaluation/finetune_corpus.EXAMPLE.json`) and save it as
   `data/evaluation/finetune_corpus.json`. It must be the 300-sample
   neutral-but-lexically-loaded set described in Section 3.3 and **disjoint**
   from the 200-sample benchmark in `app/evaluation/labeled_dataset.py`.
2. Run:
```bash
python -m app.evaluation.finetune_cv \
    --corpus data/evaluation/finetune_corpus.json \
    --seeds 13 21 42 84 168 --epochs 3 --lr 2e-5 --batch-size 16 \
    --output data/evaluation/finetune_cv_results.json
# or k-fold sensitivity over the corpus:
python -m app.evaluation.finetune_cv --corpus ... --kfold 5 --output ...
```
Outputs per-run metrics **and** `aggregate_mean_sd` for accuracy, macro-F1, and
neutral precision/recall/F1 — i.e. the variance a reviewer asks for.

### How to update the manuscript with the new numbers
Once `finetune_cv_results.json` exists, replace the single-seed Table 5 caption
language with the mean ± SD from `aggregate_mean_sd`, and add the PhraseBank row
as a new table in Section 5.1. The honest single-seed caveat in Section 5
("Targeted mitigation") can then be softened to reflect the cross-validated
result. **Do not** hand-edit the numbers — paste them from the JSON the scripts
produce.

---

## 3. Validation already performed
- `main.tex` compiles clean (pdflatex ×2).
- `stats.mcnemar_test` reproduces `mcnemar_results.json` to < 1e-9.
- `stats.stratified_bootstrap_ci`, `accuracy`, `macro_f1`, `class_recall`,
  `summarise_runs` smoke-tested.
- `finetune_cv.py` and `phrasebank_eval.py` pass `py_compile`; full runs require
  a machine with the FinBERT weights (and, for #7, the real corpus).
