# Cover letter

[Date]

To the Editors
*Data Science in Finance and Economics* (DSFE), AIMS Press
Special Issue: Machine Learning in Economics and Finance

Dear Editors,

We are pleased to submit our manuscript, "Diagnosing and mitigating the neutral-class
deficit in financial sentiment analysis: An explainability-guided approach on
multi-source investor text," for consideration in the special issue on Machine
Learning in Economics and Finance, following your kind invitation.

**Why this paper is a valuable addition.** Domain-adaptive models such as FinBERT
are the default tool for financial sentiment classification, and their weakness on
the neutral class is widely acknowledged, yet it is almost always reported as an
aggregate error rate and left unexplained. This manuscript makes that weakness the
object of study. Its contribution is methodological: it shows that the neutral-class
deficit is not diffuse noise but a single, lexically localised, and repairable
failure mode, and it demonstrates an explainability-guided loop that closes the gap
between diagnosing a model error and fixing it. Using a purpose-built multi-source
benchmark of retail, microblog and news text, and proper paired significance testing
(a McNemar test and stratified-bootstrap confidence intervals that comparable applied
studies frequently omit), the paper establishes a significant advantage of FinBERT
over a finance-lexicon baseline. It then uses LIME as a diagnostic instrument to trace
the model's low neutral recall to the systematic misattribution of contextually
neutral risk vocabulary, and shows that a small, targeted fine-tuning intervention
recovers neutral recall by 18.4 percentage points without degrading performance on the
polarised classes.

**Relation to previously published work.** The study builds on the financial
sentiment literature (Araci, 2019; Malo et al., 2014) and on recent surveys of
explainable AI in finance (Yeo et al., 2025), which identify the use of
interpretability as a feedback mechanism, rather than as a presentation or offline
audit layer, as under-explored. To my knowledge, this paper is a direct demonstration
of that loop on realistic multi-source financial text, and it foregrounds statistical
rigour (paired testing and bootstrap intervals) that is often missing from applied
sentiment comparisons. Limitations, including benchmark scale and the single-seed
nature of the fine-tuning result, are stated plainly in the manuscript, and reusable
code for a cross-validated extension and an external benchmark comparison accompanies
the submission. The work originated in the second author's undergraduate project at
Loughborough University.

**Reviewers.** I have not nominated specific reviewers and am happy to leave
referee selection to the editorial team. I have no opposed reviewers to declare.

**Declarations.** We confirm that this manuscript is original, has not been published
elsewhere, and is not under consideration for publication by another journal. Both
authors agree to publication in AIMS Press Open Access format under the Creative
Commons Attribution License. We understand the submission may be screened by the
CrossCheck originality-detection service. We confirm that permission has been
obtained for any reproduced material. Generative-AI use is disclosed in the manuscript
in the dedicated declaration section. The Article Processing Charge is covered by
Guangzhou University.

Thank you for your consideration.

Yours sincerely,

Professor Stephen Lynch (corresponding author)
Dominic Hubble
Department of Computer Science, Loughborough University, Loughborough, LE11 3TU, UK
Email: s.lynch@lboro.ac.uk
Tel: +44 (0)1509 225854
ORCID: https://orcid.org/0000-0002-4183-5122
