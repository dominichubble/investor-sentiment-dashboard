# Each row: id, comment (verbatim extract), status, response, change/location
R=[
("R1-1","Data construction is insufficiently documented. [...] the manuscript does not specify the sampling frame, date range, allocation by source, candidate pool, selection process, unit of analysis, or inclusion/exclusion rules. It is unclear whether these observations were sampled from the stated corpus of over 150,000 records or manually assembled separately.",
 "Addressed",
 "The original 200-item benchmark has been replaced by a new 338-item benchmark sampled at random from the system's database. The sampling frame, filters, per-channel allocation, replacements, negative top-up and exclusions are documented. The collection period is now stated: post timestamps span 24 Aug 2017 to 13 Aug 2026 (news 2017-08 to 2026-07, Reddit 2021-02 to 2026-08, Twitter 2022-01 to 2026-04).",
 "Sec. 3.2 (tex l.244-266, incl. new date-range sentences l.262-266): benchmark description; Table tab:source (l.~650-671)."),
("R1-2","The 300-item fine-tuning corpus is not reproducible. [...] they do not specify who assigned the neutral labels, whether the labels were independently verified, how candidate texts were retrieved, how ``lexically sentiment-loaded'' was defined, which words or phrases qualified an item, or whether they defined the criterion before inspecting LIME results.",
 "Largely addressed",
 "The original corpus could not be recovered and is no longer used; this is stated. The new training pool is fully documented: how candidates were retrieved, the labelling procedure, a blind human audit of neutral labels, and the definition of ``lexicon-loaded'' (at least one term of the system's finance sentiment lexicon). The manuscript does not list the lexicon terms and does not state explicitly when the criterion was fixed relative to the LIME analysis.",
 "Sec. 3.3 (tex l.273-323; PDF p.5-6): 2,300-candidate pool; labels 979/593/635/93; blind check 23/30, neutral precision 23/29; arm definitions; deviation note; Table 1 (tab:arms, l.311). Code and arm manifests listed under Reproducibility (Sec. 7, l.785-800)."),
("R1-3","LIME-based diagnosis requires stronger substantiation. [...] The authors should either moderate this language or provide a systematic analysis of all neutral errors [...]. I suggest adding control experiments: fine-tuning on randomly sampled neutral data of the same size; balanced fine-tuning; and, if feasible, threshold calibration or class-weighted training.",
 "Partly addressed",
 "We have moderated the language. LIME is now presented as illustrative and hypothesis-generating, and the ``single, systematic, repairable mechanism'' wording has been removed. We added a matched random-neutral control (arm b) and a mixed/balanced control (arm c). LIME-guided selection did not beat the random control (diff. $-1.6$ points, 95\\% CI $[-5.0,+1.7]$). A systematic LIME analysis of all errors (aggregate attributions, stability, a correctly-classified comparison group, counterfactuals) and the threshold or class-weighted control were not carried out.",
 "Sec. 4.4 (l.372-382, p.7) and Sec. 5.2 (l.489-529, p.9-11): LIME presented as illustrative. Arms in Sec. 3.3 / Table 1. Results in Sec. 5.3 and Tables 4-5 (l.531-596, p.11-12). LIME stability is limitation (vii), Sec. 6.2 (l.710-745, p.13-14)."),
("R1-4","The principal fine-tuning comparison has limited statistical validation. [...] The authors should report repeated-seed results and paired uncertainty estimates or tests for accuracy, macro-F1, neutral recall, and neutral F1, and validate the model on an independent held-out dataset.",
 "Largely addressed",
 "Each arm now has 5 seeds, with mean $\\pm$ sd and 95\\% CIs. Every run gets an exact McNemar test against stock, with Holm correction. A paired item-level bootstrap of seed-averaged macro-F1 covers all six comparisons. Seed-paired $t$/Wilcoxon tests are reported for macro-F1 and accuracy. External validation uses Financial PhraseBank and Twitter Financial News. Paired uncertainty for neutral recall and neutral F1 is not reported, and neutral F1 is not tabulated.",
 "Sec. 4.3 (l.352-371, p.6); Sec. 5.3, Tables 4-5 (l.531-596, p.11-12); Sec. 5.4, Table 6 (l.598-643, p.12-13). Single-seed OOD and $n=5$ power are limitations (iii)-(iv), Sec. 6.2."),
("R1-5","The fine-tuning specification is only partially complete. [...] it does not specify the parameter-update regime, [...] whether the original sentiment-classification head was retained; the optimizer, loss function, regularization, learning-rate schedule, tokenization and maximum-sequence-length settings, validation/model-selection procedure, or the exact random seed and software environment.",
 "Largely addressed",
 "Now specified: optimiser (AdamW, lr $2\\times10^{-5}$), weight decay and its exclusions, linear warm-up schedule, batch size, gradient clipping, maximum length 128, epochs, stratified 90/10 split, model selection on validation macro-F1, exact seeds and hardware. A locked package list is in the released code. Not stated explicitly in the text: that all layers are updated with the original head retained, the loss function (standard cross-entropy), and library versions.",
 "Sec. 4.5 Fine-tuning protocol (l.383-397, p.7); Reproducibility (Sec. 7, l.785-800, p.15)."),
("R2-1","The claimed ``single systematic failure mode'' is not sufficiently established. [...] The authors should provide a systematic analysis of all neutral-class errors, separately considering neutral-to-positive and neutral-to-negative cases. [...] At minimum, an additional neutral-to-positive LIME example should be included. Unless the evidence supports a genuinely unified mechanism, the phrase ``single systematic failure mode'' should be moderated.",
 "Largely addressed",
 "The phrase has been moderated and no longer appears. New Table tab:neuerr aggregates all neutral-related errors for stock, each arm (seed means), Twitter-RoBERTa and the keyword baseline. Stock FinBERT errs mainly by absorbing polar items into neutral (74 pos$\\to$neu, 44 neg$\\to$neu), with gold-neutral errors leaning slightly negative (15 neg vs 10 pos). Fine-tuning reverses this: gold-neutral items are pushed out of neutral, mostly towards positive for arms (a) and (c). A misclassified neutral LIME figure was added. No neutral-to-positive LIME example was added.",
 "Sec. 5.1, Tables 2-3; Sec. 5.2 and Fig. lime\\_neu; Sec. 5.3 neutral-error paragraph and Table tab:neuerr (l.~673-697)."),
("R2-2","A matched control experiment is necessary to demonstrate the added value of LIME-guided fine-tuning. [...] the authors should include at least one matched control in which the model is fine-tuned using the same number of randomly selected neutral financial samples under identical training conditions.",
 "Addressed",
 "Random-neutral arm (b) is matched to the lexicon-loaded arm (a): same 600-item base, 300 neutrals, identical settings, 5 seeds. Arm (a) did not outperform (b) (bootstrap diff. $-1.6$ points, Holm $p=0.51$; seed-paired $t$ $p=0.20$). The paper now reports this negative result and reframes the contribution.",
 "Sec. 3.3 / Table 1; Sec. 5.3, Table 5 (l.553-596, p.12); Abstract and Introduction contribution 3 (l.61-85; l.132-137); Conclusions (l.759+, p.14)."),
("R2-3","The composition of the fine-tuning corpus and the possibility of class-prior shift require clarification. [...] report the complete post-fine-tuning confusion matrix and per-class metrics for all three sentiment classes. The distribution of predicted classes before and after fine-tuning would also be useful. A class-balanced replay or mixed fine-tuning control would further help [...].",
 "Largely addressed",
 "We now state the exact class composition of every arm. A mixed replay arm (c) was added. Per-class recall for all three classes, neutral precision, and mean predicted-class counts are reported before and after fine-tuning; they show a shift away from neutral. Full per-arm confusion matrices and positive/negative precision and F1 are not tabulated, though per-item predictions are released.",
 "Table 1 (l.311); Sec. 5.3 text and Table 4 (l.531-550, p.11-12); predicted-neutral counts in l.~538."),
("R2-4","The values reported in Table 5 should be carefully verified. [...] I strongly recommend that the authors provide the complete confusion matrix for the fine-tuned model and ensure that all values in Table 5 can be reproduced directly from the reported counts.",
 "Addressed (by replacement)",
 "The original Table 5 could not be verified: the corpus and run behind it were lost. It has been withdrawn, and all fine-tuning results were re-run from scratch. All new values are computed from released per-item predictions, and the stock confusion matrix is given in counts.",
 "Old Table 5 removed; new Tables 3-5 (l.477, 547, 583; p.9-12). Lost data acknowledged in Sec. 3.3 (l.274-276) and limitation (vi), Sec. 6.2."),
("R2-5","Statistical significance testing should be applied to the central mitigation result, not only to the baseline comparison. [...] I recommend a paired McNemar test for stock versus fine-tuned FinBERT, together with paired bootstrap confidence intervals for changes in accuracy and macro-F1. A paired analysis restricted to the neutral subset would also be informative.",
 "Largely addressed",
 "Exact McNemar tests against stock for all 15 runs (Holm-15). Paired item-level bootstrap CIs for macro-F1 differences (Holm-6). Seed-paired tests for accuracy and macro-F1. No bootstrap CI for accuracy differences and no paired analysis restricted to the neutral subset.",
 "Sec. 4.3 (l.352-371); Sec. 5.3 (l.553-576) and Table 5 (l.583)."),
("R2-6","Single-seed fine-tuning is insufficient for supporting the robustness of the reported improvement. [...] repeat the fine-tuning experiment across several random seeds and report mean and standard deviation for accuracy, macro-F1, neutral recall, and neutral F1. Five independent seeds would provide a reasonable minimum.",
 "Largely addressed",
 "Five seeds per arm. Mean $\\pm$ sd (with 95\\% CI) are reported for macro-F1, accuracy and per-class recall, including neutral recall. Mean neutral precision is in the text. Neutral F1 is not reported. The text that described multi-seed CV as ``provided as code'' has been removed.",
 "Sec. 4.5 (l.383-386); Table 4 (l.547, p.12)."),
("R2-7","The 200-sample benchmark requires more detailed documentation and stronger external validation. [...] report the number of samples from Reddit, X/Twitter, and financial news, together with the class distribution within each source, the collection period, sampling procedure, ticker diversity, and criteria used for inclusion. It would also be useful to report source-specific performance [...] validation on at least one established external financial sentiment dataset.",
 "Largely addressed",
 "New Table tab:source reports per channel: class mix (news 32/17/66, Reddit 38/29/41, Twitter 52/35/28 pos/neg/neu), collection period, distinct tickers (49/47/79; 152 overall), top-5 ticker share, and macro-F1 for stock, arms (a)-(c), Twitter-RoBERTa and the keyword baseline. Per-channel results are flagged as descriptive (108-115 items per channel). The sampling procedure and inclusion criteria are documented. External validation is on Financial PhraseBank (two subsets) and Twitter Financial News, where all arms fall below stock; this is stated as a limitation.",
 "Sec. 3.2 (l.244-266); Sec. 5.3 and Table tab:source (l.~637-671); Sec. 5.4 and OOD table."),
("R2-8","The comparative baseline should be strengthened. [...] including at least one stronger contemporary neural financial sentiment baseline would provide a more informative reference point.",
 "Largely addressed",
 "Added an off-the-shelf neural social-media baseline, Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest), on the 338 items: macro-F1 56.8\\%, accuracy 59.2\\%. It is +7.0 points over stock FinBERT (95\\% CI $[+0.1,+14.0]$, Holm $p=0.19$) and does not beat arms (a) or (b) ($-2.3$, $-3.9$ points; Holm $p=0.67$). The keyword baseline was also re-run (macro-F1 0.467). No financial LLM baseline was added; this remains future work.",
 "Sec. 5.1 (l.~454-461); Table tab:armres row (l.~581); Sec. 5.3 (l.~561-567); Table tab:boot lower block (l.~624-627); Conclusions future work (iii) (l.~813)."),
("R2-m1","The benchmark is described as ``balanced across classes,'' although the class distribution is 70/70/60. ``Approximately balanced'' or ``near-balanced'' would be more precise.",
 "Addressed",
 "The benchmark has been replaced. Exact class counts are given (135/122/81), and the ``balanced'' description is no longer used for the benchmark.",
 "Sec. 3.2 (l.~268-271)."),
("R2-m2","Figures 1 and 2 communicate closely related information regarding the neutral-class deficit. The authors may consider reducing redundancy [...].",
 "Addressed",
 "Both figures (metrics bar chart and confusion-matrix figure), which showed the withdrawn 200-item data, have been removed. The confusion matrix is now a single table (Table 3).",
 "Sec. 5.1, Table 3 (l.477, p.9)."),
("R2-m3","The current LIME figures for correctly classified positive and negative examples are less informative [...]. Replacing or supplementing them with neutral-to-positive and neutral-to-negative examples would strengthen the presentation.",
 "Largely addressed",
 "Added Figure fig:lime\\_neu: LIME for a gold-neutral sentence that stock FinBERT misclassifies as negative (55.9\\% vs 27.5\\% neutral; top drivers \\emph{cautious}, \\emph{inflation}), referenced in Sec.~5.2. The positive and negative figures were kept. No neutral-to-positive example was added.",
 "Sec. 5.2 (l.~507-514), Figures 1-3 incl. new fig:lime\\_neu (l.~541-548)."),
("R2-m4","The statement in the abstract that the intervention produces ``no loss on the polarised classes'' is stronger than the evidence currently supports [...].",
 "Addressed",
 "The abstract has been rewritten and the claim removed. It now reports seed-averaged macro-F1, the null LIME-vs-random result and the out-of-domain drop.",
 "Abstract (l.61-85, p.1)."),
("R2-m5","The conclusion states that the methodological message ``generalises beyond FinBERT.'' [...] this claim should be softened [...].",
 "Addressed",
 "The phrase has been removed. The conclusion now limits claims to in-domain adaptation of FinBERT and notes the out-of-domain cost.",
 "Conclusions (l.759+, p.14)."),
]
import re
def tex(s):
    s=s.replace('&','\\&').replace('%','\\%').replace('#','\\#')
    s=re.sub(r'\\\\%','\\%',s)
    return s
rows=[]
for i,c,st,r,l in R:
    rows.append(f"\\textbf{{{i}}} & \\emph{{{tex(c)}}} & \\textbf{{{st}.}} {r} & {tex(l)} \\\\ \\midrule")
doc=r"""\documentclass[10pt]{article}
\usepackage[a4paper,landscape,margin=1.6cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{longtable,booktabs,array}
\renewcommand{\arraystretch}{1.15}
\begin{document}
\noindent\textbf{dsfe-635: point-by-point response table (for the letter to the Editor)}\\
\small Comments are verbatim extracts ([\dots] marks omissions) from the reviewer reports forwarded on 13 October 2026. Locations refer to \texttt{DFSE\_Lynch.tex} (line numbers) and \texttt{DFSE\_Lynch.pdf} (page numbers) as of 25 October 2026. Status is our own assessment.
\footnotesize
\begin{longtable}{>{\raggedright\arraybackslash}p{1.1cm}>{\raggedright\arraybackslash}p{8.2cm}>{\raggedright\arraybackslash}p{9.0cm}>{\raggedright\arraybackslash}p{6.0cm}}
\toprule
\textbf{ID} & \textbf{Reviewer comment} & \textbf{Response} & \textbf{Change and location}\\ \midrule\endhead
"""+"\n".join(rows)+r"""
\multicolumn{4}{p{24.9cm}}{\textbf{Also outstanding (not a reviewer item):} the Generative-AI declaration (l.~915-935) has been rewritten: it now names the labelling models (Claude claude-opus-5-5 for the benchmark; GPT-6 Astra, with one 50-item batch by gpt-6.1-sol, for the fine-tuning pool), the author audits and the use of AI assistants for code and editing, and no longer refers to the 200-sample benchmark. The journal also requires highlighted changes; \texttt{diff.pdf} (latexdiff against the October original) provides these.}\\
\bottomrule
\end{longtable}
\end{document}
"""
open('response_table.tex','w').write(doc)
md=["# dsfe-635: point-by-point response table","","Comments are verbatim extracts ([...] marks omissions) from the reviewer reports forwarded on 13 Oct 2026. Locations refer to DFSE_Lynch.tex lines and DFSE_Lynch.pdf pages as of 25 Oct 2026.","","| ID | Reviewer comment | Status | Response | Change and location |","|---|---|---|---|---|"]
clean=lambda s: s.replace('$','').replace('\\pm','±').replace('\\times10^{-5}','×10⁻⁵').replace('\\%','%').replace("``",'"').replace("''",'"').replace('|','/').replace('\\\\','')
for i,c,st,r,l in R: md.append(f"| {i} | {clean(c)} | {st} | {clean(r)} | {clean(l)} |")
md+=["","**Also outstanding (not a reviewer item):** the Generative-AI declaration (l.~915-935) has been rewritten to name the labelling models and audits; the 200-sample text is removed."]
open('response_table.md','w').write("\n".join(md)+"\n")
from collections import Counter; print(Counter(x[2] for x in R))
