# dsfe-635 pre-submission checklist (sweep of 25 Oct 2026, 17:04 GMT)
Due to Stephen: Tue 27 Oct. Send revised PDF, diff.pdf (highlighted) and the response table.
Line numbers refer to /workspace/dsfe/aims_october/DFSE_Lynch.tex.

## Blockers
- [x] 2. Remove the volunteer annotator from the acknowledgements (l.937) and the ethics statement (l.951-954). -- Done 17:12 GMT: volunteer sentences deleted from Acknowledgments and Ethics statement.
- [x] 3. Make l.891 ("code accompanies this submission") agree with Data availability (l.959-962, "on reasonable request"): upload the code or reword l.891. -- Done: l.895 now says the code is available on reasonable request (matching Data availability). Uploading a code archive is still an option.
- [x] 4. l.274 says the 200-item benchmark is unused, but the ablation (l.769) and the timing (l.776) still use it: rerun both on the 338 items or reword. -- Done: both rerun on the box on 338 items (results_v2/enrichment_timing_v2). The ablation result reverses (entropy 36.1%, lexicon 2.1%) because the original script never passed probabilities. Dom approved the corrected wording (25 Oct 17:18 GMT).
## Should do
- [x] 5. Update the stale line numbers in the response table (R2-8: 459, 586, 566, 629, 884; R2-m1 273-277; R2-m2 489; R2-m5 864). -- Done: make_response_table_v2.py, locations updated to the current tex, page refs dropped.
- [x] 6. Reword or remove the "Also outstanding" footer in the response table. -- Done: footer removed.
- [x] 7. The reproducibility paragraph (l.890-905) should mention the Twitter-RoBERTa, LIME v2 and partial_items scripts. -- Done: sentence added at l.908-912 (plus the enrichment rerun script).
- [ ] 8. Commit the work done since 479ff47 (only on Dom's go-ahead) and sync the box and Mac copies.
- [ ] 9. Dom confirms the AI declaration items: Claude for the benchmark, the 2,300 pool, no overrides recorded, the AI coding/editing sentence.
- [ ] 10. labels_0100.json metadata says Astra but the batch was Sol: fix it (with OK) or leave it with the README caveat.
- [ ] 11. Check whether the alphabetical news tickers are an artefact (tab:source, about l.652).
- [ ] 12. The rebuttal says explicitly that R1-3 is only partly addressed.
- [x] 13. Remove the uncited fleiss1971 and landis1977 references, or cite Landis & Koch at about l.818. -- Done: landis1977 cited at l.822; fleiss1971 bibitem removed.
## Nice to have
- [x] 14. Unify the LIME figure title (97%) with the caption (96.9%). -- Done: plots redrawn from lime_v2.json with 1-dp titles (96.9%; pos 95.0%, neu 55.9%). LIME not rerun.
- [ ] 15. Confirm the Guangzhou University APC line (l.940).
- [ ] 16. Optionally mention Twitter-RoBERTa in the contributions (l.120-145).

## Done
- Both PDFs build (21 and 26 pages); the main PDF has no undefined references.
- Headline numbers match the stats files (stock 0.498, a 0.591, b 0.607, c 0.534, a-b -1.6, OOD).
- 0.877 removed; no TODO markers.
- R1-1, R2-1, R2-7, R2-8 and fig:lime_neu in place.
- Title, abstract, keywords, RQs and contributions match the framing Stephen approved.
- AI declaration rewritten (Claude for the benchmark; Astra plus one Sol batch for the pool).
- No Sunday update owed (Stephen replied 25 Oct 10:04 GMT).
