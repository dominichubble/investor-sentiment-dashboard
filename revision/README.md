# DSFE revision handoff — updated 23 October 2026

## Read and resume
1. In the live workspace, read `../CLAUDE.md`, `../tasks/todo.md`, `../tasks/plan.md`, then `DSFE_Manuscript/REVISION_TASKS.md`.
2. This directory is a committed recovery snapshot of the outer workspace, not a second live source of truth. Its `tasks/` and `analysis/` preserve the same relative paths. If restoring elsewhere, use it to reconstruct the outer workspace; never overwrite a newer live artifact without comparing hashes.
3. Current task: **T5 corpus labelling**, 150 of 1,100 saved. Resume at `analysis/corpus_v2/blind_0150.json`; it was read but has no saved judgement batch. Read rubric v2 before labelling. Do not overwrite existing labels.

## Confirmed decisions
- Dom confirmed he personally chose both human benchmark check sheets. The reserve set's 19 exclusions were unjudgeable.
- Dom explicitly approved **Checkpoint A**, freezing 338 items (135 neutral, 122 positive, 81 negative). Use `analysis/benchmark_v2_frozen.json`; keep all original 359 IDs/texts out of training.
- Human checks are not replaced by model labels. No corpus human sheet has been made yet. Finish model labelling, draw a reproducible blind sample of 30 model-neutral candidates, send it to Dom, and wait for his judgements.
- Local CPU remains the plan. Python 3.11 environment exists at the workspace root; installation is being verified. No stock inference, fine-tuning or timing trial has run.
- No checkpoint B1 approval; no final corpora. No new manuscript results. Deadline still needs confirmation; target remains 27 October.

## Saved progress and verification
- Benchmark approval receipt includes its hash and both check statistics; original files preserved.
- T4 sampled 400 unrestricted candidates first and 700 lexicon-hit candidates afterwards from the remaining frame (seed and query in sampling manifest). 1,100 unique IDs and normalised texts, zero overlap with all 359 benchmark items. Two sampling tests and live assertions passed.
- Candidates: 588 Twitter, 469 Reddit, 43 news. The sample reflects the eligible database frame, not equal channel quotas.
- Label files record actual model, rubric hash, date, input hash, low-information flag and individual rationale. Batches are saved read-only. The saver rejects missing, extra and duplicate judgements.
- T2/T3 remain unticked for verification requirements despite Checkpoint A approval; stale historical approval questions in earlier notes do not revoke Dom's later approval.

## Commands and artifacts
Run preparation scripts from **the outer workspace root**, not this snapshot directory. Tests: `.venv/bin/python -m unittest discover -s analysis -p test_corpus_sampling.py`. Label saving: `.venv/bin/python analysis/save_corpus_labels.py START < explicit_judgements.txt`. Never regenerate the candidate pool over existing labels.

Scripts/data currently live outside the Git checkout; these snapshots preserve them until migration into the planned backend script/data locations. Secrets, virtual environments and package caches are excluded. Original human CSV files are not included in this snapshot; their score-file hashes preserve input identity, and rerunning those checks requires Dom's local originals.

## Commit practice
Commit after every completed task or saved 50-item labelling batch, updating this handoff and the checklist snapshot at the same boundary. Include exact artifact hashes in `snapshot_manifest.json`; record commands run and unresolved verification in the task notes. Never tick a verification that has not passed. Use local branch `dsfe-635-revision`. Do not push, open PRs, email or contact Stephen without explicit approval.

## Next required user input
Dom's 30-item blind corpus check, then Checkpoint B1 sign-off after corpora are built and verified. Before long training, measure the smoke run and stop if CPU timing exceeds the plan.
