# Revision one

## Files to use

Use only the two folders under `FINAL/` for the author handoff:

- `01_Zenodo_upload/`: one ZIP containing the revised code and numerical results.
- `02_PeerJ_upload/`: clean manuscript, tracked manuscript, reviewer response,
  six separate table documents, four separate figure images, a figure-legends
  document and a table-title/note reference document for the upload form.

The manuscript now follows the original document's structure and continuous
line numbering. `minimal_manuscript.py` applies the targeted manuscript edits;
`render_and_compare.ps1` and `verify_redline.ps1` create and check its marked copy.
The remaining files outside FINAL are working/reproduction materials, not extra
files the author must upload. Obsolete packaging/patch scripts, old ZIPs, logs and
superseded page previews have been removed. Reusable builders, original files,
raw data, numerical results, dependencies and validation records are retained.
The reference DOCX in `work/` supplies verified equations and citations to the
manuscript builder; it is not another submission version.

Each Table 1–6 DOCX now contains exactly one editable table, fits one US Letter
page with 2.5 cm margins, and is cited in numerical order in the manuscript.
The official instructions were read at
https://peerj.com/about/author-instructions/cs#table-files on 11 September 2026.
Reviewer 2 requested merging original Tables 3–6 and adding p=0.40. The expanded
comparators, ablations, stress conditions and disruption/replay diagnostics answer
the other reviewer requests. Full numerical detail stays in the archived CSVs.

Work takes place on `revision-one`. Original submitted files are preserved in
`original_submitted/`; revised upload files are in `../submitted_peerj_docs/`.

The revision corrects scientific discrepancies rather than merely editing prose.
The original simulation selected its prefix within a novelty-gated pool and
removed the cold-start penalty after gating. The revised implementation preserves
the actual penalized base prefix and retains that penalty for tail ranking.
Consequently, old numerical results cannot support claims about the revised method.
Historical `outputs/` files are retained for provenance, not for resubmission.

The historical replay merged result pages, used normalized CTR instead of binary
entropy, updated counters from hypothetical rank positions, and reported constant
CI endpoints with bootstrapping disabled. The corrected page-level diagnostic
uses logged exposures and document-aligned clicks. It is not a causal engagement
evaluation. When all ten logged documents remain candidates, set and click
preservation are structural, not evidence of successful exploration.

## Reproduction

Run from the repository root with the versions in `requirements-revision.txt`:

```powershell
python -m src.ede_sanity.revision_experiments --workers 4
python -m pytest tests -q
python -m src.ede_real_replay.revision_replay --log revision1/data/train_first_200000_sessions.txt
```

`fetch_replay_data.py` retrieves the first 200,000 complete sessions of Kaggle
`train.gz` using existing credentials. The raw log is not committed or included
in the submission. `data/source.json` records selection and its SHA-256 hash.
The default replay is a new sample, not a reconstruction of the old 61,105-session
sample, whose exact selection could not be established from the supplied files.

Validation uses seeds 100–109 at penalties 0.15 and 0.30. It selects lambda by
coverage subject to the lower paired 95% NDCG confidence bound being at least
-0.01 in each condition. Test seeds 1000–1019 are disjoint. Frozen test conditions
include penalty 0.40, eta through 1.0, all scoring ablations and eight stress
conditions. `results/protocol.json` records the selection and runtime versions.

The test suite saves every seed and reports sample SD, Student-t confidence
intervals, paired tests, and Holm correction across the primary coverage/NDCG
comparisons. The statistical unit is a simulation seed; replay bootstrap units
are entire sessions. These quantify sampling variation within the models, not
validity of the models for live traffic.

## Author actions before submission

- Confirm funding and competing-interest statements for both authors.
- Review the original AI disclosure against the revision work and supplementary prompt log.
- Archive the revised code, protocol, per-seed data and scripts in a new Zenodo
  version; replace the explicitly identified revision DOI placeholder.
- Review the changed scientific conclusions and response letter before upload.
# Latest author edits

The FINAL clean manuscript includes the author's grammar edits and the subsequent
equation-formatting repair. Edit that current document directly; rebuilding with
`minimal_manuscript.py` would discard those later author edits. The pre-repair
author copy is retained in `work/author_grammar_manuscript.docx` for recovery.
`repair_equation_formatting.py` preserves prose while fixing the targeted math.

The latest author-directed layout restores Section 3.6 verbatim, moves figure
legends out of the manuscript, and supplies table titles/notes separately.
Each Table1–Table6 DOCX contains only its grid, without titles, notes or page
footers. The clean/marked manuscripts have 13/16 pages. Earlier manuscript
builders predate these author edits and must not be used to overwrite them.
