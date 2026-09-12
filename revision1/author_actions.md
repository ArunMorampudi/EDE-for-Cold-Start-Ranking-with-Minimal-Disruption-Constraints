# Author review before resubmission

The files are revised drafts, not a statement that all author attestations are
complete. Do not upload until the following items have been reviewed.

1. **Review the changed scientific conclusion.** Correcting prefix preservation
   and penalty handling reduced the discovery gains substantially. At p=0.15,
   baseline/EDE coverage is 0.087/0.186; at p=0.30 it is 0.0055/0.0117. Novelty-only
   is competitive or better. The original large gains, engagement improvements,
   and deployment-safety claims are withdrawn, with reasons in the manuscript
   and response letter. This is a substantive scientific revision.
2. **Confirm funding for both authors.** The supplied documents do not establish
   funding status. Add the confirmed statement to the manuscript body and portal.
   If no external funding was received, confirm that fact before using a no-funding
   declaration. Independent-researcher status does not establish funding status.
3. **Confirm competing interests for both authors.** Add the agreed declaration
   to the manuscript and portal. No no-conflict declaration has been fabricated.
4. **Review the AI disclosure.** Section 3.6 retains the original author wording
   as requested. Verify that it accurately covers the revision work and check it
   against the original supplementary prompt log mentioned by Reviewer 2, which
   was not among the files supplied in this checkout.
5. **Create a new Zenodo version.** Upload the revised code and results archive,
   then provide the new version-specific DOI. The existing DOI
   `10.5281/zenodo.19242717` is explicitly labeled as the original archive and must
   not be presented as the revision archive. Add the new DOI in Section 3.5 and
   Data Availability, and update the corresponding reviewer response. Regenerate
   tracked changes after these last author edits.

## Upload mapping

Use `FINAL/01_Zenodo_upload/` for the single Zenodo ZIP and
`FINAL/02_PeerJ_upload/` for all separate PeerJ files. No working scripts, audit
reports or preview images need to be uploaded separately.

- Manuscript: `submitted_peerj_docs/cs-134580-v0.4.docx` (updated in place).
- Marked revision: `cs-134580-tracked-changes.docx`.
- Rebuttal: `cs-134580-response-to-reviewers.docx`.
- Tables: the six updated `cs-134580-Table1.docx` through `cs-134580-Table6.docx` files. Original Tables 3–6 are
  consolidated into new Table 3; new Tables 4–6 report the added analyses.
- Figures: four updated `cs-134580-Figure_*.png` files, all 300 DPI.
- Figure legends: `cs-134580-Figure-legends.docx`, supplied separately.
- Table titles/notes: `cs-134580-Table-titles-and-notes.docx`, a reference for
  entering metadata in the upload form; the six table files contain only grids.
- Supporting results: per-seed simulation CSVs, full uncertainty/statistics CSVs,
  replay summary/manifest and source-selection hash.

The new replay uses 372,282 separate pages from 200,000 complete training sessions,
not the historical 61,105-session sample. Its set/click preservation is structural
on ten logged candidates. The corrected code does not claim that this evaluates
new-item insertion, real cold-start discovery, or causal engagement effects.
