# Minimal PeerJ Computer Science revision package

This package is based on `main` and keeps the original PeerJ upload structure.
The manuscript is clean and keeps tables and figures in separate upload files.
The earlier oversized tracked-changes manuscript and generated ZIP are not
included here.

## PeerJ upload folder

Use only the files in `PeerJ_upload/` for the journal submission:

- `cs-134580-v0.4.docx` is the clean revised manuscript.
- `cs-134580-Table1.docx` through `cs-134580-Table6.docx` contain table grids only.
- `cs-134580-Table-titles-and-notes.docx` contains the titles and notes to paste into the upload form.
- `cs-134580-Figure_1.png` through `cs-134580-Figure_4.png` are the separate figures.
- `cs-134580-Figure-legends.docx` contains the figure legends.
- `cs-134580-response-to-reviewers.docx` contains point-by-point responses to the supplied review.
- `cs-134580-submission-statements.docx` contains Data Availability, AI-use, funding, and competing-interest text.
- `cs-134580-v0.4-tracked-changes.docx` is an optional Word redline against the originally submitted manuscript; use it only if PeerJ requests a marked copy.

The corrected minimal code archive for Zenodo is
`../Zenodo_upload/EDE_revision1.zip`. It contains only
the revised experiment and replay source modules, focused tests, pinned
requirements, protocols, and numerical results. It excludes manuscript files,
paper-asset builders, the separate `utils` repository, and raw Yandex logs.

The clean manuscript and response letter are the default upload files. The
tracked-changes manuscript is supplied separately for the journal's marked-copy
requirement and is not needed when PeerJ requests only a clean manuscript.

## Before upload

1. Confirm that the no-funding and no-competing-interests declarations are accurate.
2. Reconcile the AI-use wording with the complete supplementary prompt record.
3. Cite the deposited replication archive at https://doi.org/10.5281/zenodo.22726688 in the submission metadata if requested.
4. Upload the tracked-changes manuscript only if the PeerJ portal requests it.

The compact `reproducibility/` folder contains the revised protocol, pinned
dependencies, summary/per-seed results, replay manifest, and the focused
revision scripts. It does not contain raw Yandex logs or credentials.
