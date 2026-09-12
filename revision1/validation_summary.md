# Revision validation completed 11 September 2026

Branch: `revision-one`. Original submitted files remain in `original_submitted/`.

Latest upload-layout pass: Section 3.6 matches the original author wording exactly.
Four figure legends are in a separate one-page DOCX. Six table titles/notes are
in a separate six-page reference DOCX. Table1–Table6 contain only one grid each,
with no body titles, notes or header/footer text, and each renders on one page.
Three discretionary phrase edits were reverted and implementation-history
sentences were removed from the manuscript while the reviewer response retains
the explanation. The clean manuscript is 13 pages; the tracked copy is 16 pages
with 463 revisions. Word accept/reject checks pass, and all changed manuscript,
table and reference layouts were inspected. FINAL contains 15 PeerJ files,
including the two reference documents, plus the single Zenodo ZIP in its folder.
Reviewer 2's request to recalibrate the AI disclosure remains an author item
because the original disclosure was restored at the author's request.

Latest author-edit pass: preserved the author's grammar changes from the FINAL
manuscript. Repaired native Word formatting in Equations (4), (5), (7), (8) and
the inline UCB square-root expression in Section 3.4. Replaced six em dashes with
parenthetical punctuation. The clean copy has 14 pages and the regenerated marked
copy has 18 pages and 470 revisions. Both were rendered and visually inspected;
accept/reject round trips passed. Eight numbered equations and one inline equation
remain, with continuous line numbering. The earlier counts below describe prior
validation passes.

- Six targeted policy/parser tests passed, including deterministic prefix preservation,
  lambda-zero identity, page/click mapping and prevention of future-click leakage.
- The numerical audit passed for 100 validation runs and 3,900 held-out runs
  (195 conditions, 20 seeds each). Seed splits, means, sample SD, paired differences,
  multiplicity correction and metric bounds were checked against saved per-seed data.
- No protected-prefix violations were recorded.
- The corrected replay covers 372,282 result pages from 200,000 complete sessions.
  Its source hash, page counts and reported intervals were checked.
- Nine DOCX files have valid XML. The clean manuscript contains eight numbered
  equations; the reviewer response includes all 65 numbered comments.
- Word Compare generated 454 tracked revisions against the original-format
  manuscript. Accepting all changes reproduces
  the clean manuscript text exactly; rejecting all reproduces the original text exactly.
- All rendered manuscript, response, marked-copy and table pages were visually
  inspected. Adjacent table merging and misleading low-precision values were fixed.
  All four final scientific figures were inspected and have 300-DPI metadata.
- The conservative revision restores continuous line numbering in the clean and
  tracked manuscripts. The 14-page clean manuscript, 17-page marked copy and
  13-page response were rendered and inspected. Each of the six table documents
  now contains exactly one editable table on one US-letter page with 2.5 cm
  margins. Their manuscript citations occur in numerical order. The consolidated
  tables and updated response were rendered and visually inspected again.
  No manuscript wording changed in this table-formatting pass.
  Final delivery contains exactly two folders:
  one Zenodo ZIP and thirteen separate PeerJ document/image files.
- A further minimal-edit pass replaced broad paragraph rewrites with targeted phrase
  edits, shortened reviewer-requested additions, restored the original reference order,
  and preserved the original equation objects where mathematically correct. An ordered
  word comparison retains 3,069 of 4,662 original prose/reference words (65.8% versus
  54.1% in the preceding revision). This excludes equation content and is a wording
  audit, not a measure of scientific validity. Remaining large edits correct results,
  unsupported interpretations, or supply requested methods and comparisons.
- Git diff whitespace validation passed; Git reported only normal LF/CRLF conversion notices.
- Removed 16 superseded generated paths, including old ZIPs, one-time editing
  scripts, logs and preview sets. Original submissions, reusable builders, source
  data, scientific results and current validation materials remain available.

The bundled LibreOffice renderer was unavailable on this Windows host. Microsoft
Word exported the documents to PDF, and bundled Poppler generated page images for
visual inspection. Working QA files remain locally under `qa/` and are not packaged.

These checks establish internal consistency of the corrected artifacts, not
real-world validity or journal acceptance. Comparators include explicitly named
heuristics rather than exact reproductions of full published safe-ranking methods.
The replay is descriptive and cannot establish causal engagement or real cold-start
discovery. Funding, competing interests, original AI prompt-log reconciliation,
scientific author review and the new Zenodo DOI remain in `author_actions.md`.
