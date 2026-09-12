"""Build the clean, minimal PeerJ Computer Science revision package.

The manuscript and upload assets from the earlier revision are used only as
validated content sources. This builder removes no original files and writes a
new, clearly separated FINAL_minimal/PeerJ_upload package.
"""

from __future__ import annotations

import os
import shutil
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision1" / "FINAL_minimal" / "PeerJ_upload"
PRIOR = Path(os.environ.get(
    "EDE_PRIOR_REVISION_DIR",
    r"C:\Users\Arun\AppData\Local\Temp\ede_revision_one_inspect\revision1\FINAL\02_PeerJ_upload",
))

ZENODO_DOI = "https://doi.org/10.5281/zenodo.22726688"

DATA_AVAILABILITY_TEXT = (
    "The code and precomputed results for this study are available in the Zenodo "
    f"archive at {ZENODO_DOI}. The deposited archive contains the corrected policy, "
    "experiment and replay scripts, pinned dependencies, protocol, per-seed and "
    "validation results, replay summaries, focused tests, the acquisition script "
    "for the first 200,000 sessions, and a SHA-256 manifest. It excludes manuscript "
    "files, figure and table upload assets, the separate utils repository, and raw "
    "Yandex logs. Raw Yandex logs are not redistributed; they are available from the "
    "Yandex competition on Kaggle, subject to its terms."
)

FUNDING_TEXT = (
    "This research received no specific grant from any funding agency in the public, "
    "commercial, or not-for-profit sectors."
)

COMPETING_INTERESTS_TEXT = "The authors declare that they have no competing interests."


def clear_paragraph(paragraph: object) -> None:
    p = paragraph._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)


def replace_plain_paragraph(paragraph: object, text: str) -> None:
    first_rpr = None
    if paragraph.runs and paragraph.runs[0]._r.rPr is not None:
        first_rpr = deepcopy(paragraph.runs[0]._r.rPr)
    clear_paragraph(paragraph)
    run = paragraph.add_run(text)
    if first_rpr is not None:
        run._r.insert(0, first_rpr)


def insert_after_clean(paragraph: object, text: str, style: str | None = None) -> object:
    new = OxmlElement("w:p")
    paragraph._p.addnext(new)
    from docx.text.paragraph import Paragraph

    p = Paragraph(new, paragraph._parent)
    if style:
        p.style = style
    p.add_run(text)
    return p


def insert_before_clean(paragraph: object, text: str, style: str | None = None) -> object:
    new = OxmlElement("w:p")
    paragraph._p.addprevious(new)
    from docx.text.paragraph import Paragraph

    p = Paragraph(new, paragraph._parent)
    if style:
        p.style = style
    p.add_run(text)
    return p


def add_section_with_heading(doc: Document, after_text: str, heading: str, body: str) -> None:
    target = next(p for p in doc.paragraphs if p.text.strip() == after_text)
    heading_p = insert_after_clean(target, heading, "Heading 1")
    insert_after_clean(heading_p, body, "Normal")


def revise_manuscript() -> None:
    source = PRIOR / "cs-134580-v0.4.docx"
    if not source.exists():
        raise FileNotFoundError(source)
    doc = Document(source)

    # Preserve the earlier corrected evidence but make the disclosure match the
    # actual scope of the revision work. Do not rewrite surrounding prose.
    for p in doc.paragraphs:
        if p.text.strip().startswith("During the preparation of this work"):
            replace_plain_paragraph(
                p,
                "During the preparation of this work, the authors used generative "
                "AI tools, including ChatGPT-5.2 and GitHub Copilot, for manuscript "
                "editing and for assistance with code drafting, refactoring, and "
                "analysis-script development. The authors reviewed the resulting "
                "code, references, and numerical outputs and take responsibility "
                "for the final content. AI tools were not used to fabricate data or "
                "results."
            )
            break

    # The earlier draft cited Han et al. (2011), but that work is not in the
    # supplied reference list. Remove the orphan citation rather than inventing
    # bibliographic details.
    for p in doc.paragraphs:
        if "Han et al. (2011)" in p.text:
            replace_plain_paragraph(p, p.text.replace("Han et al. (2011), ", ""))
            break

    # Update only the version reference in Section 3.5 now that the corrected
    # replication archive has its version-specific DOI. Leave Section 3.6 intact.
    reproducibility_body = next(
        p for p in doc.paragraphs
        if p.text.strip().startswith("The full experimental framework, including the EDE reranking policy")
    )
    reproducibility_text = reproducibility_body.text.replace(
        "https://doi.org/10.5281/zenodo.19242717 (original archive)",
        ZENODO_DOI,
    ).replace(
        "The revised code and per-seed results require a new version-specific DOI before submission.",
        "The revised code and per-seed results are included in the deposited archive.",
    )
    replace_plain_paragraph(reproducibility_body, reproducibility_text)

    # Funding and competing-interest statements are factual author declarations.
    target = next(p for p in doc.paragraphs if p.text.strip() == "Data Availability")
    if not any(p.text.strip() == "Funding" for p in doc.paragraphs):
        insert_before_clean(target, "Funding", "Heading 1")
        insert_before_clean(target, FUNDING_TEXT, "Normal")
    target = next(p for p in doc.paragraphs if p.text.strip() == "Data Availability")
    if not any(p.text.strip() == "Competing interests" for p in doc.paragraphs):
        insert_before_clean(target, "Competing interests", "Heading 1")
        insert_before_clean(target, COMPETING_INTERESTS_TEXT, "Normal")

    # Point the manuscript to the exact existing Zenodo archive and describe the
    # corrected archive supplied for the next version. Section 3.6 is not edited.
    data_body = next(
        p for p in doc.paragraphs if p.text.strip().startswith("The original code archive is")
    )
    replace_plain_paragraph(data_body, DATA_AVAILABILITY_TEXT)

    # Keep the output clean: no inline figures, tables, tracked changes, or
    # comments. The separate upload assets carry the visuals and table grids.
    for p in list(doc.paragraphs):
        if not p.text.strip() and p._p.xpath(".//w:drawing|.//w:pict"):
            p._p.getparent().remove(p._p)

    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(OUT / "cs-134580-v0.4.docx")


def revise_response_letter() -> None:
    source = PRIOR / "cs-134580-response-to-reviewers.docx"
    doc = Document(source)
    title = next(p for p in doc.paragraphs if p.style.name == "Title")
    body = next(p for p in doc.paragraphs if p.text.strip().startswith("We have revised the manuscript"))
    locations = next(p for p in doc.paragraphs if p.text.strip().startswith("Locations below use stable section/table/figure identifiers."))
    replace_plain_paragraph(
        body,
        "We revised the manuscript while retaining its original structure and wording "
        "where the evidence permits, regenerated the simulation and replay analyses, "
        "updated the separate tables and figures, and supplied a clean manuscript for "
        "the PeerJ upload folder. The code audit uncovered discrepancies in prefix "
        "construction, penalty handling, and replay semantics that affected the "
        "submitted evidence. The revised paper reports the corrected, smaller effects "
        "and withdraws unsupported engagement and deployment-safety claims."
    )
    replace_plain_paragraph(
        locations,
        "Locations below use stable section, table, and figure identifiers. "
        "The manuscript now states that the work received no specific funding and "
        "that the authors have no competing interests. The corrected minimal "
        f"replication ZIP is deposited at {ZENODO_DOI}, which is cited in Data "
        "Availability."
    )
    for p in doc.paragraphs:
        if p.text.strip().startswith("Response. Data Availability is added."):
            replace_plain_paragraph(
                p,
                f"Response. Data Availability is included and identifies the deposited "
                f"corrected minimal replication archive at {ZENODO_DOI}. The archive "
                "contains the revised source modules, focused tests, pinned requirements, "
                "protocol, per-seed and replay results, acquisition script, and SHA-256 "
                "manifest. The manuscript now states that the work "
                "received no specific funding and that the authors have no competing "
                "interests. Section 3.6 is unchanged in this revision; the authors "
                "will reconcile its wording with the complete AI prompt record before "
                "submission."
            )
        elif p.text.strip().startswith("Response. Section 3.5 and Data Availability print"):
            replace_plain_paragraph(
                p,
                f"Response. Section 3.5 and Data Availability print {ZENODO_DOI} as "
                "the archive DOI. The deposited ZIP contains the corrected source "
                "modules, focused tests, pinned requirements, protocol, per-seed and "
                "replay results, acquisition script, and SHA-256 manifest; it excludes "
                "manuscript files, paper-asset builders, the separate utils repository, "
                "and raw logs."
            )
    doc.save(OUT / "cs-134580-response-to-reviewers.docx")


def create_statements_doc() -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10.5)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("PeerJ Submission Statements")
    doc.add_paragraph(
        "Copy the wording below into the PeerJ submission fields and the manuscript.",
    )
    sections = [
        (
            "Data Availability",
            DATA_AVAILABILITY_TEXT,
        ),
        (
            "Funding",
            FUNDING_TEXT,
        ),
        (
            "Competing interests",
            COMPETING_INTERESTS_TEXT,
        ),
        (
            "Acknowledgements",
            "The authors acknowledge using ChatGPT-5.2 to refine the grammar, clarity, "
            "and organization of the manuscript text.",
        ),
        (
            "Use of AI-assisted tools",
            "The authors used generative AI tools, including ChatGPT-5.2 and GitHub "
            "Copilot, for manuscript editing and for assistance with code drafting, "
            "refactoring, and analysis-script development. The authors reviewed the "
            "resulting code, references, and numerical outputs and take responsibility "
            "for the final content. AI tools were not used to fabricate data or results. "
            "Reconcile this wording with the complete supplementary prompt record before upload.",
        ),
    ]
    for heading, body in sections:
        doc.add_paragraph(heading, style="Heading 1")
        doc.add_paragraph(body)
    doc.save(OUT / "cs-134580-submission-statements.docx")


def copy_assets() -> None:
    names = [
        "cs-134580-Figure-legends.docx",
        "cs-134580-Table-titles-and-notes.docx",
        "cs-134580-Table1.docx",
        "cs-134580-Table2.docx",
        "cs-134580-Table3.docx",
        "cs-134580-Table4.docx",
        "cs-134580-Table5.docx",
        "cs-134580-Table6.docx",
        "cs-134580-Figure_1.png",
        "cs-134580-Figure_2.png",
        "cs-134580-Figure_3.png",
        "cs-134580-Figure_4.png",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = PRIOR / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / name)


def shorten_caption_and_note_docs() -> None:
    figure_doc = Document(OUT / "cs-134580-Figure-legends.docx")
    figure_text = {
        "Figure 1.":
            "Figure 1. Simulation coverage and NDCG@10 across cold-start penalties. "
            "Panels (a) and (b) show coverage and NDCG@10. Values are means ± SD over "
            "20 held-out seeds (1000-1019); each seed has 100 queries, 1,000 documents, "
            "200 arrivals at timestep 1,000, and T=2,000. PBM a=3, b=0.35. "
            "Nonbaseline λ=0.4, η=0.65, m=2, L=50, α=0.5, τ=50; baseline λ=0.",
        "Figure 2.":
            "Figure 2. Validation trade-off for λ. Panels (a) and (b) use penalties 0.15 "
            "and 0.30. Points are paired means versus baseline over 10 validation seeds "
            "(100-109); bars are 95% Student-t CIs. Fixed η=0.65, m=2, L=50, α=0.5, "
            "τ=50 and T=2,000. Dashed line: ΔNDCG=-0.01.",
        "Figure 3.":
            "Figure 3. Stress tests at penalty 0.30. Panels (a) and (b) show coverage and "
            "NDCG@10; values are means ± SD over 20 held-out seeds per condition. Fixed "
            "λ=0.4, η=0.65, m=2, L=50, α=0.5, τ=50 and T=2,000. Conditions are standard, "
            "cascade, noisy, strong-bias, heterogeneous, navigational, sparse, delayed "
            "and natural; sparse runs use 1,000 queries and the others use 100.",
        "Figure 4.":
            "Figure 4. Yandex logged-support displacement diagnostic. The sample contains "
            "372,282 labeled ten-result pages from the first 200,000 complete sessions. "
            "Panel (a) shows acceptance and preservation rates; panel (b) shows changed "
            "positions and absolute document rank shift. Means use 95% session-cluster "
            "bootstrap CIs (2,000 resamples; seed 20260911). λ=0.4, η=0.65, m=2, α=0.5, "
            "τ=50 and 10 logged candidates. Set/click preservation are structural logged-"
            "support diagnostics, not cold-start or engagement estimates.",
    }
    for p in figure_doc.paragraphs:
        for prefix, text in figure_text.items():
            if p.text.strip().startswith(prefix):
                replace_plain_paragraph(p, text)
                break
    figure_doc.save(OUT / "cs-134580-Figure-legends.docx")

    notes_doc = Document(OUT / "cs-134580-Table-titles-and-notes.docx")
    note_text = {
        "Table 1 Validation": (
            "Table 1 Operating-point selection",
            "EDE minus baseline. Validation uses seeds 100-109; held-out confirmation uses "
            "20 disjoint seeds (1000-1019). Values are mean [95% CI]. λ=0.4 met the "
            "validation selection rule and is used for held-out results; there is no "
            "per-query quality guarantee.",
        ),
        "Table 2 Strategy": (
            "Table 2 Strategy comparison",
            "Coverage mean ± SD over 20 held-out seeds; rows are penalties and columns are "
            "strategies. Baseline appears once and all scoring comparators share the "
            "prefix and gate. UCB, Thompson, local-swap and quota are defined heuristics. "
            "Other metrics and tests are in the archived CSVs.",
        ),
        "Table 3 Sensitivity": (
            "Table 3 Novelty and prefix sensitivity",
            "Mean ± SD over 20 held-out seeds. λ=0.4, L=50, α=0.5 and τ=50 are fixed; η "
            "and m are varied. Columns (a)-(d) use p=0.15 and 0.30; columns (e)-(f) add "
            "p=0.40. These are sensitivity tests, not further tuning.",
        ),
        "Table 4 Ablations": (
            "Table 4 Ablations and disruption",
            "Mean ± SD over 20 seeds; defaults are as in Section 3.3 unless ablated. "
            "No-decay sets N=1. RBO uses persistence 0.9; Kendall uses shared items. "
            "Loss fractions exceed ΔNDCG=0.01; worst query is the minimum query-mean "
            "ΔNDCG. Promoted relevance is continuous, while promoted relevant fraction "
            "is binary. New exposure share is age-discounted, not provider fairness.",
        ),
        "Table 5 Stress": (
            "Table 5 Stress-test results",
            "Penalty p=0.30; mean ± SD over 20 held-out seeds (1000-1019). Defaults are "
            "λ=0.4, η=0.65, m=2, L=50, α=0.5 and τ=50; baseline λ=0. Conditions are "
            "defined in Section 3.4. Coverage denominators vary by condition; full "
            "per-seed metrics are in the archived CSVs.",
        ),
        "Table 6 Corrected": (
            "Table 6 Yandex logged-support diagnostics",
            "Means [95% CI] use 2,000 whole-session bootstrap resamples (seed 20260911). "
            "The sample contains 372,282 pages from the first 200,000 complete sessions. "
            "EDE uses λ=0.4, η=0.65, m=2, α=0.5 and τ=50 on 10 logged candidates. "
            "Set/click preservation accepted all pages by construction; click preservation "
            "permits changed clicked positions. This is a logged-support diagnostic, not "
            "causal engagement or cold-start evidence.",
        ),
    }
    for p in notes_doc.paragraphs:
        for prefix, (title, note) in note_text.items():
            if p.text.strip().startswith(prefix):
                replace_plain_paragraph(p, title)
            elif p.text.strip().startswith("Paired differences are EDE minus baseline.") and prefix.startswith("Table 1"):
                replace_plain_paragraph(p, note)
            elif p.text.strip().startswith("Cells show absolute new-document coverage") and prefix.startswith("Table 2"):
                replace_plain_paragraph(p, note)
            elif p.text.strip().startswith("Original Tables 3") and prefix.startswith("Table 3"):
                replace_plain_paragraph(p, note)
            elif p.text.strip().startswith("Values are mean ± sample SD") and prefix.startswith("Table 4"):
                replace_plain_paragraph(p, note)
            elif p.text.strip().startswith("Penalty p=0.30.") and prefix.startswith("Table 5"):
                replace_plain_paragraph(p, note)
            elif p.text.strip().startswith("Means [95% CI]") and prefix.startswith("Table 6"):
                replace_plain_paragraph(p, note)
    notes_doc.save(OUT / "cs-134580-Table-titles-and-notes.docx")


def main() -> None:
    copy_assets()
    shorten_caption_and_note_docs()
    revise_manuscript()
    revise_response_letter()
    create_statements_doc()
    print(f"Created {len(list(OUT.iterdir()))} files in {OUT}")


if __name__ == "__main__":
    main()
