"""Apply the pre-formal-DART checkpoint-selection amendment to the manuscript.

This is a minimal semantic edit. The user explicitly waived render/layout QA,
so the script preserves existing styles and performs structural/text checks only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from docx import Document


ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Data_Center_VPP_English_Manuscript_and_Experimental_Protocol_NLR_NYISO_Revised.docx"
BACKUP = (
    REPO
    / "artifacts"
    / "manifests"
    / "specification_backups"
    / "Data_Center_VPP_English_Manuscript_and_Experimental_Protocol_NLR_NYISO_Revised.pre_dart_checkpoint_selection_20260812.docx"
)
MANIFEST = (
    REPO
    / "artifacts"
    / "manifests"
    / "dart_checkpoint_selection_document_amendment.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_exact(doc: Document, text: str):
    matches = [paragraph for paragraph in doc.paragraphs if paragraph.text == text]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact paragraph, found {len(matches)}: {text[:80]}")
    return matches[0]


def insert_after(doc: Document, anchor, text: str, *, style: str | None = None):
    paragraph = doc.add_paragraph(text, style=style)
    anchor._p.addnext(paragraph._p)
    return paragraph


def main() -> None:
    if not DOCX.exists():
        raise FileNotFoundError(DOCX)
    before_sha = sha256_file(DOCX)
    doc = Document(DOCX)
    marker = "three-seed group-synchronized early stopping"
    if any(marker in paragraph.text for paragraph in doc.paragraphs):
        raise RuntimeError("checkpoint-selection document amendment is already present")

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if BACKUP.exists() and sha256_file(BACKUP) != before_sha:
        raise RuntimeError("existing pre-amendment backup differs from the current input")
    if not BACKUP.exists():
        shutil.copy2(DOCX, BACKUP)

    calibration_old = (
        "Raw quantiles are first monotonically rearranged. Conformal calibration uses 2024 NYISO rolling residuals "
        "separately for the DA and RT tasks, the NLR workload validation segment for CfC-W, and the NLR physical "
        "validation segment for twin uncertainty. No residual from one source domain calibrates another. For a "
        "nominal 1-alpha interval, the calibrated interval is:"
    )
    calibration_new = (
        "Raw quantiles are first monotonically rearranged. For each formal neural DART annual refit, the final 112 "
        "eligible pre-cutoff days are partitioned chronologically into a 56-day checkpoint-calibration panel, a "
        "28-day checkpoint-selection panel, and a disjoint 28-day final-calibration and residual-coupling panel. "
        "An epoch-specific additive conformal correction fitted on the first panel is used only to assess calibration "
        "credibility and proper scores on the second panel. After the epoch is frozen, the deployment calibrator and "
        "DA/RT residual coupling are fitted on the final panel. No rolling evaluation-year outcome, market replay, "
        "optimization result, or locked-2025 target enters this process. NLR workload and physical-twin calibration "
        "remain domain-specific; no residual from one source domain calibrates another."
    )
    find_exact(doc, calibration_old).text = calibration_new

    training_old = (
        "Fit scalers and graph statistics on training data only. Early stopping and hyperparameter selection use "
        "validation data only."
    )
    training_new = (
        "Fit scalers, graph statistics, and q01/q05/q95/q99 tail thresholds on the core training history only. For "
        "every neural DART candidate, annual origin, and DA or RT head, align the complete epoch ledger across seeds "
        "20260809, 20260819, and 20260829, and average each daily checkpoint metric across the three seeds before "
        "inference."
    )
    find_exact(doc, training_old).text = training_new

    checkpoint_old = (
        "Save the best checkpoint by validation proper score, not by the test metric later highlighted in the paper."
    )
    checkpoint_new = (
        "Select the checkpoint through health and simultaneous 80%/90% calibration gates, followed by a one-sided "
        "simultaneous 95% global-CRPS confidence set and a proper-tail-twCRPS confidence set using 7-day circular "
        "moving blocks. Select the earliest surviving epoch; do not minimize one scalar score or use an arbitrary "
        "weighted composite."
    )
    checkpoint = find_exact(doc, checkpoint_old)
    checkpoint.text = checkpoint_new
    checkpoint.style = "List Bullet"
    boundary = insert_after(
        doc,
        checkpoint,
        "This checkpoint rule uses only chronologically pre-cutoff predictive evidence. Action metrics, MILP solutions, "
        "operating cost, profit, regret, CVaR, rolling evaluation-year outcomes, and locked-2025 targets are forbidden. "
        "It selects an epoch within one model family and does not change the outer minimum-MAE versus DART-ValueSelect "
        "comparison.",
        style="List Bullet",
    )
    insert_after(
        doc,
        boundary,
        "Use three-seed group-synchronized early stopping: no stop before epoch 40; assess every 5 epochs; stop all "
        "three seeds together only after three consecutive stable assessments of the selected epoch and both proper-score "
        "survivor sets with no material proper-score improvement; stop at epoch 100 at the latest. Individual-seed "
        "early stopping and per-seed epoch selection are forbidden.",
        style="List Bullet",
    )

    ablation_heading = find_exact(doc, "5.8 Forecasting baselines and ablations")
    paragraphs = doc.paragraphs
    ablation_index = next(
        index
        for index, paragraph in enumerate(paragraphs)
        if paragraph._p is ablation_heading._p
    )
    ablation_blank = paragraphs[ablation_index + 1]
    if ablation_blank.text:
        raise RuntimeError("expected the Section 5.8 placeholder paragraph to be blank")
    ablation_blank.text = (
        "Execution order is primary-before-ablation: train the full-date DGPT-E main model and the preregistered "
        "traditional/neural control candidates first. Structural DGPT-E ablations and ORACLE variants are deferred "
        "until the primary nine-candidate panel is complete and cannot alter its frozen data or checkpoint hashes."
    )

    layer1 = find_exact(
        doc,
        next(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.startswith("Layer 1, probabilistic credibility, binds every candidate")
        ),
    )
    layer1.text += (
        " Layer-1 proper scores and calibration may select a statistically credible epoch within a fixed neural model "
        "under Section 5.7, but they do not select the operational winner among model families."
    )

    split = find_exact(
        doc,
        next(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.startswith("The NLR physical twin and workload model use independent chronological")
        ),
    )
    split.text += (
        " Each annual neural DART refit freezes its checkpoint from the three strictly pre-cutoff internal panels "
        "before producing the corresponding rolling out-of-sample forecasts; those forecasts cannot feed back into "
        "epoch selection."
    )

    sensitivity = find_exact(
        doc,
        next(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.startswith("DART-ValueSelect uses the exact same physical/user problem")
        ),
    )
    sensitivity.text = (
        "Primary execution precedes every structural ablation and optional ORACLE variant. " + sensitivity.text
    )

    inference_anchor = find_exact(
        doc,
        next(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.startswith("Use Diebold-Mariano-style tests for forecast losses")
        ),
    )
    insert_after(
        doc,
        inference_anchor,
        "The within-model checkpoint confidence sets are separate from the outer A2 Model Confidence Set. The former "
        "use only pre-cutoff predictive evidence and choose the earliest statistically credible epoch; the latter uses "
        "2022-2024 causal market replay to form the DART-ValueSelect winner set.",
        style="List Bullet",
    )

    checklist_anchor = find_exact(
        doc,
        "☐ No invented revenue or assumed commercial profit is reported; claims use counterfactual operational cost "
        "reduction, incremental objective value, regret reduction, and PCR.",
    )
    insert_after(
        doc,
        checklist_anchor,
        "☐ Every neural DART checkpoint was selected from exact three-seed, within-date pre-cutoff evidence through "
        "the frozen health, calibration, global-CRPS, proper-tail-twCRPS, earliest-epoch, and synchronized-group-stop "
        "rules, with no action, MILP, value, rolling evaluation-year, or 2025 input.",
        style=checklist_anchor.style.name,
    )

    mapping_tables = [
        table
        for table in doc.tables
        if table.rows
        and len(table.rows[0].cells) == 2
        and table.rows[0].cells[0].text.strip() == "Artifact"
        and table.rows[0].cells[1].text.strip() == "Paper destination"
    ]
    if len(mapping_tables) != 1:
        raise RuntimeError("could not identify the unique Appendix D artifact mapping table")
    mapping = mapping_tables[0]
    artifact_path = "artifacts/manifests/dart_checkpoint_selection_manifest.json"
    if any(row.cells[0].text.strip() == artifact_path for row in mapping.rows):
        raise RuntimeError("checkpoint-selection artifact row is already present")
    cells = mapping.add_row().cells
    cells[0].text = artifact_path
    cells[1].text = (
        "Sections 5.6-5.7 and 8.2; pre-cutoff three-seed epoch-selection evidence"
    )

    doc.save(DOCX)
    after_sha = sha256_file(DOCX)
    check = Document(DOCX)
    required_fragments = (
        marker,
        "proper-tail-twCRPS confidence set",
        "primary-before-ablation",
        "within-model checkpoint confidence sets are separate",
        "Every neural DART checkpoint was selected",
    )
    all_text = "\n".join(paragraph.text for paragraph in check.paragraphs)
    missing = [fragment for fragment in required_fragments if fragment not in all_text]
    if missing:
        raise RuntimeError(f"post-save structural text check failed: {missing}")
    if not any(
        row.cells[0].text.strip() == artifact_path
        for table in check.tables
        for row in table.rows
    ):
        raise RuntimeError("post-save artifact mapping row is absent")

    manifest = {
        "schema_version": "dcvpp.dart_checkpoint_selection_document_amendment/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_path": str(DOCX),
        "document_sha256_before": before_sha,
        "document_sha256_after": after_sha,
        "backup_path": str(BACKUP),
        "backup_sha256": sha256_file(BACKUP),
        "structural_checks_passed": True,
        "render_or_layout_qa_performed": False,
        "render_qa_omission_reason": "user_explicitly_requested_no_format_check",
        "training_or_experiment_started": False,
        "changes": [
            "three_disjoint_pre_cutoff_checkpoint_panels",
            "three_seed_within_date_epoch_aggregation",
            "proper_score_confidence_sets_and_earliest_epoch",
            "three_seed_group_synchronized_early_stopping",
            "primary_before_ablation_execution_order",
            "within_model_vs_outer_selector_boundary",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "manifest_sha256": sha256_file(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
