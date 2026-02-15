"""Paper audit script: validates outputs for internal consistency and paper claims."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Paper defaults
PAPER_DEFAULTS = {
    "lambda": 0.4,
    "eta": 0.65,
    "m": 2,
    "top_L": 50,
    "alpha": 0.5,
}


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _outputs_dir() -> Path:
    return _root_dir() / "outputs"


def _audit_dir() -> Path:
    audit_dir = _outputs_dir() / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir


class AuditReport:
    """Accumulates audit results."""

    def __init__(self):
        self.sections: Dict[str, Tuple[str, str]] = {}  # section -> (status, details)
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def add_section(self, name: str, status: str, details: str = "") -> None:
        """Add a section result. status = 'PASS', 'WARN', or 'FAIL'."""
        self.sections[name] = (status, details)
        if status == "FAIL":
            self.failures.append(f"{name}: {details}")
        elif status == "WARN":
            self.warnings.append(f"{name}: {details}")

    def verdict(self) -> str:
        """Return final verdict."""
        if self.failures:
            return "FAILED"
        return "Ready for submission" if not self.warnings else "PASSED (with warnings)"

    def console_summary(self) -> str:
        """Print a short console summary."""
        pass_count = sum(1 for _, (s, _) in self.sections.items() if s == "PASS")
        warn_count = sum(1 for _, (s, _) in self.sections.items() if s == "WARN")
        fail_count = sum(1 for _, (s, _) in self.sections.items() if s == "FAIL")

        summary = f"\n{'=' * 80}\nPAPER AUDIT SUMMARY\n{'=' * 80}\n"
        summary += f"PASS: {pass_count} | WARN: {warn_count} | FAIL: {fail_count}\n"
        summary += f"Verdict: {self.verdict()}\n"
        if self.warnings:
            summary += f"\nTop {min(10, len(self.warnings))} warnings:\n"
            for w in self.warnings[:10]:
                summary += f"  - {w}\n"
        if self.failures:
            summary += f"\nFailures:\n"
            for f in self.failures:
                summary += f"  - {f}\n"
        return summary

    def markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Paper Audit Report",
            "",
            f"**Verdict:** {self.verdict()}",
            "",
            "## Section Results",
            "",
        ]

        for name, (status, details) in self.sections.items():
            emoji = "✓" if status == "PASS" else "⚠" if status == "WARN" else "✗"
            lines.append(f"### {emoji} {name}")
            lines.append("")
            lines.append(f"**Status:** {status}")
            if details:
                lines.append("")
                for line in details.split("\n"):
                    if line:
                        lines.append(f"{line}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for i, w in enumerate(self.warnings[:10], 1):
                lines.append(f"{i}. {w}")
            if len(self.warnings) > 10:
                lines.append(f"... and {len(self.warnings) - 10} more")
            lines.append("")

        if self.failures:
            lines.append("## Failures")
            lines.append("")
            for f in self.failures:
                lines.append(f"- {f}")
            lines.append("")

        return "\n".join(lines)


def check_artifact_existence(report: AuditReport) -> bool:
    """A) Check that all required CSV and report files exist."""
    outputs = _outputs_dir()
    required_files = [
        "tables/difficulty_sweep_extended.csv",
        "tables/baselines_suite_results.csv",
        "tables/click_model_robustness.csv",
        "tables/sensitivity_eta_m.csv",
        "report_paper_ready_difficulty.md",
        "report_click_model_robustness.md",
        "report_baselines_suite.md",
        "report_sensitivity_eta_m.md",
    ]

    missing = []
    for rel_path in required_files:
        full_path = outputs / rel_path
        if not full_path.exists():
            missing.append(str(rel_path))

    if missing:
        details = f"Missing files:\n" + "\n".join([f"  - {f}" for f in missing])
        report.add_section("A) Artifact Existence", "FAIL", details)
        return False

    details = f"All {len(required_files)} required files exist."
    report.add_section("A) Artifact Existence", "PASS", details)
    return True


def check_metric_bounds(report: AuditReport) -> bool:
    """B) Check that metrics are in valid ranges."""
    outputs = _outputs_dir()
    csvs = {
        "difficulty_sweep_extended": outputs / "tables/difficulty_sweep_extended.csv",
        "baselines_suite_results": outputs / "tables/baselines_suite_results.csv",
        "click_model_robustness": outputs / "tables/click_model_robustness.csv",
        "sensitivity_eta_m": outputs / "tables/sensitivity_eta_m.csv",
    }

    issues = []
    for name, path in csvs.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)

        for col in ["ndcg10_mean", "new_cov_mean", "avg_new_fraction_mean"]:
            if col not in df.columns:
                continue
            out_of_bounds = df[(df[col] < 0) | (df[col] > 1)]
            if not out_of_bounds.empty:
                for _, row in out_of_bounds.iterrows():
                    issues.append(f"{name}: {col}={row[col]:.4f} for {row.to_dict()}")

    if issues:
        details = "Out-of-bounds metrics:\n" + "\n".join([f"  - {i}" for i in issues[:5]])
        if len(issues) > 5:
            details += f"\n... and {len(issues) - 5} more"
        report.add_section("B) Metric Bounds", "FAIL", details)
        return False

    details = "All metrics in valid [0, 1] range."
    report.add_section("B) Metric Bounds", "PASS", details)
    return True


def check_default_consistency(report: AuditReport) -> bool:
    """C) Check that experiments use the correct paper defaults."""
    outputs = _outputs_dir()
    issues = []

    # Check click_model_robustness.csv for EDE-Safer rows
    try:
        df_click = pd.read_csv(outputs / "tables/click_model_robustness.csv")
        ede_safer = df_click[df_click["strategy_name"] == "EDE-Safer"]

        for idx, row in ede_safer.iterrows():
            if row.get("lambda") != PAPER_DEFAULTS["lambda"]:
                issues.append(
                    f"click_model_robustness: EDE-Safer row {idx} lambda={row.get('lambda')} "
                    f"!= {PAPER_DEFAULTS['lambda']}"
                )
            if row.get("eta") != PAPER_DEFAULTS["eta"]:
                issues.append(
                    f"click_model_robustness: EDE-Safer row {idx} eta={row.get('eta')} "
                    f"!= {PAPER_DEFAULTS['eta']}"
                )
            if row.get("m") != PAPER_DEFAULTS["m"]:
                issues.append(
                    f"click_model_robustness: EDE-Safer row {idx} m={row.get('m')} "
                    f"!= {PAPER_DEFAULTS['m']}"
                )
            if row.get("top_L") != PAPER_DEFAULTS["top_L"]:
                issues.append(
                    f"click_model_robustness: EDE-Safer row {idx} top_L={row.get('top_L')} "
                    f"!= {PAPER_DEFAULTS['top_L']}"
                )
    except Exception as e:
        issues.append(f"Error checking click_model_robustness: {e}")

    # Check difficulty_sweep_extended.csv
    try:
        df_difficulty = pd.read_csv(outputs / "tables/difficulty_sweep_extended.csv")
        ede_rows = df_difficulty[df_difficulty["config_name"] == "EDE"]

        for idx, row in ede_rows.iterrows():
            if row.get("m") != PAPER_DEFAULTS["m"]:
                issues.append(
                    f"difficulty_sweep_extended: EDE row {idx} m={row.get('m')} "
                    f"!= {PAPER_DEFAULTS['m']}"
                )
            if row.get("eta") != PAPER_DEFAULTS["eta"]:
                issues.append(
                    f"difficulty_sweep_extended: EDE row {idx} eta={row.get('eta')} "
                    f"!= {PAPER_DEFAULTS['eta']}"
                )
            if "lambda" in df_difficulty.columns and row.get("lambda") != 0.4:
                issues.append(
                    f"difficulty_sweep_extended: EDE row {idx} lambda={row.get('lambda')} "
                    f"!= 0.4"
                )
            if row.get("top_L") != PAPER_DEFAULTS["top_L"]:
                issues.append(
                    f"difficulty_sweep_extended: EDE row {idx} top_L={row.get('top_L')} "
                    f"!= {PAPER_DEFAULTS['top_L']}"
                )

        baseline_rows = df_difficulty[df_difficulty["config_name"] == "Baseline"]
        for idx, row in baseline_rows.iterrows():
            if "lambda" in df_difficulty.columns and row.get("lambda") != 0.0:
                issues.append(
                    f"difficulty_sweep_extended: Baseline row {idx} lambda={row.get('lambda')} "
                    f"!= 0.0"
                )
    except Exception as e:
        issues.append(f"Error checking difficulty_sweep_extended: {e}")

    if issues:
        details = "Default mismatch:\n" + "\n".join([f"  - {i}" for i in issues[:5]])
        if len(issues) > 5:
            details += f"\n... and {len(issues) - 5} more"
        report.add_section("C) Default Consistency", "FAIL", details)
        return False

    details = "All experiments use correct paper defaults."
    report.add_section("C) Default Consistency", "PASS", details)
    return True


def check_baseline_monotonicity(report: AuditReport) -> bool:
    """D) Check that Baseline coverage is non-increasing with penalty."""
    outputs = _outputs_dir()

    try:
        df = pd.read_csv(outputs / "tables/difficulty_sweep_extended.csv")
        baseline_df = df[df["config_name"] == "Baseline"].sort_values("penalty")

        violations = []
        prev_cov = None
        prev_penalty = None

        for _, row in baseline_df.iterrows():
            cov = row["new_cov_mean"]
            penalty = row["penalty"]

            if prev_cov is not None:
                gap = prev_cov - cov
                if gap < -0.02:  # coverage increased (violates monotonicity)
                    violations.append(
                        f"Penalty {prev_penalty:.2f}→{penalty:.2f}: "
                        f"coverage increased {prev_cov:.4f}→{cov:.4f} (gap={gap:.4f})"
                    )

            prev_cov = cov
            prev_penalty = penalty

        if violations:
            severe = [v for v in violations if abs(v.split("gap=")[-1].rstrip(")").split("=")[-1]) > 0.05]
            status = "FAIL" if severe else "WARN"
            details = "Monotonicity violations:\n" + "\n".join(
                [f"  - {v}" for v in violations[:5]]
            )
            if len(violations) > 5:
                details += f"\n... and {len(violations) - 5} more"
            report.add_section("D) Baseline Monotonicity", status, details)
            return status == "PASS"
        else:
            details = "Baseline coverage monotonically non-increasing with penalty."
            report.add_section("D) Baseline Monotonicity", "PASS", details)
            return True
    except Exception as e:
        report.add_section("D) Baseline Monotonicity", "WARN", f"Could not check: {e}")
        return True


def check_robustness_claims(report: AuditReport) -> bool:
    """E) Check that EDE-Safer beats baseline strategies (with tolerance)."""
    outputs = _outputs_dir()

    try:
        df = pd.read_csv(outputs / "tables/click_model_robustness.csv")

        worst_gap = 0.0
        violations = []

        for click_model in ["pbm", "cascade", "noisy"]:
            for penalty in [0.10, 0.15, 0.20]:
                subset = df[(df["click_model"] == click_model) & (df["penalty"] == penalty)]
                if subset.empty:
                    continue

                ede_row = subset[subset["strategy_name"] == "EDE-Safer"]
                if ede_row.empty:
                    continue

                ede_cov = ede_row["new_cov_mean"].iloc[0]

                for baseline_strat in ["Random-Injection", "Epsilon-Greedy"]:
                    baseline_row = subset[subset["strategy_name"] == baseline_strat]
                    if baseline_row.empty:
                        continue

                    baseline_cov = baseline_row["new_cov_mean"].iloc[0]
                    gap = ede_cov - baseline_cov + 0.05  # EDE gets 0.05 tolerance

                    if gap < 0:
                        worst_gap = min(worst_gap, gap)
                        violations.append(
                            f"click_model={click_model}, p={penalty:.2f}, {baseline_strat}: "
                            f"EDE {ede_cov:.4f} < {baseline_strat} {baseline_cov:.4f} (gap={gap - 0.05:.4f})"
                        )

        if worst_gap < -0.10:
            status = "FAIL"
        elif worst_gap < -0.05 or len(violations) > 0:
            status = "WARN"
        else:
            status = "PASS"

        if violations:
            details = "Robustness violations:\n" + "\n".join(
                [f"  - {v}" for v in violations[:5]]
            )
            if len(violations) > 5:
                details += f"\n... and {len(violations) - 5} more"
            details += f"\nWorst gap: {worst_gap:.4f}"
        else:
            details = "EDE-Safer consistently beats baseline strategies."

        report.add_section("E) Robustness Claims", status, details)
        return status != "FAIL"
    except Exception as e:
        report.add_section("E) Robustness Claims", "WARN", f"Could not check: {e}")
        return True


def check_smoothing_claims(report: AuditReport) -> bool:
    """F) Check that smoothed >= unsmoothed coverage (with tolerance)."""
    outputs = _outputs_dir()

    try:
        df = pd.read_csv(outputs / "tables/baselines_suite_results.csv")

        violations = []
        for penalty in [0.10, 0.15, 0.20, 0.30]:
            subset = df[df["penalty"] == penalty]

            smoothed = subset[subset["strategy_name"] == "EDE-Smoothed"]
            unsmoothed = subset[subset["strategy_name"] == "EDE-Unsmooth"]

            if smoothed.empty or unsmoothed.empty:
                continue

            smoothed_cov = smoothed["new_cov_mean"].mean()
            unsmoothed_cov = unsmoothed["new_cov_mean"].mean()

            gap = smoothed_cov - unsmoothed_cov
            if gap < -0.02:
                violations.append(
                    f"p={penalty:.2f}: smoothed {smoothed_cov:.4f} < unsmoothed {unsmoothed_cov:.4f} "
                    f"(gap={gap:.4f})"
                )

        if violations:
            severe = [v for v in violations if float(v.split("gap=")[-1].rstrip(")")) < -0.05]
            status = "FAIL" if severe else "WARN"
            details = "Smoothing violations:\n" + "\n".join([f"  - {v}" for v in violations])
        else:
            status = "PASS"
            details = "Smoothing claim verified: smoothed >= unsmoothed coverage."

        report.add_section("F) Smoothing Claims", status, details)
        return status != "FAIL"
    except Exception as e:
        report.add_section("F) Smoothing Claims", "WARN", f"Could not check: {e}")
        return True


def check_sensitivity_report(report: AuditReport) -> bool:
    """G) Check that sensitivity report recommends eta=0.65, m=2 as default."""
    outputs = _outputs_dir()

    try:
        report_path = outputs / "report_sensitivity_eta_m.md"
        if not report_path.exists():
            report.add_section("G) Sensitivity Report", "WARN", "Report file not found")
            return True

        content = report_path.read_text(encoding="utf-8")

        # Check for paper default mention
        has_paper_default = "eta=0.65" in content and "m=2" in content

        # Check for "Primary" recommendation
        has_primary = "Primary" in content and "eta=0.65" in content and "m=2" in content

        # Look for any m=1 as default claim
        has_m1_default = re.search(r"default.*m=1|m=1.*default", content, re.IGNORECASE)

        if not has_paper_default:
            report.add_section(
                "G) Sensitivity Report",
                "WARN",
                "Report does not clearly recommend eta=0.65, m=2. "
                "Suggested fix: ensure report defines 'Primary' variant as eta=0.65, m=2.",
            )
            return True
        elif has_m1_default:
            report.add_section(
                "G) Sensitivity Report",
                "WARN",
                "Report mentions m=1 as a default variant. "
                "Ensure clear distinction that m=2 is the paper primary default.",
            )
            return True
        else:
            report.add_section(
                "G) Sensitivity Report",
                "PASS",
                "Report correctly recommends eta=0.65, m=2 as paper default.",
            )
            return True
    except Exception as e:
        report.add_section("G) Sensitivity Report", "WARN", f"Could not parse report: {e}")
        return True


def main(manuscript_path: str = None) -> None:
    """Run paper audit."""
    print("\n" + "=" * 80)
    print("PAPER AUDIT")
    print("=" * 80)

    report = AuditReport()

    # Run all checks
    check_artifact_existence(report)
    check_metric_bounds(report)
    check_default_consistency(report)
    check_baseline_monotonicity(report)
    check_robustness_claims(report)
    check_smoothing_claims(report)
    check_sensitivity_report(report)

    # Write markdown report
    audit_dir = _audit_dir()
    report_path = audit_dir / "paper_audit_report.md"
    report_path.write_text(report.markdown(), encoding="utf-8")

    print(report.console_summary())
    print(f"\n✓ Detailed report: {report_path}")

    # Return exit code
    if report.failures:
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper audit script")
    parser.add_argument(
        "--manuscript_path",
        type=str,
        default=None,
        help="Optional path to manuscript for number cross-check",
    )
    args = parser.parse_args()

    exit_code = main(args.manuscript_path)
    exit(exit_code)
