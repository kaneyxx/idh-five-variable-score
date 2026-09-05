"""Compare a credentialed MIMIC-IV run with the published reference aggregate.

The reference numbers in ``verification_targets.json`` are the values reported
in the manuscript's MIMIC-IV analysis.  A credentialed runner with the official
MIMIC-IV v2.2 files should reproduce them exactly; ``run_pipeline`` exits with
status 2 when any check fails, so a silent divergence cannot be mistaken for a
successful reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_targets_path() -> Path:
    return Path(__file__).with_name("verification_targets.json")


def load_targets(path: str | Path | None = None) -> dict[str, Any]:
    target_path = Path(path) if path is not None else default_targets_path()
    return json.loads(target_path.read_text(encoding="utf-8"))


def verify_targets(
    result: dict[str, Any], targets: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a per-field pass/fail report for one credentialed run."""

    reference = load_targets() if targets is None else targets
    checks: list[dict[str, Any]] = []

    exact_fields = (
        (
            "score.public_score_specification_sha256",
            reference["score_specification_sha256"],
            result["score"]["public_score_specification_sha256"],
        ),
        ("score.minimum", reference["score"]["minimum"], result["score"]["minimum"]),
        ("score.maximum", reference["score"]["maximum"], result["score"]["maximum"]),
        (
            "score.uf_bw_missing_points",
            reference["score"]["uf_bw_missing_points"],
            result["score"]["uf_bw_missing_points"],
        ),
        (
            "score.idh_7d_points",
            reference["score"]["idh_7d_points"],
            result["score"]["idh_7d_points"],
        ),
        (
            "score.risk_equation",
            reference["score"]["risk_equation"],
            result["score"]["risk_equation"],
        ),
        (
            "history_feature.definition",
            reference["history_feature"]["definition"],
            result["history_feature"]["definition"],
        ),
        (
            "history_feature.pre_sbp_dependent_switch_used",
            reference["history_feature"]["pre_sbp_dependent_switch_used"],
            result["history_feature"]["pre_sbp_dependent_switch_used"],
        ),
        (
            "history_feature.score_refit",
            reference["history_feature"]["score_refit"],
            result["history_feature"]["score_refit"],
        ),
        (
            "history_feature.score_recalibrated",
            reference["history_feature"]["score_recalibrated"],
            result["history_feature"]["score_recalibrated"],
        ),
    )
    for field, expected, observed in exact_fields:
        checks.append(
            {
                "field": field,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    for cohort_name in ("primary", "sensitivity"):
        expected_counts = reference[cohort_name]["counts"]
        observed_counts = result[cohort_name]["counts"]
        for metric, expected in expected_counts.items():
            observed = int(observed_counts[metric])
            checks.append(
                {
                    "field": f"{cohort_name}.counts.{metric}",
                    "expected": int(expected),
                    "observed": observed,
                    "absolute_tolerance": 0,
                    "passed": observed == int(expected),
                }
            )
    for metric, specification in reference["primary"]["metrics"].items():
        expected = float(specification["value"])
        tolerance = float(specification["absolute_tolerance"])
        observed = float(result["primary"]["metrics"][metric])
        checks.append(
            {
                "field": f"primary.metrics.{metric}",
                "expected": expected,
                "observed": observed,
                "absolute_difference": abs(observed - expected),
                "absolute_tolerance": tolerance,
                "passed": abs(observed - expected) <= tolerance,
            }
        )
    failures = [item for item in checks if not item["passed"]]
    return {
        "status": "passed" if not failures else "failed",
        "reference_status": str(reference["status"]),
        "checks": checks,
        "failure_count": len(failures),
    }
