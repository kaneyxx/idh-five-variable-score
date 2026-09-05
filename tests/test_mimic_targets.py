"""The automatic checks, the schema, and the bundled reference must agree.

``mimic/verification_targets.json`` is what ``run_pipeline`` compares a
credentialed run against.  ``mimic/verification/mimic_aggregate_reference.json``
is the full expected output; it is produced by
``scripts/make_reference_aggregate.py`` from a complete, hash-verified run, so
it is absent until someone with PhysioNet credentials generates it.  The tests
that need it skip cleanly when it is not there.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mimic.metrics import BOOTSTRAP_METRICS, assert_aggregate_only, probability_bootstrap
from mimic.targets import default_targets_path, load_targets, verify_targets


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "mimic" / "verification" / "mimic_aggregate_reference.json"
SCHEMA = ROOT / "mimic" / "aggregate_result.schema.json"


def _reference() -> dict:
    if not REFERENCE.exists():
        pytest.skip(
            "No bundled reference aggregate. Generate one with "
            "`python scripts/make_reference_aggregate.py --mimic-root <ROOT>`."
        )
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def test_targets_file_is_the_one_the_pipeline_loads() -> None:
    assert default_targets_path() == ROOT / "mimic" / "verification_targets.json"
    assert set(load_targets()) >= {
        "score_specification_sha256",
        "score",
        "history_feature",
        "primary",
        "sensitivity",
    }


def test_targets_pin_the_repository_score_specification() -> None:
    expected = hashlib.sha256(
        (ROOT / "specification" / "score_specification.json").read_bytes()
    ).hexdigest().upper()
    assert load_targets()["score_specification_sha256"] == expected


def test_targets_declare_a_nadir90_only_history_and_no_refit() -> None:
    history = load_targets()["history_feature"]
    assert history["pre_sbp_dependent_switch_used"] is False
    assert history["score_refit"] is False
    assert history["score_recalibrated"] is False
    assert "<90 mm Hg" in history["definition"]


def test_metric_tolerances_are_tight_enough_to_matter() -> None:
    targets = load_targets()["primary"]["metrics"]
    assert targets["auroc"]["absolute_tolerance"] == pytest.approx(1e-12)
    assert targets["brier"]["absolute_tolerance"] == pytest.approx(1e-12)
    # The two BFGS-solved quantities are the only loosened checks.
    loosened = {
        name
        for name, spec in targets.items()
        if float(spec["absolute_tolerance"]) > 1e-12
    }
    assert loosened == {"calibration_intercept", "calibration_slope"}


def test_a_diverging_run_is_reported_as_failed() -> None:
    """A wrong count must not be able to pass silently."""

    targets = load_targets()
    minimal = {
        "score": {
            "public_score_specification_sha256": targets["score_specification_sha256"],
            **{k: targets["score"][k] for k in
               ("minimum", "maximum", "uf_bw_missing_points", "idh_7d_points")},
            "risk_equation": targets["score"]["risk_equation"],
        },
        "history_feature": dict(targets["history_feature"]),
        "primary": {
            "counts": dict(targets["primary"]["counts"]),
            "metrics": {
                name: spec["value"]
                for name, spec in targets["primary"]["metrics"].items()
            },
        },
        "sensitivity": {"counts": dict(targets["sensitivity"]["counts"])},
    }
    assert verify_targets(minimal)["status"] == "passed"

    minimal["primary"]["counts"]["events"] += 1
    report = verify_targets(minimal)
    assert report["status"] == "failed"
    assert [c["field"] for c in report["checks"] if not c["passed"]] == [
        "primary.counts.events"
    ]


def test_bootstrap_returns_aggregate_intervals_around_the_point_estimates() -> None:
    rng = np.random.default_rng(11)
    patients = np.repeat(np.arange(80), 6)
    probability = rng.uniform(0.05, 0.9, size=patients.size)
    outcome = (rng.uniform(size=patients.size) < probability).astype(int)

    result = probability_bootstrap(
        outcome, probability, patients, replicates=200, seed=20260725
    )
    assert result["replicate_values_returned"] is False
    assert result["patient_multiplicities_returned"] is False
    assert result["n_bootstrap"] == 200
    assert result["patients"] == 80
    assert set(result["metrics"]) <= set(BOOTSTRAP_METRICS)
    for name, interval in result["metrics"].items():
        assert interval["lower"] <= interval["upper"], name
    assert_aggregate_only(result)


def test_bundled_reference_passes_its_own_targets() -> None:
    report = verify_targets(_reference())
    assert report["failure_count"] == 0, [
        check for check in report["checks"] if not check["passed"]
    ]


def test_bundled_reference_came_from_a_complete_verified_run() -> None:
    reference = _reference()
    assert reference["source_verification"]["status"] == "passed"
    # The complete official chartevents table, not a filtered extract.
    chartevents = reference["cohort_flow"]["chartevents"]
    assert chartevents["source_rows_scanned"] > chartevents["target_item_rows"]


def test_bundled_reference_satisfies_the_public_schema_shape() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    reference = _reference()
    for key in schema["required"]:
        assert key in reference, key
    assert reference["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert reference["artifact_status"] == schema["properties"]["artifact_status"]["const"]
    for key in schema["properties"]["score"]["required"]:
        assert key in reference["score"], key


def test_bundled_reference_carries_no_row_level_output() -> None:
    assert_aggregate_only(_reference())
