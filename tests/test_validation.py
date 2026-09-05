from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from idh_score.validation import (
    BootstrapFiniteReplicatesError,
    CSVContractError,
    DEFAULT_BOOTSTRAP_REPLICATES,
    VALIDATION_METRICS,
    compute_validation_metrics,
    patient_cluster_bootstrap,
    validate_csv,
    validation_cli,
)


FIELDNAMES = [
    "patient_id",
    "sbp_mm_hg",
    "idh_events_prior_7d",
    "uf_bw_percent",
    "dbp_mm_hg",
    "heart_rate_bpm",
    "outcome",
]


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    selected_fields = FIELDNAMES if fieldnames is None else fieldnames
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in selected_fields})


def test_point_metrics_have_known_values() -> None:
    metrics = compute_validation_metrics(
        [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    )
    assert metrics["sessions"] == 4
    assert metrics["events"] == 2
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5.0 / 6.0)
    assert metrics["brier"] == pytest.approx(0.158125)
    assert metrics["observed_expected"] == pytest.approx(2.0 / 1.65)
    assert math.isfinite(float(metrics["calibration_intercept"]))
    assert math.isfinite(float(metrics["calibration_slope"]))


def test_joint_calibration_recovers_known_intercept_and_slope() -> None:
    probability = np.repeat([0.2, 0.8], 10)
    target = np.array([1, 1, *([0] * 8), *([1] * 8), 0, 0])
    metrics = compute_validation_metrics(target, probability)
    assert float(metrics["calibration_intercept"]) == pytest.approx(0.0, abs=1e-8)
    assert float(metrics["calibration_slope"]) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    ("target", "probability"),
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        ([0, 0, 1, 1], [0.2, 0.5, 0.5, 0.8]),
    ],
    ids=("complete-separation", "quasi-complete-separation"),
)
def test_joint_calibration_is_not_estimable_under_separation(
    target: list[int], probability: list[float]
) -> None:
    metrics = compute_validation_metrics(target, probability)
    assert math.isnan(float(metrics["calibration_intercept"]))
    assert math.isnan(float(metrics["calibration_slope"]))


def test_fixed_slope_delta_solves_observed_equals_expected() -> None:
    target = [0, 1] * 5
    probability = [0.2] * 10
    metrics = compute_validation_metrics(target, probability)
    expected_delta = -math.log(0.2 / 0.8)
    assert float(metrics["delta"]) == pytest.approx(expected_delta, abs=1e-10)
    assert float(metrics["fixed_slope_intercept_offset"]) == pytest.approx(
        expected_delta, abs=1e-10
    )


def test_patient_cluster_bootstrap_is_reproducible_and_aggregate_only() -> None:
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    probability = np.array(
        [0.08, 0.22, 0.31, 0.44, 0.53, 0.66, 0.72, 0.81, 0.89, 0.95]
    )
    patients = np.array([f"p{index}" for index in range(10)], dtype=object)
    first = patient_cluster_bootstrap(
        target, probability, patients, n_bootstrap=40, seed=1234
    )
    second = patient_cluster_bootstrap(
        target, probability, patients, n_bootstrap=40, seed=1234
    )
    assert first == second
    assert first["replicate_values_returned"] is False
    assert first["patient_multiplicities_returned"] is False
    assert first["patients"] == 10
    assert first["minimum_finite_replicates"] == 38
    assert set(first["metrics"]) == set(VALIDATION_METRICS)
    expected_intervals = {
        "auroc": (40, 0.206125, 0.959375),
        "auprc": (40, 0.33125, 0.9767857142857141),
        "brier": (40, 0.11783174999999999, 0.44638925),
        "calibration_intercept": (39, -2.755572041931832, 1.8201839733365488),
        "calibration_slope": (39, -0.9477392928167525, 3.131891646083854),
        "observed_expected": (40, 0.3144645209862601, 1.2248129156234822),
        "delta": (40, -2.700405867588989, 0.6482979651478962),
    }
    for metric, (finite, lower, upper) in expected_intervals.items():
        interval = first["metrics"][metric]
        assert interval["finite_replicates"] == finite
        assert interval["lower"] == pytest.approx(lower, rel=1e-7, abs=1e-9)
        assert interval["upper"] == pytest.approx(upper, rel=1e-7, abs=1e-9)


def test_separated_bootstrap_fails_the_default_finite_replicate_gate() -> None:
    target = np.array([0] * 5 + [1] * 5)
    probability = np.array(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    patients = np.array([f"p{index}" for index in range(10)], dtype=object)

    with pytest.raises(BootstrapFiniteReplicatesError) as caught:
        patient_cluster_bootstrap(
            target, probability, patients, n_bootstrap=40, seed=1234
        )

    assert caught.value.required == 38
    assert caught.value.observed["calibration_intercept"] == 0
    assert caught.value.observed["calibration_slope"] == 0


def test_bootstrap_requires_at_least_one_finite_replicate() -> None:
    with pytest.raises(ValueError, match="between 1 and n_bootstrap"):
        patient_cluster_bootstrap(
            [0, 1], [0.2, 0.8], ["p0", "p1"], n_bootstrap=10, min_finite=0
        )


def _valid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(12):
        rows.append(
            {
                "patient_id": f"p{index // 2}",
                "sbp_mm_hg": 95 + index * 4,
                "idh_events_prior_7d": index % 3,
                "uf_bw_percent": 1.5 + (index % 5),
                "dbp_mm_hg": 42 + index * 3,
                "heart_rate_bpm": 65 + index * 4,
                "outcome": int(index % 3 == 0),
            }
        )
    return rows


def test_validate_csv_runs_cluster_bootstrap_when_patient_id_complete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.csv"
    _write_csv(path, _valid_rows())
    result = validate_csv(
        path,
        patient_id_column="patient_id",
        bootstrap_replicates=30,
        bootstrap_seed=17,
        minimum_finite_replicates=1,
    )
    assert result["schema_version"] == "2.0.0"
    assert result["status"] == "complete"
    assert result["scope"] == {
        "rule": "fixed_reported_TN_Nadir90_score",
        "result": "local_input_aggregate_validation",
        "private_cohort_reproduction": False,
    }
    assert result["outcome_contract"]["measurement_window"] == {
        "notation": "(T0, recorded treatment end]",
        "start": "treatment initiation (T0)",
        "end": "recorded treatment end",
        "start_inclusive": False,
        "end_inclusive": True,
    }
    assert result["probability_mapping"]["source"] == "published_score_equation"
    assert result["probability_mapping"]["applied_intercept_offset"] == 0.0
    assert result["audit"]["rows"] == 12
    assert result["audit"]["outcome"]["events"] == 4
    assert result["audit"]["patient_id"] == {
        "column_requested": True,
        "column_present": True,
        "eligible_rows": 12,
        "complete_for_eligible_rows": True,
        "missing_in_eligible_rows": 0,
        "unique_patients": 6,
    }
    assert result["bootstrap"]["status"] == "complete"
    assert result["bootstrap"]["n_bootstrap"] == 30
    assert "delta" in result["bootstrap"]["metrics"]
    assert result["privacy"]["aggregate_only"] is True
    assert result["privacy"]["patient_identifiers_returned"] is False
    assert '"p0"' not in json.dumps(result)
    assert len(
        result["audit"]["score"]["counts_0_to_48_all_input_rows"]
    ) == 49


def test_missing_outcomes_are_excluded_and_patient_ids_are_subset_in_sync(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blank_outcome.csv"
    rows = _valid_rows()
    for index, row in enumerate(rows):
        row["cluster_key"] = "" if index < 2 else f"eligible-{index // 2}"
    rows[0]["outcome"] = ""
    rows[1]["outcome"] = "N/A"
    fields = ["cluster_key", *FIELDNAMES[1:]]
    _write_csv(path, rows, fieldnames=fields)

    result = validate_csv(
        path,
        patient_id_column="cluster_key",
        bootstrap_replicates=30,
        bootstrap_seed=17,
        minimum_finite_replicates=1,
    )
    assert result["metrics"]["sessions"] == 10
    assert result["audit"]["outcome"]["input_rows"] == 12
    assert result["audit"]["outcome"]["eligible_rows"] == 10
    assert result["audit"]["outcome"]["missing_excluded"] == 2
    assert result["audit"]["outcome"]["blank_excluded"] == 1
    assert result["audit"]["patient_id"]["eligible_rows"] == 10
    assert result["audit"]["patient_id"]["complete_for_eligible_rows"] is True
    assert result["bootstrap"]["patients"] == 5


def test_arbitrary_patient_identifier_column_enables_cluster_bootstrap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "arbitrary_patient_column.csv"
    rows = _valid_rows()
    for index, row in enumerate(rows):
        row["dialysis_person"] = f"person-{index // 2}"
    fields = ["dialysis_person", *FIELDNAMES[1:]]
    _write_csv(path, rows, fieldnames=fields)

    result = validate_csv(
        path,
        patient_id_column="dialysis_person",
        bootstrap_replicates=30,
        bootstrap_seed=17,
        minimum_finite_replicates=1,
    )
    assert result["bootstrap"]["status"] == "complete"
    assert result["bootstrap"]["patients"] == 6
    assert result["audit"]["patient_id"]["unique_patients"] == 6


def test_patient_id_is_not_auto_detected_without_explicit_option(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not_auto_detected.csv"
    _write_csv(path, _valid_rows()[:6])
    result = validate_csv(path)
    assert result["bootstrap"]["status"] == "not_performed"
    assert result["bootstrap"]["reason"] == "patient_id_column_not_requested"
    assert result["audit"]["patient_id"]["column_requested"] is False


def test_custom_probability_mapping_is_generic_intercept_offset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "offset.csv"
    _write_csv(path, _valid_rows(), fieldnames=FIELDNAMES[1:])
    result = validate_csv(path, intercept_offset=-0.25, bootstrap_replicates=0)
    assert result["probability_mapping"]["source"] == "custom_intercept_offset"
    assert result["probability_mapping"]["applied_intercept_offset"] == -0.25


def test_validate_csv_audits_invalid_features_and_warns_on_uf_units(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.csv"
    rows = _valid_rows()[:6]
    rows[0]["sbp_mm_hg"] = ""
    rows[1]["sbp_mm_hg"] = "not-a-number"
    rows[2]["sbp_mm_hg"] = 999
    rows[3]["idh_events_prior_7d"] = 1.5
    for index, row in enumerate(rows):
        row["uf_bw_percent"] = 0.03 + index * 0.01
    _write_csv(path, rows, fieldnames=FIELDNAMES[1:])

    result = validate_csv(path, bootstrap_replicates=1000)
    sbp = result["audit"]["features"]["sbp_mm_hg"]
    assert sbp["blank"] == 1
    assert sbp["nonnumeric"] == 1
    assert sbp["above_range"] == 1
    assert sbp["mapped_to_missing"] == 3
    history = result["audit"]["features"]["idh_events_prior_7d"]
    assert history["noninteger"] == 1
    warning_codes = {warning["code"] for warning in result["warnings"]}
    assert "uf_bw_median_outside_plausible_percentage_point_range" in warning_codes
    assert "invalid_feature_values_used_missing_branches" in warning_codes
    assert "cluster_bootstrap_not_performed" in warning_codes
    assert result["bootstrap"]["reason"] == "patient_id_column_not_requested"


@pytest.mark.parametrize(
    ("uf_median", "warning_expected"),
    [(0.49, True), (0.5, False), (10.0, False), (10.01, True)],
)
def test_uf_median_warning_uses_strict_half_to_ten_boundaries(
    tmp_path: Path, uf_median: float, warning_expected: bool
) -> None:
    path = tmp_path / f"uf-{uf_median}.csv"
    rows = _valid_rows()[:6]
    for row in rows:
        row["uf_bw_percent"] = uf_median
    _write_csv(path, rows, fieldnames=FIELDNAMES[1:])
    result = validate_csv(path, bootstrap_replicates=0)
    codes = {warning["code"] for warning in result["warnings"]}
    assert (
        "uf_bw_median_outside_plausible_percentage_point_range" in codes
    ) is warning_expected


def test_incomplete_patient_id_does_not_fall_back_to_row_bootstrap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete_patient.csv"
    rows = _valid_rows()[:6]
    rows[2]["patient_id"] = ""
    _write_csv(path, rows)
    result = validate_csv(path, patient_id_column="patient_id")
    assert result["bootstrap"] == {
        "status": "not_performed",
        "reason": "patient_id_column_incomplete_for_metric_eligible_rows",
        "default_when_complete_patient_id": 1000,
    }
    assert result["audit"]["patient_id"]["missing_in_eligible_rows"] == 1


@pytest.mark.parametrize("bad_outcome", ["yes", 2, float("inf")])
def test_nonmissing_outcome_must_be_finite_binary_nadir90(
    tmp_path: Path, bad_outcome: object
) -> None:
    path = tmp_path / "bad_outcome.csv"
    rows = _valid_rows()[:6]
    rows[0]["outcome"] = bad_outcome
    _write_csv(path, rows, fieldnames=FIELDNAMES[1:])
    with pytest.raises(CSVContractError, match="nonmissing outcome"):
        validate_csv(path, bootstrap_replicates=0)


def test_csv_contract_requires_all_five_inputs(tmp_path: Path) -> None:
    path = tmp_path / "missing_input.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sbp_mm_hg", "idh_events_prior_7d", "outcome"],
        )
        writer.writeheader()
        writer.writerow({"sbp_mm_hg": 120, "idh_events_prior_7d": 0, "outcome": 0})
    with pytest.raises(CSVContractError, match="missing required columns"):
        validate_csv(path, bootstrap_replicates=0)


def test_complete_patient_id_defaults_to_1000_bootstrap_replicates() -> None:
    signature = inspect.signature(validate_csv)
    assert signature.parameters["bootstrap_replicates"].default == 1000
    assert DEFAULT_BOOTSTRAP_REPLICATES == 1000


def test_cli_function_emits_json_without_row_level_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cli.csv"
    _write_csv(path, _valid_rows()[:6], fieldnames=FIELDNAMES[1:])
    assert validation_cli([str(path), "--no-bootstrap"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["status"] == "complete"
    assert rendered["bootstrap"]["reason"] == "disabled_by_caller"
    assert rendered["privacy"]["row_level_scores_or_probabilities_returned"] is False


def test_cli_accepts_arbitrary_patient_id_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cli_cluster.csv"
    rows = _valid_rows()
    for index, row in enumerate(rows):
        row["cohort_person"] = f"p-{index // 2}"
    _write_csv(path, rows, fieldnames=["cohort_person", *FIELDNAMES[1:]])
    assert validation_cli(
        [
            str(path),
            "--patient-id",
            "cohort_person",
            "--bootstrap-replicates",
            "30",
            "--bootstrap-seed",
            "17",
            "--minimum-finite-replicates",
            "1",
        ]
    ) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["bootstrap"]["status"] == "complete"
    assert rendered["bootstrap"]["patients"] == 6
