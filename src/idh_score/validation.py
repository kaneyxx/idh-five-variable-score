"""Aggregate-only local validation for the frozen IDH score.

The CSV contract requires five score inputs and a binary Nadir90 outcome.
Rows with a declared missing outcome token are excluded from validation metrics;
a nonmissing outcome that is not finite 0/1 fails the contract.  When the caller
explicitly selects a complete patient-identifier column, the default is a
1,000-replicate patient-cluster bootstrap.  Patient identifiers, row-level
scores, predictions, and bootstrap multiplicities are never returned or written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from .score import (
    SCORE_INTERCEPT,
    SCORE_COEFFICIENT,
    VALID_RANGES,
    calculate_idh_score,
    probability_with_intercept_offset,
)


CSV_SCHEMA_VERSION: Final = "2.0.0"
PROBABILITY_CLIP: Final = 1e-15
DEFAULT_BOOTSTRAP_REPLICATES: Final = 1_000
DEFAULT_BOOTSTRAP_SEED: Final = 20_260_725
DEFAULT_MIN_FINITE_FRACTION: Final = 0.95
UF_PLAUSIBLE_MEDIAN_MIN: Final = 0.5
UF_PLAUSIBLE_MEDIAN_MAX: Final = 10.0

FEATURE_COLUMNS: Final = (
    "sbp_mm_hg",
    "idh_events_prior_7d",
    "uf_bw_percent",
    "dbp_mm_hg",
    "heart_rate_bpm",
)
REQUIRED_COLUMNS: Final = (*FEATURE_COLUMNS, "outcome")
MISSING_TOKENS: Final = frozenset(
    {"", ".", "na", "n/a", "nan", "none", "null", "missing"}
)
VALIDATION_METRICS: Final = (
    "auroc",
    "auprc",
    "brier",
    "calibration_intercept",
    "calibration_slope",
    "observed_expected",
    "delta",
)


class CSVContractError(ValueError):
    """Raised when a CSV cannot satisfy the public validation contract."""


class BootstrapFiniteReplicatesError(RuntimeError):
    """Raised when bootstrap metrics miss the finite-replicate requirement."""

    def __init__(self, required: int, observed: dict[str, int]) -> None:
        self.required = required
        self.observed = observed
        failures = {
            metric: count for metric, count in observed.items() if count < required
        }
        super().__init__(
            "Bootstrap metrics have too few finite replicates "
            f"(required={required}, observed={failures})"
        )


def _as_float_vector(values: Sequence[Any], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    try:
        result = array.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_metric_inputs(
    y_true: Sequence[Any], y_probability: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray]:
    target = _as_float_vector(y_true, "y_true")
    probability = _as_float_vector(y_probability, "y_probability")
    if target.size != probability.size:
        raise ValueError("y_true and y_probability must have equal length")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary (0/1)")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("y_probability must be within [0, 1]")
    return target, probability


def _calibration_intercept_slope(
    target: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[float, float]:
    """Joint unpenalized logistic recalibration on logit(predicted risk)."""

    positive_weight = sample_weight > 0
    observed_target = target[positive_weight]
    if np.unique(observed_target).size != 2:
        return float("nan"), float("nan")

    clipped = np.clip(
        probability[positive_weight], PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP
    )
    logit_probability = np.log(clipped) - np.log1p(-clipped)
    if np.ptp(logit_probability) == 0.0:
        return float("nan"), float("nan")

    calibration_target = target[positive_weight]
    event_logit = logit_probability[calibration_target == 1.0]
    non_event_logit = logit_probability[calibration_target == 0.0]
    if (
        np.max(non_event_logit) <= np.min(event_logit)
        or np.max(event_logit) <= np.min(non_event_logit)
    ):
        # With one predictor plus an intercept, non-overlapping class ranges
        # imply complete separation; ranges touching at one boundary imply
        # quasi-complete separation.  In either case the unpenalized joint
        # logistic maximum-likelihood estimate is not finite.  Optimizers can
        # nevertheless report convergence at an arbitrary large coefficient
        # because the gradient approaches zero along the separating direction.
        return float("nan"), float("nan")

    weight = sample_weight[positive_weight].astype(np.float64, copy=True)
    weight /= weight.sum()
    design = np.column_stack(
        (np.ones(logit_probability.size, dtype=np.float64), logit_probability)
    )
    prevalence = float(np.dot(weight, calibration_target))
    initial_intercept = math.log(prevalence) - math.log1p(-prevalence)
    initial = np.array([initial_intercept, 0.0], dtype=np.float64)

    def objective(coefficients: np.ndarray) -> float:
        linear_predictor = design @ coefficients
        losses = np.logaddexp(0.0, linear_predictor)
        losses -= calibration_target * linear_predictor
        return float(np.dot(weight, losses))

    def gradient(coefficients: np.ndarray) -> np.ndarray:
        residual = expit(design @ coefficients) - calibration_target
        return design.T @ (weight * residual)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 1_000},
        )

    coefficients = np.asarray(result.x, dtype=np.float64)
    score = np.asarray(gradient(coefficients), dtype=np.float64)
    if (
        not result.success
        or not np.isfinite(coefficients).all()
        or not np.isfinite(score).all()
        or np.max(np.abs(score)) > 1e-6
    ):
        return float("nan"), float("nan")
    return float(coefficients[0]), float(coefficients[1])


def _fixed_slope_intercept_offset(
    target: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Estimate delta with the supplied prediction logit held at slope one."""

    positive_weight = sample_weight > 0
    effective_target = target[positive_weight]
    effective_probability = probability[positive_weight]
    effective_weight = sample_weight[positive_weight].astype(np.float64)
    total_weight = float(effective_weight.sum())
    observed = float(np.dot(effective_weight, effective_target))
    if observed <= 0.0 or observed >= total_weight:
        return float("nan")

    clipped = np.clip(
        effective_probability, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP
    )
    base_logit = np.log(clipped) - np.log1p(-clipped)

    def score_equation(delta: float) -> float:
        expected = float(np.dot(effective_weight, expit(base_logit + delta)))
        return expected - observed

    return float(brentq(score_equation, -50.0, 50.0, xtol=1e-12, rtol=1e-12))


def _performance_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    positive_weight = sample_weight > 0
    effective_target = target[positive_weight]
    effective_probability = probability[positive_weight]
    effective_weight = sample_weight[positive_weight]
    total_weight = float(effective_weight.sum())
    observed = float(np.dot(effective_weight, effective_target))
    expected = float(np.dot(effective_weight, effective_probability))

    if np.unique(effective_target).size == 2:
        auroc = float(
            roc_auc_score(
                effective_target,
                effective_probability,
                sample_weight=effective_weight,
            )
        )
        auprc = float(
            average_precision_score(
                effective_target,
                effective_probability,
                sample_weight=effective_weight,
            )
        )
    else:
        auroc = float("nan")
        auprc = float("nan")

    brier = float(
        np.dot(effective_weight, (effective_probability - effective_target) ** 2)
        / total_weight
    )
    calibration_intercept, calibration_slope = _calibration_intercept_slope(
        target, probability, sample_weight
    )
    observed_expected = observed / expected if expected > 0.0 else float("nan")
    fixed_offset = _fixed_slope_intercept_offset(
        target, probability, sample_weight
    )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "observed_expected": observed_expected,
        "delta": fixed_offset,
        "fixed_slope_intercept_offset": fixed_offset,
    }


def compute_validation_metrics(
    y_true: Sequence[Any], y_probability: Sequence[Any]
) -> dict[str, float | int]:
    """Compute point metrics, joint calibration, O:E, and fixed-slope delta."""

    target, probability = _validate_metric_inputs(y_true, y_probability)
    sample_weight = np.ones(target.size, dtype=np.float64)
    result: dict[str, float | int] = {
        "sessions": int(target.size),
        "events": int(target.sum()),
        "non_events": int(target.size - target.sum()),
        "prevalence": float(target.mean()),
    }
    result.update(_performance_metrics(target, probability, sample_weight))
    return result


def _patient_key(value: Any) -> tuple[str, str]:
    if value is None:
        raise ValueError("patient_ids must not contain missing values")
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            if not math.isfinite(float(value)):
                raise ValueError("patient_ids must not contain missing values")
        except OverflowError:
            pass
    if isinstance(value, str) and not value.strip():
        raise ValueError("patient_ids must not contain missing values")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("patient_ids must contain scalar hashable values") from error
    return type(value).__name__, str(value)


def patient_cluster_bootstrap(
    y_true: Sequence[Any],
    y_probability: Sequence[Any],
    patient_ids: Sequence[Any],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    min_finite: int | None = None,
) -> dict[str, Any]:
    """Return aggregate patient-cluster percentile confidence intervals."""

    target, probability = _validate_metric_inputs(y_true, y_probability)
    patients = np.asarray(patient_ids, dtype=object)
    if patients.ndim != 1:
        raise ValueError("patient_ids must be one-dimensional")
    if patients.size != target.size:
        raise ValueError("patient_ids must have the same length as y_true")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral):
        raise ValueError("n_bootstrap must be a positive integer")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")

    key_to_code: dict[tuple[str, str], int] = {}
    patient_codes = np.empty(patients.size, dtype=np.int64)
    for index, value in enumerate(patients):
        key = _patient_key(value)
        if key not in key_to_code:
            key_to_code[key] = len(key_to_code)
        patient_codes[index] = key_to_code[key]
    n_patients = len(key_to_code)
    if n_patients == 0:
        raise ValueError("patient_ids must not be empty")

    if min_finite is None:
        min_finite = int(math.ceil(DEFAULT_MIN_FINITE_FRACTION * n_bootstrap))
    if isinstance(min_finite, bool) or not isinstance(min_finite, Integral):
        raise ValueError("min_finite must be an integer")
    min_finite = int(min_finite)
    if min_finite < 1 or min_finite > n_bootstrap:
        raise ValueError("min_finite must be between 1 and n_bootstrap")

    rng = np.random.default_rng(int(seed))
    replicate_values = {
        metric: np.full(n_bootstrap, np.nan, dtype=np.float64)
        for metric in VALIDATION_METRICS
    }
    cluster_probability = np.full(n_patients, 1.0 / n_patients)
    for replicate in range(n_bootstrap):
        cluster_multiplicity = rng.multinomial(n_patients, cluster_probability)
        session_weight = cluster_multiplicity[patient_codes].astype(np.float64)
        metrics = _performance_metrics(target, probability, session_weight)
        for metric in VALIDATION_METRICS:
            replicate_values[metric][replicate] = metrics[metric]

    finite_counts = {
        metric: int(np.isfinite(values).sum())
        for metric, values in replicate_values.items()
    }
    if any(count < min_finite for count in finite_counts.values()):
        raise BootstrapFiniteReplicatesError(min_finite, finite_counts)

    intervals: dict[str, dict[str, float | int | None]] = {}
    for metric, values in replicate_values.items():
        finite = values[np.isfinite(values)]
        if finite.size:
            lower, upper = np.percentile(finite, [2.5, 97.5])
            lower_value: float | None = float(lower)
            upper_value: float | None = float(upper)
        else:
            lower_value = None
            upper_value = None
        intervals[metric] = {
            "lower": lower_value,
            "upper": upper_value,
            "finite_replicates": finite_counts[metric],
        }
    return {
        "status": "complete",
        "method": "patient-cluster multinomial percentile bootstrap",
        "n_bootstrap": n_bootstrap,
        "seed": int(seed),
        "minimum_finite_replicates": min_finite,
        "patients": n_patients,
        "interval": [0.025, 0.975],
        "metrics": intervals,
        "replicate_values_returned": False,
        "patient_multiplicities_returned": False,
    }


def _is_missing_token(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_TOKENS


def _is_blank_outcome(value: Any) -> bool:
    """Only an actually blank outcome is eligible for row exclusion."""

    return value is None or not str(value).strip()


def _empty_feature_audit(rows: int, *, history: bool) -> dict[str, int]:
    audit = {
        "rows": rows,
        "valid": 0,
        "blank": 0,
        "nonnumeric": 0,
        "nonfinite": 0,
        "below_range": 0,
        "above_range": 0,
        "mapped_to_missing": 0,
    }
    if history:
        audit.update({"noninteger": 0, "negative": 0})
    return audit


def _parse_feature_column(
    records: list[dict[str, Any]], column: str
) -> tuple[list[float | int | None], dict[str, int]]:
    audit = _empty_feature_audit(
        len(records), history=column == "idh_events_prior_7d"
    )
    parsed: list[float | int | None] = []
    range_by_column = {
        "sbp_mm_hg": VALID_RANGES["Pre_HD_SBP"],
        "uf_bw_percent": VALID_RANGES["UF_BW_Perc"],
        "dbp_mm_hg": VALID_RANGES["Start_DBP"],
        "heart_rate_bpm": VALID_RANGES["Heart_Rate"],
    }
    for record in records:
        raw = record.get(column)
        category: str | None = None
        value: float | int | None = None
        if _is_missing_token(raw):
            category = "blank"
        else:
            try:
                numeric = float(str(raw).strip())
            except (TypeError, ValueError):
                category = "nonnumeric"
            else:
                if not math.isfinite(numeric):
                    category = "nonfinite"
                elif column == "idh_events_prior_7d":
                    if numeric < 0.0:
                        category = "negative"
                    elif not numeric.is_integer():
                        category = "noninteger"
                    else:
                        value = int(numeric)
                else:
                    lower, upper = range_by_column[column]
                    if numeric < lower:
                        category = "below_range"
                    elif numeric > upper:
                        category = "above_range"
                    else:
                        value = numeric
        if category is None:
            audit["valid"] += 1
        else:
            audit[category] += 1
            audit["mapped_to_missing"] += 1
        parsed.append(value)

    accounted = audit["valid"] + audit["mapped_to_missing"]
    if accounted != len(records):
        raise AssertionError(f"feature audit did not account for every {column} row")
    return parsed, audit


def _parse_outcome(
    records: list[dict[str, Any]], column: str = "outcome"
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    values: list[int] = []
    eligible = np.zeros(len(records), dtype=bool)
    failures = {
        "nonnumeric": 0,
        "nonfinite": 0,
        "not_binary_0_or_1": 0,
    }
    missing_excluded = 0
    blank_excluded = 0
    for index, record in enumerate(records):
        raw = record.get(column)
        if _is_missing_token(raw):
            missing_excluded += 1
            if _is_blank_outcome(raw):
                blank_excluded += 1
            continue
        try:
            numeric = float(str(raw).strip())
        except (TypeError, ValueError):
            failures["nonnumeric"] += 1
            continue
        if not math.isfinite(numeric):
            failures["nonfinite"] += 1
        elif numeric not in (0.0, 1.0):
            failures["not_binary_0_or_1"] += 1
        else:
            eligible[index] = True
            values.append(int(numeric))
    if any(failures.values()):
        raise CSVContractError(
            "Every nonmissing outcome must be finite and encoded only as 0/1; "
            f"invalid_counts={failures}"
        )
    target = np.asarray(values, dtype=np.int8)
    if target.size == 0:
        raise CSVContractError("outcome has no nonmissing metric-eligible rows")
    if np.unique(target).size != 2:
        raise CSVContractError(
            "nonmissing outcome rows must include at least one event and one non-event"
        )
    events = int(target.sum())
    return target, eligible, {
        "input_rows": len(records),
        "eligible_rows": int(target.size),
        "missing_excluded": missing_excluded,
        "blank_excluded": blank_excluded,
        "events": events,
        "non_events": int(target.size - events),
        "prevalence": float(target.mean()),
        **failures,
    }


def _read_csv(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CSVContractError("CSV must contain a header row")
        fieldnames = list(reader.fieldnames)
        if len(fieldnames) != len(set(fieldnames)):
            raise CSVContractError("CSV header contains duplicate column names")
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing_columns:
            raise CSVContractError(
                f"CSV is missing required columns: {', '.join(missing_columns)}"
            )
        records = list(reader)
    if not records:
        raise CSVContractError("CSV must contain at least one data row")
    if any(None in record for record in records):
        raise CSVContractError("CSV contains rows with more fields than the header")
    return records, fieldnames


def _patient_audit(
    records: list[dict[str, Any]],
    fieldnames: list[str],
    eligibility_mask: np.ndarray,
    *,
    column: str | None,
) -> tuple[list[str] | None, dict[str, Any]]:
    eligible_rows = int(eligibility_mask.sum())
    if column is None:
        return None, {
            "column_requested": False,
            "column_present": None,
            "eligible_rows": eligible_rows,
            "complete_for_eligible_rows": False,
            "missing_in_eligible_rows": None,
            "unique_patients": None,
        }
    if not isinstance(column, str) or not column.strip():
        raise ValueError("patient_id_column must be a nonblank column name")
    column = column.strip()
    if column not in fieldnames:
        raise CSVContractError(
            f"Requested patient identifier column is absent: {column}"
        )
    values: list[str | None] = []
    for record, eligible in zip(records, eligibility_mask, strict=True):
        if not eligible:
            continue
        raw = record.get(column)
        values.append(None if _is_missing_token(raw) else str(raw).strip())
    missing = sum(value is None for value in values)
    if missing:
        return None, {
            "column_requested": True,
            "column_present": True,
            "eligible_rows": eligible_rows,
            "complete_for_eligible_rows": False,
            "missing_in_eligible_rows": missing,
            "unique_patients": None,
        }
    complete = [str(value) for value in values]
    return complete, {
        "column_requested": True,
        "column_present": True,
        "eligible_rows": eligible_rows,
        "complete_for_eligible_rows": True,
        "missing_in_eligible_rows": 0,
        "unique_patients": len(set(complete)),
    }


def _finite_or_none(value: float | int) -> float | int | None:
    if isinstance(value, Integral):
        return int(value)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def validate_csv(
    path: str | Path,
    *,
    intercept_offset: float | None = None,
    patient_id_column: str | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    minimum_finite_replicates: int | None = None,
) -> dict[str, Any]:
    """Validate one canonical CSV and return aggregate-only JSON-safe results."""

    if isinstance(bootstrap_replicates, bool) or not isinstance(
        bootstrap_replicates, Integral
    ):
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    bootstrap_replicates = int(bootstrap_replicates)
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be a nonnegative integer")

    records, fieldnames = _read_csv(path)
    target, outcome_eligibility, outcome_audit = _parse_outcome(records)
    features: dict[str, list[float | int | None]] = {}
    feature_audit: dict[str, dict[str, Any]] = {}
    for column in FEATURE_COLUMNS:
        features[column], feature_audit[column] = _parse_feature_column(
            records, column
        )

    if intercept_offset is None:
        selected_offset = 0.0
        mapping_source = "published_score_equation"
    else:
        # The score helper performs strict real/finite validation.
        probability_with_intercept_offset(0, intercept_offset)
        selected_offset = float(intercept_offset)
        mapping_source = "custom_intercept_offset"

    totals: list[int] = []
    probabilities: list[float] = []
    component_counts = {
        component: Counter()
        for component in (
            "Pre_HD_SBP",
            "IDH_7D",
            "UF_BW_Perc",
            "Start_DBP",
            "Heart_Rate",
        )
    }
    for index in range(len(records)):
        result = calculate_idh_score(
            sbp_mm_hg=features["sbp_mm_hg"][index],
            idh_events_prior_7d=features["idh_events_prior_7d"][index],
            uf_bw_percent=features["uf_bw_percent"][index],
            dbp_mm_hg=features["dbp_mm_hg"][index],
            heart_rate_bpm=features["heart_rate_bpm"][index],
        )
        totals.append(result.total_points)
        probabilities.append(
            probability_with_intercept_offset(result.total_points, selected_offset)
        )
        for component, points in result.component_points.items():
            component_counts[component][points] += 1

    score_array = np.asarray(totals, dtype=np.int16)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    eligible_score = score_array[outcome_eligibility]
    eligible_probability = probability_array[outcome_eligibility]
    point = compute_validation_metrics(target, eligible_probability)
    point_json = {name: _finite_or_none(value) for name, value in point.items()}
    local_delta = point["delta"]
    if math.isfinite(float(local_delta)):
        total_local_offset: float | None = selected_offset + float(local_delta)
    else:
        total_local_offset = None

    warnings: list[dict[str, Any]] = []
    valid_uf = [
        float(value) for value in features["uf_bw_percent"] if value is not None
    ]
    uf_median: float | None = float(np.median(valid_uf)) if valid_uf else None
    if uf_median is not None and (
        uf_median < UF_PLAUSIBLE_MEDIAN_MIN
        or uf_median > UF_PLAUSIBLE_MEDIAN_MAX
    ):
        warnings.append(
            {
                "code": "uf_bw_median_outside_plausible_percentage_point_range",
                "observed_valid_median": uf_median,
                "message": (
                    "The valid UF/BW median is outside 0.5-10 percentage points. "
                    "Confirm units and source-field mapping (enter 3.5 for 3.5%)."
                ),
            }
        )
    invalid_feature_rows = sum(
        audit["nonnumeric"]
        + audit["nonfinite"]
        + audit["below_range"]
        + audit["above_range"]
        + audit.get("noninteger", 0)
        + audit.get("negative", 0)
        for audit in feature_audit.values()
    )
    if invalid_feature_rows:
        warnings.append(
            {
                "code": "invalid_feature_values_used_missing_branches",
                "count_across_feature_cells": invalid_feature_rows,
                "message": (
                    "Invalid feature cells were retained and assigned their "
                    "predictor-specific frozen missing points."
                ),
            }
        )

    patient_ids, patient_audit = _patient_audit(
        records,
        fieldnames,
        outcome_eligibility,
        column=patient_id_column,
    )
    if patient_ids is not None and bootstrap_replicates > 0:
        bootstrap = patient_cluster_bootstrap(
            target,
            eligible_probability,
            patient_ids,
            n_bootstrap=bootstrap_replicates,
            seed=bootstrap_seed,
            min_finite=minimum_finite_replicates,
        )
    elif bootstrap_replicates == 0:
        bootstrap = {
            "status": "not_performed",
            "reason": "disabled_by_caller",
            "default_when_complete_patient_id": DEFAULT_BOOTSTRAP_REPLICATES,
        }
    else:
        reason = (
            "patient_id_column_not_requested"
            if not patient_audit["column_requested"]
            else "patient_id_column_incomplete_for_metric_eligible_rows"
        )
        bootstrap = {
            "status": "not_performed",
            "reason": reason,
            "default_when_complete_patient_id": DEFAULT_BOOTSTRAP_REPLICATES,
        }
        warnings.append(
            {
                "code": "cluster_bootstrap_not_performed",
                "message": (
                    "A patient-cluster bootstrap requires an explicitly selected "
                    "identifier column complete for metric-eligible rows; no row "
                    "bootstrap was substituted."
                ),
            }
        )

    score_counts = Counter(int(value) for value in score_array)
    eligible_score_counts = Counter(int(value) for value in eligible_score)
    known_columns = set(REQUIRED_COLUMNS)
    if patient_id_column is not None:
        known_columns.add(patient_id_column.strip())
    extra_columns = [
        name
        for name in fieldnames
        if name not in known_columns
    ]
    for column, audit in feature_audit.items():
        audit["component_point_counts"] = {
            str(points): int(count)
            for points, count in sorted(
                component_counts[
                    {
                        "sbp_mm_hg": "Pre_HD_SBP",
                        "idh_events_prior_7d": "IDH_7D",
                        "uf_bw_percent": "UF_BW_Perc",
                        "dbp_mm_hg": "Start_DBP",
                        "heart_rate_bpm": "Heart_Rate",
                    }[column]
                ].items()
            )
        }

    return {
        "schema_version": CSV_SCHEMA_VERSION,
        "status": "complete",
        "scope": {
            "rule": "fixed_reported_TN_Nadir90_score",
            "result": "local_input_aggregate_validation",
            "private_cohort_reproduction": False,
        },
        "outcome_contract": {
            "name": "Nadir90",
            "csv_column": "outcome",
            "event_encoding": 1,
            "non_event_encoding": 0,
            "definition": (
                "current-session intradialytic nadir systolic blood pressure "
                "below 90 mm Hg"
            ),
            "comparison_operator": "<",
            "threshold_mm_hg": 90.0,
            "measurement_window": {
                "notation": "(T0, recorded treatment end]",
                "start": "treatment initiation (T0)",
                "end": "recorded treatment end",
                "start_inclusive": False,
                "end_inclusive": True,
            },
        },
        "probability_mapping": {
            "source": mapping_source,
            "score_intercept": SCORE_INTERCEPT,
            "score_coefficient": SCORE_COEFFICIENT,
            "applied_intercept_offset": selected_offset,
            "formula": "expit(score intercept + applied offset + coefficient * total score)",
            "estimated_fixed_slope_delta_from_applied_map": _finite_or_none(
                local_delta
            ),
            "estimated_total_offset": total_local_offset,
        },
        "metrics": point_json,
        "bootstrap": bootstrap,
        "audit": {
            "rows": len(records),
            "required_columns_present": True,
            "extra_columns_count": len(extra_columns),
            "outcome": outcome_audit,
            "patient_id": patient_audit,
            "features": feature_audit,
            "uf_bw_percent_valid_median": uf_median,
            "score": {
                "input_rows": int(score_array.size),
                "metric_eligible_rows": int(eligible_score.size),
                "minimum_observed": int(score_array.min()),
                "maximum_observed": int(score_array.max()),
                "mean": float(score_array.mean()),
                "median": float(np.median(score_array)),
                "counts_0_to_48_all_input_rows": {
                    str(score): int(score_counts.get(score, 0))
                    for score in range(49)
                },
                "counts_0_to_48_metric_eligible_rows": {
                    str(score): int(eligible_score_counts.get(score, 0))
                    for score in range(49)
                },
            },
        },
        "warnings": warnings,
        "privacy": {
            "aggregate_only": True,
            "row_level_scores_or_probabilities_returned": False,
            "patient_identifiers_returned": False,
            "bootstrap_replicates_or_multiplicities_returned": False,
            "input_path_returned": False,
        },
    }


def build_validation_parser() -> argparse.ArgumentParser:
    """Build the parser used by the package-level validation entry point."""

    parser = argparse.ArgumentParser(
        prog="idh-validate",
        description=(
            "Score and locally validate a canonical CSV; output is aggregate-only."
        ),
    )
    parser.add_argument("csv_path", help="Input CSV satisfying the public contract")
    parser.add_argument(
        "--output",
        help="Optional aggregate JSON output path; stdout is used when omitted",
    )
    parser.add_argument(
        "--intercept-offset",
        type=float,
        help="Finite custom offset from the published score intercept",
    )
    parser.add_argument(
        "--patient-id",
        dest="patient_id_column",
        metavar="COLUMN",
        help=(
            "Column to use for patient clustering; without this option no "
            "cluster bootstrap is performed"
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="Patient-cluster replicates when patient_id is complete (default: 1000)",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument("--minimum-finite-replicates", type=int)
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap even when patient_id is complete",
    )
    return parser


def validation_cli(argv: Sequence[str] | None = None) -> int:
    """Run CSV validation and emit aggregate-only JSON."""

    parser = build_validation_parser()
    args = parser.parse_args(argv)
    result = validate_csv(
        args.csv_path,
        intercept_offset=args.intercept_offset,
        patient_id_column=args.patient_id_column,
        bootstrap_replicates=(
            0 if args.no_bootstrap else args.bootstrap_replicates
        ),
        bootstrap_seed=args.bootstrap_seed,
        minimum_finite_replicates=args.minimum_finite_replicates,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Alias for entry-point wiring by the repository integrator."""

    return validation_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
