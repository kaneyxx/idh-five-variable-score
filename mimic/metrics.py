"""Aggregate-only probability metrics for the MIMIC-IV stress test."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def probability_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(outcome, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0 or np.unique(y).size != 2:
        raise ValueError("Probability metrics require a nonempty binary-outcome cohort")
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("Probabilities must be finite, strictly between 0 and 1, and aligned")
    # Reuse the public core's unpenalized joint calibration implementation;
    # this keeps local validation and the MIMIC workflow on one definition.
    from idh_score.validation import compute_validation_metrics

    computed = compute_validation_metrics(y, p)
    names = (
        "auroc",
        "auprc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "observed_expected",
    )
    return {name: float(computed[name]) for name in names}


BOOTSTRAP_METRICS = (
    "auroc",
    "auprc",
    "brier",
    "calibration_intercept",
    "calibration_slope",
    "observed_expected",
)


def probability_bootstrap(
    outcome: np.ndarray,
    probability: np.ndarray,
    patient_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Return aggregate patient-cluster confidence intervals for one cohort.

    Only interval bounds and replicate bookkeeping are returned. Bootstrap
    draws, per-replicate values, and patient multiplicities are never exported.
    """

    from idh_score.validation import patient_cluster_bootstrap

    raw = patient_cluster_bootstrap(
        np.asarray(outcome, dtype=int),
        np.asarray(probability, dtype=float),
        np.asarray(patient_ids, dtype=object),
        n_bootstrap=int(replicates),
        seed=int(seed),
    )
    computed = raw["metrics"]
    intervals = {
        name: {
            "lower": float(computed[name]["lower"]),
            "upper": float(computed[name]["upper"]),
            "finite_replicates": int(computed[name]["finite_replicates"]),
        }
        for name in BOOTSTRAP_METRICS
        if name in computed and computed[name]["lower"] is not None
    }
    non_estimable = sorted(
        name
        for name in BOOTSTRAP_METRICS
        if name in computed and computed[name]["lower"] is None
    )
    return {
        "status": raw["status"],
        "method": raw["method"],
        "n_bootstrap": int(raw["n_bootstrap"]),
        "seed": int(raw["seed"]),
        "patients": int(raw["patients"]),
        "minimum_finite_replicates": int(raw["minimum_finite_replicates"]),
        "interval": list(raw["interval"]),
        "replicate_values_returned": False,
        "patient_multiplicities_returned": False,
        "non_estimable": non_estimable,
        "metrics": intervals,
    }


def cohort_counts(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "sessions": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "events": int(frame["target_nadir90"].sum()),
        "prevalence": float(frame["target_nadir90"].mean()),
    }


FORBIDDEN_OUTPUT_KEYS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "session_id",
    "session_index",
    "records",
    "row_data",
}


def assert_aggregate_only(value: Any, *, location: str = "root") -> None:
    """Reject accidental row identifiers before JSON serialization."""

    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_OUTPUT_KEYS:
                raise ValueError(f"Restricted row-level key at {location}.{key}")
            assert_aggregate_only(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_aggregate_only(item, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite aggregate value at {location}")
