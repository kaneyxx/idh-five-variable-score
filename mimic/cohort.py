"""Cohort, split, history, and current-score construction for MIMIC-IV."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .contracts import ScoreContract


SPLIT_SEED = 20260724


def canonicalize_sessions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep the latest end time for duplicate subject/stay/start groups."""

    required = {
        "session_index",
        "subject_id",
        "stay_id",
        "starttime",
        "endtime",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Session frame lacks columns: {sorted(missing)}")
    key = ["subject_id", "stay_id", "starttime"]
    sizes = frame.groupby(key, dropna=False).size()
    canonical = (
        frame.sort_values(key + ["endtime", "session_index"], kind="stable")
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )
    return canonical, {
        "raw_hd_sessions": int(len(frame)),
        "duplicate_start_groups": int(sizes.gt(1).sum()),
        "duplicate_rows_removed": int(len(frame) - len(canonical)),
        "canonical_hd_sessions": int(len(canonical)),
    }


def build_patient_disjoint_split(
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the historical patient-disjoint split from raw MIMIC rows.

    The split population deliberately follows its historical construction:
    pre-SBP 40--300 mmHg, a broad-screen six-hour nadir, at least one outcome
    measurement, positive documented duration, and no CRRT active at T0.
    """

    frame = canonical.copy()
    frame["starttime"] = pd.to_datetime(frame["starttime"], errors="raise")
    frame["endtime"] = pd.to_datetime(frame["endtime"], errors="raise")
    pre_sbp = pd.to_numeric(frame["pre_sbp"], errors="coerce")
    nadir = pd.to_numeric(frame["primary_nadir_sbp_6h"], errors="coerce")
    count = pd.to_numeric(frame["primary_poststart_sbp_n_6h"], errors="coerce")
    duration_valid = frame["endtime"].gt(frame["starttime"])
    label_known = (
        duration_valid
        & pre_sbp.between(40.0, 300.0, inclusive="both")
        & nadir.notna()
        & count.ge(1)
    )
    # Historical split stratification variable.  It is NOT the score history
    # event and NOT the analysis outcome: both of those are Nadir90 (nadir
    # SBP < 90 mm Hg).  This pre-SBP-dependent rule is retained verbatim for
    # one purpose only -- reproducing the pre-existing patient-disjoint fold
    # assignment bit for bit.  Changing it would silently change the test set.
    target = np.where(
        pre_sbp.lt(160.0),
        nadir.lt(90.0),
        nadir.lt(100.0),
    )
    frame["split_stratum"] = np.where(label_known, target.astype(float), np.nan)
    crrt = frame["crrt_active_at_start"].fillna(False).astype(bool)

    # This stable ordering is part of the historical SGKF contract.
    frame = frame.sort_values(
        ["subject_id", "starttime", "endtime", "session_index"], kind="stable"
    ).reset_index(drop=True)
    crrt = frame["crrt_active_at_start"].fillna(False).astype(bool)
    eligible = frame["split_stratum"].notna()
    population = frame.loc[eligible & ~crrt].copy().reset_index(drop=True)
    population["split_stratum"] = population["split_stratum"].astype(
        np.int8
    )
    if population["split_stratum"].nunique() != 2:
        raise ValueError("Split population must contain both outcome classes")

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED,
    )
    y = population["split_stratum"].to_numpy(dtype=np.int8)
    groups = population["subject_id"].to_numpy(dtype=np.int64)
    folds = np.full(len(population), -1, dtype=np.int8)
    for fold, (_, heldout) in enumerate(splitter.split(population, y, groups)):
        folds[heldout] = fold
    if (folds < 0).any():
        raise RuntimeError("Patient-disjoint split did not assign every row")
    population["split"] = np.select(
        [folds == 0, folds == 1],
        ["locked_test", "validation"],
        default="train",
    )
    if population.groupby("subject_id", sort=False)["split"].nunique().gt(1).any():
        raise RuntimeError("A patient appears in more than one split")

    by_split: dict[str, Any] = {}
    for split in ("train", "validation", "locked_test"):
        selected = population["split"].eq(split)
        by_split[split] = {
            "sessions": int(selected.sum()),
            "patients": int(population.loc[selected, "subject_id"].nunique()),
            "split_stratum_positive": int(
                population.loc[selected, "split_stratum"].sum()
            ),
        }
    audit = {
        "split_seed": SPLIT_SEED,
        "label_eligible_before_crrt_exclusion": int(eligible.sum()),
        "label_eligible_crrt_at_start_excluded": int((eligible & crrt).sum()),
        "split_ledger_sessions": int(len(population)),
        "split_ledger_patients": int(population["subject_id"].nunique()),
        "patient_overlap": 0,
        "by_split": by_split,
    }
    return population, audit


def add_observable_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add MIMIC-observable prior Nadir90 events in seven calendar days.

    A prior session is labelled for history only when it satisfies the
    historical primary endpoint eligibility contract (pre-SBP and selected
    nadir both 50--250 mm Hg, at least one post-start SBP, positive duration,
    and no CRRT at T0).  The broader 40--300 mm Hg split label must not make an
    otherwise primary-ineligible prior session observable history.

    No observable prior labelled HD is missing, not zero, because unobserved
    outpatient dialysis history cannot be inferred from MIMIC-IV.
    """

    result = frame.reset_index(drop=True).copy()
    required = {
        "pre_sbp",
        "primary_nadir_sbp_6h",
        "primary_poststart_sbp_n_6h",
        "duration_hours",
        "crrt_active_at_start",
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"History frame lacks columns: {sorted(missing)}")

    pre_sbp = pd.to_numeric(result["pre_sbp"], errors="coerce")
    nadir = pd.to_numeric(result["primary_nadir_sbp_6h"], errors="coerce")
    measurement_count = pd.to_numeric(
        result["primary_poststart_sbp_n_6h"], errors="coerce"
    )
    duration_valid = pd.to_numeric(
        result["duration_hours"], errors="coerce"
    ).gt(0)
    no_crrt = ~result["crrt_active_at_start"].fillna(False).astype(bool)
    history_label_known = (
        pre_sbp.between(50.0, 250.0, inclusive="both")
        & nadir.between(50.0, 250.0, inclusive="both")
        & measurement_count.ge(1)
        & duration_valid
        & no_crrt
    )
    history_target = np.where(history_label_known, nadir.lt(90.0), np.nan).astype(
        float
    )

    history = np.full(len(result), np.nan, dtype=float)
    labelled_count = np.zeros(len(result), dtype=np.int32)
    ordered = result.sort_values(
        ["subject_id", "starttime", "endtime", "session_index"], kind="stable"
    )
    for _, group in ordered.groupby("subject_id", sort=False):
        positions = group.index.to_numpy(dtype=np.int64)
        starts = pd.to_datetime(group["starttime"], errors="raise").to_numpy(
            dtype="datetime64[ns]"
        )
        ends = pd.to_datetime(group["endtime"], errors="raise").to_numpy(
            dtype="datetime64[ns]"
        )
        days = pd.to_datetime(group["starttime"], errors="raise").dt.normalize().to_numpy(
            dtype="datetime64[ns]"
        )
        labels = history_target[positions]
        known = np.isfinite(labels)
        for position, row_index in enumerate(positions):
            lower = days[position] - np.timedelta64(7, "D")
            prior = (
                (days[:position] >= lower)
                & (days[:position] < days[position])
                & (ends[:position] < starts[position])
                & known[:position]
            )
            count = int(prior.sum())
            labelled_count[row_index] = count
            if count:
                history[row_index] = float(labels[:position][prior].sum())
    result["observable_labelled_hd_7d"] = labelled_count
    result["idh_events_prior_7d"] = history
    return result


def build_analysis_cohorts(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return historical-primary and per-measurement sensitivity cohorts."""

    locked = frame.loc[frame["split"].astype(str).eq("locked_test")].copy()
    pre_sbp = pd.to_numeric(locked["pre_sbp"], errors="coerce")
    duration = pd.to_numeric(locked["duration_hours"], errors="coerce").gt(0)
    no_crrt = ~locked["crrt_active_at_start"].fillna(False).astype(bool)
    common = duration & pre_sbp.between(50.0, 250.0, inclusive="both") & no_crrt

    primary_nadir = pd.to_numeric(
        locked["primary_nadir_sbp_6h"], errors="coerce"
    )
    primary_count = pd.to_numeric(
        locked["primary_poststart_sbp_n_6h"], errors="coerce"
    )
    primary_mask = (
        common
        & primary_nadir.between(50.0, 250.0, inclusive="both")
        & primary_count.ge(1)
    )
    primary = locked.loc[primary_mask].copy()
    primary["target_nadir90"] = primary_nadir.loc[primary_mask].lt(90.0).astype(
        np.int8
    )

    sensitivity_nadir = pd.to_numeric(
        locked["sensitivity_nadir_sbp_6h"], errors="coerce"
    )
    sensitivity_count = pd.to_numeric(
        locked["sensitivity_poststart_sbp_n_6h"], errors="coerce"
    )
    sensitivity_mask = common & sensitivity_nadir.notna() & sensitivity_count.ge(1)
    sensitivity = locked.loc[sensitivity_mask].copy()
    sensitivity["target_nadir90"] = sensitivity_nadir.loc[sensitivity_mask].lt(
        90.0
    ).astype(np.int8)
    return primary.reset_index(drop=True), sensitivity.reset_index(drop=True)


def score_current_rule(
    frame: pd.DataFrame,
    contract: ScoreContract,
) -> pd.DataFrame:
    """Apply the repository's current 0--48 score without exporting rows."""

    from idh_score import (  # Imported lazily so extraction helpers stay standalone.
        SPECIFICATION_VERSION,
        SCORE_INTERCEPT,
        SCORE_COEFFICIENT,
        calculate_idh_score,
    )

    if str(SPECIFICATION_VERSION) != contract.specification_version:
        raise RuntimeError("Score JSON and idh_score module versions differ")
    if float(SCORE_INTERCEPT) != contract.alpha or float(SCORE_COEFFICIENT) != contract.beta:
        raise RuntimeError("Score JSON and idh_score probability maps differ")

    totals: list[int] = []
    probabilities: list[float] = []
    for row in frame.itertuples(index=False):
        scored = calculate_idh_score(
            sbp_mm_hg=row.pre_sbp,
            idh_events_prior_7d=row.idh_events_prior_7d,
            uf_bw_percent=None,
            dbp_mm_hg=row.pre_dbp,
            heart_rate_bpm=row.heart_rate,
        )
        observed_version = str(scored.specification.get("schema_version"))
        if observed_version != contract.specification_version:
            raise RuntimeError(
                "A score result reported an unexpected specification version"
            )
        totals.append(int(scored.total_points))
        probabilities.append(float(scored.predicted_probability))
    result = frame.copy()
    result["frozen_score"] = np.asarray(totals, dtype=np.int16)
    result["frozen_probability"] = np.asarray(probabilities, dtype=float)
    if not result["frozen_score"].between(
        contract.minimum, contract.maximum, inclusive="both"
    ).all():
        raise RuntimeError("Current score produced a value outside 0--48")
    return result
