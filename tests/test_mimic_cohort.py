from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mimic.cohort import (
    add_observable_history,
    build_analysis_cohorts,
    build_patient_disjoint_split,
    canonicalize_sessions,
    score_current_rule,
)
from mimic.contracts import load_score_contract


def _base_row(subject: int, day: int, event: int) -> dict[str, object]:
    start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day)
    return {
        "session_index": subject,
        "subject_id": subject,
        "hadm_id": 1000 + subject,
        "stay_id": 2000 + subject,
        "starttime": start,
        "endtime": start + pd.Timedelta(hours=4),
        "duration_hours": 4.0,
        "crrt_active_at_start": False,
        "pre_sbp": 120.0,
        "pre_dbp": 70.0,
        "heart_rate": 80.0,
        "primary_nadir_sbp_6h": 80.0 if event else 100.0,
        "primary_poststart_sbp_n_6h": 2,
        "sensitivity_nadir_sbp_6h": 80.0 if event else 100.0,
        "sensitivity_poststart_sbp_n_6h": 2,
    }


def test_canonicalization_keeps_latest_endtime() -> None:
    row = _base_row(1, 0, 1)
    earlier = dict(row)
    earlier["session_index"] = 0
    earlier["endtime"] = pd.Timestamp(row["starttime"]) + pd.Timedelta(hours=3)
    later = dict(row)
    later["session_index"] = 1
    frame = pd.DataFrame([earlier, later])
    canonical, audit = canonicalize_sessions(frame)
    assert len(canonical) == 1
    assert canonical.iloc[0]["session_index"] == 1
    assert audit["duplicate_rows_removed"] == 1


def test_patient_split_is_deterministic_and_patient_disjoint() -> None:
    frame = pd.DataFrame(
        [_base_row(subject, subject, subject % 2) for subject in range(20)]
    )
    first, _ = build_patient_disjoint_split(frame)
    second, _ = build_patient_disjoint_split(frame)
    assert first[["subject_id", "split"]].equals(second[["subject_id", "split"]])
    assert not first.groupby("subject_id")["split"].nunique().gt(1).any()
    assert set(first["split"]) == {"train", "validation", "locked_test"}


def test_history_excludes_same_day_and_leaves_unobservable_history_missing() -> None:
    starts = pd.to_datetime(
        ["2020-01-01 08:00", "2020-01-02 08:00", "2020-01-02 18:00", "2020-01-10 08:00"]
    )
    frame = pd.DataFrame(
        {
            "session_index": [0, 1, 2, 3],
            "subject_id": [1, 1, 1, 1],
            "starttime": starts,
            "endtime": starts + pd.Timedelta(hours=4),
            "duration_hours": [4.0] * 4,
            "crrt_active_at_start": [False] * 4,
            "pre_sbp": [120.0] * 4,
            "primary_nadir_sbp_6h": [80.0, 100.0, 80.0, 100.0],
            "primary_poststart_sbp_n_6h": [2] * 4,
            "split_stratum": [1, 0, 1, 0],
        }
    )
    observed = add_observable_history(frame)
    assert np.isnan(observed.loc[0, "idh_events_prior_7d"])
    assert observed.loc[1, "idh_events_prior_7d"] == 1
    assert observed.loc[2, "idh_events_prior_7d"] == 1
    assert np.isnan(observed.loc[3, "idh_events_prior_7d"])


def test_history_excludes_broad_split_rows_that_are_not_primary_eligible() -> None:
    starts = pd.date_range("2020-01-01 08:00", periods=4, freq="D")
    frame = pd.DataFrame(
        {
            "session_index": range(4),
            "subject_id": [1] * 4,
            "starttime": starts,
            "endtime": starts + pd.Timedelta(hours=4),
            "duration_hours": [4.0] * 4,
            "crrt_active_at_start": [False] * 4,
            # The first two rows are eligible for the broader split ledger but
            # not the historical primary endpoint used to define history.
            "pre_sbp": [260.0, 120.0, 120.0, 120.0],
            "primary_nadir_sbp_6h": [80.0, 40.0, 80.0, 100.0],
            "primary_poststart_sbp_n_6h": [2] * 4,
            "split_stratum": [1, 1, 1, 0],
        }
    )
    observed = add_observable_history(frame)
    assert observed.loc[3, "observable_labelled_hd_7d"] == 1
    assert observed.loc[3, "idh_events_prior_7d"] == 1


def test_history_uses_nadir90_without_a_pre_sbp_switch() -> None:
    starts = pd.date_range("2020-01-01 08:00", periods=4, freq="D")
    frame = pd.DataFrame(
        {
            "session_index": range(4),
            "subject_id": [1] * 4,
            "starttime": starts,
            "endtime": starts + pd.Timedelta(hours=4),
            "duration_hours": [4.0] * 4,
            "crrt_active_at_start": [False] * 4,
            "pre_sbp": [170.0, 170.0, 170.0, 120.0],
            # 95 is a non-event: the history event is Nadir90 only, so the
            # pre-SBP-dependent split stratifier never applies here.
            # Exactly 90 is a non-event because the rule is strictly <90.
            "primary_nadir_sbp_6h": [95.0, 90.0, 89.0, 100.0],
            "primary_poststart_sbp_n_6h": [2] * 4,
        }
    )
    observed = add_observable_history(frame)
    assert observed.loc[1, "idh_events_prior_7d"] == 0
    assert observed.loc[2, "idh_events_prior_7d"] == 0
    assert observed.loc[3, "observable_labelled_hd_7d"] == 3
    assert observed.loc[3, "idh_events_prior_7d"] == 1


def test_primary_and_per_measurement_sensitivity_are_not_interchanged() -> None:
    first = _base_row(1, 0, 0)
    first.update({"split": "locked_test", "split_stratum": 0, "idh_events_prior_7d": np.nan})
    second = _base_row(2, 0, 0)
    second.update(
        {
            "split": "locked_test",
            "split_stratum": 0,
            "idh_events_prior_7d": np.nan,
            "primary_nadir_sbp_6h": 40.0,
            "sensitivity_nadir_sbp_6h": 100.0,
        }
    )
    primary, sensitivity = build_analysis_cohorts(pd.DataFrame([first, second]))
    assert len(primary) == 1
    assert len(sensitivity) == 2
    assert int(primary["target_nadir90"].sum()) == 0
    assert int(sensitivity["target_nadir90"].sum()) == 0


def test_current_rule_requires_matching_public_specification_and_probability_map() -> None:
    frame = pd.DataFrame([_base_row(1, 0, 0)])
    frame["idh_events_prior_7d"] = 0.0
    contract = load_score_contract()

    scored = score_current_rule(frame, contract)
    assert scored["frozen_score"].tolist() == [10]
    assert scored["frozen_probability"].between(0.0, 1.0).all()

    with pytest.raises(RuntimeError, match="versions differ"):
        score_current_rule(
            frame,
            replace(contract, specification_version="1.0.0"),
        )
    with pytest.raises(RuntimeError, match="probability maps differ"):
        score_current_rule(frame, replace(contract, alpha=-4.0))
