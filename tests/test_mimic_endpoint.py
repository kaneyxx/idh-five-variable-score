from __future__ import annotations

import pandas as pd

from mimic.contracts import NIBP_DBP_ITEMID, NIBP_SBP_ITEMID, HEART_RATE_ITEMID
from mimic.extraction import (
    endpoint_window_end,
    extract_procedure_sessions,
    is_in_outcome_window,
    scan_vitals,
)


def test_outcome_window_is_open_left_and_closed_right() -> None:
    start = pd.Timestamp("2020-01-01 00:00:00")
    documented_end = pd.Timestamp("2020-01-01 08:00:00")
    assert endpoint_window_end(start, documented_end) == pd.Timestamp(
        "2020-01-01 06:00:00"
    )
    assert not is_in_outcome_window(start, start, documented_end)
    assert is_in_outcome_window(start + pd.Timedelta(microseconds=1), start, documented_end)
    assert is_in_outcome_window(start + pd.Timedelta(hours=6), start, documented_end)
    assert not is_in_outcome_window(
        start + pd.Timedelta(hours=6, microseconds=1), start, documented_end
    )


def test_documented_end_is_included_when_before_six_hours() -> None:
    start = pd.Timestamp("2020-01-01 00:00:00")
    documented_end = pd.Timestamp("2020-01-01 04:00:00")
    assert is_in_outcome_window(documented_end, start, documented_end)
    assert not is_in_outcome_window(
        documented_end + pd.Timedelta(microseconds=1), start, documented_end
    )


def test_scanner_includes_exact_six_hours_and_separates_screening(tmp_path) -> None:
    start = pd.Timestamp("2020-01-01 00:00:00")
    sessions = pd.DataFrame(
        {
            "session_index": [0, 1],
            "subject_id": [1, 2],
            "hadm_id": [11, 22],
            "stay_id": [101, 202],
            "starttime": [start, start],
            "endtime": [start + pd.Timedelta(hours=8)] * 2,
            "duration_hours": [8.0, 8.0],
            "status": ["FinishedRunning", "FinishedRunning"],
            "crrt_active_at_start": [False, False],
            "is_raw_canonical_session": [True, True],
        }
    )
    rows = [
        # Session 0: T0 is pre-index only; exactly +6 h is retained; +6 h +1 s is not.
        (101, start, NIBP_SBP_ITEMID, 120.0),
        (101, start, NIBP_DBP_ITEMID, 70.0),
        (101, start, HEART_RATE_ITEMID, 80.0),
        (101, start + pd.Timedelta(hours=1), NIBP_SBP_ITEMID, 100.0),
        (101, start + pd.Timedelta(hours=6), NIBP_SBP_ITEMID, 80.0),
        (101, start + pd.Timedelta(hours=6, seconds=1), NIBP_SBP_ITEMID, 60.0),
        # Session 1: broad primary selects 40, while per-measurement sensitivity selects 100.
        (202, start, NIBP_SBP_ITEMID, 140.0),
        (202, start + pd.Timedelta(hours=1), NIBP_SBP_ITEMID, 40.0),
        (202, start + pd.Timedelta(hours=2), NIBP_SBP_ITEMID, 100.0),
    ]
    chart = pd.DataFrame(rows, columns=["stay_id", "charttime", "itemid", "valuenum"])
    path = tmp_path / "chartevents.csv.gz"
    chart.to_csv(path, index=False, compression="gzip")

    scanned, _ = scan_vitals(sessions, path, chunksize=2)
    assert scanned.loc[0, "primary_poststart_sbp_n_6h"] == 2
    assert scanned.loc[0, "primary_nadir_sbp_6h"] == 80.0
    assert scanned.loc[0, "sensitivity_nadir_sbp_6h"] == 80.0
    assert scanned.loc[1, "primary_nadir_sbp_6h"] == 40.0
    assert scanned.loc[1, "sensitivity_nadir_sbp_6h"] == 100.0
    assert scanned.loc[1, "primary_poststart_sbp_n_6h"] == 2
    assert scanned.loc[1, "sensitivity_poststart_sbp_n_6h"] == 1


def test_historical_split_treats_crrt_ending_at_t0_as_active(tmp_path) -> None:
    start = pd.Timestamp("2020-01-01 08:00:00")
    procedure = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 11,
                "stay_id": 101,
                "starttime": start,
                "endtime": start + pd.Timedelta(hours=4),
                "itemid": 225441,
                "statusdescription": "FinishedRunning",
            },
            {
                "subject_id": 1,
                "hadm_id": 11,
                "stay_id": 101,
                "starttime": start - pd.Timedelta(hours=1),
                "endtime": start,
                "itemid": 225802,
                "statusdescription": "FinishedRunning",
            },
        ]
    )
    path = tmp_path / "procedureevents.csv.gz"
    procedure.to_csv(path, index=False, compression="gzip")
    sessions, audit = extract_procedure_sessions(path)
    assert bool(sessions.loc[0, "crrt_active_at_start"])
    assert audit["hd_sessions_with_crrt_active_at_start"] == 1
