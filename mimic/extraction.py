"""Raw MIMIC-IV v2.2 HD-session and blood-pressure extraction.

Restricted identifiers exist only in memory during a credentialed run.  The
public CLI serializes aggregate counts and metrics only.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .contracts import (
    ABP_DBP_ITEMID,
    ABP_SBP_ITEMID,
    CRRT_ITEMIDS,
    HD_ITEMID,
    HEART_RATE_ITEMID,
    NIBP_DBP_ITEMID,
    NIBP_SBP_ITEMID,
    VITAL_ITEMIDS,
)


PRE_WINDOW_HOURS = 2
OUTCOME_HOURS = 6


def endpoint_window_end(
    start: pd.Timestamp, documented_end: pd.Timestamp
) -> pd.Timestamp:
    """Return ``min(documented end, T0 + 6 h)``."""

    return min(documented_end, start + pd.Timedelta(hours=OUTCOME_HOURS))


def is_in_outcome_window(
    charttime: pd.Timestamp,
    start: pd.Timestamp,
    documented_end: pd.Timestamp,
) -> bool:
    """Implement ``(T0, min(documented end, T0 + 6 h)]`` exactly."""

    return bool(start < charttime <= endpoint_window_end(start, documented_end))


def _as_datetime(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def extract_procedure_sessions(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load intermittent HD sessions and raw CRRT-at-start status."""

    usecols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "starttime",
        "endtime",
        "itemid",
        "statusdescription",
    ]
    procedures = pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        low_memory=False,
    )
    procedures["itemid"] = pd.to_numeric(procedures["itemid"], errors="coerce")
    procedures = procedures.loc[
        procedures["itemid"].isin({HD_ITEMID, *CRRT_ITEMIDS})
    ].copy()
    _as_datetime(procedures, ["starttime", "endtime"])
    for column in ("subject_id", "hadm_id", "stay_id"):
        procedures[column] = pd.to_numeric(procedures[column], errors="coerce")

    required = ["subject_id", "hadm_id", "stay_id", "starttime", "endtime"]
    valid = procedures.dropna(subset=required).copy()
    hd = valid.loc[valid["itemid"].eq(HD_ITEMID)].copy().reset_index(drop=True)
    crrt = valid.loc[valid["itemid"].isin(CRRT_ITEMIDS)].copy().reset_index(drop=True)
    valid_crrt = crrt["endtime"].gt(crrt["starttime"])
    invalid_crrt_count = int((~valid_crrt).sum())
    crrt = crrt.loc[valid_crrt].copy()

    for frame in (hd, crrt):
        for column in ("subject_id", "hadm_id", "stay_id"):
            frame[column] = frame[column].astype("int64")
    hd["session_index"] = np.arange(len(hd), dtype=np.int64)
    hd["duration_hours"] = (
        (hd["endtime"] - hd["starttime"]).dt.total_seconds() / 3600.0
    )
    hd["status"] = hd["statusdescription"].astype("string")
    hd = hd.loc[
        :,
        [
            "session_index",
            "subject_id",
            "hadm_id",
            "stay_id",
            "starttime",
            "endtime",
            "duration_hours",
            "status",
        ],
    ].copy()

    crrt_by_stay: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for row in crrt.itertuples(index=False):
        crrt_by_stay[int(row.stay_id)].append((row.starttime, row.endtime))
    active = np.zeros(len(hd), dtype=bool)
    for position, row in enumerate(hd.itertuples(index=False)):
        for crrt_start, crrt_end in crrt_by_stay.get(int(row.stay_id), []):
            # Historical split compatibility rule: a CRRT interval ending
            # exactly at HD T0 was treated as active.  This closed-at-end rule
            # is retained only to reconstruct the pre-existing patient split.
            if crrt_start <= row.starttime <= crrt_end:
                active[position] = True
                break
    hd["crrt_active_at_start"] = active

    key = ["subject_id", "stay_id", "starttime"]
    group_sizes = hd.groupby(key, dropna=False).size()
    canonical_indices = set(
        hd.sort_values(key + ["endtime", "session_index"], kind="stable")
        .drop_duplicates(key, keep="last")["session_index"]
        .astype(int)
    )
    hd["is_raw_canonical_session"] = hd["session_index"].isin(canonical_indices)
    audit = {
        "logical_source": "mimiciv_icu.procedureevents",
        "hd_itemid": HD_ITEMID,
        "crrt_itemids": sorted(CRRT_ITEMIDS),
        "raw_hd_sessions": int(len(hd)),
        "canonical_hd_sessions": int(len(canonical_indices)),
        "duplicate_start_groups": int(group_sizes.gt(1).sum()),
        "duplicate_rows_removed_by_canonicalization": int(len(hd) - len(canonical_indices)),
        "valid_positive_duration_crrt_rows": int(len(crrt)),
        "invalid_crrt_intervals_excluded": invalid_crrt_count,
        "hd_sessions_with_crrt_active_at_start": int(active.sum()),
        "crrt_at_start_rule": "crrt_start <= T0 <= crrt_end (historical split compatibility)",
    }
    return hd, audit


def _empty_choice(size: int) -> dict[str, np.ndarray]:
    return {
        "value": np.full(size, np.nan, dtype=float),
        "charttime": np.full(size, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "priority": np.full(size, -1, dtype=np.int16),
    }


def _event_type(itemid: int) -> tuple[str, int] | None:
    if itemid == NIBP_SBP_ITEMID:
        return "sbp", 2
    if itemid == ABP_SBP_ITEMID:
        return "sbp", 1
    if itemid == NIBP_DBP_ITEMID:
        return "dbp", 2
    if itemid == ABP_DBP_ITEMID:
        return "dbp", 1
    if itemid == HEART_RATE_ITEMID:
        return "hr", 1
    return None


def _valid_pre_value(field: str, value: float) -> bool:
    if field == "sbp":
        return 20.0 <= value <= 350.0
    if field == "dbp":
        return 5.0 <= value <= 250.0
    return 10.0 <= value <= 350.0


def _update_choice(
    choice: dict[str, np.ndarray],
    position: int,
    value: float,
    charttime: np.datetime64,
    priority: int,
) -> None:
    prior = choice["charttime"][position]
    if (
        np.isnat(prior)
        or charttime > prior
        or (charttime == prior and priority > choice["priority"][position])
    ):
        choice["value"][position] = value
        choice["charttime"][position] = charttime
        choice["priority"][position] = priority


def scan_vitals(
    sessions: pd.DataFrame,
    path: str | Path,
    *,
    chunksize: int = 1_000_000,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach pre-index vitals and both six-hour endpoint constructions.

    ``primary_*`` first admits individual SBP values in 20--350 mmHg and applies
    the 50--250 rule to the selected session nadir later.  ``sensitivity_*``
    applies 50--250 mmHg to each measurement before selecting the nadir.
    """

    frame = sessions.reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("No HD sessions supplied")
    n = len(frame)
    starts = pd.to_datetime(frame["starttime"], errors="raise").to_numpy(
        dtype="datetime64[ns]"
    )
    documented_ends = pd.to_datetime(frame["endtime"], errors="raise").to_numpy(
        dtype="datetime64[ns]"
    )
    ends = np.minimum(documented_ends, starts + np.timedelta64(OUTCOME_HOURS, "h"))
    pre_starts = starts - np.timedelta64(PRE_WINDOW_HOURS, "h")
    sessions_by_stay: dict[int, list[int]] = defaultdict(list)
    for position, stay_id in enumerate(frame["stay_id"].to_numpy(dtype=np.int64)):
        sessions_by_stay[int(stay_id)].append(position)
    tracked_stays = set(sessions_by_stay)

    choices = {field: _empty_choice(n) for field in ("sbp", "dbp", "hr")}
    primary_nadir = np.full(n, np.nan, dtype=float)
    primary_count = np.zeros(n, dtype=np.int64)
    sensitivity_nadir = np.full(n, np.nan, dtype=float)
    sensitivity_count = np.zeros(n, dtype=np.int64)
    audit = {
        "logical_source": "mimiciv_icu.chartevents",
        "itemids": {
            "nibp_sbp": NIBP_SBP_ITEMID,
            "abp_sbp": ABP_SBP_ITEMID,
            "nibp_dbp": NIBP_DBP_ITEMID,
            "abp_dbp": ABP_DBP_ITEMID,
            "heart_rate": HEART_RATE_ITEMID,
        },
        "source_rows_scanned": 0,
        "target_item_rows": 0,
        "target_rows_in_tracked_stays": 0,
        "primary_poststart_measurements_20_350": 0,
        "sensitivity_poststart_measurements_50_250": 0,
    }

    reader = pd.read_csv(
        path,
        compression="gzip",
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        audit["source_rows_scanned"] += int(len(chunk))
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        target = chunk["itemid"].isin(VITAL_ITEMIDS)
        audit["target_item_rows"] += int(target.sum())
        if not target.any():
            continue
        chunk = chunk.loc[target].copy()
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce")
        chunk = chunk.loc[chunk["stay_id"].isin(tracked_stays)].copy()
        audit["target_rows_in_tracked_stays"] += int(len(chunk))
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.loc[
            chunk["charttime"].notna()
            & chunk["valuenum"].notna()
            & np.isfinite(chunk["valuenum"])
        ]

        for row in chunk.itertuples(index=False):
            event = _event_type(int(row.itemid))
            if event is None:
                continue
            field, priority = event
            value = float(row.valuenum)
            charttime = np.datetime64(row.charttime.to_datetime64())
            for position in sessions_by_stay.get(int(row.stay_id), []):
                if (
                    _valid_pre_value(field, value)
                    and pre_starts[position] <= charttime <= starts[position]
                ):
                    _update_choice(choices[field], position, value, charttime, priority)
                if field != "sbp" or not starts[position] < charttime <= ends[position]:
                    continue
                if 20.0 <= value <= 350.0:
                    primary_count[position] += 1
                    prior = primary_nadir[position]
                    if not np.isfinite(prior) or value < prior:
                        primary_nadir[position] = value
                    audit["primary_poststart_measurements_20_350"] += 1
                if 50.0 <= value <= 250.0:
                    sensitivity_count[position] += 1
                    prior = sensitivity_nadir[position]
                    if not np.isfinite(prior) or value < prior:
                        sensitivity_nadir[position] = value
                    audit["sensitivity_poststart_measurements_50_250"] += 1
        if progress is not None and chunk_number % 25 == 0:
            progress(
                f"raw vital scan: {audit['source_rows_scanned']:,} source rows scanned"
            )

    frame["pre_sbp"] = choices["sbp"]["value"]
    frame["pre_dbp"] = choices["dbp"]["value"]
    frame["heart_rate"] = choices["hr"]["value"]
    frame["primary_nadir_sbp_6h"] = primary_nadir
    frame["primary_poststart_sbp_n_6h"] = primary_count
    frame["sensitivity_nadir_sbp_6h"] = sensitivity_nadir
    frame["sensitivity_poststart_sbp_n_6h"] = sensitivity_count
    audit["sessions_with_pre_sbp"] = int(np.isfinite(frame["pre_sbp"]).sum())
    audit["sessions_with_pre_dbp"] = int(np.isfinite(frame["pre_dbp"]).sum())
    audit["sessions_with_heart_rate"] = int(np.isfinite(frame["heart_rate"]).sum())
    audit["sessions_with_primary_endpoint"] = int(np.isfinite(primary_nadir).sum())
    audit["sessions_with_sensitivity_endpoint"] = int(
        np.isfinite(sensitivity_nadir).sum()
    )
    return frame, audit
