from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from mimic.contracts import (
    EXPECTED_SCORE_SPECIFICATION_VERSION,
    OUTCOME_WINDOW,
    load_score_contract,
)
from mimic.metrics import assert_aggregate_only
from mimic.pipeline import score_input_missingness


def _score_spec() -> dict[str, object]:
    missing = {
        "Pre_HD_SBP": 1,
        "IDH_7D": 2,
        "UF_BW_Perc": 4,
        "Start_DBP": 3,
        "Heart_Rate": 2,
    }
    return {
        "schema_version": "2.0.0",
        "score": {
            "minimum": 0,
            "maximum": 48,
            "features": [
                {"id": feature, "missing_points": points}
                for feature, points in missing.items()
            ],
        },
        "probability_maps": {
            "score_intercept": -4.321773159969571,
            "score_coefficient": 0.17227817775653662,
        },
    }


def test_current_score_contract_accepts_only_locked_0_to_48_spec(tmp_path) -> None:
    path = tmp_path / "score_specification.json"
    path.write_text(json.dumps(_score_spec()), encoding="utf-8")
    contract = load_score_contract(path)
    assert (contract.minimum, contract.maximum) == (0, 48)
    assert contract.missing_points["UF_BW_Perc"] == 4
    assert contract.specification_version == EXPECTED_SCORE_SPECIFICATION_VERSION
    assert contract.specification_sha256 == (
        hashlib.sha256(path.read_bytes()).hexdigest().upper()
    )

    stale = _score_spec()
    stale["score"]["maximum"] = 45
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="Score range"):
        load_score_contract(path)


def test_repository_probability_maps_schema_loads_directly() -> None:
    contract = load_score_contract()
    assert contract.specification_version == EXPECTED_SCORE_SPECIFICATION_VERSION
    assert re.fullmatch(r"[0-9A-F]{64}", contract.specification_sha256)
    assert contract.alpha == pytest.approx(-4.321773159969571, abs=0.0)
    assert contract.beta == pytest.approx(0.17227817775653662, abs=0.0)


def test_current_score_contract_rejects_wrong_version_or_probability_map(tmp_path) -> None:
    path = tmp_path / "score_specification.json"
    stale = _score_spec()
    stale["schema_version"] = "1.0.0"
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_score_contract(path)

    altered = _score_spec()
    altered["probability_maps"]["score_intercept"] = -4.0
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)probability map"):
        load_score_contract(path)


def test_current_score_contract_rejects_old_uf_missing_branch(tmp_path) -> None:
    document = _score_spec()
    for feature in document["score"]["features"]:
        if feature["id"] == "UF_BW_Perc":
            feature["missing_points"] = 3
    path = tmp_path / "score_specification.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="missing branches"):
        load_score_contract(path)


def test_aggregate_guard_rejects_identifiers() -> None:
    assert_aggregate_only({"counts": {"sessions": 817, "patients": 308}})
    with pytest.raises(ValueError, match="Restricted row-level key"):
        assert_aggregate_only({"subject_id": [1, 2]})


def test_reported_missingness_matches_frozen_score_branches() -> None:
    frame = pd.DataFrame(
        {
            "pre_sbp": [120.0, np.nan, 260.0],
            "idh_events_prior_7d": [np.nan, 0.0, 1.0],
            "pre_dbp": [70.0, np.nan, 10.0],
            "heart_rate": [80.0, 10.0, np.nan],
        }
    )
    assert score_input_missingness(frame) == {
        "pre_sbp": 2,
        "idh_history": 1,
        "uf_bw_percent_structural": 3,
        "pre_dbp": 2,
        "heart_rate": 2,
    }


def test_public_schema_and_provenance_are_current_and_path_free() -> None:
    mimic_root = Path(__file__).resolve().parents[1] / "mimic"
    schema = json.loads(
        (mimic_root / "aggregate_result.schema.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (mimic_root / "provenance.json").read_text(encoding="utf-8")
    )
    assert schema["properties"]["schema_version"]["const"] == "2.1.0"
    assert schema["properties"]["artifact_status"]["const"] == (
        "credentialed_run_verified_against_published_reference"
    )
    # The released history feature is Nadir90 only: the schema pins the run to
    # report that no pre-SBP-dependent switch was used, and that the score was
    # neither refitted nor recalibrated for MIMIC-IV.
    history = schema["properties"]["history_feature"]["properties"]
    assert history["pre_sbp_dependent_switch_used"]["const"] is False
    assert history["score_refit"]["const"] is False
    assert history["score_recalibrated"]["const"] is False
    assert provenance["score"]["public_specification_version"] == (
        EXPECTED_SCORE_SPECIFICATION_VERSION
    )
    assert provenance["outcome"]["window"] == OUTCOME_WINDOW["notation"]
    assert provenance["outcome"]["exactly_six_hours_included"] is True

    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in mimic_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".sql", ".txt"}
    )
    assert re.search(r"(?i)\b[a-z]:[\\/]", text) is None
    assert ("0" + "-" + "45") not in text
