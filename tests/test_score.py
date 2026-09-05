from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from idh_score import (
    MAXIMUM_SCORE,
    SPECIFICATION_VERSION,
    SCORE_COEFFICIENT,
    SCORE_INTERCEPT,
    calculate_idh_score,
    probability_lookup,
    probability_with_intercept_offset,
    source_probability,
)


ROOT = Path(__file__).resolve().parents[1]


def _baseline(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "sbp_mm_hg": 140,
        "idh_events_prior_7d": 0,
        "uf_bw_percent": 2,
        "dbp_mm_hg": 75,
        "heart_rate_bpm": 60,
    }
    values.update(updates)
    return values


def test_zero_maximum_missing_and_cold_start_profiles() -> None:
    zero = calculate_idh_score(**_baseline())
    assert zero.total_points == 0
    assert zero.predicted_probability == pytest.approx(source_probability(0))
    assert zero.specification["schema_version"] == "2.0.0"
    assert zero.specification["reported_rule_scope"] == (
        "code-only implementation of the fixed Nadir90 "
        "five-variable point rule and its probability mapping; model "
        "fitting, variable selection, and private-data analyses are "
        "out of scope"
    )

    maximum = calculate_idh_score(
        sbp_mm_hg=99,
        idh_events_prior_7d=2,
        uf_bw_percent=5,
        dbp_mm_hg=44,
        heart_rate_bpm=100,
    )
    assert maximum.total_points == 48 == MAXIMUM_SCORE

    missing = calculate_idh_score(
        sbp_mm_hg=None,
        idh_events_prior_7d=None,
        uf_bw_percent=None,
        dbp_mm_hg=None,
        heart_rate_bpm=None,
    )
    assert missing.total_points == 12
    assert missing.component_points == {
        "Pre_HD_SBP": 1,
        "IDH_7D": 2,
        "UF_BW_Perc": 4,
        "Start_DBP": 3,
        "Heart_Rate": 2,
    }
    assert zero.component_points["IDH_7D"] == 0


@pytest.mark.parametrize(
    ("component", "argument", "value", "points"),
    [
        ("Pre_HD_SBP", "sbp_mm_hg", 99, 11),
        ("Pre_HD_SBP", "sbp_mm_hg", 100, 5),
        ("Pre_HD_SBP", "sbp_mm_hg", 110, 3),
        ("Pre_HD_SBP", "sbp_mm_hg", 120, 1),
        ("Pre_HD_SBP", "sbp_mm_hg", 130, 0),
        ("Pre_HD_SBP", "sbp_mm_hg", None, 1),
        ("IDH_7D", "idh_events_prior_7d", 0, 0),
        ("IDH_7D", "idh_events_prior_7d", 1, 8),
        ("IDH_7D", "idh_events_prior_7d", 2, 14),
        ("IDH_7D", "idh_events_prior_7d", None, 2),
        ("UF_BW_Perc", "uf_bw_percent", 0.9, 1),
        ("UF_BW_Perc", "uf_bw_percent", 1, 0),
        ("UF_BW_Perc", "uf_bw_percent", 3, 2),
        ("UF_BW_Perc", "uf_bw_percent", 4, 4),
        ("UF_BW_Perc", "uf_bw_percent", 5, 5),
        ("UF_BW_Perc", "uf_bw_percent", None, 4),
        ("Start_DBP", "dbp_mm_hg", 44, 12),
        ("Start_DBP", "dbp_mm_hg", 45, 7),
        ("Start_DBP", "dbp_mm_hg", 55, 4),
        ("Start_DBP", "dbp_mm_hg", 65, 2),
        ("Start_DBP", "dbp_mm_hg", 75, 0),
        ("Start_DBP", "dbp_mm_hg", None, 3),
        ("Heart_Rate", "heart_rate_bpm", 69, 0),
        ("Heart_Rate", "heart_rate_bpm", 70, 1),
        ("Heart_Rate", "heart_rate_bpm", 80, 3),
        ("Heart_Rate", "heart_rate_bpm", 90, 4),
        ("Heart_Rate", "heart_rate_bpm", 100, 6),
        ("Heart_Rate", "heart_rate_bpm", None, 2),
    ],
)
def test_every_component_bin_and_missing_branch(
    component: str, argument: str, value: object, points: int
) -> None:
    result = calculate_idh_score(**_baseline(**{argument: value}))
    assert result.component_points[component] == points


@pytest.mark.parametrize(
    ("argument", "value", "component", "missing_points"),
    [
        ("sbp_mm_hg", 49.99, "Pre_HD_SBP", 1),
        ("sbp_mm_hg", 250.01, "Pre_HD_SBP", 1),
        ("idh_events_prior_7d", -1, "IDH_7D", 2),
        ("idh_events_prior_7d", 1.5, "IDH_7D", 2),
        ("uf_bw_percent", -0.01, "UF_BW_Perc", 4),
        ("uf_bw_percent", 25.01, "UF_BW_Perc", 4),
        ("dbp_mm_hg", 19.99, "Start_DBP", 3),
        ("dbp_mm_hg", 150.01, "Start_DBP", 3),
        ("heart_rate_bpm", 19.99, "Heart_Rate", 2),
        ("heart_rate_bpm", 250.01, "Heart_Rate", 2),
        ("heart_rate_bpm", float("nan"), "Heart_Rate", 2),
    ],
)
def test_invalid_numeric_values_use_missing_branches(
    argument: str, value: object, component: str, missing_points: int
) -> None:
    result = calculate_idh_score(**_baseline(**{argument: value}))
    assert result.component_points[component] == missing_points


def test_paper_contract_examples() -> None:
    clinical = calculate_idh_score(
        sbp_mm_hg=105,
        idh_events_prior_7d=2,
        uf_bw_percent=4.2,
        dbp_mm_hg=60,
        heart_rate_bpm=85,
    )
    assert clinical.component_points == {
        "Pre_HD_SBP": 5,
        "IDH_7D": 14,
        "UF_BW_Perc": 4,
        "Start_DBP": 4,
        "Heart_Rate": 3,
    }
    assert clinical.total_points == 30

    assert calculate_idh_score(**_baseline(sbp_mm_hg=140, dbp_mm_hg=40)).total_points == 12
    assert calculate_idh_score(**_baseline(sbp_mm_hg=105, dbp_mm_hg=75)).total_points == 5


def test_source_and_offset_probability_api() -> None:
    assert source_probability(0) == pytest.approx(0.013102370334544101, abs=1e-15)
    assert source_probability(15) == pytest.approx(0.1496179729649258, abs=1e-15)
    assert source_probability(48) == pytest.approx(0.9810641216365715, abs=1e-15)
    for score in range(49):
        assert probability_with_intercept_offset(score, 0.0) == source_probability(score)
        assert probability_with_intercept_offset(
            score, -0.5
        ) < source_probability(score)
    with pytest.raises(ValueError):
        source_probability(49)
    with pytest.raises(ValueError):
        probability_with_intercept_offset(10, float("inf"))


def test_complete_lookup_matches_the_specified_probability_map() -> None:
    lookup = probability_lookup()
    assert len(lookup) == 49
    assert [row["total_score"] for row in lookup] == list(range(49))
    for row in lookup:
        score = int(row["total_score"])
        assert set(row) == {"total_score", "predicted_risk"}
        assert row["predicted_risk"] == source_probability(score)
    # Published lookup percentages are rounded to one decimal place.
    assert 100 * float(lookup[0]["predicted_risk"]) == pytest.approx(1.3102370334544101)
    assert round(100 * float(lookup[15]["predicted_risk"]), 1) == 15.0
    assert round(100 * float(lookup[48]["predicted_risk"]), 1) == 98.1


# The bundled table stores full double precision, but the value comes from
# math.exp, and a C library is not required to round exp correctly. glibc and
# the macOS libm can therefore return adjacent doubles for the same input, so
# the stored table is checked to within a couple of units in the last place
# rather than bit for bit. One ULP here is about 3e-17 in absolute terms and
# 2e-16 relative, far below any difference that could matter clinically or
# change a reported metric; test_a_wrong_equation_is_still_caught below shows
# the bound is still tight enough to reject a wrong table.
LOOKUP_TOLERANCE_ULP = 2


def _within_ulp(observed: float, expected: float, tolerance: int) -> bool:
    return abs(observed - expected) <= tolerance * math.ulp(expected)


def test_bundled_lookup_csv_matches_public_api_to_within_two_ulp() -> None:
    with (ROOT / "reference" / "risk_lookup_0_48.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    lookup = probability_lookup()
    assert len(rows) == len(lookup) == 49
    for row, expected in zip(rows, lookup, strict=True):
        assert int(row["total_score"]) == expected["total_score"]
        probability = float(expected["predicted_risk"])
        assert _within_ulp(
            float(row["predicted_risk"]), probability, LOOKUP_TOLERANCE_ULP
        ), row["total_score"]
        assert _within_ulp(
            float(row["predicted_risk_percent"]),
            100 * probability,
            LOOKUP_TOLERANCE_ULP,
        ), row["total_score"]


def _rows_rejected(alpha: float, beta: float) -> int:
    """How many of the 49 rows a given equation fails the bundled table on."""

    rejected = 0
    for entry in probability_lookup():
        score = int(entry["total_score"])
        expected = float(entry["predicted_risk"])
        wrong = 1.0 / (1.0 + math.exp(-(alpha + beta * score)))
        if not _within_ulp(wrong, expected, LOOKUP_TOLERANCE_ULP):
            rejected += 1
    return rejected


def test_a_wrong_equation_is_still_caught() -> None:
    """The relaxed bound must still reject a genuinely different map.

    Two ULP absorbs the C library's rounding of exp, and nothing more. A
    perturbation one part in a trillion -- far smaller than any transcription
    error, truncated constant, or refitted coefficient -- is rejected on every
    row it can reach.
    """

    assert _rows_rejected(SCORE_INTERCEPT * (1 + 1e-12), SCORE_COEFFICIENT) == 49
    assert _rows_rejected(SCORE_INTERCEPT, SCORE_COEFFICIENT * (1 + 1e-12)) == 48
    # A score of 0 cancels the slope, so no slope change can ever reach that row.
    assert _rows_rejected(-4.321773, 0.172278) == 49

    # Even the smallest representable change is caught somewhere: shifting the
    # slope by a single ULP fails 11 rows, and the intercept by a single ULP
    # fails 27. The bound is not vacuous.
    assert _rows_rejected(SCORE_INTERCEPT, math.nextafter(SCORE_COEFFICIENT, math.inf)) >= 10
    assert _rows_rejected(math.nextafter(SCORE_INTERCEPT, math.inf), SCORE_COEFFICIENT) >= 25


def test_human_readable_score_rule_csv_matches_machine_specification() -> None:
    spec = json.loads(
        (ROOT / "specification" / "score_specification.json").read_text("utf-8")
    )
    expected: list[dict[str, str]] = []
    for feature in spec["score"]["features"]:
        if feature["id"] == "IDH_7D":
            labels = ["0", "1", ">=2"]
            missing_label = "missing, nonfinite, negative, or noninteger"
        else:
            valid = feature["valid_range"]

            def display(value: float) -> str:
                return f"{value:g}"

            labels = []
            for score_bin in feature["bins"]:
                lower = (
                    valid["minimum"]
                    if score_bin["lower"] is None
                    else score_bin["lower"]
                )
                if score_bin["upper"] is None:
                    labels.append(
                        f"{display(lower)} to {display(valid['maximum'])} inclusive"
                    )
                else:
                    labels.append(
                        f"{display(lower)} to <{display(score_bin['upper'])}"
                    )
            missing_label = (
                "missing, nonfinite, or outside "
                f"{display(valid['minimum'])}-{display(valid['maximum'])}"
            )
        for label, score_bin in zip(labels, feature["bins"], strict=True):
            expected.append(
                {
                    "feature": feature["id"],
                    "public_argument": feature["public_argument"],
                    "unit": feature["unit"],
                    "bin": label,
                    "points": str(score_bin["points"]),
                }
            )
        expected.append(
            {
                "feature": feature["id"],
                "public_argument": feature["public_argument"],
                "unit": feature["unit"],
                "bin": missing_label,
                "points": str(feature["missing_points"]),
            }
        )

    with (ROOT / "reference" / "score_rule.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        observed = list(csv.DictReader(handle))
    assert observed == expected


def test_machine_readable_spec_matches_implementation() -> None:
    spec = json.loads(
        (ROOT / "specification" / "score_specification.json").read_text("utf-8")
    )
    assert spec["schema_version"] == SPECIFICATION_VERSION
    assert spec["outcome"]["measurement_window"] == {
        "start": "treatment initiation",
        "end": "recorded treatment duration",
        "notation": "(T0, recorded treatment end]",
        "start_inclusive": False,
        "end_inclusive": True,
        "source_time_rule": "0 < minutes_from_start <= actual_duration_min",
    }
    assert spec["history_feature"]["source_outcome"] == {
        "name": "Nadir90",
        "definition": "prior-session post-start nadir systolic blood pressure below 90 mm Hg",
        "threshold_mm_hg": 90.0,
        "comparison_operator": "<",
    }
    assert spec["specification"]["semantic_alignment"][
        "minimum_sessions_per_patient"
    ] is None
    assert spec["score"]["maximum"] == 48
    assert spec["probability_maps"]["score_intercept"] == pytest.approx(SCORE_INTERCEPT)
    assert spec["probability_maps"]["score_coefficient"] == pytest.approx(
        0.17227817775653662
    )
    assert spec["probability_maps"]["complete_lookup"]["probability_column"] == "predicted_risk"
    assert "sites" not in spec["probability_maps"]
    assert spec["provenance"] == {
        "artifact_scope": "code-only fixed rule",
        "raw_patient_or_session_data_included": False,
        "model_training_or_refitting_included": False,
        "row_level_outputs_written": False,
    }


def test_non_numeric_python_api_inputs_raise_instead_of_silent_coercion() -> None:
    with pytest.raises(TypeError):
        calculate_idh_score(**_baseline(sbp_mm_hg="120"))
    with pytest.raises(TypeError):
        source_probability(True)
    with pytest.raises(TypeError):
        probability_with_intercept_offset(10, True)
