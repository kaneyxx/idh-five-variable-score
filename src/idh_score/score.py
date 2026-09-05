"""Reference implementation of the five-variable IDH point rule.

This module implements the fixed 0--48 point rule and its probability
mapping. It does not reproduce model fitting, variable
selection, or any analysis of private patient data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Final


MINIMUM_SCORE: Final = 0
MAXIMUM_SCORE: Final = 48
SCORE_INTERCEPT: Final = -4.321773159969571
SCORE_COEFFICIENT: Final = 0.17227817775653662
SPECIFICATION_VERSION: Final = "2.0.0"

# Public API units are used here. UF/BW is entered in percentage points, so
# 3.5 means 3.5%, rather than the model-scale value 0.035.
VALID_RANGES: Final[dict[str, tuple[float, float]]] = {
    "Pre_HD_SBP": (50.0, 250.0),
    "UF_BW_Perc": (0.0, 25.0),
    "Start_DBP": (20.0, 150.0),
    "Heart_Rate": (20.0, 250.0),
}
MISSING_POINTS: Final[dict[str, int]] = {
    "Pre_HD_SBP": 1,
    "IDH_7D": 2,
    "UF_BW_Perc": 4,
    "Start_DBP": 3,
    "Heart_Rate": 2,
}


def _specification_metadata() -> dict[str, object]:
    """Return compact rule-scope and input-validity metadata for each result."""

    return {
        "schema_version": SPECIFICATION_VERSION,
        "valid_ranges_public_units": {
            feature: [lower, upper]
            for feature, (lower, upper) in VALID_RANGES.items()
        },
        "range_semantics": "closed intervals; both endpoints are valid",
        "invalid_numeric_policy": (
            "nonfinite or out-of-range numeric inputs use the predictor-specific "
            "missing branch"
        ),
        "outcome_rule": (
            "current-session post-start nadir SBP below 90 mm Hg in "
            "(T0, recorded treatment end]"
        ),
        "history_rule": (
            "prior post-start Nadir90 events in [index date - 7 days, index date); "
            "same-day and future sessions excluded; cold start is observed 0"
        ),
        "reported_rule_scope": (
            "code-only implementation of the fixed Nadir90 "
            "five-variable point rule and its probability mapping; model "
            "fitting, variable selection, and private-data analyses are "
            "out of scope"
        ),
    }


@dataclass(frozen=True)
class IDHScoreResult:
    """Calculated total, predicted probability, and component points."""

    total_points: int
    predicted_probability: float
    component_points: dict[str, int]
    specification: dict[str, object]


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return not math.isfinite(float(value))
    except OverflowError:
        # An arbitrarily large integer is finite. Bounded predictors handle it
        # as out of range before conversion; an integer history remains valid.
        return False


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or missing.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or missing.")
    return result


def _bounded_number_or_missing(
    name: str, value: object, valid_range: tuple[float, float]
) -> float | None:
    """Return a valid finite number, or None for the frozen missing branch."""

    if _missing(value):
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or missing.")
    lower, upper = valid_range
    if value < lower or value > upper:
        return None
    return _finite_number(name, value)


def _sbp_points(value: object) -> int:
    x = _bounded_number_or_missing(
        "sbp_mm_hg", value, VALID_RANGES["Pre_HD_SBP"]
    )
    if x is None:
        return MISSING_POINTS["Pre_HD_SBP"]
    if x < 100:
        return 11
    if x < 110:
        return 5
    if x < 120:
        return 3
    if x < 130:
        return 1
    return 0


def _idh_history_points(value: object) -> int:
    if _missing(value):
        return MISSING_POINTS["IDH_7D"]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("idh_events_prior_7d must be a real number or missing.")
    if isinstance(value, Integral):
        x = int(value)
    else:
        numeric = _finite_number("idh_events_prior_7d", value)
        if not numeric.is_integer():
            return MISSING_POINTS["IDH_7D"]
        x = int(numeric)
    if x < 0:
        return MISSING_POINTS["IDH_7D"]
    if x == 0:
        return 0
    if x == 1:
        return 8
    return 14


def _uf_bw_points(value: object) -> int:
    x = _bounded_number_or_missing(
        "uf_bw_percent", value, VALID_RANGES["UF_BW_Perc"]
    )
    if x is None:
        return MISSING_POINTS["UF_BW_Perc"]
    if x < 1:
        return 1
    if x < 3:
        return 0
    if x < 4:
        return 2
    if x < 5:
        return 4
    return 5


def _dbp_points(value: object) -> int:
    x = _bounded_number_or_missing(
        "dbp_mm_hg", value, VALID_RANGES["Start_DBP"]
    )
    if x is None:
        return MISSING_POINTS["Start_DBP"]
    if x < 45:
        return 12
    if x < 55:
        return 7
    if x < 65:
        return 4
    if x < 75:
        return 2
    return 0


def _heart_rate_points(value: object) -> int:
    x = _bounded_number_or_missing(
        "heart_rate_bpm", value, VALID_RANGES["Heart_Rate"]
    )
    if x is None:
        return MISSING_POINTS["Heart_Rate"]
    if x < 70:
        return 0
    if x < 80:
        return 1
    if x < 90:
        return 3
    if x < 100:
        return 4
    return 6


def _validate_total_points(total_points: object) -> int:
    if isinstance(total_points, bool) or not isinstance(total_points, Integral):
        raise TypeError("total_points must be an integer.")
    total = int(total_points)
    if not MINIMUM_SCORE <= total <= MAXIMUM_SCORE:
        raise ValueError("total_points must be between 0 and 48 inclusive.")
    return total


def _validate_offset(intercept_offset: object) -> float:
    if isinstance(intercept_offset, bool) or not isinstance(intercept_offset, Real):
        raise TypeError("intercept_offset must be a finite real number.")
    offset = float(intercept_offset)
    if not math.isfinite(offset):
        raise ValueError("intercept_offset must be finite.")
    return offset


def _expit_scalar(linear_predictor: float) -> float:
    """Numerically stable scalar logistic transform."""

    if linear_predictor >= 0.0:
        return 1.0 / (1.0 + math.exp(-linear_predictor))
    exponential = math.exp(linear_predictor)
    return exponential / (1.0 + exponential)


def probability_with_intercept_offset(
    total_points: int, intercept_offset: float
) -> float:
    """Return risk after adding a fixed offset to the score intercept.

    The score coefficient and all point rules remain unchanged.  An offset of
    zero is exactly ``source_probability(total_points)``.
    """

    total = _validate_total_points(total_points)
    offset = _validate_offset(intercept_offset)
    return _expit_scalar(SCORE_INTERCEPT + offset + SCORE_COEFFICIENT * total)


def source_probability(total_points: int) -> float:
    """Return the probability for a total from 0 to 48."""

    return probability_with_intercept_offset(total_points, 0.0)


def probability_lookup() -> tuple[dict[str, float | int], ...]:
    """Return the complete 0--48 probability lookup."""

    return tuple(
        {
            "total_score": score,
            "predicted_risk": source_probability(score),
        }
        for score in range(MINIMUM_SCORE, MAXIMUM_SCORE + 1)
    )


def calculate_idh_score(
    *,
    sbp_mm_hg: object,
    idh_events_prior_7d: object,
    uf_bw_percent: object,
    dbp_mm_hg: object,
    heart_rate_bpm: object,
) -> IDHScoreResult:
    """Calculate frozen component points, total points, and probability."""

    components = {
        "Pre_HD_SBP": _sbp_points(sbp_mm_hg),
        "IDH_7D": _idh_history_points(idh_events_prior_7d),
        "UF_BW_Perc": _uf_bw_points(uf_bw_percent),
        "Start_DBP": _dbp_points(dbp_mm_hg),
        "Heart_Rate": _heart_rate_points(heart_rate_bpm),
    }
    total = sum(components.values())
    return IDHScoreResult(
        total_points=total,
        predicted_probability=source_probability(total),
        component_points=components,
        specification=_specification_metadata(),
    )
