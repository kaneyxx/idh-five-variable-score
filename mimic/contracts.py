"""Public contracts for the credentialed MIMIC-IV workflow.

Only logical MIMIC table names and public metadata appear here. No local path,
row-level identifier, or restricted record is part of the release contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


MIMIC_VERSION = "2.2"

HD_ITEMID = 225441
CRRT_ITEMIDS = frozenset({225802, 225803, 225809, 225955})
NIBP_SBP_ITEMID = 220179
ABP_SBP_ITEMID = 220050
NIBP_DBP_ITEMID = 220180
ABP_DBP_ITEMID = 220051
HEART_RATE_ITEMID = 220045
VITAL_ITEMIDS = frozenset(
    {
        NIBP_SBP_ITEMID,
        ABP_SBP_ITEMID,
        NIBP_DBP_ITEMID,
        ABP_DBP_ITEMID,
        HEART_RATE_ITEMID,
    }
)

EXPECTED_SCORE_SPECIFICATION_VERSION = "2.0.0"
EXPECTED_SCORE_RANGE = (0, 48)
EXPECTED_SCORE_INTERCEPT = -4.321773159969571
EXPECTED_SCORE_COEFFICIENT = 0.17227817775653662
EXPECTED_MISSING_POINTS = {
    "Pre_HD_SBP": 1,
    "IDH_7D": 2,
    "UF_BW_Perc": 4,
    "Start_DBP": 3,
    "Heart_Rate": 2,
}

EXPECTED_RAW_SHA256 = {
    "procedureevents.csv.gz": (
        "FB01265DEDC45A0C66DD5E1B44B84CDF096878FB8543E3E8CCE8DADFB977EF57"
    ),
    "chartevents.csv.gz": (
        "451E55859336059A83135B0DCDD3631B413B26DD5553ADD220B0F0944ADA6A25"
    ),
}

OUTCOME_WINDOW = {
    "notation": "(T0, min(documented HD end, T0 + 6 h)]",
    "start_inclusive": False,
    "end_inclusive": True,
    "exactly_six_hours_included": True,
}


@dataclass(frozen=True)
class ScoreContract:
    """Validated metadata from the repository's current public score spec."""

    path: Path
    specification_sha256: str
    specification_version: str
    minimum: int
    maximum: int
    missing_points: dict[str, int]
    alpha: float
    beta: float


def default_score_spec_path() -> Path:
    """Return the expected repository-root public score specification path."""

    return (
        Path(__file__).resolve().parents[1]
        / "specification"
        / "score_specification.json"
    )


def _feature_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        features = document["score"]["features"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Score specification lacks score.features") from exc
    if not isinstance(features, list):
        raise ValueError("score.features must be a list")
    mapped = {str(item.get("id")): item for item in features if isinstance(item, dict)}
    if set(EXPECTED_MISSING_POINTS).difference(mapped):
        raise ValueError("Score specification does not contain all five frozen inputs")
    return mapped


def load_score_contract(path: str | Path | None = None) -> ScoreContract:
    """Read and validate the current 0--48 public score specification.

    ``specification_sha256`` identifies the public JSON file that was actually
    read. It is not a reference to an unavailable internal source artifact.
    """

    score_path = Path(path) if path is not None else default_score_spec_path()
    if not score_path.is_file():
        raise FileNotFoundError(f"Public score specification not found: {score_path}")
    raw_document = score_path.read_bytes()
    specification_sha = hashlib.sha256(raw_document).hexdigest().upper()
    document = json.loads(raw_document.decode("utf-8"))
    try:
        specification_version = str(document["schema_version"])
        minimum = int(document["score"]["minimum"])
        maximum = int(document["score"]["maximum"])
        probability = document["probability_maps"]
        alpha = float(probability["score_intercept"])
        beta = float(probability["score_coefficient"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed public score specification") from exc

    if specification_version != EXPECTED_SCORE_SPECIFICATION_VERSION:
        raise ValueError(
            "Score specification version must be "
            f"{EXPECTED_SCORE_SPECIFICATION_VERSION}, observed {specification_version}"
        )
    if (minimum, maximum) != EXPECTED_SCORE_RANGE:
        raise ValueError(
            f"Score range must be {EXPECTED_SCORE_RANGE[0]}--{EXPECTED_SCORE_RANGE[1]}"
        )

    features = _feature_map(document)
    missing_points = {
        feature: int(features[feature]["missing_points"])
        for feature in EXPECTED_MISSING_POINTS
    }
    if missing_points != EXPECTED_MISSING_POINTS:
        raise ValueError(
            "Frozen missing branches differ from the current 0--48 score: "
            f"{missing_points}"
        )
    if alpha != EXPECTED_SCORE_INTERCEPT or beta != EXPECTED_SCORE_COEFFICIENT:
        raise ValueError(
            "Probability map differs from the score equation in this repository"
        )
    return ScoreContract(
        path=score_path.resolve(),
        specification_sha256=specification_sha,
        specification_version=specification_version,
        minimum=minimum,
        maximum=maximum,
        missing_points=missing_points,
        alpha=alpha,
        beta=beta,
    )


def resolve_mimic_paths(root: str | Path) -> dict[str, Path]:
    """Resolve a credentialed MIMIC-IV root without exporting the local path."""

    base = Path(root).expanduser().resolve()
    candidates = (base / MIMIC_VERSION, base)
    for version_root in candidates:
        paths = {
            "procedureevents.csv.gz": version_root / "icu" / "procedureevents.csv.gz",
            "chartevents.csv.gz": version_root / "icu" / "chartevents.csv.gz",
        }
        if all(path.is_file() for path in paths.values()):
            return paths
    logical = ", ".join(f"icu/{name}" for name in EXPECTED_RAW_SHA256)
    raise FileNotFoundError(
        f"Could not find MIMIC-IV {MIMIC_VERSION} inputs below the supplied root: {logical}"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_raw_hashes(paths: dict[str, Path]) -> dict[str, Any]:
    """Verify official compressed MIMIC-IV v2.2 inputs by public hashes."""

    files: dict[str, Any] = {}
    for logical_name, expected in EXPECTED_RAW_SHA256.items():
        observed = sha256_file(paths[logical_name])
        files[logical_name] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": observed == expected,
        }
    return {
        "status": "passed" if all(item["matches"] for item in files.values()) else "failed",
        "files": files,
    }
