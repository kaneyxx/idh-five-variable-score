"""Guards on what this repository is allowed to contain.

These tests encode the two release constraints for the package:

1. no code, constant, table, or wording derived from the private development
   or external-validation cohorts; and
2. one seven-day history definition only -- Nadir90 -- with the single
   documented exception of the historical split stratification variable, which
   exists solely to reproduce the pre-existing patient-disjoint fold
   assignment.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scan_sensitive_content import scan_repository  # noqa: E402


SKIP_DIRECTORIES = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
SKIP_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SKIP_DIRECTORY_SUFFIXES = (".egg-info", ".dist-info")
TEXT_SUFFIXES = {
    "", ".cff", ".cfg", ".csv", ".ini", ".json", ".md", ".py",
    ".sql", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}

# The split stratification variable is a pre-SBP-dependent rule. It may only be
# described in the files that implement, document, or test the split.
STRATIFICATION_ALLOWLIST = {
    "mimic/cohort.py",
    "mimic/provenance.json",
    "mimic/README.md",
    "mimic/sql/mimic_v2_2_endpoint_provenance.sql",
    "tests/test_mimic_cohort.py",
}


SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()


def _repository_text_files() -> list[tuple[str, str]]:
    """Every scannable text file except this guard module.

    This file necessarily spells out the patterns it forbids, so scanning it
    would always fail.  Nothing else is exempt.
    """

    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.name in SKIP_FILENAMES:
            continue
        if any(part.endswith(SKIP_DIRECTORY_SUFFIXES) for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative == SELF:
            continue
        files.append((relative, path.read_text(encoding="utf-8-sig")))
    return files


def test_minimal_public_tree_passes_sensitive_content_scan() -> None:
    report = scan_repository(ROOT)
    assert report["status"] == "passed"
    assert report["findings"] == []
    assert report["inventoried_aggregate_tables"] == 2


def test_private_cohort_machinery_is_absent() -> None:
    for relative in (
        "translation",
        "release",
        "MANUSCRIPT_RELEASE_UPDATES.md",
        "reference/site_probability_lookup_0_48.csv",
        "reference/native_shape_function_data.csv",
        "reference/distilled_shape_bins.csv",
        "reference/point_scale_search.csv",
        "reference/locked_translation_specification.json",
        "reference/global_fidelity_sufficient_statistics.json",
        "scripts/verify_translation.py",
    ):
        assert not (ROOT / relative).exists(), relative


def test_no_file_names_a_private_cohort_or_a_site_specific_constant() -> None:
    """No private-cohort label, offset, or per-site probability may appear."""

    banned = re.compile(
        r"(?:\bTN\b|\bD6\b|\bCY\b"
        r"|derivation[_ -]site|source[_ -]cohort"
        r"|site_probability|SITE_INTERCEPT_OFFSETS"
        r"|D6_INTERCEPT_OFFSET|CY_INTERCEPT_OFFSET"
        r"|probability-site)"
    )
    offenders = []
    for relative, text in _repository_text_files():
        for match in banned.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line}: {match.group(0)}")
    assert offenders == [], offenders


def test_history_and_outcome_are_nadir90_only() -> None:
    """The pre-SBP-dependent rule may only appear where the split needs it."""

    named = re.compile(r"(?i)nadir\s*90\s*/\s*100|nadir90_100")
    switch = re.compile(r"\b160(?:\.0)?\b")
    offenders = []
    for relative, text in _repository_text_files():
        if named.search(text):
            offenders.append(f"{relative}: names the two-threshold rule")
        if relative in STRATIFICATION_ALLOWLIST:
            continue
        if switch.search(text) and re.search(r"(?i)pre[_ -]?sbp", text):
            offenders.append(f"{relative}: pre-SBP-dependent threshold outside the split")
    assert offenders == [], offenders


def test_specification_declares_a_single_threshold_history_event() -> None:
    import json

    spec = json.loads(
        (ROOT / "specification" / "score_specification.json").read_text(
            encoding="utf-8"
        )
    )
    source = spec["history_feature"]["source_outcome"]
    assert source["name"] == "Nadir90"
    assert source["threshold_mm_hg"] == 90.0
    assert source["comparison_operator"] == "<"
    assert "pre_dialysis_sbp_switch_mm_hg" not in source
    assert "higher_pre_dialysis_sbp_threshold_mm_hg" not in source
