"""Fail when a public package contains sensitive or private artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPOSITORY_ROOT / "PUBLIC_DATA_INVENTORY.json"
SKIP_DIRECTORIES = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
# Operating-system sidecar files. They carry no repository content and are
# gitignored, so they must not fail release QA when a checkout is browsed in
# Finder or Explorer.
SKIP_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
# Directories produced by installing the package. They are gitignored build
# artefacts, not repository content.
SKIP_DIRECTORY_SUFFIXES = (".egg-info", ".dist-info")
FORBIDDEN_MODEL_SUFFIXES = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".sav",
    ".onnx",
    ".h5",
    ".keras",
    ".pt",
    ".pth",
}
TABULAR_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather"}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".cfg",
    ".csv",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
ROW_IDENTIFIER_FIELDS = {
    "patient_id",
    "session_id",
    "subject_id",
    "hadm_id",
    "stay_id",
    "mrn",
    "charttime",
    "date_of_birth",
    "dob",
}


TEXT_PATTERNS = {
    "windows_absolute_path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"),
    "unc_path": re.compile(r"\\\\[A-Za-z0-9._-]+[\\/]"),
    "private_posix_path": re.compile(
        r"(?<![:A-Za-z0-9_])/(?:home|Users|root|mnt/[a-z])(?:/|\\)"
    ),
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b"
        r"\s*[:=]\s*['\"][^'\"]+['\"]"
    ),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    "connection_string": re.compile(
        r"(?i)\b(?:server|data source)\s*=.+;\s*(?:database|initial catalog)\s*="
    ),
    "internal_artifact_path": re.compile(
        r"(?i)(?:outputs|side_projects|work)[\\/]"
    ),
    "private_qualified_table": re.compile(
        r"(?i)\b(?:dbo|prod|production|warehouse|internal)\."
        r"[A-Za-z_][A-Za-z0-9_]*\b"
    ),
}


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.name in SKIP_FILENAMES:
            continue
        if any(part.endswith(SKIP_DIRECTORY_SUFFIXES) for part in path.parts):
            continue
        yield path


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _contains_identifier_records(value: Any) -> bool:
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, dict) for item in value):
            shared = set.intersection(*(set(item) for item in value))
            if {str(key).lower() for key in shared} & ROW_IDENTIFIER_FIELDS:
                return True
        return any(_contains_identifier_records(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_identifier_records(item) for item in value.values())
    return False


def scan_repository(
    root: Path = REPOSITORY_ROOT,
    inventory_path: Path = DEFAULT_INVENTORY,
) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    approved_tables = inventory["aggregate_tables"]
    findings: list[dict[str, Any]] = []
    scanned_files = 0

    for path in _iter_files(root):
        scanned_files += 1
        relative = _relative(path, root)
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_MODEL_SUFFIXES:
            findings.append({"file": relative, "kind": "trained_model_or_pickle"})
            continue
        if suffix in TABULAR_SUFFIXES:
            if relative not in approved_tables:
                findings.append({"file": relative, "kind": "unapproved_tabular_file"})
            elif suffix not in {".csv", ".tsv"}:
                findings.append(
                    {"file": relative, "kind": "non_text_tabular_file_not_permitted"}
                )
            else:
                delimiter = "\t" if suffix == ".tsv" else ","
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle, delimiter=delimiter)
                    header = next(reader, [])
                    # Blank physical lines are not table records in the
                    # inventoried lookup CSV files.
                    row_count = sum(1 for row in reader if any(cell.strip() for cell in row))
                expected = approved_tables[relative]
                if header != expected["columns"]:
                    findings.append({"file": relative, "kind": "aggregate_schema_mismatch"})
                if row_count != int(expected["rows"]):
                    findings.append({"file": relative, "kind": "aggregate_row_count_mismatch"})
                if {field.lower() for field in header} & ROW_IDENTIFIER_FIELDS:
                    findings.append({"file": relative, "kind": "row_identifier_column"})
        if suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            findings.append({"file": relative, "kind": "unexpected_binary_content"})
            continue
        for kind, pattern in TEXT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"file": relative, "line": line, "kind": kind})
        if suffix == ".json":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                findings.append({"file": relative, "kind": "invalid_json"})
            else:
                if _contains_identifier_records(parsed):
                    findings.append({"file": relative, "kind": "row_identifier_records"})

    missing_tables = sorted(set(approved_tables) - {
        _relative(path, root) for path in _iter_files(root)
    })
    findings.extend(
        {"file": relative, "kind": "inventoried_table_missing"}
        for relative in missing_tables
    )
    return {
        "status": "passed" if not findings else "failed",
        "root": ".",
        "files_scanned": scanned_files,
        "inventoried_aggregate_tables": len(approved_tables),
        "findings": findings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = scan_repository(args.root, args.inventory)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["findings"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
