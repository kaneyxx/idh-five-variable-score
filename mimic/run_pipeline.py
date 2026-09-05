"""Command-line entry point for aggregate-only MIMIC-IV v2.2 point estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .pipeline import DEFAULT_BOOTSTRAP_SEED, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mimic-root",
        required=True,
        type=Path,
        help="Credentialed root containing 2.2/icu or the 2.2 directory itself.",
    )
    parser.add_argument(
        "--score-spec",
        type=Path,
        default=None,
        help=(
            "Current repository specification/score_specification.json "
            "(default: resolved automatically)."
        ),
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help=(
            "Patient-cluster bootstrap replicates for the primary cohort. "
            "0 (default) reports point estimates only; the reported confidence "
            "intervals use 1000."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Seed for the patient-cluster bootstrap.",
    )
    parser.add_argument(
        "--verify-source-hashes",
        action="store_true",
        help="Hash the two large compressed source files before analysis.",
    )
    parser.add_argument(
        "--output-json",
        default="-",
        help="Aggregate JSON destination, or '-' for stdout. Row-level output is unsupported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing aggregate JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        mimic_root=args.mimic_root,
        score_spec_path=args.score_spec,
        chunksize=args.chunksize,
        verify_source_hashes=args.verify_source_hashes,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output_json == "-":
        sys.stdout.write(rendered)
    else:
        destination = Path(args.output_json)
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite aggregate output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Write UTF-8 bytes so the aggregate uses LF newlines on every
        # platform, including Windows.
        destination.write_bytes(rendered.encode("utf-8"))
        print(f"Wrote aggregate-only result: {destination}", file=sys.stderr)
    verification = result["target_verification"]
    if verification["status"] != "passed":
        print(
            f"Reference verification FAILED: {verification['failure_count']} "
            "check(s) differ from the reference values; see "
            "target_verification in the output.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
