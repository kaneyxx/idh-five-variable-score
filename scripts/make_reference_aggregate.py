"""Generate the bundled reference aggregate from a complete credentialed run.

The file this writes, ``mimic/verification/mimic_aggregate_reference.json``, is
the expected output of the MIMIC-IV analysis. It is produced only from a run
that (a) reads the complete official source files, (b) verifies their SHA-256
hashes, and (c) passes every automatic check in
``mimic/verification_targets.json``. Any of those failing aborts without
writing, so the bundled reference cannot silently come from a partial extract
or a diverging run.

    python scripts/make_reference_aggregate.py --mimic-root <CREDENTIALED_ROOT>

Add ``--bootstrap-replicates 1000`` to include the reported patient-cluster
confidence intervals. Expect a long run: the complete chartevents table is
about 313 million rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "mimic" / "verification" / "mimic_aggregate_reference.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mimic-root",
        required=True,
        type=Path,
        help="Credentialed root containing 2.2/icu or the 2.2 directory itself.",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="Patient-cluster bootstrap replicates; the reported intervals use 1000.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from mimic.pipeline import DEFAULT_BOOTSTRAP_SEED, run_pipeline

    args = parse_args()
    result = run_pipeline(
        mimic_root=args.mimic_root,
        chunksize=args.chunksize,
        verify_source_hashes=True,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )

    if result["source_verification"]["status"] != "passed":
        print(
            "Refusing to write a reference: source hash verification did not pass.",
            file=sys.stderr,
        )
        return 2
    if result["target_verification"]["status"] != "passed":
        failed = [
            check["field"]
            for check in result["target_verification"]["checks"]
            if not check["passed"]
        ]
        print(
            "Refusing to write a reference: this run disagrees with "
            f"verification_targets.json on {failed}.",
            file=sys.stderr,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    args.output.write_bytes(rendered.encode("utf-8"))
    scanned = result["cohort_flow"]["chartevents"]["source_rows_scanned"]
    print(f"Wrote {args.output} from {scanned:,} scanned chartevents rows.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
