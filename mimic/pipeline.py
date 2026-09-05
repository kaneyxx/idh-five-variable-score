"""End-to-end raw MIMIC-IV v2.2 to aggregate point-estimate pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from idh_score import VALID_RANGES

from .cohort import (
    add_observable_history,
    build_analysis_cohorts,
    build_patient_disjoint_split,
    canonicalize_sessions,
    score_current_rule,
)
from .contracts import (
    MIMIC_VERSION,
    OUTCOME_WINDOW,
    load_score_contract,
    resolve_mimic_paths,
    verify_raw_hashes,
)
from .extraction import extract_procedure_sessions, scan_vitals
from .metrics import (
    assert_aggregate_only,
    cohort_counts,
    probability_bootstrap,
    probability_metrics,
)
from .targets import verify_targets


# The seed the reported confidence intervals were produced with.
DEFAULT_BOOTSTRAP_SEED = 20_260_725


def score_input_missingness(frame: pd.DataFrame) -> dict[str, int]:
    """Count rows routed to each frozen score's missing branch."""

    def bounded_missing(column: str, feature: str) -> int:
        values = pd.to_numeric(frame[column], errors="coerce")
        lower, upper = VALID_RANGES[feature]
        return int((~values.between(lower, upper, inclusive="both")).sum())

    history = pd.to_numeric(frame["idh_events_prior_7d"], errors="coerce")
    history_missing = history.isna() | history.lt(0) | history.mod(1).ne(0)
    return {
        "pre_sbp": bounded_missing("pre_sbp", "Pre_HD_SBP"),
        "idh_history": int(history_missing.sum()),
        "uf_bw_percent_structural": int(len(frame)),
        "pre_dbp": bounded_missing("pre_dbp", "Start_DBP"),
        "heart_rate": bounded_missing("heart_rate", "Heart_Rate"),
    }


def run_pipeline(
    *,
    mimic_root: str | Path,
    score_spec_path: str | Path | None = None,
    chunksize: int = 1_000_000,
    verify_source_hashes: bool = False,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the credentialed workflow and return aggregate-only JSON data."""

    paths = resolve_mimic_paths(mimic_root)
    contract = load_score_contract(score_spec_path)
    if verify_source_hashes:
        source_verification = verify_raw_hashes(paths)
        if source_verification["status"] != "passed":
            raise RuntimeError("MIMIC-IV compressed source hash verification failed")
    else:
        source_verification = {
            "status": "not_checked",
            "note": "Use --verify-source-hashes for byte-level MIMIC-IV v2.2 verification.",
        }

    if progress is not None:
        progress("reading MIMIC-IV v2.2 procedureevents")
    sessions, procedure_audit = extract_procedure_sessions(
        paths["procedureevents.csv.gz"]
    )
    if progress is not None:
        progress("scanning credentialed chartevents; no row-level output will be written")
    sessions, vital_audit = scan_vitals(
        sessions,
        paths["chartevents.csv.gz"],
        chunksize=chunksize,
        progress=progress,
    )
    canonical, canonical_audit = canonicalize_sessions(sessions)
    split_frame, split_audit = build_patient_disjoint_split(canonical)
    split_frame = add_observable_history(split_frame)
    primary, sensitivity = build_analysis_cohorts(split_frame)
    scored_primary = score_current_rule(primary, contract)

    primary_counts = cohort_counts(scored_primary)
    sensitivity_counts = cohort_counts(sensitivity)
    primary_metrics = probability_metrics(
        scored_primary["target_nadir90"].to_numpy(),
        scored_primary["frozen_probability"].to_numpy(),
    )
    primary_missingness = score_input_missingness(scored_primary)

    primary_bootstrap: dict[str, Any] | None = None
    if bootstrap_replicates:
        if progress is not None:
            progress(
                f"patient-cluster bootstrap: {bootstrap_replicates} replicates"
            )
        primary_bootstrap = probability_bootstrap(
            scored_primary["target_nadir90"].to_numpy(),
            scored_primary["frozen_probability"].to_numpy(),
            scored_primary["subject_id"].to_numpy(),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )

    result: dict[str, Any] = {
        "artifact_status": "credentialed_run_verified_against_reference_values",
        "schema_version": "2.1.0",
        "analysis": "MIMIC-IV feature-constrained partial-rule transport stress test",
        "dataset": {"name": "MIMIC-IV", "version": MIMIC_VERSION},
        "privacy": {
            "output_level": "aggregate_only",
            "row_level_data_written": False,
            "identifiers_written": False,
        },
        "score": {
            "specification_version": contract.specification_version,
            "public_score_specification_sha256": contract.specification_sha256,
            "minimum": contract.minimum,
            "maximum": contract.maximum,
            "uf_bw_missing_points": contract.missing_points["UF_BW_Perc"],
            "idh_7d_points": {"0": 0, "1": 8, ">=2": 14, "missing": 2},
            "risk_equation": {
                "link": "logistic",
                "alpha": contract.alpha,
                "beta": contract.beta,
            },
            "probability_mapping": "score equation as specified; no MIMIC refit",
            "history_event": "prior-session post-start nadir SBP below 90 mm Hg",
        },
        "history_feature": {
            "definition": (
                "count of observable prior primary-eligible HD sessions with "
                "selected nadir SBP <90 mm Hg in the preceding seven calendar days"
            ),
            "pre_sbp_dependent_switch_used": False,
            "score_refit": False,
            "score_recalibrated": False,
        },
        "endpoint": {
            **OUTCOME_WINDOW,
            "primary": (
                "historical operational construction: admit individual SBP 20--350 "
                "mm Hg, select session nadir, then require selected nadir 50--250 mm Hg"
            ),
            "sensitivity": (
                "protocol-concordant construction: require each SBP measurement 50--250 "
                "mm Hg before selecting the session nadir"
            ),
        },
        "source_verification": source_verification,
        "uncertainty": {
            "patient_cluster_bootstrap_computed": primary_bootstrap is not None,
            "note": (
                "Patient-cluster confidence intervals are computed only when "
                "--bootstrap-replicates is given; the default run reports point "
                "estimates alone."
            ),
        },
        "cohort_flow": {
            "procedureevents": procedure_audit,
            "chartevents": vital_audit,
            "canonicalization": canonical_audit,
            "patient_disjoint_split": split_audit,
        },
        "primary": {
            "role": "primary_20_350_measurement_screen_then_50_250_nadir_eligibility",
            "counts": primary_counts,
            "missingness_definition": (
                "score-native missing branch: nonfinite or out-of-contract "
                "numeric inputs; UF/body weight is structurally unavailable"
            ),
            "missingness": primary_missingness,
            "metrics": primary_metrics,
        },
        "sensitivity": {
            "role": "per_measurement_50_250_sensitivity",
            "counts": sensitivity_counts,
        },
    }
    if primary_bootstrap is not None:
        result["primary"]["bootstrap"] = primary_bootstrap
    result["target_verification"] = verify_targets(result)
    assert_aggregate_only(result)
    return result
