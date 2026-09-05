# MIMIC-IV v2.2 credentialed workflow

This directory converts the two required MIMIC-IV v2.2 source files into an
aggregate-only point-estimate report. It never writes identifiers, row-level
features, outcomes, scores, or predictions. MIMIC-IV data are not redistributed;
each runner must obtain PhysioNet credentials and accept the applicable data-use
agreement.

The repository bundles the reference values for this analysis in
`verification_targets.json`. Every run compares itself with them field by field
and writes the outcome into `target_verification`. The command exits `0` only
when every check passes and `2` otherwise. No MIMIC-IV record, identifier, or
row-level value is bundled -- only counts and the reported point estimates.

The complete expected aggregate lives in
`verification/mimic_aggregate_reference.json`. It is not committed here, because
producing it needs credentialed access; generate your own with
`python scripts/make_reference_aggregate.py --mimic-root <ROOT>` from the
repository root.

## Endpoint contract

Both endpoint constructions use

`(T0, min(documented HD end, T0 + 6 h)]`

where `T0` is `procedureevents.starttime`. A measurement at `T0` is excluded;
the right endpoint is included. A measurement exactly 6 hours after `T0` is
included only when the documented procedure has not already ended.

- **Primary:** admit individual SBP measurements from 20 through 350 mm Hg,
  select the session nadir, then require the selected nadir to be from 50
  through 250 mm Hg.
- **Sensitivity:** restrict every individual SBP measurement to 50 through
  250 mm Hg before selecting the nadir.

The 7-day score input counts observable completed prior sessions with a
post-start nadir SBP below 90 mm Hg (Nadir90; one threshold, no pre-SBP switch). Same-day and future sessions are excluded,
and a prior session must end before the index session. If no qualifying prior
session is observable in MIMIC-IV, history remains missing and uses the score's
fixed missing branch.

The patient-disjoint allocation retains its documented historical split
algorithm and its historical stratification variable (nadir SBP <90 mm Hg when
pre-dialysis SBP <160 mm Hg, otherwise nadir SBP <100 mm Hg) for one reason
only: reproducing the pre-existing fold assignment bit for bit. That variable
is neither the analysis outcome nor the seven-day history event. Both of those
are Nadir90 -- nadir SBP below 90 mm Hg -- everywhere in this package.

## Inputs and execution

Supply either the directory containing `2.2/` or the `2.2/` directory itself:

```text
2.2/
└── icu/
    ├── procedureevents.csv.gz
    └── chartevents.csv.gz
```

From the repository root:

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m mimic.run_pipeline `
  --mimic-root <CREDENTIALED_MIMIC_ROOT> `
  --score-spec specification/score_specification.json `
  --output-json mimic_aggregate_result.json
```

Use `--verify-source-hashes` to compare the two compressed source files with the
documented official-file hashes. This adds substantial I/O.

Use `--bootstrap-replicates 1000` to add the reported 95% patient-cluster
confidence intervals under `primary.bootstrap`. The bootstrap resamples patients
with a multinomial draw and reports percentile bounds; replicate values and
patient multiplicities are never written. A metric whose replicates are all
non-estimable -- joint calibration under separation, for example -- is listed in
`primary.bootstrap.non_estimable` instead of being given a bound.

The executable MIMIC modules are run from a repository checkout; they are not
part of the installable base scorer wheel. `provenance.json`, the JSON schema,
and the SQL crosswalk document the inputs and endpoint, but the Python modules
are authoritative for execution.

## Output boundary

The output contains cohort counts, missingness, endpoint metadata, point
estimates, and, when `--bootstrap-replicates` is given, interval bounds. Passing
the synthetic tests is not a raw MIMIC numerical replay: only a credentialed run
whose `target_verification.status` is `passed` reproduces the reported
analysis.
