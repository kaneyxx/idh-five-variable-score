# IDH five-variable score — MIMIC-IV reproduction package

This repository contains everything needed to recompute the MIMIC-IV v2.2
analysis of the five-variable intradialytic hypotension (IDH) score,
and nothing else. It holds the fixed 0–48 point rule, its probability equation,
the raw-to-aggregate MIMIC-IV workflow, and the reference numbers that a
credentialed run is expected to reproduce.

> **Status.** The manuscript describing this score is under review and is not
> yet published. The reference values in `mimic/verification_targets.json` are
> the values that manuscript reports; nothing here should be read as a
> published result.

It contains **no** patient- or session-level data, no model-training or
variable-selection code, and no data, constants, or results from the private
development and external-validation cohorts. See
[`PUBLIC_SCOPE.md`](PUBLIC_SCOPE.md) for the exact claim boundary.

## Quick start

The scorer uses only the Python standard library and supports Python 3.11 or
newer. The dependency lock was resolved for the CPython 3.14.2 reference
environment.

```bash
python -m pip install -e .
```

```python
from idh_score import calculate_idh_score

result = calculate_idh_score(
    sbp_mm_hg=105,
    idh_events_prior_7d=2,
    uf_bw_percent=4.2,
    dbp_mm_hg=60,
    heart_rate_bpm=85,
)

print(result.total_points)          # 30
print(result.predicted_probability) # probability equation
print(result.component_points)      # five component contributions
```

The complete arithmetic is also in
[`reference/score_rule.csv`](reference/score_rule.csv), and all 49 predicted
risks are in
[`reference/risk_lookup_0_48.csv`](reference/risk_lookup_0_48.csv).

To verify the repository itself:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest
python scripts/scan_sensitive_content.py
```

## Definitions

- **Outcome.** Current-session nadir systolic blood pressure below 90 mm Hg
  (Nadir90), from measurements strictly after treatment initiation through the
  recorded treatment end: `(T0, recorded treatment end]`.
- **Seven-day history.** `idh_events_prior_7d` is the number of prior-session
  Nadir90 events in `[index date − 7 days, index date)`. Same-day and future
  sessions are excluded. A single threshold of 90 mm Hg is used everywhere; the
  package contains no pre-SBP-dependent alternative, and the tests enforce that.
- **Units.** `uf_bw_percent` is in percentage points: enter `3.2` for 3.2%, not
  `0.032`.
- **Valid ranges.** SBP 50–250 mm Hg, UF/BW 0–25 percentage points, DBP
  20–150 mm Hg, heart rate 20–250 beats/min; both endpoints are valid. Missing,
  nonfinite, or out-of-range inputs take the predictor-specific fixed missing
  branch.
- There is no minimum-sessions-per-patient eligibility rule in this score
  definition.

The machine-readable authoritative contract is
[`specification/score_specification.json`](specification/score_specification.json).

## Probability mapping

```text
P(Nadir90) = expit(-4.321773159969571 + 0.17227817775653662 * score)
```

Each additional point multiplies the modelled odds of Nadir90 by
`exp(0.172278) = 1.19`.

`probability_with_intercept_offset(score, offset)` applies an externally
estimated intercept adjustment while keeping the score coefficient fixed.
Absolute risk should be recalibrated and evaluated before use in a different
institution or time period.

### Why the constants are written to full precision

The two coefficients above are stored as the shortest decimal strings that
round-trip to the exact IEEE-754 doubles produced by the fit. That is a
reproducibility requirement, not a precision claim: parsing a shorter string
yields a different double, and the aggregate checks in
`mimic/verification_targets.json` compare four of the six metrics to 1e-12.

For reading and for hand calculation, far fewer digits are needed. Rounding both
coefficients to six decimals reproduces all 49 risks to within 1.1e-6, which is
identical to the full-precision value once the risk is expressed as a percentage
to two decimal places; five decimals already suffice for one decimal place. The
accompanying manuscript therefore reports the equation as
`p = expit(-4.321773 + 0.172278 x score)`.

Neither figure says anything about statistical precision. Both coefficients are
estimates from a finite sample, and their sampling uncertainty is many orders of
magnitude larger than the last digits shown here.

## Reproducing the MIMIC-IV analysis

[`mimic/`](mimic/) converts two credentialed MIMIC-IV v2.2 source files into an
aggregate-only report. MIMIC-IV data are not redistributed: each runner needs
their own PhysioNet credentials and must accept the data-use agreement.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m mimic.run_pipeline \
  --mimic-root <CREDENTIALED_MIMIC_ROOT> \
  --score-spec specification/score_specification.json \
  --output-json mimic_aggregate_result.json
```

The run compares itself with the reference values in
[`mimic/verification_targets.json`](mimic/verification_targets.json) and writes
the per-field result into `target_verification`. It exits `0` only when every
check passes and `2` otherwise, so a divergence cannot be mistaken for a
successful reproduction.

Add `--bootstrap-replicates 1000` to reproduce the reported 95% patient-cluster
confidence intervals; they appear under `primary.bootstrap`. Without the flag the
run reports point estimates only, which is much faster.

The output is aggregate-only: cohort counts, missingness, endpoint metadata,
point estimates, and — when requested — interval bounds. It never contains
identifiers, row-level features, outcomes, scores, predictions, bootstrap
replicate values, or patient multiplicities.

### The bundled reference aggregate

`scripts/make_reference_aggregate.py` writes the complete expected output to
`mimic/verification/mimic_aggregate_reference.json`:

```bash
python scripts/make_reference_aggregate.py \
  --mimic-root <CREDENTIALED_MIMIC_ROOT> \
  --bootstrap-replicates 1000
```

It refuses to write unless the source files pass SHA-256 verification and the
run passes every check in `verification_targets.json`, so the bundled reference
can only come from a complete, verified run. The file is not included in this
repository, because generating it requires credentialed access; the tests that
compare against it skip until it exists.

[`mimic/README.md`](mimic/README.md) documents the endpoint contract, the
primary and sensitivity screens, and the historical patient-disjoint split.

## Applying the score to your own data

`idh-validate` scores a CSV and returns aggregates only — AUROC, average
precision, Brier score, observed-to-expected ratio, joint calibration intercept
and slope, and the fixed-slope intercept adjustment. It is the same metric code
the MIMIC-IV pipeline uses.

```bash
python -m pip install -e ".[validation]"
idh-validate input.csv --output results.json
idh-validate input.csv --patient-id patient_id --output results.json
```

Required columns are `sbp_mm_hg`, `idh_events_prior_7d`, `uf_bw_percent`,
`dbp_mm_hg`, `heart_rate_bpm`, and binary `outcome`. A patient identifier column
is optional and must be selected explicitly with `--patient-id`; with a complete
one, the validator defaults to 1,000 patient-cluster bootstrap replicates. Joint
calibration is reported as non-estimable when complete or quasi-complete
separation is detected. The validator never returns row-level scores,
predictions, identifiers, or bootstrap multiplicities.

## Repository map

- `src/idh_score/` — scorer, probability equation, and aggregate-only validator.
- `specification/` — machine-readable fixed rule and definitions.
- `reference/` — human-readable score and predicted-risk tables.
- `mimic/` — credentialed MIMIC-IV workflow and its reference values.
- `tests/` — synthetic scientific and software regression tests, plus the
  release-boundary guards in `tests/test_public_package.py`.
- `scripts/` — sensitive-content scanner and the reference-aggregate generator.

## Intended use and licenses

This research prediction instrument is not a treatment directive and does not
replace clinical judgment, prospective implementation review, or local safety
evaluation. Source code is BSD-3-Clause (`LICENSE`); tables and documentation
are CC BY 4.0 (`LICENSE_DATA_DOCUMENTATION.txt`).
