# Public scope

This repository is the MIMIC-IV reproduction package for the published
five-variable IDH score. It is code plus two score lookup tables plus the
reference numbers for one analysis.

## Included

- the fixed five-variable 0–48 point rule and its probability equation;
- a generic intercept-offset function for a locally estimated recalibration;
- aggregate-only validation with optional patient-cluster bootstrap;
- the credentialed MIMIC-IV v2.2 raw-to-aggregate workflow;
- the published MIMIC-IV reference numbers, which every credentialed run is
  checked against automatically;
- the patient-cluster bootstrap that produces the reported confidence intervals;
- synthetic tests, machine-readable contracts, and two score lookup tables.

## Not included or claimed

- any patient- or session-level data, private or credentialed;
- any data, count, constant, table, or result from the private development and
  external-validation cohorts, including their probability mappings and
  intercept offsets;
- reconstruction of those cohorts or of the study flow;
- model training, variable selection, or re-estimation of the point rule;
- fitted shape functions or the integer-score derivation procedure;
- trained models, row-level predictions, manuscript files, or manuscript hashes;
- bootstrap replicate values or patient multiplicities.

## One documented exception

The MIMIC-IV patient-disjoint split was originally built with a stratification
variable that uses a pre-dialysis-SBP-dependent threshold. That variable is
retained verbatim in `mimic/cohort.py`, and only there, because the fold
assignment cannot be reproduced without it. It is neither the analysis outcome
nor the seven-day history event: both of those are Nadir90 — nadir SBP below
90 mm Hg — throughout this package. `tests/test_public_package.py` enforces that
the exception stays confined to the split.

## Interpretation

The scorer reproduces the published arithmetic from five supplied inputs. It
does not reproduce the private-data study that produced the constants. Users
should evaluate calibration in their own population before interpreting absolute
risk.
