# Environment

The reference scorer in `src/idh_score/score.py` uses only the Python standard
library and supports Python 3.11 or newer. The reproducibility lock below was
resolved for the CPython 3.14.2 reference environment.

Optional workflows use these pinned direct dependencies:

- local validation: NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0;
- credentialed MIMIC-IV workflow: the validation dependencies plus pandas 3.0.0;
- tests: the preceding packages plus pytest 9.1.1.

Install only the component needed:

```powershell
python -m pip install -e .
python -m pip install -e ".[validation]"
python -m pip install -e ".[mimic]"
python -m pip install -e ".[test]"
```

`requirements-lock.txt` records the complete clean-environment dependency set.
The MIMIC-IV reference numbers were produced in exactly this environment; the
patient-disjoint split depends on the pinned scikit-learn version, so use the
lock file rather than newer releases when reproducing them.

## Floating-point reproducibility

The probability equation is evaluated with `math.exp`, and a C library is not
required to round `exp` correctly. glibc and the macOS libm can therefore return
adjacent doubles for the same input, which is why
`reference/risk_lookup_0_48.csv` is checked to within two units in the last
place rather than bit for bit.

The effect is bounded and does not reach any reported figure. Across the 49
probabilities the largest ULP is 1.1e-16, so over the 817-session MIMIC-IV
cohort the Brier score can move by at most 2.2e-16 and the observed-to-expected
ratio by at most 4.2e-16 -- roughly 2,000 to 4,500 times smaller than the 1e-12
tolerances in `mimic/verification_targets.json`. AUROC and average precision are
rank statistics over at most 49 distinct probabilities, and the closest gap
between two of them is 2.4e-03, about 2e13 ULP, so no ordering and no tie group
can change; both are unaffected. Joint calibration is solved by BFGS and is
checked to 5e-4 for the same reason.

This repository does not contain or execute any model-training pipeline.
