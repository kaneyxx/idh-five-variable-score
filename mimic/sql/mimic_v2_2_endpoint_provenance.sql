-- MIMIC-IV v2.2 endpoint provenance for the public Python workflow.
-- PostgreSQL-style notation is used for readability. Run only in a
-- credentialed MIMIC environment. Do not export the row-level CTEs.

WITH hd_raw AS (
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        starttime AS t0,
        endtime AS documented_hd_end,
        statusdescription
    FROM mimiciv_icu.procedureevents
    WHERE itemid = 225441
      AND subject_id IS NOT NULL
      AND hadm_id IS NOT NULL
      AND stay_id IS NOT NULL
      AND starttime IS NOT NULL
      AND endtime IS NOT NULL
),
hd AS (
    -- Python resolves duplicate (subject_id, stay_id, t0) rows by the latest
    -- documented end, preserving source order as the final tie breaker.
    SELECT *, LEAST(documented_hd_end, t0 + INTERVAL '6 hour') AS outcome_end
    FROM hd_raw
),
sbp AS (
    SELECT stay_id, charttime, itemid, valuenum
    FROM mimiciv_icu.chartevents
    WHERE itemid IN (220179, 220050)
      AND valuenum IS NOT NULL
),
endpoint_rows AS (
    SELECT
        hd.subject_id,
        hd.stay_id,
        hd.t0,
        sbp.charttime,
        sbp.valuenum,
        CASE WHEN sbp.valuenum BETWEEN 20 AND 350 THEN sbp.valuenum END
            AS primary_broad_screen_sbp,
        CASE WHEN sbp.valuenum BETWEEN 50 AND 250 THEN sbp.valuenum END
            AS sensitivity_per_measurement_sbp
    FROM hd
    JOIN sbp
      ON sbp.stay_id = hd.stay_id
     AND sbp.charttime > hd.t0
     AND sbp.charttime <= hd.outcome_end
     -- The window is (T0, min(documented HD end, T0 + 6 h)].
     -- T0 is excluded. The right endpoint, including exactly T0+6h, is included.
),
session_endpoints AS (
    SELECT
        subject_id,
        stay_id,
        t0,
        MIN(primary_broad_screen_sbp) AS primary_operational_nadir,
        COUNT(primary_broad_screen_sbp) AS primary_operational_measurements,
        MIN(sensitivity_per_measurement_sbp) AS sensitivity_nadir,
        COUNT(sensitivity_per_measurement_sbp) AS sensitivity_measurements
    FROM endpoint_rows
    GROUP BY subject_id, stay_id, t0
)
SELECT
    COUNT(*) FILTER (
        WHERE primary_operational_measurements >= 1
          AND primary_operational_nadir BETWEEN 50 AND 250
    ) AS primary_eligible_sessions,
    COUNT(*) FILTER (
        WHERE sensitivity_measurements >= 1
          AND sensitivity_nadir IS NOT NULL
    ) AS sensitivity_eligible_sessions
FROM session_endpoints;

-- Pre-index mappings used by Python:
--   SBP itemids 220179/220050; DBP 220180/220051; HR 220045.
--   Window [T0-2h, T0], latest charttime; NIBP wins exact SBP/DBP ties.
-- CRRT-at-start itemids: 225802, 225803, 225809, 225955.
-- Historical split compatibility uses crrt_start <= T0 <= crrt_end; the
-- end-equality case is retained only to reconstruct the locked split ledger.
-- Observable seven-day history uses only prior primary-eligible labelled HD
-- sessions (pre-SBP and selected nadir 50--250, count >=1, duration >0, no
-- CRRT at T0). Its event is post-start nadir SBP <90 mm Hg; it does not use
-- the pre-SBP-dependent stratification variable that reproduces the split.
-- Patient-disjoint SGKF, history construction, current 0--48 scoring, and
-- aggregate metrics are implemented in the version-controlled Python modules.
