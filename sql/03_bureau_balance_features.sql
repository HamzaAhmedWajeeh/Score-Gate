-- Monthly bureau balance rolled up to the applicant. Window function 1 (ROW_NUMBER)
-- picks each credit's most recent month to read its current delinquency standing.
CREATE OR REPLACE TABLE feat_bureau_balance AS
WITH bucketed AS (
    -- STATUS: 'C' (closed) and 'X' (unknown) are not delinquency and map to 0; '0'
    -- is on-time; '1'..'5' are days-past-due buckets in 30-day steps.
    SELECT
        SK_ID_BUREAU,
        MONTHS_BALANCE,
        CASE WHEN STATUS IN ('1', '2', '3', '4', '5')
            THEN CAST(STATUS AS INTEGER) ELSE 0 END AS dpd_bucket
    FROM raw_bureau_balance
),
ranked AS (
    SELECT
        SK_ID_BUREAU,
        dpd_bucket,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_BUREAU ORDER BY MONTHS_BALANCE DESC) AS rn
    FROM bucketed
),
latest AS (
    -- Each credit's current standing: its status in the most recent month.
    SELECT SK_ID_BUREAU, dpd_bucket AS current_bucket
    FROM ranked
    WHERE rn = 1
),
ever AS (
    -- Whether the credit ever reached 90+ days past due across its whole history.
    SELECT SK_ID_BUREAU, MAX(dpd_bucket) >= 3 AS ever_dpd3
    FROM bucketed
    GROUP BY SK_ID_BUREAU
),
per_credit AS (
    SELECT b.SK_ID_CURR, l.current_bucket, e.ever_dpd3
    FROM raw_bureau b
    JOIN latest l USING (SK_ID_BUREAU)
    JOIN ever e USING (SK_ID_BUREAU)
)
SELECT
    SK_ID_CURR,
    -- Credits currently past due; live delinquency is the sharpest warning.
    COUNT(*) FILTER (WHERE current_bucket >= 1) AS bureau_current_dpd_count,
    -- Share of the applicant's credits that ever hit 90+ DPD; chronic delinquency.
    AVG(CASE WHEN ever_dpd3 THEN 1.0 ELSE 0.0 END) AS bureau_ever_dpd3_share
FROM per_credit
GROUP BY SK_ID_CURR;
