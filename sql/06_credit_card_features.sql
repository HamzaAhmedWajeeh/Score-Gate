-- Credit-card utilisation rolled up to the applicant. Window function 3 (a
-- frame-based rolling AVG) smooths utilisation over each card's trailing 6 months.
CREATE OR REPLACE TABLE feat_credit_card AS
WITH base AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        -- The credit limit is genuinely 0 on some rows, so this NULLIF is load-bearing.
        AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS utilization
    FROM raw_credit_card_balance
),
rolled AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        -- Trailing 6-month average utilisation; smooths month-to-month noise.
        AVG(utilization) OVER (
            PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS rolling_util_6m,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE DESC
        ) AS recency_rank
    FROM base
),
latest AS (
    -- One row per card, holding its most recent trailing-6-month utilisation.
    SELECT SK_ID_CURR, rolling_util_6m
    FROM rolled
    WHERE recency_rank = 1
)
SELECT
    SK_ID_CURR,
    -- Number of distinct card contracts on file; a presence signal that restores
    -- source symmetry. A NULL card_count downstream means "no card history in this
    -- source", so a NULL card_rolling_util_6m among holders cleanly means the
    -- utilisation was undefined (a zero credit limit), not missing history.
    COUNT(*) AS card_count,
    -- Mean latest rolling utilisation across the applicant's cards; sustained high
    -- utilisation is a classic early distress signal.
    AVG(rolling_util_6m) AS card_rolling_util_6m
FROM latest
GROUP BY SK_ID_CURR;
