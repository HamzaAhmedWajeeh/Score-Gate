-- Application-level features, one row per SK_ID_CURR, carrying the TARGET label.
-- Every ratio guards its denominator with NULLIF so a zero or missing denominator
-- becomes NULL instead of an error; optbinning's missing bin then handles it.
CREATE OR REPLACE TABLE feat_application AS
WITH cleaned AS (
    SELECT
        SK_ID_CURR,
        TARGET,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        AMT_ANNUITY,
        AMT_GOODS_PRICE,
        DAYS_BIRTH,
        -- 365243 is a sentinel for "not employed" (pensioners, unemployed); null it
        -- so the ratio and the scorecard's missing bin treat that group honestly.
        NULLIF(DAYS_EMPLOYED, 365243) AS DAYS_EMPLOYED,
        -- 1 when a real employment tenure is on file, 0 when it was the sentinel.
        CASE WHEN DAYS_EMPLOYED = 365243 THEN 0 ELSE 1 END AS employment_recorded,
        CNT_CHILDREN,
        EXT_SOURCE_1,
        EXT_SOURCE_2,
        EXT_SOURCE_3
    FROM raw_application_train
)
SELECT
    SK_ID_CURR,
    TARGET,
    -- Leverage: loan size relative to yearly income.
    AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0) AS credit_to_income,
    -- Debt-burden-ratio proxy: the annuity-to-income figure regulators cap in
    -- responsible-lending rules.
    AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) AS annuity_to_income,
    -- Financed share of the asset; a lower value implies a larger down payment.
    AMT_CREDIT / NULLIF(AMT_GOODS_PRICE, 0) AS credit_to_goods,
    -- Fraction of life spent in the current job; longer tenure signals stability.
    DAYS_EMPLOYED / NULLIF(DAYS_BIRTH, 0) AS employed_share,
    -- External bureau scores are the strongest single signal; average the three,
    -- ignoring missing ones by dividing by the count actually present.
    (COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) + COALESCE(EXT_SOURCE_3, 0))
        / NULLIF(
            (CASE WHEN EXT_SOURCE_1 IS NOT NULL THEN 1 ELSE 0 END)
            + (CASE WHEN EXT_SOURCE_2 IS NOT NULL THEN 1 ELSE 0 END)
            + (CASE WHEN EXT_SOURCE_3 IS NOT NULL THEN 1 ELSE 0 END), 0) AS ext_source_mean,
    -- Worst of the three external scores; the weakest source often drives risk.
    LEAST(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) AS ext_source_min,
    -- How many external scores exist; a thin bureau file is itself informative.
    (CASE WHEN EXT_SOURCE_1 IS NOT NULL THEN 1 ELSE 0 END)
        + (CASE WHEN EXT_SOURCE_2 IS NOT NULL THEN 1 ELSE 0 END)
        + (CASE WHEN EXT_SOURCE_3 IS NOT NULL THEN 1 ELSE 0 END) AS ext_source_count,
    -- Pass-through raw fields kept for direct use or later binning.
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    AMT_ANNUITY,
    DAYS_BIRTH,
    DAYS_EMPLOYED,
    employment_recorded,
    CNT_CHILDREN,
    EXT_SOURCE_1,
    EXT_SOURCE_2,
    EXT_SOURCE_3
FROM cleaned;
