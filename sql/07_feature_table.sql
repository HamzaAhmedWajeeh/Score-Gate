-- Final assembly. Each source was aggregated to one row per applicant first, so
-- these are one-to-one LEFT JOINs onto the application base by SK_ID_CURR. A NULL
-- here means no history in that source (a thin file), which is distinct from a real
-- 0; we never COALESCE it away, because optbinning's missing bin learns thin-file
-- risk on its own. The feature table deliberately never selects CODE_GENDER.
CREATE OR REPLACE TABLE features AS
SELECT
    a.*,
    b.bureau_credit_count,
    b.bureau_active_count,
    b.bureau_debt_to_credit,
    b.bureau_overdue_max,
    b.bureau_prolong_sum,
    b.bureau_recency,
    bb.bureau_current_dpd_count,
    bb.bureau_ever_dpd3_share,
    p.prev_app_count,
    p.prev_refused_share,
    p.prev_approved_credit_mean,
    i.inst_late_share,
    i.inst_shortfall_share,
    i.inst_mean_delay,
    i.inst_delay_trend,
    c.card_count,
    c.card_rolling_util_6m
FROM feat_application a
LEFT JOIN feat_bureau b USING (SK_ID_CURR)
LEFT JOIN feat_bureau_balance bb USING (SK_ID_CURR)
LEFT JOIN feat_prev_app p USING (SK_ID_CURR)
LEFT JOIN feat_installments i USING (SK_ID_CURR)
LEFT JOIN feat_credit_card c USING (SK_ID_CURR);

-- Fairness metadata, kept in its own table keyed by SK_ID_CURR. Gender and age band
-- live here only, feeding the fairness snapshot; they are never model features.
CREATE OR REPLACE TABLE fairness_metadata AS
SELECT
    SK_ID_CURR,
    CODE_GENDER,
    CASE
        WHEN -DAYS_BIRTH / 365.25 < 25 THEN '<25'
        WHEN -DAYS_BIRTH / 365.25 < 35 THEN '25-34'
        WHEN -DAYS_BIRTH / 365.25 < 45 THEN '35-44'
        WHEN -DAYS_BIRTH / 365.25 < 55 THEN '45-54'
        WHEN -DAYS_BIRTH / 365.25 < 65 THEN '55-64'
        ELSE '65+'
    END AS age_band
FROM raw_application_train;
