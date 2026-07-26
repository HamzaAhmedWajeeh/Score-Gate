-- Credit-bureau history aggregated to one row per applicant.
CREATE OR REPLACE TABLE feat_bureau AS
SELECT
    SK_ID_CURR,
    -- Total prior bureau-reported credits; the breadth of external credit history.
    COUNT(*) AS bureau_credit_count,
    -- Currently open external credits; concurrent obligations raise repayment strain.
    COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bureau_active_count,
    -- Outstanding debt against total credit granted; an external utilisation proxy.
    SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0) AS bureau_debt_to_credit,
    -- Largest single overdue amount on file; a severity marker for past arrears.
    MAX(AMT_CREDIT_SUM_OVERDUE) AS bureau_overdue_max,
    -- Count of credit prolongations; restructures signal earlier distress.
    SUM(CNT_CREDIT_PROLONG) AS bureau_prolong_sum,
    -- Days (negative, so closest to zero) since the most recent credit opened; recency.
    MAX(DAYS_CREDIT) AS bureau_recency
FROM raw_bureau
GROUP BY SK_ID_CURR;
