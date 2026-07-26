-- Prior Home Credit applications aggregated to the applicant.
CREATE OR REPLACE TABLE feat_prev_app AS
SELECT
    SK_ID_CURR,
    -- Number of previous applications; the depth of the internal relationship.
    COUNT(*) AS prev_app_count,
    -- Share previously refused; prior internal declines predict future default.
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END)
        AS prev_refused_share,
    -- Mean credit on past approvals; the typical size of prior granted facilities.
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN AMT_CREDIT END)
        AS prev_approved_credit_mean
FROM raw_previous_application
GROUP BY SK_ID_CURR;
