-- Installment repayment behaviour aggregated to the applicant. Window function 2
-- (LAG) builds a per-installment deterioration trend within each prior credit.
CREATE OR REPLACE TABLE feat_installments AS
WITH base AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        NUM_INSTALMENT_NUMBER,
        -- Positive means the payment landed after the due date.
        DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT AS pay_delay,
        -- Positive means the applicant paid less than the amount due.
        AMT_INSTALMENT - AMT_PAYMENT AS pay_shortfall
    FROM raw_installments_payments
),
trended AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        pay_delay,
        pay_shortfall,
        -- Change in lateness versus the previous installment; a rising delay means
        -- behaviour is deteriorating, and the trend predicts better than the level.
        pay_delay - LAG(pay_delay) OVER (
            PARTITION BY SK_ID_PREV ORDER BY NUM_INSTALMENT_NUMBER
        ) AS delay_trend,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_PREV ORDER BY NUM_INSTALMENT_NUMBER DESC
        ) AS recency_rank
    FROM base
),
credit_trend AS (
    -- Mean deterioration over each credit's last 12 installments.
    SELECT SK_ID_CURR, SK_ID_PREV, AVG(delay_trend) AS credit_delay_trend
    FROM trended
    WHERE recency_rank <= 12
    GROUP BY SK_ID_CURR, SK_ID_PREV
),
applicant_trend AS (
    -- Average the per-credit trend equally across the applicant's credits.
    SELECT SK_ID_CURR, AVG(credit_delay_trend) AS inst_delay_trend
    FROM credit_trend
    GROUP BY SK_ID_CURR
),
applicant_base AS (
    SELECT
        SK_ID_CURR,
        -- Share of installments paid late; how often the applicant is tardy.
        AVG(CASE WHEN pay_delay > 0 THEN 1.0 ELSE 0.0 END) AS inst_late_share,
        -- Share of installments underpaid; how often payments fall short.
        AVG(CASE WHEN pay_shortfall > 0 THEN 1.0 ELSE 0.0 END) AS inst_shortfall_share,
        -- Average days late across all installments; the level of tardiness.
        AVG(pay_delay) AS inst_mean_delay
    FROM base
    GROUP BY SK_ID_CURR
)
SELECT
    b.SK_ID_CURR,
    b.inst_late_share,
    b.inst_shortfall_share,
    b.inst_mean_delay,
    t.inst_delay_trend
FROM applicant_base b
LEFT JOIN applicant_trend t USING (SK_ID_CURR);
