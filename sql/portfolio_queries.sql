-- Monthly delinquency trend
SELECT
    DATE_TRUNC('month', issue_date) AS month,
    COUNT(*) AS loan_count,
    AVG(days_past_due) AS avg_days_past_due,
    AVG(default_probability) AS avg_default_probability,
    AVG(predicted_default::int) AS predicted_default_rate
FROM scored_loans
GROUP BY 1
ORDER BY 1;

-- Risk segmentation distribution
SELECT
    risk_segment,
    COUNT(*) AS loan_count,
    SUM(loan_amnt) AS exposure,
    AVG(default_probability) AS avg_pd,
    SUM(expected_loss) AS expected_loss
FROM scored_loans
GROUP BY risk_segment
ORDER BY avg_pd;

-- Top risky loans for underwriting review
SELECT
    loan_id,
    loan_amnt,
    int_rate,
    annual_inc,
    dti,
    fico_score,
    days_past_due,
    default_probability,
    risk_segment
FROM scored_loans
ORDER BY default_probability DESC
LIMIT 50;
