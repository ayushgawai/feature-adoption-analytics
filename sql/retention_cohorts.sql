-- Retention cohorts: % of each signup-month cohort active in later months
WITH act AS (SELECT DISTINCT user_id, date_trunc('month', ts) AS m FROM events WHERE event_type='session_start')
SELECT date_trunc('month', u.signup_ts) AS cohort,
       datediff('month', date_trunc('month', u.signup_ts), a.m) AS month_n,
       count(DISTINCT a.user_id) AS active_users
FROM users u JOIN act a USING (user_id)
WHERE a.m >= date_trunc('month', u.signup_ts)
GROUP BY 1, 2 ORDER BY 1, 2
