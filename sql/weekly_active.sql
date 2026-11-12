-- Weekly active subscriptions (paid plans with a session that week) for forecasting
SELECT date_trunc('week', e.ts) AS week, count(DISTINCT e.user_id) AS was
FROM events e JOIN users u USING (user_id)
WHERE e.event_type='session_start' AND u.plan_initial IN ('pro','team')
GROUP BY 1 ORDER BY 1
