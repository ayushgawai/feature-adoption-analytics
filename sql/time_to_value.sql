-- Time to value: days from signup to first export (activation), by experiment group
SELECT u.experiment_group,
       count(*) AS users,
       round(median(datediff('day', u.signup_ts, f.first_export)),2) AS median_days,
       round(avg(datediff('day', u.signup_ts, f.first_export)),2) AS mean_days
FROM users u JOIN (
  SELECT user_id, min(ts) AS first_export FROM events
  WHERE event_type='feature_use' AND feature LIKE 'export%' GROUP BY 1) f USING (user_id)
GROUP BY 1 ORDER BY 1
