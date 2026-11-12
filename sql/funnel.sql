-- Adoption funnel: signup -> first session -> first feature -> first export -> upgrade
WITH f AS (
  SELECT u.user_id,
    min(e.ts) FILTER (WHERE e.event_type='session_start') AS s1,
    min(e.ts) FILTER (WHERE e.event_type='feature_use') AS s2,
    min(e.ts) FILTER (WHERE e.event_type='feature_use' AND e.feature LIKE 'export%') AS s3,
    min(e.ts) FILTER (WHERE e.event_type='upgrade') AS s4
  FROM users u LEFT JOIN events e USING (user_id) GROUP BY 1)
SELECT count(*) AS signed_up,
  count(s1) AS first_session, count(s2) AS first_feature,
  count(s3) AS first_export, count(s4) AS upgraded,
  round(100.0*count(s3)/count(*),2) AS pct_activated
FROM f
