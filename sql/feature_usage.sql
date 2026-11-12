-- Feature usage: high- and low-usage features (last 90 days of data)
SELECT feature,
       count(*) AS uses,
       count(DISTINCT user_id) AS users,
       round(100.0 * count(DISTINCT user_id) /
             (SELECT count(DISTINCT user_id) FROM events WHERE event_type='feature_use'), 2) AS pct_reach
FROM events WHERE event_type='feature_use' AND feature IS NOT NULL
GROUP BY feature ORDER BY uses DESC
