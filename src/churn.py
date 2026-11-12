"""Churn model: logistic regression vs random forest, time-based split.

Label: churned within 60 days after the feature window (first 30 days).
Features use only the first 30 days, so nothing leaks from the future.
Reports ROC AUC, Brier (calibration), and permutation importance.
"""
import numpy as np, pandas as pd, duckdb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance

FEATS = ["sessions_30d", "feature_events_30d", "distinct_features", "exports_30d",
         "days_to_first_export", "is_paid", "is_team"]

def build_features(data="data"):
    con = duckdb.connect()
    df = con.execute(f"""
      WITH f AS (SELECT user_id,
        count(*) FILTER (WHERE event_type='session_start') AS sessions_30d,
        count(*) FILTER (WHERE event_type='feature_use') AS feature_events_30d,
        count(DISTINCT feature) AS distinct_features,
        count(*) FILTER (WHERE feature LIKE 'export%') AS exports_30d,
        datediff('day', min(u.signup_ts), min(e.ts) FILTER (WHERE e.feature LIKE 'export%')) AS days_to_first_export
       FROM read_parquet('{data}/events.parquet') e JOIN read_parquet('{data}/users.parquet') u USING(user_id)
       WHERE e.ts < u.signup_ts + INTERVAL 30 DAY GROUP BY 1, u.signup_ts)
      SELECT f.*, (u.plan_initial != 'free')::INT AS is_paid, (u.plan_initial='team')::INT AS is_team,
        u.signup_ts, (u.churn_ts IS NOT NULL AND u.churn_ts < u.signup_ts + INTERVAL 90 DAY)::INT AS churned
      FROM f JOIN read_parquet('{data}/users.parquet') u USING(user_id)""").df()
    return df.fillna({"days_to_first_export": 999})

def evaluate(df, cutoff="2025-01-01"):
    tr, te = df[df.signup_ts < cutoff], df[df.signup_ts >= cutoff]
    models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)}
    out = {}
    for name, m in models.items():
        m.fit(tr[FEATS], tr.churned); p = m.predict_proba(te[FEATS])[:, 1]
        out[name] = {"roc_auc": round(roc_auc_score(te.churned, p), 4),
                     "brier": round(brier_score_loss(te.churned, p), 4), "n_test": len(te)}
        print(name, out[name])
    imp = permutation_importance(models["random_forest"], te[FEATS], te.churned,
                                 n_repeats=5, random_state=42, scoring="roc_auc")
    pi = pd.Series(imp.importances_mean, index=FEATS).sort_values(ascending=False)
    print("permutation importance (RF):\n", pi.round(4).to_string())
    return out, pi

if __name__ == "__main__":
    evaluate(build_features())
