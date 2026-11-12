"""Tests for metric SQL (DuckDB on tiny fixtures) and experiment analysis."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import numpy as np, pandas as pd, duckdb, pytest
from experiment import analyse

def run_sql(sql, users, events):
    con = duckdb.connect()
    con.register("users", users); con.register("events", events)
    return con.execute(sql).df()

@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    d = tmp_path_factory.mktemp("d")
    users = pd.DataFrame({"user_id": [1, 2], "signup_ts": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                          "plan_initial": ["pro", "free"], "experiment_group": ["control", "treatment"],
                          "churn_ts": pd.to_datetime([None, None]), "engagement": [0.9, 0.1]})
    events = pd.DataFrame({"user_id": [1, 1, 1, 2], "ts": pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-02"]),
        "event_type": ["signup", "session_start", "feature_use", "session_start"],
        "feature": [None, None, "export_png", None], "plan_to": [None] * 4, "plan": ["pro", "pro", "pro", "free"]})
    users.to_parquet(d / "users.parquet"); events.to_parquet(d / "events.parquet")
    return users, events

def sql(name): return pathlib.Path(f"sql/{name}.sql").read_text()

def test_funnel_counts(fixture):
    out = run_sql(sql("funnel"), *fixture)
    assert out.signed_up[0] == 2 and out.first_export[0] == 1 and out.pct_activated[0] == 50.0

def test_feature_usage(fixture):
    out = run_sql(sql("feature_usage"), *fixture)
    assert out.feature[0] == "export_png" and out.uses[0] == 1

def test_time_to_value(fixture):
    out = run_sql(sql("time_to_value"), *fixture)
    row = out[out.experiment_group == "control"]
    assert row.median_days[0] == 2.0

def test_experiment_detects_lift_and_srm():
    rng = np.random.default_rng(0); n = 4000
    df = pd.DataFrame({"experiment_group": rng.choice(["control", "treatment"], n),
                       "pre_sessions": rng.poisson(2, n)})
    df["activated"] = (rng.random(n) < np.where(df.experiment_group == "treatment", .4, .25)).astype(int)
    r = analyse(df, n_boot=500)
    assert r["srm_ok"] and r["t_test_p"] < 0.01 and r["bootstrap_ci95"][0] > 0
    assert abs(r["cuped_diff"] - (r["rate_treatment"] - r["rate_control"])) < 0.05

def test_srm_flags_imbalance():
    df = pd.DataFrame({"experiment_group": ["control"] * 900 + ["treatment"] * 100,
                       "activated": [1] * 1000, "pre_sessions": [1] * 1000})
    assert not analyse(df, n_boot=50)["srm_ok"]
