"""A/B test: onboarding change. Outcome = activated (exported) within 14 days.
t-test, bootstrap CI, CUPED (pre-period covariate: signup-week engagement
proxy = sessions in first 2 days, measured before the new flow diverges),
and a sample-ratio-mismatch (SRM) chi-square check vs the 50/50 design.
"""
import numpy as np, pandas as pd
from scipy import stats

def build(data="data", seed=42):
    u = pd.read_parquet(f"{data}/users.parquet"); e = pd.read_parquet(f"{data}/events.parquet")
    exp = u[u.experiment_group.isin(["control", "treatment"])].copy()
    first_exp = e[e.feature.fillna("").str.startswith("export")].groupby("user_id").ts.min()
    fe = first_exp.rename("first_export").reset_index()
    exp = exp.merge(fe, on="user_id", how="left")
    exp["activated"] = ((exp.first_export - exp.signup_ts).dt.days <= 14).fillna(False).astype(int)
    early = e[(e.event_type == "session_start")].merge(exp[["user_id", "signup_ts"]], on="user_id")
    early = early[(early.ts - early.signup_ts).dt.days <= 2].groupby("user_id").size()
    exp["pre_sessions"] = exp.user_id.map(early).fillna(0)
    return exp

def analyse(df, seed=42, n_boot=2000):
    c, t = df[df.experiment_group == "control"], df[df.experiment_group == "treatment"]
    res = {"n_control": len(c), "n_treatment": len(t),
           "rate_control": round(c.activated.mean(), 4), "rate_treatment": round(t.activated.mean(), 4)}
    res["t_test_p"] = float(stats.ttest_ind(t.activated, c.activated, equal_var=False).pvalue)
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(t.activated, len(t)).mean() - rng.choice(c.activated, len(c)).mean()
             for _ in range(n_boot)]
    res["bootstrap_ci95"] = [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])]
    # CUPED
    theta = np.cov(df.activated, df.pre_sessions)[0, 1] / np.var(df.pre_sessions)
    adj = df.activated - theta * (df.pre_sessions - df.pre_sessions.mean())
    a_c, a_t = adj[df.experiment_group == "control"], adj[df.experiment_group == "treatment"]
    res["cuped_diff"] = round(float(a_t.mean() - a_c.mean()), 4)
    res["cuped_var_reduction_pct"] = round(100 * (1 - adj.var() / df.activated.var()), 2)
    # SRM: chi-square against 50/50
    obs = np.array([len(c), len(t)]); exp_n = obs.sum() / 2
    res["srm_chi2_p"] = float(stats.chisquare(obs, [exp_n, exp_n]).pvalue)
    res["srm_ok"] = bool(res["srm_chi2_p"] > 0.001)
    return res

if __name__ == "__main__":
    import json; print(json.dumps(analyse(build()), indent=2))
