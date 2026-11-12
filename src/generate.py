"""Synthetic event-log generator for a subscription creative-software product.

18 months (2024-01-01..2025-06-30), seasonality (weekly + annual), growth,
plans, sessions, feature events, upgrades, cancellations, and an onboarding
A/B test (from 2025-01-01, treatment activates faster / churns less).
Full scale targets ~2M events; --scale small is the laptop sanity size.
"""
import argparse
import numpy as np
import pandas as pd

FEATURES = ["editor", "templates", "export_png", "export_video",
            "ai_assist", "collab", "brand_kit", "scheduler", "analytics_view"]
POP = np.array([.30, .18, .14, .08, .10, .06, .05, .05, .04]); POP /= POP.sum()
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30")


def generate(n_users=12000, seed=42):
    rng = np.random.default_rng(seed)
    days = (END - START).days + 1
    # signup dates: growth + annual seasonality + weekday effect
    t = np.arange(days)
    w = (1 + 0.004 * t) * (1 + 0.35 * np.sin(2 * np.pi * t / 365.25))
    w *= [0.75 if (START + pd.Timedelta(days=int(d))).weekday() >= 5 else 1.1 for d in t]
    signup = START + pd.to_timedelta(rng.choice(days, n_users, p=w / w.sum()), unit="D")
    signup = signup + pd.to_timedelta(rng.integers(0, 86400, n_users), unit="s")
    plan0 = rng.choice(["free", "pro", "team"], n_users, p=[.62, .28, .10])
    engagement = rng.beta(2, 3, n_users)                      # latent stickiness
    exp_start = pd.Timestamp("2025-01-01")
    group = np.where(signup < exp_start, "pre",
                     rng.choice(["control", "treatment"], n_users))
    # lifetime: churn hazard falls with engagement; treatment helps a little
    hazard = 0.16 * np.exp(-3.0 * engagement) * np.where(group == "treatment", 0.8, 1.0)
    lifetime = rng.exponential(1 / np.clip(hazard, .01, None))  # months
    churn_dt = signup + pd.to_timedelta(lifetime * 30.4, unit="D")
    cancelled = churn_dt <= END
    churn_s = pd.Series(pd.DatetimeIndex(churn_dt)); churn_s[~cancelled] = pd.NaT
    churn_dt = churn_s
    active_end = churn_s.fillna(END).reset_index(drop=True)

    users = pd.DataFrame({"user_id": np.arange(1, n_users + 1), "signup_ts": signup,
                          "plan_initial": plan0, "experiment_group": group,
                          "churn_ts": churn_dt, "engagement": engagement.round(4)})

    rows = []  # (user_id, ts, event_type, feature, plan)
    for uid, su, ae, eng, grp in zip(users.user_id, signup, active_end,
                                     engagement, group):
        months = max(1, int((ae - su).days // 30) + 1)
        # time-to-value: days to first export; treatment is faster
        ttv = rng.gamma(2, 14 if grp == "treatment" else 24) * (1.4 - eng)
        n_sess = rng.poisson((2 + 22 * eng) * months *
                             (1 + 0.3 * np.sin(2 * np.pi * su.dayofyear / 365.25)))
        if n_sess == 0:
            rows.append((uid, su, "signup", None, None)); continue
        span = max(1, (ae - su).days)
        sess = su + pd.to_timedelta(np.sort(rng.integers(0, span * 86400, n_sess)), unit="s")
        rows.append((uid, su, "signup", None, None))
        aff = POP * rng.lognormal(0, .7, len(FEATURES)); aff /= aff.sum()
        first_export_done = False
        for s in sess:
            rows.append((uid, s, "session_start", None, None))
            for _ in range(rng.integers(2, 7)):
                f = FEATURES[rng.choice(len(FEATURES), p=aff)]
                if f.startswith("export") and s < su + pd.Timedelta(days=float(ttv)):
                    continue  # not yet activated: no exports before time-to-value
                rows.append((uid, s + pd.Timedelta(minutes=int(rng.integers(1, 40))),
                             "feature_use", f, None))
                if not first_export_done and f.startswith("export") and s >= su + pd.Timedelta(days=float(ttv)):
                    first_export_done = True
        if rng.random() < .18 + .3 * eng:  # upgrade
            rows.append((uid, su + (ae - su) * rng.random(), "upgrade", None,
                         rng.choice(["pro", "team"])))
        if not pd.isna(users.loc[uid - 1, "churn_ts"]):
            rows.append((uid, ae, "cancel", None, None))
    ev = pd.DataFrame(rows, columns=["user_id", "ts", "event_type", "feature", "plan_to"])
    ev["plan"] = ev.user_id.map(dict(zip(users.user_id, plan0)))
    return users, ev.sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scale", choices=["small", "full"], default="small")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    n = 2500 if a.scale == "small" else 6000  # full ≈ 2M events at ~350 events/user
    u, e = generate(n, a.seed)
    import pathlib; pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    u.to_parquet(f"{a.out}/users.parquet", index=False); e.to_parquet(f"{a.out}/events.parquet", index=False)
    print(f"users={len(u)} events={len(e)} range={e.ts.min()}..{e.ts.max()}")
