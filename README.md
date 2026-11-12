# Feature Adoption Analytics

Product analytics on synthetic SaaS usage data for a subscription
creative-software product: feature adoption, retention, churn modelling,
an onboarding A/B test, and weekly-active-subscription forecasting.
Python + SQL (DuckDB), scikit-learn, statsmodels. All data is synthetic.

## Design

```
 src/generate.py          sql/*.sql              src/churn.py
 (users, plans,      ┌──► feature_usage ──┐       (logreg vs RF,
  sessions, feature  │    funnel          │        time split, AUC,
  events, upgrades,  │    retention_cohorts        Brier, perm. imp.)
  cancels, A/B)      │    time_to_value   │
        │            │    weekly_active ──┼──► src/forecast.py
        ▼            │         │          │       (ETS vs seasonal
 data/users.parquet  │         ▼          │        naive, MAPE holdout)
 data/events.parquet └─► data/metrics/*.parquet
                              │
              src/experiment.py (t-test, bootstrap, CUPED, SRM)
                              │
                       src/dashboard.py ──► dashboard.html (static)
```

| Component | File | What it does |
|---|---|---|
| Generator | `src/generate.py` | 18 months (2024-01→2025-06), growth + weekly/annual seasonality, latent engagement drives sessions, feature choice, upgrades, churn; onboarding A/B from 2025-01 |
| SQL metrics | `sql/*.sql` + `src/run_sql.py` | DuckDB templates: feature usage/reach, adoption funnel, retention cohorts, time-to-value, weekly actives; each extracted to `data/metrics/<name>.parquet` |
| Churn model | `src/churn.py` | Features from first 30 days only; label = churn by day 90; train pre-2025 / test 2025+; logreg vs random forest, ROC AUC, Brier, permutation importance |
| Experiment | `src/experiment.py` | Activation (export ≤14d): Welch t-test, bootstrap 95% CI, CUPED, SRM chi-square vs 50/50 |
| Forecast | `src/forecast.py` | Weekly active paid subscriptions, last-12-week holdout, ETS (damped trend) or SARIMA vs seasonal-naive (lag 52), MAPE |
| Dashboard | `src/dashboard.py` | Static `dashboard.html` from metric parquets, no server |
| Tests | `tests/test_metrics.py` | Metric SQL on tiny fixtures + experiment lift/SRM detection |
| Benchmark | `bench.py` | Times the whole pipeline at `--scale small|full` |

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/generate.py --scale full     # ~2.2M events (small: 2500 users)
.venv/bin/python src/run_sql.py                   # SQL -> data/metrics/*.parquet
.venv/bin/python src/churn.py
.venv/bin/python src/experiment.py
.venv/bin/python src/forecast.py                  # --model sarima also available
.venv/bin/python src/dashboard.py                 # writes dashboard.html
```

## Test

```bash
.venv/bin/python -m pytest tests -q
# 5 passed
```

## Bench (laptop sanity runs, copied verbatim)

Full scale ran fine on a laptop, so no scale-down was needed for the bench;
`--scale small` (2,500 users, ~0.87M events, ~10s) is the quick check.

```
$ .venv/bin/python bench.py --scale full
users=6000 events=2213373 range=2024-01-01 00:12:14..2025-07-01 22:07:52
feature_usage: 9 rows | funnel: 1 | retention_cohorts: 172 | time_to_value: 3 | weekly_active: 79
logreg        {'roc_auc': 0.6283, 'brier': 0.0872, 'n_test': 3008}
random_forest {'roc_auc': 0.5177, 'brier': 0.115,  'n_test': 3008}
experiment: control 0.0995 vs treatment 0.1903 activation, t-test p≈1e-12,
  bootstrap CI95 [0.0661, 0.1163], CUPED diff 0.0909, SRM p=0.093 (ok)
forecast (ETS): {'mape_model_pct': 6.02, 'mape_seasonal_naive_pct': 72.89}
TOTAL 20.5s scale=full
```

## Findings (synthetic, expected by construction)

- `editor`/`templates` dominate usage; `analytics_view`/`scheduler` are niche.
- Treatment roughly doubles 14-day activation (faster time-to-value was
  built in); SRM check passes.
- Churn is only weakly predictable from 30-day behaviour: logreg AUC 0.63
  beats the random forest (0.52) — reported as-is, not tuned away.
- Seasonal naive is a poor baseline here because the series grows fast
  (lag-52 values are far below current levels); ETS MAPE 6.0% on holdout.

## Limitations

- Synthetic data: every "effect" (treatment lift, churn drivers, seasonality)
  was put in by the generator, so results validate the pipeline, not any
  real product claim.
- 18 months is under two annual cycles, so ETS uses damped trend only and
  cannot learn annual seasonality; SARIMA(52) is offered but equally starved.
- CUPED's covariate (sessions in first 2 days) is barely correlated with
  activation in this generator, so variance reduction is ~0% — honest null.
- Generator is a per-user Python loop: ~15s for 2.2M events. Fine here;
  a vectorised generator would be needed at 100M+ events.
- No hyperparameter search, no calibration curves beyond Brier score,
  no multiple-testing correction, dashboard is static tables (no charts).

## License

MIT, Copyright (c) 2026 Ayush Gawai. See `LICENSE`.
