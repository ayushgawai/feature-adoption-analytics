"""Weekly active subscription forecast: ETS (statsmodels) vs seasonal-naive.
Holdout = last 12 weeks; metric = MAPE. SARIMA available via --model sarima.
"""
import argparse, warnings, pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def mape(a, f): return float((abs(a - f) / a).mean() * 100)

def forecast(metrics="data/metrics", model="ets", holdout=12):
    df = pd.read_parquet(f"{metrics}/weekly_active.parquet")
    s = df.iloc[:-1].set_index("week")["was"]  # drop partial final week
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    naive = pd.Series(s.iloc[-holdout - 52:-52].values, index=test.index) if len(s) >= holdout + 52 \
        else pd.Series([train.iloc[-1]] * holdout, index=test.index)
    if model == "sarima":
        pred = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 52)).fit(disp=False).forecast(holdout)
    else:
        # ponytail: 18 months < 2 annual cycles, so ETS gets trend only;
        # annual seasonality is exactly what the seasonal-naive baseline carries.
        pred = ExponentialSmoothing(train, trend="add", damped_trend=True).fit().forecast(holdout)
    r = {"model": model, "holdout_weeks": holdout,
         "mape_model_pct": round(mape(test, pred), 2), "mape_seasonal_naive_pct": round(mape(test, naive), 2)}
    print(r); return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="ets", choices=["ets", "sarima"])
    forecast(model=ap.parse_args().model)
