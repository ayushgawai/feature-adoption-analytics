"""Static HTML dashboard from data/metrics/*.parquet (no server needed)."""
import pandas as pd, pathlib

def table(name):
    p = pathlib.Path(f"data/metrics/{name}.parquet")
    if not p.exists(): return f"<p>missing {name} — run src/run_sql.py</p>"
    return pd.read_parquet(p).head(50).to_html(index=False, border=0)

if __name__ == "__main__":
    html = f"""<!doctype html><meta charset="utf-8"><title>Feature Adoption Analytics</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1000px}}table{{border-collapse:collapse;margin-bottom:2rem}}td,th{{padding:4px 10px;border-bottom:1px solid #ccc;text-align:right}}</style>
<h1>Feature Adoption Analytics</h1>
<h2>Feature usage</h2>{table('feature_usage')}
<h2>Adoption funnel</h2>{table('funnel')}
<h2>Time to value</h2>{table('time_to_value')}
<h2>Weekly active subscriptions</h2>{table('weekly_active')}
<p>Synthetic data. See README for churn / experiment / forecast results.</p>"""
    pathlib.Path("dashboard.html").write_text(html); print("wrote dashboard.html")
