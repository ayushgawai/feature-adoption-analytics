"""Run every sql/*.sql template in DuckDB, extract each result to parquet."""
import pathlib, duckdb

def run(data="data", out="data/metrics", sql_dir="sql"):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW users AS SELECT * FROM '{data}/users.parquet'")
    con.execute(f"CREATE VIEW events AS SELECT * FROM '{data}/events.parquet'")
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    results = {}
    for f in sorted(pathlib.Path(sql_dir).glob("*.sql")):
        df = con.execute(f.read_text()).df()
        df.to_parquet(f"{out}/{f.stem}.parquet", index=False)
        results[f.stem] = df
        print(f"{f.stem}: {len(df)} rows -> {out}/{f.stem}.parquet")
    return results

if __name__ == "__main__":
    run()
