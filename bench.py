"""Benchmark / simulation: time the full pipeline at a given scale, print a summary."""
import argparse, subprocess, sys, time

STEPS = [["src/generate.py"], ["src/run_sql.py"], ["src/churn.py"], ["src/experiment.py"], ["src/forecast.py"]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scale", default="small", choices=["small", "full"])
    a = ap.parse_args(); t0 = time.time()
    for s in STEPS:
        t = time.time()
        cmd = [sys.executable] + s + (["--scale", a.scale] if "generate" in s[0] else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(f"$ {' '.join(cmd)}\n{out.stdout.strip()}\n[{time.time()-t:.1f}s]")
        if out.returncode: print(out.stderr[-2000:]); sys.exit(1)
    print(f"TOTAL {time.time()-t0:.1f}s scale={a.scale}")
