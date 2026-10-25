"""Resumable driver: stock FinBERT baseline, then 3 arms x 5 seeds (plan.md T8/T10).

Each run writes <out>/runs/<name>.json atomically; existing files are skipped on
restart. <out>/progress.json is rewritten after each run (done/total, per-run
wall time, ETA from mean fine-tune run time).
Usage: .venv311/bin/python analysis/finetune_v1/run_all.py --out analysis/results_v1
"""
from __future__ import annotations
import argparse, datetime as dt, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_runner as fr

ARMS = ["arm_a", "arm_b", "arm_c"]
SEEDS = [13, 21, 42, 84, 168]


def now(): return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--arms", nargs="*", default=ARMS); ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--no-stock", action="store_true"); ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--arms-dir", default=None, help="default analysis/training_sets_v1")
    a = ap.parse_args()
    out = Path(a.out); (out / "runs").mkdir(parents=True, exist_ok=True)
    jobs = ([] if a.no_stock else [("stock", None, None)]) + [(f"{arm}_seed{s}", arm, s) for arm in a.arms for s in a.seeds]
    prog_path = out / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"runs": {}}
    prog.update({"started_or_resumed": now(), "total_finetune_runs": sum(1 for j in jobs if j[1]),
                 "epochs": a.epochs, "arms_dir": a.arms_dir, "pid": __import__("os").getpid()})

    def write_prog(current=None):
        done = [n for n, arm, _ in jobs if arm and (out / "runs" / f"{n}.json").exists()]
        times = [prog["runs"][n]["wall_seconds"] for n in done if n in prog["runs"]]
        remaining = prog["total_finetune_runs"] - len(done)
        mean = sum(times) / len(times) if times else None
        prog.update({"updated": now(), "finetune_runs_done": len(done), "current": current,
                     "stock_done": (out / "runs" / "stock.json").exists(),
                     "mean_run_seconds": round(mean, 1) if mean else None,
                     "eta_seconds": round(mean * remaining) if mean else None,
                     "eta_finish": (dt.datetime.now().astimezone() + dt.timedelta(seconds=mean * remaining)).isoformat(timespec="seconds") if mean else None})
        prog_path.write_text(json.dumps(prog, indent=1))

    for name, arm, seed in jobs:
        p = out / "runs" / f"{name}.json"
        if p.exists():
            print(f"{now()} skip {name} (exists)", flush=True); continue
        write_prog(current=name); print(f"{now()} start {name}", flush=True)
        try:
            r = fr.run_stock(p, threads=a.threads) if arm is None else fr.run_condition(arm, seed, p, epochs=a.epochs, threads=a.threads, arms_dir=a.arms_dir)
        except Exception:
            traceback.print_exc(); prog.setdefault("errors", []).append({"run": name, "at": now(), "error": traceback.format_exc()[-2000:]})
            write_prog(); continue
        prog["runs"][name] = {"finished": now(), "wall_seconds": r["wall_seconds"], "best_epoch": r.get("best_epoch"),
                              "val_macro_f1": r.get("val", {}).get("macro_f1"), "bench_macro_f1": r["benchmark"]["macro_f1"],
                              "bench_accuracy": r["benchmark"]["accuracy"]}
        print(f"{now()} done {name} {r['wall_seconds']}s bench_macroF1={r['benchmark']['macro_f1']:.4f}", flush=True)
        write_prog()
    write_prog(); print(f"{now()} ALL DONE", flush=True)


if __name__ == "__main__":
    main()
