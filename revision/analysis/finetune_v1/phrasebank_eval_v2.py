"""Financial PhraseBank evaluation (plan.md T12) for stock FinBERT and v2 fine-tuned arms.

Loader fix: the HF `financial_phrasebank` loading script is unsupported by current `datasets` (and
`datasets` is not installed in .venv311), so this reads the local official zip extract
external/phrasebank/FinancialPhraseBank-v1.0/Sentences_*.txt (ISO-8859-1, "sentence@label", CR/CRLF).
Model checkpoints were not saved by the v2 run, so for each arm the seed with the best *validation*
macro-F1 is retrained with identical code/seed (keep_best_model=True); reproduction is verified by
comparing the 338 benchmark predictions with the saved run JSON.
CAVEAT: ProsusAI/finbert was fine-tuned on Financial PhraseBank, so stock scores are inflated/in-sample.
"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_runner as fr

ROOT = fr.ROOT
PB = ROOT / "external/phrasebank/FinancialPhraseBank-v1.0"
OUT = ROOT / "analysis/results_v2/phrasebank"
RUNS = ROOT / "analysis/results_v2/runs"
CONFIGS = {"sentences_allagree": "Sentences_AllAgree.txt", "sentences_50agree": "Sentences_50Agree.txt"}


def load_pb(fname):
    raw = (PB / fname).read_bytes().decode("latin-1")
    rows = []
    for line in re.split(r"\r\n|\r|\n", raw):
        if not line.strip():
            continue
        text, lab = line.rsplit("@", 1)
        lab = lab.strip().lower(); assert lab in fr.LABELS, lab
        rows.append({"text": text.strip(), "label": lab})
    return rows


def evaluate(model, tok, data):
    out = {}
    for cfg, rows in data.items():
        lg = fr.predict(model, tok, [r["text"] for r in rows], 128, 64)
        preds = [fr.LABELS[i] for i in lg.argmax(1)]
        out[cfg] = {"metrics": fr.metrics([r["label"] for r in rows], preds),
                    "predictions": [{"i": i, "gold": r["label"], "pred": p, "logits": [round(float(x), 6) for x in l]}
                                    for i, (r, p, l) in enumerate(zip(rows, preds, lg))]}
    return out


def main():
    t0 = time.time()
    data = {c: load_pb(f) for c, f in CONFIGS.items()}
    info = {c: {"n": len(r), "n_unique_text": len({x["text"] for x in r}),
                "label_counts": {l: sum(x["label"] == l for x in r) for l in fr.LABELS}} for c, r in data.items()}
    print("loaded", json.dumps(info), flush=True)
    summary = {"data": info, "max_length": 128, "models": {}}
    jobs = [("stock", None, None)]
    for arm in ("arm_a", "arm_b", "arm_c"):
        runs = [json.loads((RUNS / f"{arm}_seed{s}.json").read_text()) for s in (13, 21, 42, 84, 168)]
        best = max(runs, key=lambda r: (r["val"]["macro_f1"], -r["seed"]))
        jobs.append((arm, best["seed"], best))
    for name, seed, saved in jobs:
        p = OUT / f"{name if seed is None else f'{name}_seed{seed}'}.json"
        if p.exists():
            summary["models"][p.stem] = json.loads(p.read_text())["summary"]; print("skip", p.name, flush=True); continue
        ts = time.time()
        if seed is None:
            tok, model = fr.load_model(); extra = {}
        else:
            res, model, tok = fr.run_condition(name, seed, OUT / "retrain" / f"{name}_seed{seed}.json", epochs=3,
                                               arms_dir=ROOT / "analysis/training_sets_v2", keep_best_model=True)
            same = [a["pred"] for a in res["benchmark_predictions"]] == [b["pred"] for b in saved["benchmark_predictions"]]
            extra = {"selected_by": "best validation macro-F1 over 5 seeds", "seed": seed, "best_epoch": res["best_epoch"],
                     "retrain_reproduces_saved_benchmark_preds": same,
                     "retrain_bench_macro_f1": res["benchmark"]["macro_f1"], "saved_bench_macro_f1": saved["benchmark"]["macro_f1"]}
        ev = evaluate(model, tok, data)
        summ = {**extra, **{c: {k: ev[c]["metrics"][k] for k in ("accuracy", "macro_f1", "pred_counts")} |
                {"recall": {l: v["recall"] for l, v in ev[c]["metrics"]["per_class"].items()}} for c in ev},
                "wall_seconds": round(time.time() - ts, 1)}
        p.write_text(json.dumps({"summary": summ, "results": ev}, indent=1))
        summary["models"][p.stem] = summ
        print("done", p.stem, json.dumps({c: round(summ[c]["macro_f1"], 4) for c in ev}), extra.get("retrain_reproduces_saved_benchmark_preds"), summ["wall_seconds"], flush=True)
    summary["caveat"] = "ProsusAI/finbert was fine-tuned on Financial PhraseBank; stock results are in-sample and inflated."
    summary["wall_seconds_total_this_invocation"] = round(time.time() - t0, 1)
    (OUT / "phrasebank_summary.json").write_text(json.dumps(summary, indent=1))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
