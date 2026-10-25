"""Re-draw the LIME v2 figures from the saved lime_v2.json (no LIME rerun). Same layout as run_lime_v2.py,
but the title shows confidence to one decimal place so it matches the captions (e.g. 96.9%)."""
import json, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
HERE = Path(__file__).parent; out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
d = json.loads((HERE / "lime_v2.json").read_text())["stock"]
for k, r in d.items():
    w = r["weights"]; s = r["text"]; fig, ax = plt.subplots(figsize=(8, 4.2)); ww = w[::-1]
    ax.barh([t for t, _ in ww], [v for _, v in ww], color=["#2ca02c" if v > 0 else "#d62728" for _, v in ww])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("LIME feature weight")
    short = s if len(s) <= 72 else s[:69] + "..."
    ax.set_title(f"Prediction: {r['pred'].upper()} | confidence {r['confidence']*100:.1f}%\n\"{short}\"", fontsize=10)
    ax.legend(handles=[Patch(color="#2ca02c", label="Supports predicted class"), Patch(color="#d62728", label="Contradicts predicted class")], loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"lime_{k}.pdf"); plt.close(fig); print(k, f"{r['confidence']*100:.1f}%")
