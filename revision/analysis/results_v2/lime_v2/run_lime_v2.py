"""LIME v2 for the manuscript (stock ProsusAI/finbert, CPU). Mirrors backend/app/explainability/lime_explainer.py:
LimeTextExplainer(class_names=[positive,negative,neutral], split_expression=r"\s+", bow=False, random_state=42),
num_samples=5000, num_features=8, explaining the predicted class. Also records predictions of saved checkpoints."""
import json, re, numpy as np, torch, lime, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.manual_seed(0); torch.set_num_threads(8)
HERE = Path(__file__).parent; ROOT = HERE.parents[2]
LABELS = ["positive", "negative", "neutral"]; NS, NF = 5000, 8
S = {"positive": "NVIDIA shares surged 12% after the company raised its full-year guidance significantly.",
     "negative": "Intel reported a massive quarterly loss of $7 billion and announced 15,000 layoffs.",
     "neutral": "The Federal Reserve kept interest rates unchanged, maintaining its cautious outlook on inflation."}
def loader(path):
    tok = AutoTokenizer.from_pretrained(path); m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    order = [ {v.lower(): int(k) for k, v in m.config.id2label.items()}[l] for l in LABELS]
    def proba(texts):
        texts = [t if (t and t.strip()) else "neutral" for t in texts]; out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad(): p = torch.softmax(m(**enc).logits, -1).numpy()
            out.append(p[:, order])
        return np.vstack(out)
    return proba
res = {"settings": dict(model="ProsusAI/finbert", num_samples=NS, num_features=NF, random_state=42, split_expression=r"\s+", bow=False,
       class_order=LABELS, lime_version=getattr(lime, "__version__", "0.2.0.1"), torch=torch.__version__), "stock": {}, "checkpoints": {}}
stock = loader("ProsusAI/finbert")
for k, s in S.items():
    p = stock([s])[0]; idx = int(p.argmax())
    exp = LimeTextExplainer(class_names=LABELS, split_expression=r"\s+", bow=False, random_state=42).explain_instance(
        s, stock, num_features=NF, num_samples=NS, labels=(idx,))
    w = exp.as_list(label=idx)
    res["stock"][k] = dict(text=s, probs=dict(zip(LABELS, map(float, p))), pred=LABELS[idx], confidence=float(p[idx]), weights=w)
    fig, ax = plt.subplots(figsize=(8, 4.2)); ww = w[::-1]
    ax.barh([t for t, _ in ww], [v for _, v in ww], color=["#2ca02c" if v > 0 else "#d62728" for _, v in ww])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("LIME feature weight")
    short = s if len(s) <= 72 else s[:69] + "..."
    ax.set_title(f"Prediction: {LABELS[idx].upper()} | confidence {p[idx]*100:.0f}%\n\"{short}\"", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ca02c", label="Supports predicted class"), Patch(color="#d62728", label="Contradicts predicted class")], loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / f"lime_{k}.pdf"); plt.close(fig)
CK = ROOT / "analysis/results_v2/ood_twitterfin/checkpoints"
for name in ["arm_a_seed84", "arm_b_seed84", "arm_c_seed21"]:
    f = loader(str(CK / name)); res["checkpoints"][name] = {}
    for k, s in S.items():
        p = f([s])[0]; res["checkpoints"][name][k] = dict(probs=dict(zip(LABELS, map(float, p))), pred=LABELS[int(p.argmax())], confidence=float(p.max()))
(HERE / "lime_v2.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
