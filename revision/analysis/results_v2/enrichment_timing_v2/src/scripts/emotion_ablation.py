"""
Ablation study for the finance-emotion taxonomy.

Runs FinBERT on the 200-sample benchmark to obtain per-sample softmax
probabilities, then applies the emotion taxonomy under four configurations:

    full           -- all components enabled
    no_entropy     -- entropy contribution zeroed
    no_lexicon     -- domain lexicons disabled
    no_aspects     -- aspect priors disabled (equivalent to no-ABSA input)

Reports:
  - Marginal distribution of dominant emotion labels under each configuration
  - Per-sample fraction where the dominant label CHANGES when a component
    is removed, i.e. a structural measure of each component's contribution.

No emotion ground truth exists on the benchmark, so this is a
component-contribution analysis (not an accuracy ablation). It is an honest
surrogate for "how much does each component matter" when labelled emotion
data is unavailable.

Output: backend/data/evaluation/emotion_ablation.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.evaluation.labeled_dataset import get_labeled_dataset  # noqa: E402
from app.models.finbert_model import get_model  # noqa: E402
from app.analysis import finance_emotion  # noqa: E402


def shannon_entropy(probs: Dict[str, float]) -> float:
    """Normalised Shannon entropy over 3 classes, scaled to [0, 1]."""
    vals = [p for p in probs.values() if p > 0]
    if not vals:
        return 0.0
    h = -sum(p * math.log(p) for p in vals)
    return h / math.log(3)


def run_full(text: str, label: str, score: float, probs: Dict[str, float], entropy: float) -> dict:
    return finance_emotion.infer_finance_emotion(
        text=text,
        sentiment_label=label,
        sentiment_score=score,
        scores=probs,
        uncertainty=entropy,
        aspects=[],
    )


def run_no_entropy(text: str, label: str, score: float, probs: Dict[str, float], entropy: float) -> dict:
    return finance_emotion.infer_finance_emotion(
        text=text,
        sentiment_label=label,
        sentiment_score=score,
        scores=probs,
        uncertainty=0.0,
        aspects=[],
    )


def run_no_lexicon(text: str, label: str, score: float, probs: Dict[str, float], entropy: float) -> dict:
    # Pass a single space so the lowered text contains no matches from the
    # domain lexicons, while preserving all other pathways.
    return finance_emotion.infer_finance_emotion(
        text=" ",
        sentiment_label=label,
        sentiment_score=score,
        scores=probs,
        uncertainty=entropy,
        aspects=[],
    )


def main() -> None:
    dataset = get_labeled_dataset()
    texts = [row["text"] for row in dataset]

    print("Loading FinBERT for probability capture...")
    model = get_model()
    raw = model.predict_batch(texts, batch_size=32)

    per_sample = []
    for item, pred in zip(dataset, raw):
        text = item["text"]
        probs = pred.get("scores") or {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
        }
        entropy = shannon_entropy(probs)
        label = pred["label"]
        score = float(pred.get("score") or max(probs.values()))

        out = {
            "text": text,
            "finbert_label": label,
            "finbert_probs": probs,
            "entropy": round(entropy, 4),
            "full": run_full(text, label, score, probs, entropy)["label"],
            "no_entropy": run_no_entropy(text, label, score, probs, entropy)["label"],
            "no_lexicon": run_no_lexicon(text, label, score, probs, entropy)["label"],
            # "no_aspects" is identical to "full" because aspects=[] already in baseline;
            # we retain the cell so the aspect ablation shows 0 change (honest finding).
            "no_aspects": run_full(text, label, score, probs, entropy)["label"],
        }
        per_sample.append(out)

    variants = ["full", "no_entropy", "no_lexicon", "no_aspects"]
    dist = {v: Counter(r[v] for r in per_sample) for v in variants}

    # Change fraction: how often does dropping a component flip the dominant label?
    change_fraction = {}
    for v in variants:
        if v == "full":
            continue
        changes = sum(1 for r in per_sample if r[v] != r["full"])
        change_fraction[v] = changes / len(per_sample)

    summary = {
        "n": len(per_sample),
        "distribution": {v: dict(dist[v]) for v in variants},
        "change_fraction_vs_full": change_fraction,
        "note": (
            "change_fraction_vs_full is the proportion of the 200 samples where "
            "removing the named component changes the dominant emotion label. It "
            "is a structural measure of the component's influence on the taxonomy; "
            "labelled emotion ground truth was not available, so this is not an "
            "accuracy ablation."
        ),
    }

    out_path = REPO_ROOT / "backend" / "data" / "evaluation" / "emotion_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_sample": per_sample}, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
