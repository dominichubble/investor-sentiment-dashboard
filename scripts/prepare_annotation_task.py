"""Extract the 200 benchmark texts into a JSON file (no labels) so a second
annotator can label them independently."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.evaluation.labeled_dataset import get_labeled_dataset  # noqa: E402


def main() -> None:
    ds = get_labeled_dataset()
    out = [{"idx": i, "text": row["text"]} for i, row in enumerate(ds)]
    out_path = REPO_ROOT / "backend" / "data" / "evaluation" / "annotation_task.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {len(out)} texts to {out_path}")


if __name__ == "__main__":
    main()
