import json, sys
# usage: save_batch.py START END p="1,2" n="3" x="4"  (everything else in range = neutral)
a, b = int(sys.argv[1]), int(sys.argv[2])
sets = {k: {int(i) for i in v.split(",") if i} for k, v in (arg.split("=") for arg in sys.argv[3:])}
name = {"p": "positive", "n": "negative", "x": "exclude"}
out = {}
for i in range(a, b):
    lab = next((name[k] for k, s in sets.items() if i in s), "neutral")
    out[f"b2_{i:03d}"] = lab
json.dump(out, open(f"analysis/labels/llm_{a:03d}_{b-1:03d}.json", "w"), indent=0)
from collections import Counter; print(Counter(out.values()))
