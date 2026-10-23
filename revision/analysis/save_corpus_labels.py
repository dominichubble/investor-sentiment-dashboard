"""Save explicit in-context judgements; never infer or default missing labels.

Usage: python analysis/save_corpus_labels.py START < judgements.txt
Each line: c2_0000 label low_information(0/1) rationale
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'analysis/corpus_v2'
start = int(sys.argv[1])
source = base / f'blind_{start:04d}.json'
items = json.loads(source.read_text())
labels = {}
for line in sys.stdin:
    if not line.strip():
        continue
    item, label, flag, reason = line.strip().split(maxsplit=3)
    assert item not in labels, 'Duplicate judgement'
    assert label in ('positive', 'negative', 'neutral', 'exclude')
    assert flag in ('0', '1')
    assert flag == '0' or label == 'neutral'
    labels[item] = dict(label=label, low_information=flag == '1', rationale=reason)
assert set(labels) == {x['item'] for x in items}, 'Missing or extra judgements'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out = dict(meta=dict(labeller='OpenAI GPT-6 Astra (gpt-6-astra)', reasoning_effort='high',
                    method='in-context manual model judgement of each full text',
                    rubric='v2', rubric_sha256=sha(ROOT / 'analysis/LABELLING_RUBRIC_v2.md'),
                    date=datetime.now(timezone.utc).date().isoformat(),
                    input_sha256=sha(source),
                    fields_shown=['item', 'text', 'ticker', 'channel'],
                    blindness='No candidate human labels, database sentiment predictions, FinBERT outputs, sampling group or lexicon-hit metadata shown.'),
           labels=labels)
target = base / f'labels_{start:04d}.json'
with target.open('x') as f:
    json.dump(out, f, indent=1)
    f.write('\n')
target.chmod(0o444)
print(f'Saved {len(labels)} explicit judgements to {target.name}')
