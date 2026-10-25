"""Shared paths and helpers for the corpus_v2 labelling pipeline (stdlib only).

Nothing here writes to existing files. Label files are only ever created with mode 'x'.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]          # ~/Work/dsfe-revision
ANALYSIS = ROOT / 'analysis'
PIPE = ANALYSIS / 'labelling_pipeline'
CORPUS = ANALYSIS / 'corpus_v2'
CANDIDATES = CORPUS / 'candidates.json'
RUBRIC = ANALYSIS / 'LABELLING_RUBRIC_v2.md'
BENCH_FROZEN = ANALYSIS / 'benchmark_v2_frozen.json'   # 338 approved items
BENCH_ORIGINAL = ANALYSIS / 'benchmark_v2_final.json'  # all 359 original ids (plan.md amendment: keep excluding)
OVERRIDES = PIPE / 'overrides_log.jsonl'
BLIND_LOG = PIPE / 'blind_check_log.jsonl'

ALLOWED = ('positive', 'negative', 'neutral', 'exclude')
FIELDS_SHOWN = ['item', 'text', 'ticker', 'channel']
DEFAULT_LABELLER = 'OpenAI GPT-6 Astra (gpt-6-astra)'
DEFAULT_METHOD = 'in-context manual model judgement of each full text'
BLINDNESS = ('No candidate human labels, database sentiment predictions, FinBERT outputs, '
             'sampling group or lexicon-hit metadata shown.')

sys.dont_write_bytecode = True
sys.path.insert(0, str(ANALYSIS))
from corpus_rule import clean, normalise  # noqa: E402  same normalisation as T4 sampling


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidates():
    """Original 1,100-item pool followed by any extension draws (candidates_ext_NNNN.json, 23 Oct 2026)."""
    pool = json.loads(CANDIDATES.read_text())
    for ext in sorted(CORPUS.glob('candidates_ext_*.json')):
        pool += json.loads(ext.read_text())
    for i, row in enumerate(pool):
        if row['item'] != f'c2_{i:04d}':
            raise SystemExit(f'Pool order broken at index {i}: {row["item"]}')
    return pool


def benchmark_blocklist():
    """ids and normalised texts of the frozen 338 plus all 359 original benchmark items."""
    ids, texts = set(), set()
    frozen = json.loads(BENCH_FROZEN.read_text())['items']
    rows = list(frozen)
    if BENCH_ORIGINAL.exists():
        rows += json.loads(BENCH_ORIGINAL.read_text())
    for b in rows:
        ids.add(b['id'])
        ids.add(b.get('item'))
        texts.add(normalise(clean(b['text'])))
    ids.discard(None)
    return ids, texts


def label_dirs(extra=()):
    dirs = [CORPUS] + [Path(d) for d in extra if d]
    seen, out = set(), []
    for d in dirs:
        r = d.resolve()
        if r not in seen and r.exists():
            seen.add(r)
            out.append(d)
    return out


def load_labels(extra_dirs=()):
    """Return ({item: record+{'_file'}}, [(path, meta, n)]). Fails loudly on an item labelled twice."""
    labels, files = {}, []
    for d in label_dirs(extra_dirs):
        for p in sorted(Path(d).glob('labels_*.json')):
            data = json.loads(p.read_text())
            files.append((p, data.get('meta', {}), len(data['labels'])))
            for item, rec in data['labels'].items():
                if item in labels:
                    raise SystemExit(f'{item} labelled in both {labels[item]["_file"]} and {p}')
                labels[item] = dict(rec, _file=str(p), _labeller=data.get('meta', {}).get('labeller'))
    return labels, files


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path, record):
    with Path(path).open('a') as f:            # append-only; never rewritten
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def apply_overrides(labels, overrides):
    """Final label per item: model label, replaced by the latest 'override' entry. Computed on read."""
    final = {k: v['label'] for k, v in labels.items()}
    n = 0
    for o in overrides:
        if o.get('action') == 'override' and o['id'] in final:
            final[o['id']] = o['new']
            n += 1
    return final, n


def counter_str(c, keys=ALLOWED):
    return ' / '.join(f'{k} {c.get(k, 0)}' for k in keys)


def summary(labels):
    return Counter(v['label'] for v in labels.values())
