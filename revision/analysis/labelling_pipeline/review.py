#!/usr/bin/env python3
"""Terminal review of model labels (stdlib only). Never edits labels_*.json.

  review.py               accept / override / skip model labels; writes to overrides_log.jsonl (append-only)
  review.py --blind       Dom's 30-item blind neutral check: model label hidden; answers go to blind_check_log.jsonl
  review.py --report      agreement report for the blind check (no prompts)

Final labels = model labels with the latest 'override' entry per item applied, computed on read (common.apply_overrides).
Answers are read from stdin, so a file can be piped in for non-interactive use.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from datetime import datetime, timezone
import random
import textwrap

import common as C

KEYS = {'p': 'positive', 'n': 'neutral', 'g': 'negative', 'x': 'exclude'}
BLIND_SEED = 20261023
BLIND_N = 30


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def ask(prompt):
    print(prompt, end='', flush=True)
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    if not sys.stdin.isatty():
        print(line.rstrip())
    return line.strip()


def show(i, total, row, label=None, rationale=None):
    print(f'\n[{i}/{total}] {row["item"]}  ({row["channel"]}, {row["ticker"]})')
    print(textwrap.fill(row['text'], 100, initial_indent='  ', subsequent_indent='  '))
    if label is not None:
        print(f'  model: {label}' + (f'  - {rationale}' if rationale else ''))


def blind_sample(labels, n=BLIND_N, seed=BLIND_SEED):
    """n model-neutral items, fixed seed, sampled from items sorted by id (deterministic for a given label set)."""
    neutrals = sorted(k for k, v in labels.items() if v['label'] == 'neutral')
    return sorted(random.Random(seed).sample(neutrals, min(n, len(neutrals))))


def run_review(a, pool, labels, log):
    overrides = C.read_jsonl(log)
    done = {o['id'] for o in overrides}
    final, _ = C.apply_overrides(labels, overrides)
    todo = [r for r in pool if r['item'] in labels and (a.all or r['item'] not in done)
            and (not a.label or final[r['item']] == a.label)]
    print(f'{len(todo)} items to review (log: {log}). Keys: [a]ccept, [o]verride, [s]kip, [q]uit')
    n = Counter()
    try:
        for i, r in enumerate(todo, 1):
            rec = labels[r['item']]
            show(i, len(todo), r, final[r['item']], rec.get('rationale'))
            choice = ask('  a/o/s/q > ').lower()
            if choice == 'q':
                break
            if choice == 'a':
                C.append_jsonl(log, dict(timestamp=now(), id=r['item'], action='accept', old=final[r['item']],
                                         new=final[r['item']], note='', reviewer=a.reviewer))
                n['accept'] += 1
            elif choice == 'o':
                new = KEYS.get(ask('  new label [p]os/[n]eu/ne[g]/e[x]clude > ').lower())
                if not new:
                    print('  not a label; skipped'); n['skip'] += 1; continue
                note = ask('  note (optional) > ')
                C.append_jsonl(log, dict(timestamp=now(), id=r['item'], action='override', old=final[r['item']],
                                         new=new, note=note, reviewer=a.reviewer))
                n['override'] += 1
            else:
                n['skip'] += 1
    except EOFError:
        print('\n(end of input)')
    print(f'\nsession: accept {n["accept"]}, override {n["override"]}, skip {n["skip"]}')
    return 0


def get_sample(a, labels, create):
    """The 30 items are frozen in a sample file the first time --blind runs, so later labelling cannot change them."""
    import json
    from pathlib import Path
    path = Path(a.sample_file) if a.sample_file else C.PIPE / 'blind_check_sample.json'
    if path.exists():
        return json.loads(path.read_text())['items'], path
    sample = blind_sample(labels, a.n, a.seed)
    if create:
        with path.open('x') as f:
            json.dump(dict(created=now(), seed=a.seed, n=a.n, drawn_from_model_neutrals=sum(
                v['label'] == 'neutral' for v in labels.values()), items=sample), f, indent=1)
            f.write('\n')
        print(f'Froze blind sample in {path}')
    return sample, None


def run_blind(a, pool, labels, log):
    rows = {r['item']: r for r in pool}
    sample, _ = get_sample(a, labels, create=True)
    answered = {o['id'] for o in C.read_jsonl(log)}
    todo = [s for s in sample if s not in answered]
    print(f'Blind neutral check: {len(sample)} items (seed {a.seed}), {len(todo)} still to answer. Model labels hidden.')
    print('Keys: [p]ositive [n]eutral ne[g]ative e[x]clude [s]kip [q]uit')
    try:
        for i, item in enumerate(todo, 1):
            show(i, len(todo), rows[item])
            k = ask('  p/n/g/x/s/q > ').lower()
            if k == 'q':
                break
            if k not in KEYS:
                continue
            C.append_jsonl(log, dict(timestamp=now(), id=item, action='blind_check', human=KEYS[k],
                                     seed=a.seed, n=a.n, reviewer=a.reviewer))
    except EOFError:
        print('\n(end of input)')
    return run_report(a, pool, labels, log)


def run_report(a, pool, labels, log):
    sample, path = get_sample(a, labels, create=False)
    if path is None:
        print('(no frozen sample file yet; showing what the sample would be from current labels)')
    sample = set(sample)
    ans = {}
    for o in C.read_jsonl(log):
        if o.get('action') == 'blind_check' and o['id'] in sample:
            ans[o['id']] = o['human']               # latest answer wins
    c = Counter(ans.values())
    judged = sum(v for k, v in c.items() if k != 'exclude')
    print(f'\nBlind check report: answered {len(ans)}/{len(sample)} (log {log})')
    print('  human labels on model-neutral items:', C.counter_str(c))
    if judged:
        print(f'  neutral precision (human neutral / human judged, excludes removed): '
              f'{c["neutral"]}/{judged} = {c["neutral"] / judged:.3f}')
    if ans:
        print(f'  agreement incl. excludes as disagreement: {c["neutral"]}/{len(ans)} = {c["neutral"] / len(ans):.3f}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--blind', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--labels-dir', action='append', default=[], help='extra folder with labels_*.json (tests)')
    ap.add_argument('--log', help='override log path (default overrides_log.jsonl / blind_check_log.jsonl)')
    ap.add_argument('--label', choices=C.ALLOWED, help='review only items whose current label is this')
    ap.add_argument('--all', action='store_true', help='include items already reviewed')
    ap.add_argument('--n', type=int, default=BLIND_N)
    ap.add_argument('--seed', type=int, default=BLIND_SEED)
    ap.add_argument('--reviewer', default='Dom')
    ap.add_argument('--sample-file', help='frozen blind sample (default blind_check_sample.json)')
    ap.add_argument('--only-extra', action='store_true', help='ignore analysis/corpus_v2 labels (fixtures)')
    a = ap.parse_args(argv)
    pool = C.load_candidates()
    if a.only_extra:
        labels = {}
        import json
        from pathlib import Path
        for d in a.labels_dir:
            for p in sorted(Path(d).glob('labels_*.json')):
                labels.update(json.loads(p.read_text())['labels'])
    else:
        labels, _ = C.load_labels(a.labels_dir)
    if a.blind or a.report:
        log = a.log or C.BLIND_LOG
        return (run_report if a.report else run_blind)(a, pool, labels, log)
    return run_review(a, pool, labels, a.log or C.OVERRIDES)


if __name__ == '__main__':
    sys.exit(main())
