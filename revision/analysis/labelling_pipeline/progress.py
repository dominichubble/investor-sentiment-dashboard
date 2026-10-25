#!/usr/bin/env python3
"""Progress of corpus_v2 labelling and of the three T6 training arms (read-only).

Composition rule implemented (tasks/plan.md line 40, T6):
  (a) LIME-guided   : 300 neutral, lexicon-loaded
  (b) random-neutral: 300 neutral drawn from all labelled neutrals, disjoint from (a) "where possible"
  (c) mixed         : 100 loaded-neutral + 100 positive + 100 negative
"Loaded" is read as the T4 rule (plan.md line 38): >= 1 token from the repo keyword lexicon, i.e. a
non-empty `lexicon_hits` in candidates.json. ASSUMPTIONS (plan is silent or ambiguous):
  A1  174 'random'-group candidates also have lexicon hits; they count as loaded (rule-based reading).
      The stricter group-based count (sampling_group == 'lexicon_hit') is printed alongside.
  A2  low_information neutrals count toward the arms (plan does not say); their number is shown.
  A3  (c)'s 100 loaded neutrals may reuse (a)'s items (plan only asks (b) to be disjoint from (a)).
  A4  'exclude' items are never used. Final labels = model labels with review overrides applied.
Plan.md line 83 fallback if loaded neutrals < 300: relax to >= 1 hit from an extended list (not implemented here).
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
import math

import common as C

TARGET = 300


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels-dir', action='append', default=[], help='extra folder with labels_*.json (tests)')
    ap.add_argument('--log', help='overrides log (default overrides_log.jsonl)')
    a = ap.parse_args(argv)

    pool = C.load_candidates()
    rows = {r['item']: r for r in pool}
    labels, files = C.load_labels(a.labels_dir)
    unknown = set(labels) - set(rows)
    if unknown:
        raise SystemExit(f'labels for ids not in the pool: {sorted(unknown)[:5]}')
    overrides = C.read_jsonl(a.log or C.OVERRIDES)
    final, n_over = C.apply_overrides(labels, overrides)
    groups = Counter(r['sampling_group'] for r in pool)

    print('== Label files')
    for p, meta, n in files:
        print(f'  {p.name}: {n} labels, labeller={meta.get("labeller")!r}, effort={meta.get("reasoning_effort")}, '
              f'date={meta.get("date")}')
    print(f'\n== Coverage: labelled {len(labels)} / {len(pool)}, remaining {len(pool) - len(labels)}')
    for g in sorted(groups):
        lab = sum(1 for k in labels if rows[k]['sampling_group'] == g)
        print(f'  {g:12s} labelled {lab} / {groups[g]}, remaining {groups[g] - lab}')

    model = C.summary(labels)
    fin = Counter(final.values())
    print(f'\n== Class distribution (model labels): {C.counter_str(model)}')
    print(f'   low_information (all neutral by rule): {sum(v["low_information"] for v in labels.values())}')
    print(f'   overrides applied: {n_over} (log entries: {len(overrides)}); final labels: {C.counter_str(fin)}')
    for g in sorted(groups):
        c = Counter(final[k] for k in final if rows[k]['sampling_group'] == g)
        print(f'   {g:12s} {C.counter_str(c)}')
    print(f'   exclusions: {fin.get("exclude", 0)}')

    neutral = [k for k, v in final.items() if v == 'neutral']
    loaded = [k for k in neutral if rows[k]['lexicon_hits']]
    loaded_strict = [k for k in neutral if rows[k]['sampling_group'] == 'lexicon_hit']
    unloaded = [k for k in neutral if not rows[k]['lexicon_hits']]
    lowinfo_loaded = sum(labels[k]['low_information'] for k in loaded)
    a_n = min(TARGET, len(loaded))
    b_disjoint = len(neutral) - a_n
    c_neu, c_pos, c_neg = min(100, len(loaded)), min(100, fin.get('positive', 0)), min(100, fin.get('negative', 0))
    print('\n== Arms toward 300 (plan.md line 40; assumptions A1-A4 in --help)')
    print(f'  loaded neutrals: {len(loaded)} (A1 rule-based; strict lexicon_hit group only: {len(loaded_strict)}); '
          f'of which low_information {lowinfo_loaded}; neutrals without lexicon hits: {len(unloaded)}')
    print(f'  (a) LIME-guided   : {a_n} / {TARGET}')
    print(f'  (b) random-neutral: {min(TARGET, b_disjoint)} / {TARGET} disjoint from (a) '
          f'[{min(TARGET, len(neutral))} / {TARGET} if overlap with (a) is allowed]')
    print(f'  (c) mixed         : loaded-neutral {c_neu}/100, positive {c_pos}/100, negative {c_neg}/100 '
          f'-> {c_neu + c_pos + c_neg} / {TARGET}')
    print(f'  fully disjoint (a)+(b) needs 600 neutrals, 300 of them loaded; have {len(neutral)} neutrals.')

    print('\n== ESTIMATE (projection, not a result): remaining items assumed to label like the labelled ones,')
    print('   per sampling group, at the current final-label rates. 95% Wilson interval on each rate only.')
    exp = {'loaded': [len(loaded)] * 3, 'neutral': [len(neutral)] * 3,
           'positive': [fin.get('positive', 0)] * 3, 'negative': [fin.get('negative', 0)] * 3}
    for g in sorted(groups):
        keys = [k for k in final if rows[k]['sampling_group'] == g]
        n, rem = len(keys), groups[g] - len(keys)
        tests = {'loaded': lambda k: final[k] == 'neutral' and rows[k]['lexicon_hits'],
                 'neutral': lambda k: final[k] == 'neutral',
                 'positive': lambda k: final[k] == 'positive', 'negative': lambda k: final[k] == 'negative'}
        for name, t in tests.items():
            p, lo, hi = wilson(sum(1 for k in keys if t(k)), n)
            for i, r in enumerate((p, lo, hi)):
                exp[name][i] += r * rem
        p, lo, hi = wilson(sum(1 for k in keys if final[k] == 'neutral'), n)
        print(f'   {g:12s} neutral rate {p:.3f} (95% {lo:.3f}-{hi:.3f}) from n={n}; {rem} remaining')
    for name, (pt, lo, hi) in exp.items():
        print(f'   expected {name:8s} at 1,100 labelled: ~{pt:.0f} (range {lo:.0f}-{hi:.0f})')
    pt, lo, hi = exp['loaded']
    print(f'   expected loaded-neutral shortfall for arm (a): ~{max(0, TARGET - pt):.0f} '
          f'(range {max(0, TARGET - hi):.0f}-{max(0, TARGET - lo):.0f})')
    lk = [k for k in final if rows[k]['sampling_group'] == 'lexicon_hit']
    rem = groups['lexicon_hit'] - len(lk)
    p, lo, hi = wilson(len(loaded_strict), len(lk))
    s_pt, s_lo, s_hi = (len(loaded_strict) + r * rem for r in (p, lo, hi))
    print(f'   strict reading (lexicon_hit group only): expected ~{s_pt:.0f} loaded neutrals (range {s_lo:.0f}-{s_hi:.0f}), '
          f'shortfall ~{max(0, TARGET - s_pt):.0f} (range {max(0, TARGET - s_hi):.0f}-{max(0, TARGET - s_lo):.0f})')
    pt, lo, hi = exp['neutral']
    print(f'   expected shortfall vs 600 neutrals for disjoint (a)+(b): ~{max(0, 600 - pt):.0f} '
          f'(range {max(0, 600 - hi):.0f}-{max(0, 600 - lo):.0f})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
