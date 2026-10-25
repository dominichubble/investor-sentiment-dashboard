"""Build v2 training arms: one shared pos/neg base + the v1 arm-specific component.

Design (Dom, 23 Oct 2026, option 2; deviation from plan.md T6 line 40 documented in BUILD_NOTES.md):
  BASE   = 300 positive + 300 negative, sampled (seed SEED_V2) from labelled pool positives/negatives,
           excluding benchmark (359 final + 338 frozen, by id and normalised text) and excluding
           v1 arm_c's positive/negative items. Identical in all three arms.
  arm_a  = BASE + v1 arm_a (300 lexicon-loaded neutrals)
  arm_b  = BASE + v1 arm_b (300 random neutrals)
  arm_c  = BASE + v1 arm_c (100 loaded neutrals + 100 positive + 100 negative)
Each arm has 900 rows. Rows are written in a seeded shuffled order. Format [{text,label}].
Reads v1 manifests and pool labels only; writes new files only (refuses to overwrite).
"""
import hashlib, json, random, sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / 'analysis/training_sets_v1'
sys.path.insert(0, str(ROOT / 'analysis/labelling_pipeline'))
import common  # noqa: E402

SEED_V2 = 20261024
N_BASE_PER_CLASS = 300


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(path, text):
    path = Path(path)
    if path.exists() and path.read_text() != text:
        raise SystemExit(f'refusing to overwrite differing {path}')
    path.write_text(text)


def main():
    C = {c['item']: c for c in common.load_candidates()}
    labels, _ = common.load_labels()
    final, _ = common.apply_overrides(labels, common.read_jsonl(common.OVERRIDES))
    assert len(labels) == len(C) == 2300
    lowinfo = {k: bool(v.get('low_information')) for k, v in labels.items()}
    v1 = {a: [r['item'] for r in json.loads((V1 / f'arm_{a}_manifest.json').read_text())['rows']] for a in 'abc'}
    for a in 'abc':  # v1 data files must match manifests (sanity)
        m = json.loads((V1 / f'arm_{a}_manifest.json').read_text())
        assert sha(V1 / f'arm_{a}.json') == m['data_sha256'], f'v1 arm_{a} changed'
    block_ids, block_texts = common.benchmark_blocklist()
    norm = lambda k: common.normalise(common.clean(C[k]['text']))
    in_bench = lambda k: k in block_ids or C[k]['id'] in block_ids or norm(k) in block_texts

    c_set = set(v1['c'])
    specific_texts = {norm(k) for a in 'abc' for k in v1[a]}
    elig = {}
    for lab in ('positive', 'negative'):
        elig[lab] = sorted(k for k in C if final[k] == lab and k not in c_set and not in_bench(k)
                           and norm(k) not in specific_texts)
    r = random.Random(SEED_V2)
    base = {lab: sorted(r.sample(elig[lab], N_BASE_PER_CLASS)) for lab in ('positive', 'negative')}
    base_items = sorted(base['positive'] + base['negative'])

    arms = {a: sorted(base_items + v1[a]) for a in 'abc'}
    summary = {'seed_v2': SEED_V2, 'base_per_class': N_BASE_PER_CLASS,
               'eligible_for_base': {k: len(v) for k, v in elig.items()},
               'pool_label_counts': dict(Counter(final.values())), 'arms': {}, 'overlaps': {}}
    for a, ids in arms.items():
        assert len(ids) == len(set(ids)) == 900, (a, len(ids))
        nt = [norm(k) for k in ids]
        assert len(nt) == len(set(nt)), f'duplicate texts in arm {a}'
        assert not [k for k in ids if in_bench(k)], f'benchmark overlap in arm {a}'
        assert set(base_items) <= set(ids)
        order = list(ids); random.Random(SEED_V2 + ord(a)).shuffle(order)
        data = [{'text': common.clean(C[k]['text']), 'label': final[k]} for k in order]
        assert all(d['label'] in ('positive', 'negative', 'neutral') for d in data)
        write_new(HERE / f'arm_{a}.json', json.dumps(data, ensure_ascii=False, indent=1))
        rows = [dict(item=k, id=C[k]['id'], label=final[k], component='base' if k in base_items else 'arm_specific',
                     low_information=lowinfo[k], channel=C[k]['channel'], sampling_group=C[k]['sampling_group'])
                for k in order]
        man = {'arm': a, 'n': len(ids), 'seed_v2': SEED_V2, 'row_order_seed': SEED_V2 + ord(a),
               'label_counts': dict(Counter(final[k] for k in ids)),
               'arm_specific_label_counts': dict(Counter(final[k] for k in v1[a])),
               'low_information': sum(lowinfo[k] for k in ids),
               'per_channel': dict(Counter(C[k]['channel'] for k in ids)),
               'v1_source': f'training_sets_v1/arm_{a}.json', 'v1_sha256': sha(V1 / f'arm_{a}.json'),
               'data_file': f'arm_{a}.json', 'data_sha256': sha(HERE / f'arm_{a}.json'), 'rows': rows}
        write_new(HERE / f'arm_{a}_manifest.json', json.dumps(man, indent=1))
        summary['arms'][a] = {k: man[k] for k in ('n', 'label_counts', 'arm_specific_label_counts', 'low_information', 'per_channel', 'data_sha256')}
    write_new(HERE / 'base_manifest.json', json.dumps({'seed_v2': SEED_V2, 'items': base_items,
              'label_counts': {k: len(v) for k, v in base.items()},
              'per_channel': dict(Counter(C[k]['channel'] for k in base_items))}, indent=1))
    S = {a: set(v) for a, v in arms.items()}
    sp = {a: set(v1[a]) for a in 'abc'}
    summary['overlaps'] = {'full_arms': {'a&b': len(S['a'] & S['b']), 'a&c': len(S['a'] & S['c']), 'b&c': len(S['b'] & S['c']),
                                         'a&b&c': len(S['a'] & S['b'] & S['c'])},
                           'arm_specific_only': {'a&b': len(sp['a'] & sp['b']), 'a&c': len(sp['a'] & sp['c']),
                                                 'b&c': len(sp['b'] & sp['c']), 'a&b&c': len(sp['a'] & sp['b'] & sp['c'])},
                           'base_vs_arm_specific': {a: len(set(base_items) & sp[a]) for a in 'abc'}}
    summary['label_file_sha256'] = {p.name: sha(p) for p in sorted(common.CORPUS.glob('labels_*.json'))}
    write_new(HERE / 'build_summary.json', json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'label_file_sha256'}, indent=1))


if __name__ == '__main__':
    main()
