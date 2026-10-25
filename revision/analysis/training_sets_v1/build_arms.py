"""Build the three T6 training arms from all labelled corpus_v2 items (v2 build, 23 Oct 2026 evening).

Pool = original 1,100 candidates + 1,200-item extension (candidates_ext_1100.json), all model-labelled under rubric v2.
Reading of tasks/plan.md line 40 (T6): (a) 300 lexicon-loaded neutrals; (b) 300 neutrals drawn at random from all
labelled neutrals, disjoint from (a) "where possible"; (c) 100 loaded neutrals + 100 positive + 100 negative.
"Loaded" = plan.md line 38 rule (>=1 token of _POSITIVE_WORDS | _NEGATIVE_WORDS), the PRIMARY definition.
  (a) if >= 300 loaded neutrals exist: seeded sample of 300 of them (no widening). Otherwise the pre-specified
      Loughran-McDonald widening (plan.md line 83) runs exactly as in the 19:10 build (see previous_2026-10-23_1910/).
  (b) seeded sample of 300 from neutrals NOT in (a); only if fewer than 300 exist, all of them plus a seeded top-up
      from (a).
  (c) seeded: 100 base-loaded neutrals + 100 positive + 100 negative.
Low-information neutrals count (progress.py A2). Final labels = model labels + review overrides; 'exclude' never used.
Outputs are created new; the script refuses to overwrite anything.
"""
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/labelling_pipeline'))
import common  # noqa: E402  (labelling_pipeline/common.py; imports corpus_rule)
from corpus_rule import read_lexicon  # noqa: E402

SEED = 20261023
N = 300
LM_CSV = HERE / 'sources/LM_MasterDictionary_1993-2025.csv'
KEYWORDS = ROOT / 'investor-sentiment-dashboard/backend/app/models/keyword_sentiment.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tokens(text):
    return set(re.findall(r'\b[a-z]+\b', text.lower()))


def write_new(path, obj=None, text=None):
    with open(path, 'x') as f:
        f.write(text if text is not None else json.dumps(obj, indent=1, ensure_ascii=False) + '\n')


def lm_terms():
    pos, neg = set(), set()
    with open(LM_CSV, newline='') as f:
        for r in csv.DictReader(f):
            w = r['Word'].lower()
            if int(float(r['Positive'] or 0)) > 0:
                pos.add(w)
            if int(float(r['Negative'] or 0)) > 0:
                neg.add(w)
    return pos, neg


def main():
    cands = common.load_candidates()
    C = {c['item']: c for c in cands}
    labels, _files = common.load_labels()
    final, n_over = common.apply_overrides(labels, common.read_jsonl(common.OVERRIDES))
    assert len(labels) == len(C), f'expected all {len(C)} pool items labelled, have {len(labels)}'
    lowinfo = {k: bool(v.get('low_information')) for k, v in labels.items()}
    base = read_lexicon(KEYWORDS)
    toks = {k: tokens(c['text']) for k, c in C.items()}
    mism = [k for k in C if sorted(toks[k] & base) != C[k]['lexicon_hits']]
    assert not mism, f'base lexicon does not reproduce candidates.json lexicon_hits: {mism[:5]}'

    neutrals = sorted(k for k in C if final[k] == 'neutral')
    loaded_by = {k: 'base' for k in neutrals if toks[k] & base}

    # ---- pre-specified widening (label-blind order), only if the base lexicon gives < 300
    widening_needed = len(loaded_by) < N
    lm_pos, lm_neg = lm_terms()
    pool = (lm_pos | lm_neg) - base
    df = Counter(t for k in C for t in toks[k] & pool)
    order = sorted((t for t in pool if df[t] > 0), key=lambda t: (-df[t], t))
    # Single pass over the fixed order. Arm (a) adds terms until loaded neutrals (low-info counted) >= 300.
    # A second, informative-only tally continues to see how many terms would be needed if low-info
    # neutrals were NOT counted; it never changes arm (a).
    log, added = [], []
    var_loaded = {k for k in loaded_by if not lowinfo[k]}
    alt_stop = None
    for t in (order if widening_needed else []):
        a_done = len(loaded_by) >= N
        v_done = len(var_loaded) >= N
        if a_done and v_done:
            break
        new = [k for k in neutrals if k not in loaded_by and t in toks[k]]
        new_v = [k for k in neutrals if not lowinfo[k] and k not in var_loaded and t in toks[k]]
        entry = dict(term=t, lm_list='+'.join(x for x, ss in (('positive', lm_pos), ('negative', lm_neg)) if t in ss),
                     doc_freq_1100=df[t], used_for_arm_a=not a_done)
        if not a_done:
            added.append(t)
            for k in new:
                loaded_by[k] = t
            entry.update(new_loaded_neutrals=len(new),
                         new_loaded_neutrals_lowinfo=sum(lowinfo[k] for k in new),
                         cumulative_loaded_neutrals=len(loaded_by))
        var_loaded.update(new_v)
        entry.update(variant_excl_lowinfo_new=len(new_v), variant_excl_lowinfo_cumulative=len(var_loaded))
        log.append(entry)
        if alt_stop is None and len(var_loaded) >= N:
            alt_stop = dict(term=t, terms_needed=len(log))
    variant_terms = alt_stop['terms_needed'] if alt_stop else None
    n_loaded = len(loaded_by)
    reached = n_loaded >= N

    # ---- arm (a)
    if not widening_needed:
        arm_a = sorted(random.Random(SEED + 1).sample(sorted(loaded_by), N))
    else:
        rank = {'base': 0, **{t: i + 1 for i, t in enumerate(added)}}
        ordered = sorted(loaded_by, key=lambda k: (rank[loaded_by[k]], k))
        if len(ordered) > N:
            last = added[-1] if added else 'base'
            keep = [k for k in ordered if loaded_by[k] != last]
            tail = sorted(k for k in ordered if loaded_by[k] == last)
            arm_a = keep + random.Random(SEED + 1).sample(tail, N - len(keep))
        else:
            arm_a = ordered
        arm_a = sorted(arm_a)

    # ---- arm (b): neutrals not in (a) first; overlap only if fewer than 300 such neutrals exist
    rb = random.Random(SEED + 2)
    non_a = sorted(set(neutrals) - set(arm_a))
    if len(non_a) >= N:
        arm_b = sorted(rb.sample(non_a, N))
    else:
        arm_b = sorted(non_a + rb.sample(sorted(arm_a), min(N - len(non_a), len(arm_a))))

    # ---- arm (c): 100 base-loaded neutrals + 100 positive + 100 negative
    base_loaded = sorted(k for k in neutrals if toks[k] & base)
    pos = sorted(k for k in C if final[k] == 'positive')
    neg = sorted(k for k in C if final[k] == 'negative')
    r = random.Random(SEED + 3)
    arm_c = sorted(r.sample(base_loaded, 100) + r.sample(pos, 100) + r.sample(neg, 100))

    # ---- checks
    block_ids, block_texts = common.benchmark_blocklist()
    arms = dict(a=arm_a, b=arm_b, c=arm_c)
    for name, ids in arms.items():
        assert len(ids) == len(set(ids)), f'duplicate ids in arm {name}'
        nt = [common.normalise(common.clean(C[k]['text'])) for k in ids]
        assert len(nt) == len(set(nt)), f'duplicate texts in arm {name}'
        bad = [k for k in ids if k in block_ids or C[k]['id'] in block_ids
               or common.normalise(common.clean(C[k]['text'])) in block_texts]
        assert not bad, f'benchmark overlap in arm {name}: {bad[:5]}'
        assert all(final[k] in ('positive', 'negative', 'neutral') for k in ids)

    # ---- write
    def manifest(name, ids, note):
        rows = [dict(item=k, id=C[k]['id'], label=final[k], low_information=lowinfo[k], channel=C[k]['channel'],
                     ticker=C[k]['ticker'], sampling_group=C[k]['sampling_group'],
                     base_lexicon_hits=C[k]['lexicon_hits'],
                     loaded_by=loaded_by.get(k) if final[k] == 'neutral' else None) for k in ids]
        return dict(arm=name, note=note, seed=SEED, n=len(ids), label_counts=dict(Counter(final[k] for k in ids)),
                    low_information=sum(lowinfo[k] for k in ids),
                    per_channel=dict(Counter(C[k]['channel'] for k in ids)),
                    per_sampling_group=dict(Counter(C[k]['sampling_group'] for k in ids)),
                    data_file=f'arm_{name}.json', data_sha256=None, rows=rows)
    notes = dict(a='LIME-guided: lexicon-loaded neutrals' + (' (base + pre-specified LM widening)' if widening_needed else ' (original lexicon, plan.md line 38; no widening)'),
                 b='random-neutral: seeded sample of labelled neutrals not in (a) (overlap only if too few)',
                 c='mixed: 100 base-loaded neutrals + 100 positive + 100 negative')
    for name, ids in arms.items():
        data = [dict(text=C[k]['text'], label=final[k]) for k in ids]
        write_new(HERE / f'arm_{name}.json', data)
        m = manifest(name, ids, notes[name])
        m['data_sha256'] = sha(HERE / f'arm_{name}.json')
        write_new(HERE / f'arm_{name}_manifest.json', m)
    lex = ['# Keyword list used for arm (a)' + ('' if widening_needed else ' (NO widening needed: base lexicon only)'),
           '# Extended keyword list for arm (a). Base: _POSITIVE_WORDS | _NEGATIVE_WORDS of keyword_sentiment.py',
           f'# ({len(base)} terms, sha256 {sha(KEYWORDS)}).',
           f'# Added: {len(added)} Loughran-McDonald 1993-2025 terms (sha256 {sha(LM_CSV)}), in order of addition.',
           '# Lines starting with # are comments.', '[base]', *sorted(base), '[added_in_order]', *added]
    write_new(HERE / 'lexicon_extended.txt', text='\n'.join(lex) + '\n')
    ov = lambda x, y: len(set(x) & set(y))
    summary = dict(
        seed=SEED, inputs=dict(candidates_sha256=sha(common.CANDIDATES), keyword_lexicon_sha256=sha(KEYWORDS),
                               lm_csv_sha256=sha(LM_CSV), rubric_sha256=sha(common.RUBRIC),
                               label_files={p.name: sha(p) for p in sorted(common.CORPUS.glob('labels_*.json'))},
                               overrides_entries=len(common.read_jsonl(common.OVERRIDES)), overrides_applied=n_over),
        neutrals_total=len(neutrals), neutrals_lowinfo=sum(lowinfo[k] for k in neutrals),
        base_loaded_neutrals=len(base_loaded), base_loaded_lowinfo=sum(lowinfo[k] for k in base_loaded),
        widening_needed=widening_needed, non_a_neutrals=len(non_a), lm_pool_terms=len(pool), lm_pool_terms_in_corpus=len(order), terms_added=added,
        loaded_after_widening=n_loaded, arm_a_reached_300=reached,
        variant_lowinfo_excluded=dict(variant_loaded_final=len(var_loaded), terms_scanned=len(log), base_loaded=sum(not lowinfo[k] for k in base_loaded),
                                      reached_300=alt_stop is not None, terms_needed=variant_terms,
                                      stop_term=alt_stop['term'] if alt_stop else None),
        arms={n: dict(n=len(i), labels=dict(Counter(final[k] for k in i)), low_information=sum(lowinfo[k] for k in i))
              for n, i in arms.items()},
        overlap=dict(a_b=ov(arm_a, arm_b), a_c=ov(arm_a, arm_c), b_c=ov(arm_b, arm_c),
                     a_b_c=len(set(arm_a) & set(arm_b) & set(arm_c))))
    write_new(HERE / 'widening_log.json', dict(order_rule='LM pos|neg minus base; doc freq over 1,100 texts desc, ties alpha',
                                               terms=log))
    write_new(HERE / 'build_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'inputs'}, indent=1))


if __name__ == '__main__':
    main()
