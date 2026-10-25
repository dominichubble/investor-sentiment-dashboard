"""Read-only corpus_v2 EXTENSION draw (23 Oct 2026). Same code path as build_corpus_candidates.py.

Reuses corpus_rule (eligible_frame filters, lexicon rule, sample_candidates) and db_lookup.connect unchanged.
Differences from the original T4 draw, all fixed before any extension item is labelled:
  * new seed SEED_EXT = 20261023;
  * the 1,100 existing candidates are removed from the eligible frame (by id and normalised text) before sampling;
  * benchmark exclusion = original 359 (the frozen 338 are a subset; both checked);
  * size 1,200 at the original 7:4 lexicon-hit:random ratio (700:400) -> 764 lexicon-hit + 436 random;
  * items are numbered c2_1100.. in the order sample_candidates returns them (seeded shuffle).
Raw DB errors are suppressed to protect credentials. Writes NEW files only (refuses to overwrite).
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from corpus_rule import clean, normalise, read_lexicon, eligible_frame, sample_candidates  # noqa: E402
from db_lookup import connect  # noqa: E402

SEED_EXT = 20261023
N_LOADED_EXT, N_RANDOM_EXT = 764, 436
FIRST = 1100
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/corpus_v2'
TARGET = OUT / 'candidates_ext_1100.json'
MANIFEST = OUT / 'sampling_manifest_ext_1100.json'
QUERY = ('SELECT id, text, ticker, source, data_source, published_at '
         'FROM sentiment_records WHERE ticker IS NOT NULL ORDER BY id')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    n = N_LOADED_EXT + N_RANDOM_EXT
    blinds = [OUT / f'blind_{s:04d}.json' for s in range(FIRST, FIRST + n, 50)]
    clash = [p.name for p in [TARGET, MANIFEST, *blinds] if p.exists()]
    if clash:
        raise FileExistsError(f'refusing to overwrite: {clash[:5]}')
    bp = ROOT / 'analysis/benchmark_v2_final.json'
    fp = ROOT / 'analysis/benchmark_v2_frozen.json'
    rp = ROOT / 'analysis/corpus_rule.py'
    kp = ROOT / 'investor-sentiment-dashboard/backend/app/models/keyword_sentiment.py'
    cp = OUT / 'candidates.json'
    benchmark = json.loads(bp.read_text())
    frozen = json.loads(fp.read_text())['items']
    existing = json.loads(cp.read_text())
    lexicon = read_lexicon(kp)
    rule_hash = sha(rp)
    try:
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute('SHOW transaction_read_only')
                if cursor.fetchone()[0] != 'on':
                    raise RuntimeError('Read-only transaction required')
            with connection.cursor(name='corpus_candidates_ext') as cursor:
                cursor.itersize = 5000
                cursor.execute(QUERY)
                frame, counts = eligible_frame(cursor, benchmark + frozen, lexicon)
    except Exception as exc:
        print(f'Database query failed ({type(exc).__name__}); no connection details logged.', file=sys.stderr)
        return 2
    ex_ids = {r['id'] for r in existing}
    ex_texts = {normalise(clean(r['text'])) for r in existing}
    before = len(frame)
    frame = [r for r in frame if r['id'] not in ex_ids and normalise(r['text']) not in ex_texts]
    counts['existing_candidates_removed'] = before - len(frame)
    counts['eligible_after_extension_exclusions'] = len(frame)
    sel = sample_candidates(frame, seed=SEED_EXT, n_random=N_RANDOM_EXT, n_loaded=N_LOADED_EXT)
    assert sel == sample_candidates(list(reversed(frame)), seed=SEED_EXT, n_random=N_RANDOM_EXT, n_loaded=N_LOADED_EXT)
    sel = [dict(r, item=f'c2_{FIRST + i:04d}') for i, r in enumerate(sel)]
    ids, texts = {r['id'] for r in sel}, {normalise(r['text']) for r in sel}
    bl_ids = {b['id'] for b in benchmark + frozen}
    bl_texts = {normalise(clean(b['text'])) for b in benchmark + frozen}
    assert len(ids) == len(texts) == n
    assert not ids & bl_ids and not texts & bl_texts
    assert not ids & ex_ids and not texts & ex_texts
    assert sha(rp) == rule_hash
    with open(TARGET, 'x') as f:
        f.write(json.dumps(sel, indent=1) + '\n')
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(), seed=SEED_EXT, first_item=f'c2_{FIRST:04d}',
        n_random=N_RANDOM_EXT, n_lexicon_hit=N_LOADED_EXT, query=QUERY, transaction_read_only=True,
        filters=counts, eligible_by_channel=dict(Counter(r['channel'] for r in frame)),
        eligible_with_lexicon_hit=sum(bool(r['lexicon_hits']) for r in frame),
        sampled_by_group_and_channel=dict(Counter(r['sampling_group'] + '/' + r['channel'] for r in sel)),
        benchmark_exclusion=dict(original_359_sha256=sha(bp), frozen_338_sha256=sha(fp), n_ids=len(bl_ids)),
        existing_candidates_sha256=sha(cp), rule_sha256=rule_hash, lexicon_source_sha256=sha(kp),
        eligible_frame_sha256=hashlib.sha256(json.dumps(frame, sort_keys=True).encode()).hexdigest(),
        sampling=f'{N_RANDOM_EXT} unrestricted random first, then {N_LOADED_EXT} lexicon-hit from remaining frame '
                 '(corpus_rule.sample_candidates, unchanged); 7:4 ratio as in the original 700:400 draw; '
                 'existing 1,100 candidates removed from the frame first',
        verification=f'{n} unique ids and normalised texts; zero overlap with 359+338 benchmark and the 1,100 existing '
                     'candidates by id and normalised text; same sample after reversing frame input',
        labelling_status='not labelled at draw time; no FinBERT or stored database labels queried',
        candidates_sha256=sha(TARGET))
    with open(MANIFEST, 'x') as f:
        f.write(json.dumps(manifest, indent=2) + '\n')
    for i, p in enumerate(blinds):
        view = [{k: r[k] for k in ('item', 'text', 'ticker', 'channel')} for r in sel[i * 50:(i + 1) * 50]]
        with open(p, 'x') as f:
            f.write(json.dumps(view, indent=1) + '\n')
    print(json.dumps({k: manifest[k] for k in ('seed', 'filters', 'eligible_by_channel', 'eligible_with_lexicon_hit',
                                                'sampled_by_group_and_channel', 'verification')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
