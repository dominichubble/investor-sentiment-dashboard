"""Read-only T4 pool build. Raw DB errors are suppressed to protect credentials."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from corpus_rule import SEED, N_RANDOM, N_LOADED, clean, normalise, read_lexicon, eligible_frame, sample_candidates
from db_lookup import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/corpus_v2'
QUERY = ('SELECT id, text, ticker, source, data_source, published_at '
         'FROM sentiment_records WHERE ticker IS NOT NULL ORDER BY id')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'candidates.json').exists():
        raise FileExistsError('Candidate file already exists; do not overwrite labelled data')
    bp = ROOT / 'analysis/benchmark_v2_final.json'
    rp = ROOT / 'analysis/corpus_rule.py'
    kp = ROOT / 'investor-sentiment-dashboard/backend/app/models/keyword_sentiment.py'
    benchmark = json.loads(bp.read_text())
    lexicon = read_lexicon(kp)
    rule_hash = sha(rp)
    try:
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute('SHOW transaction_read_only')
                if cursor.fetchone()[0] != 'on':
                    raise RuntimeError('Read-only transaction required')
            with connection.cursor(name='corpus_candidates') as cursor:
                cursor.itersize = 5000
                cursor.execute(QUERY)
                frame, counts = eligible_frame(cursor, benchmark, lexicon)
    except Exception as exc:
        # Never emit the exception message: connection errors can contain secrets.
        print(f'Database query failed ({type(exc).__name__}); no connection details logged.', file=sys.stderr)
        return 2
    selected = sample_candidates(frame)
    assert selected == sample_candidates(list(reversed(frame)))
    ids, texts = {r['id'] for r in selected}, {normalise(r['text']) for r in selected}
    assert len(ids) == len(texts) == N_RANDOM + N_LOADED
    assert not ids & {b['id'] for b in benchmark}
    assert not texts & {normalise(clean(b['text'])) for b in benchmark}
    assert sha(rp) == rule_hash
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(), seed=SEED,
        n_random=N_RANDOM, n_lexicon_hit=N_LOADED, query=QUERY,
        transaction_read_only=True, filters=counts,
        eligible_by_channel=dict(Counter(r['channel'] for r in frame)),
        eligible_with_lexicon_hit=sum(bool(r['lexicon_hits']) for r in frame),
        sampled_by_group_and_channel=dict(Counter(r['sampling_group'] + '/' + r['channel'] for r in selected)),
        benchmark_exclusion_n=len(benchmark), benchmark_sha256=sha(bp),
        rule_sha256=rule_hash, lexicon_source_sha256=sha(kp), lexicon=sorted(lexicon),
        eligible_frame_sha256=hashlib.sha256(json.dumps(frame, sort_keys=True).encode()).hexdigest(),
        sampling='400 unrestricted random first, then 700 lexicon-hit from remaining frame; without replacement; no channel quotas',
        language_filter='ASCII >=95% and at least one 3-letter English-alphabet run; heuristic only, actual language checked during labelling',
        deduplication='URL-stripped whitespace-cleaned text, lowercased, nonword runs collapsed; not semantic near-duplicate detection',
        verification='1100 unique ids and normalised texts; zero overlap with all original 359 benchmark ids/texts; same sample after reversing frame input',
        labelling_status='not labelled; no FinBERT or stored database labels queried',
    )
    target = OUT / 'candidates.json'
    target.write_text(json.dumps(selected, indent=1) + '\n')
    manifest['candidates_sha256'] = sha(target)
    (OUT / 'sampling_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Blind view omits sampling group, matched words and all stored predictions.
    for start in range(0, len(selected), 50):
        view = [{k: r[k] for k in ('item', 'text', 'ticker', 'channel')} for r in selected[start:start+50]]
        (OUT / f'blind_{start:04d}.json').write_text(json.dumps(view, indent=1) + '\n')
    print(json.dumps({k: manifest[k] for k in ('filters', 'eligible_by_channel', 'eligible_with_lexicon_hit', 'sampled_by_group_and_channel', 'verification')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
