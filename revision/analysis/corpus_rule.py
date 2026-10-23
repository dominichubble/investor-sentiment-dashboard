"""T4 sampling rules fixed before candidate labelling; no model predictions used."""

import ast
from collections import Counter
import random
import re

SEED = 20261021
N_RANDOM = 400
N_LOADED = 700
MIN_CHARS, MAX_CHARS = 60, 500
CHANNELS = {'reddit', 'news', 'twitter'}


def clean(text):
    return re.sub(r'\s+', ' ', re.sub(r'https?://\S+', '', text or '')).strip()


def normalise(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()


def read_lexicon(path):
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', '')
            if name in ('_POSITIVE_WORDS', '_NEGATIVE_WORDS'):
                values[name] = ast.literal_eval(node.value)
    if len(values) != 2:
        raise ValueError('Expected both keyword lexicons')
    return set.union(*values.values())


def eligible_frame(rows, benchmark, lexicon):
    blocked_ids = {b['id'] for b in benchmark}
    blocked_texts = {normalise(clean(b['text'])) for b in benchmark}
    seen, frame, counts = set(), [], Counter()
    # The query orders by id; first eligible row is the deterministic duplicate representative.
    for rid, raw, ticker, source, channel, published in rows:
        counts['queried'] += 1
        text = clean(raw)
        key = normalise(text)
        if rid in blocked_ids or key in blocked_texts:
            counts['benchmark_overlap_removed'] += 1
        elif channel not in CHANNELS or ticker is None:
            counts['channel_or_ticker_removed'] += 1
        elif not (MIN_CHARS <= len(text) <= MAX_CHARS):
            counts['length_removed'] += 1
        elif not re.search(r'[a-zA-Z]{3}', text) or sum(map(str.isascii, text)) < .95 * len(text):
            counts['language_proxy_removed'] += 1
        elif key in seen:
            counts['duplicate_removed'] += 1
        else:
            seen.add(key)
            tokens = set(re.findall(r'\b[a-z]+\b', text.lower()))
            frame.append(dict(id=rid, text=text, ticker=ticker, source=source,
                              channel=channel, published_at=str(published),
                              lexicon_hits=sorted(tokens & lexicon)))
    counts['eligible'] = len(frame)
    return frame, dict(counts)


def sample_candidates(frame, seed=SEED, n_random=N_RANDOM, n_loaded=N_LOADED):
    rng = random.Random(seed)
    frame = sorted(frame, key=lambda row: row['id'])
    unrestricted = rng.sample(frame, n_random)
    used = {row['id'] for row in unrestricted}
    loaded = rng.sample([row for row in frame if row['lexicon_hits'] and row['id'] not in used], n_loaded)
    selected = [dict(row, sampling_group='random') for row in unrestricted]
    selected += [dict(row, sampling_group='lexicon_hit') for row in loaded]
    rng.shuffle(selected)
    return [dict(row, item=f'c2_{i:04d}') for i, row in enumerate(selected)]
