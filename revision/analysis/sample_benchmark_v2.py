"""Draw the sampling frame for benchmark v2: random, stratified by channel only.
FinBERT labels are NOT read (so sampling and labels stay independent of the model under test)."""
import re, sys, json, random, collections
sys.path.insert(0, "analysis")
from db_lookup import connect

SEED, PER_CHANNEL = 20261020, 100
MIN_CH, MAX_CH = 60, 500          # shortcut: short texts only, so items are sentence-scale like the original benchmark
clean = lambda s: re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", s or "")).strip()
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()

frame = collections.defaultdict(list); seen = set()
with connect() as c, c.cursor(name="cur") as cur:
    cur.itersize = 5000
    cur.execute("select id, text, ticker, source, data_source, published_at from sentiment_records where ticker is not null")
    for rid, text, ticker, source, ch, pub in cur:
        t = clean(text)
        if not (MIN_CH <= len(t) <= MAX_CH) or not re.search(r"[a-zA-Z]{3}", t): continue
        if sum(map(str.isascii, t)) < 0.95 * len(t): continue      # crude English filter
        k = norm(t)
        if k in seen: continue
        seen.add(k)
        frame[ch].append(dict(id=rid, text=t, ticker=ticker, source=source, channel=ch, published_at=str(pub)))

rng = random.Random(SEED); sample = []
print("eligible per channel:", {k: len(v) for k, v in frame.items()})
for ch in ("reddit", "news", "twitter"):
    sample += rng.sample(frame[ch], PER_CHANNEL)
rng.shuffle(sample)
for i, s in enumerate(sample): s["item"] = f"b2_{i:03d}"
json.dump(sample, open("analysis/benchmark_v2_sample.json", "w"), indent=1)
print("sampled", len(sample), "| distinct tickers", len({s["ticker"] for s in sample}),
      "| distinct sources", len({s["source"] for s in sample}))
print("date range", min(s["published_at"] for s in sample)[:10], max(s["published_at"] for s in sample)[:10])
for s in sample[:4]: print("-", s["channel"], "|", s["text"][:110])
