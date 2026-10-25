"""Replacements for excluded items (random, same channel) + negative-lexicon top-up. FinBERT labels not read."""
import re, sys, ast, json, random, collections
sys.path.insert(0, "analysis")
from db_lookup import connect

SEED = 20261021
TOPUP = {"news": 25, "reddit": 15, "twitter": 20}
clean = lambda s: re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", s or "")).strip()
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
tok = lambda s: set(re.findall(r"[a-z']+", s.lower()))
src = open("investor-sentiment-dashboard/backend/app/models/keyword_sentiment.py").read()
neg = next(ast.literal_eval(n.value) for n in ast.parse(src).body
           if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_NEGATIVE_WORDS")

done = json.load(open("analysis/benchmark_v2_labelled.json"))
used_ids = {x["id"] for x in done}; used_txt = {norm(x["text"]) for x in done}
excl = collections.Counter(x["channel"] for x in done if x["llm_label"] == "exclude")

pool = collections.defaultdict(list); seen = set()
with connect() as c, c.cursor(name="cur") as cur:
    cur.itersize = 5000
    cur.execute("select id, text, ticker, source, data_source, published_at from sentiment_records where ticker is not null")
    for rid, text, ticker, source, ch, pub in cur:
        t = clean(text)
        if not (60 <= len(t) <= 500) or not re.search(r"[a-zA-Z]{3}", t): continue
        if sum(map(str.isascii, t)) < 0.95 * len(t): continue
        k = norm(t)
        if k in seen or k in used_txt or rid in used_ids: continue
        seen.add(k)
        pool[ch].append(dict(id=rid, text=t, ticker=ticker, source=source, channel=ch, published_at=str(pub), neg_hit=bool(tok(t) & neg)))

rng = random.Random(SEED); out = []
for ch, n in excl.items():
    out += [dict(x, kind="replacement") for x in rng.sample(pool[ch], n)]
taken = {x["id"] for x in out}
for ch, n in TOPUP.items():
    cand = [x for x in pool[ch] if x["neg_hit"] and x["id"] not in taken]
    print(ch, "negative-lexicon candidates:", len(cand))
    out += [dict(x, kind="topup_neg") for x in rng.sample(cand, min(n, len(cand)))]
rng.shuffle(out)
for i, x in enumerate(out): x["item"] = f"b2_{300+i:03d}"; x.pop("neg_hit")
json.dump(out, open("analysis/benchmark_v2_topup_sample.json", "w"), indent=1)
print(len(out), collections.Counter((x["kind"], x["channel"]) for x in out))
