import re, ast, json, collections, sys
sys.path.insert(0, "analysis")
from db_lookup import connect

norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
src = open("investor-sentiment-dashboard/backend/app/evaluation/labeled_dataset.py").read()
ds = ast.literal_eval(src[src.index("LABELED_DATASET"):].split("=",1)[1].split("\n]\n")[0].strip() + "\n]") if False else None
# parse the list safely via exec of just the literal
ns = {}
exec(src.split("def ")[0].replace("from typing import Dict, List", "from typing import Dict, List"), ns)
bench = ns["LABELED_DATASET"]
print("benchmark items:", len(bench))

with connect() as c, c.cursor() as cur:
    cur.execute("select source, data_source, count(*), min(published_at), max(published_at), count(distinct ticker) from sentiment_records group by 1,2 order by 3 desc")
    print("\nsource | data_source | n | first | last | tickers")
    for r in cur.fetchall(): print(r)
    cur.execute("select id, text, source, data_source, ticker, published_at, sentiment_label from sentiment_records")
    rows = cur.fetchall()
print("\nfetched", len(rows))
index = collections.defaultdict(list)
for r in rows: index[norm(r[1])].append(r)
hits, miss = [], []
for b in bench:
    m = index.get(norm(b["text"]))
    (hits if m else miss).append((b, m))
print("exact (normalised) matches in DB:", len(hits), "of", len(bench))
cnt = collections.Counter((m[0][2], m[0][3], b["label"]) for b, m in hits)
for k, v in sorted(cnt.items()): print(k, v)
if hits:
    ds_ = [m[0][5] for _, m in hits if m[0][5]]
    print("matched date range:", min(ds_), max(ds_), "| distinct tickers:", len({m[0][4] for _, m in hits}))
json.dump([{"text": b["text"], "label": b["label"], "db": [dict(id=x[0], source=x[2], data_source=x[3], ticker=x[4], published_at=str(x[5]), finbert_label=x[6]) for x in m]} for b, m in hits],
          open("analysis/benchmark_db_matches.json", "w"), indent=1)
print("unmatched sample:", [b["text"][:70] for b, _ in miss[:3]])
