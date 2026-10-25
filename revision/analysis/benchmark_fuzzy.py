import re, sys, collections
sys.path.insert(0, "analysis")
from db_lookup import connect
ns = {}; src = open("investor-sentiment-dashboard/backend/app/evaluation/labeled_dataset.py").read()
exec(src.split("def ")[0], ns); bench = ns["LABELED_DATASET"]
words = lambda s: re.findall(r"[a-z0-9$%.]+", (s or "").lower())
sh = lambda w, k=4: {" ".join(w[i:i+k]) for i in range(max(len(w)-k+1, 1))}
bsh = [sh(words(b["text"])) for b in bench]
inv = collections.defaultdict(set)
for i, s in enumerate(bsh):
    for g in s: inv[g].add(i)
best = [0.0]*len(bench); bestrow = [None]*len(bench)
with connect() as c, c.cursor(name="cur") as cur:
    cur.itersize = 5000
    cur.execute("select id, text, source from sentiment_records")
    totals = collections.Counter(); n = 0
    for rid, text, source in cur:
        n += 1
        s = sh(words(text)); cand = collections.Counter()
        for g in s:
            for i in inv.get(g, ()): cand[i] += 1
        for i, k in cand.items():
            j = k / len(bsh[i])           # fraction of benchmark shingles found in this record
            if j > best[i]: best[i], bestrow[i] = j, (rid, source, (text or "")[:90])
print("records scanned:", n)
import statistics
print("benchmark items whose best DB record shares >=50% of 4-word shingles:", sum(b >= .5 for b in best))
print(">=80%:", sum(b >= .8 for b in best), "| max overlap:", round(max(best), 2), "| median:", round(statistics.median(best), 2))
for i in sorted(range(len(best)), key=lambda i: -best[i])[:3]:
    print(round(best[i], 2), "|", bench[i]["text"][:80], "||", bestrow[i])
