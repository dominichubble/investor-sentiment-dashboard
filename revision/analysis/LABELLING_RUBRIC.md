# Benchmark v2 labelling rubric (fixed 20 Oct 2026, before any labelling)

Unit: one text item with its ticker. Label = likely effect on the value of the **mentioned ticker/company/asset**,
from an investor's point of view (Financial PhraseBank taxonomy, Malo et al. 2014).

- **positive**: the text reports or argues for something that would raise the value (beats, growth, upgrades, bullish thesis, favourable news).
- **negative**: something that would lower the value (misses, downgrades, losses, legal/regulatory trouble, bearish thesis, sell-off).
- **neutral**: factual or procedural content with no clear value impact (announcements, filings, scheduled events, price quotes without direction, questions, mixed or balanced statements).
- **exclude** (rare, flag only): not English, spam/advert/SEO market-research boilerplate with no financial content, or too fragmentary to judge. Excluded items are replaced by resampling.

Rules:
1. Judge the value of the asset the text is actually about, not the author's mood or the general market. Bad macro news with no stated effect on that asset is neutral. The ticker field is sometimes a spurious match (e.g. AGI, A, GOLD); if so, judge the asset the text discusses and do not exclude the item for that reason alone.
2. A question or request for advice is neutral unless it states a clear view.
3. Mixed items (good and bad) are neutral unless one side clearly dominates.
4. Do not use FinBERT's output or any stored model label. Labels are assigned blind to them.
5. Labeller: Claude (claude-sonnet-5-5), read item text, ticker and channel only. Dom independently labels a random 100 for agreement.
