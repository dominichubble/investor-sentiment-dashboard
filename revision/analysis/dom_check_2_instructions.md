# Check #2: instructions (50 fresh items)

File to fill in: `dom_label_check_2.csv`, last column. Use exactly: `positive`, `negative`, `neutral` or `exclude`.

Label = likely effect on the value of the asset the text is actually about, from an investor's point of view
(Financial PhraseBank taxonomy). The `ticker` column is sometimes a spurious match: judge the asset the text discusses.

- **positive**: reports or argues for something that would raise the value (beats, growth, upgrades, bullish thesis or position).
- **negative**: something that would lower the value (misses, downgrades, losses, legal trouble, bearish thesis or position, falling price).
- **neutral**: factual or procedural content with no clear value impact; questions; balanced or mixed statements.
  **Decided 20 Oct: factual filler is neutral, not exclude.** This covers fund-holdings lists ("X Capital Buys..."),
  earnings-call transcripts, scheduled-event notices, generic how-to articles and roundups with no stated direction.
- **exclude**: only for spam/advert/SEO market-research boilerplate, a bare mention list or thread header with no content,
  non-English text, or text too fragmentary to judge.

Please label by your own judgement and do not look at any model output or at the earlier label files.
These 50 items are a **reserve set**: they will not be used to tune the rubric, so the agreement figure is unbiased.
Save the finished file as `dom_label_check_2_completed.csv` (Downloads is fine) and tell me.
