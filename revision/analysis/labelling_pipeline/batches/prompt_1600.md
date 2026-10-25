You are labelling financial social-media/news items for the DSFE-635 training corpus.
Apply rubric v2 below exactly (sha256 610ca38a19cd1e4249b452d285f3c8c62bcdd8ca92cd5bd4269b908afe2906df). Judge each item independently from its text, ticker and channel only.

Key rules (summary; the full rubric text follows and wins on any conflict):
- Label = likely effect on the value of the asset the text is actually about, from an investor's point of view
  (Financial PhraseBank taxonomy). The ticker can be a spurious match; judge the asset discussed.
- R1 position gains follow the asset direction: profit on puts/shorts = negative; on calls/longs = positive;
  bare P&L with no view on the asset = neutral.
- R2 chart calls and quoted price moves take their stated direction; neutral only if no direction or balanced.
- R3 net-skeptical / risk-framed text = negative; a question with no lean = neutral.
- R4 new initiatives / favourable policy / naming favourite holdings = positive.
- R5 mixed analyst or rotation lists and bare launch headlines = neutral unless one side clearly dominates.
- R6 factual filler (holdings lists, transcripts, event notices, how-to, roundups) = neutral with
  "low_information": true. low_information may be true ONLY for neutral.
- R7 exclude ONLY spam/advert/SEO boilerplate, bare mention lists/thread headers, non-English, or too fragmentary.

Reply with ONE JSON object and nothing else, covering every item below exactly once:
{"labels": {"<item>": {"label": "positive|negative|neutral|exclude", "low_information": false, "rationale": "<one short sentence>"}, ...}}

----- RUBRIC v2 (verbatim) -----
# Benchmark v2 labelling rubric, version 2 (approved by Dom 20 Oct 2026)

Supersedes `LABELLING_RUBRIC.md` (v1, kept unchanged). v2 was written from the 21 disagreements and 11 exclusions in
Dom's blind check #1 (kappa v1 = 0.614, n = 89). Check #2's 50 items are a reserve set and were not used to write these rules.

Unit and target are as in v1: one text item; label = likely effect on the value of the asset the text is actually about,
from an investor's point of view (Financial PhraseBank taxonomy). The ticker field can be a spurious match; judge the asset discussed.

## Changes from v1

| # | Rule | Example items (check #1) |
|---|------|--------------------------|
| R1 | **Asset direction, not the author's profit.** A gain from a bearish position (puts, shorts) is **negative**; a gain from a bullish position is **positive**. Posts that only report P&L or a position with no view on the asset stay neutral. | b2_196 "$100k -> $165k on TSLA puts" -> negative; b2_274 "bear gains, playing from the short side" -> negative (Dom labelled both positive; this is a deliberate rule choice, see note) |
| R2 | **Follow the stated direction in charts and prices.** A technical-analysis call with a clear bearish or bullish direction takes that label, including conditional calls that name the bearish case ("top area if support fails"). A quoted price fall or rise with no commentary takes its direction. Neutral only if there is no direction or the directions balance. | b2_207 -> negative; b2_173 -> negative; b2_145 "GameStop -2.27%" -> negative; b2_024 -> negative; b2_261 "pain lol" -> negative |
| R3 | **Skeptical or risk-framed text is negative when the net tone is skeptical**, even without a stated loss. A question with no lean either way is neutral. | b2_278 "Excellent Business, Down 40% And Still Expensive" -> negative; b2_141 "alts still look pretty dead" -> negative; b2_257 "how will they monetize AI?" -> negative; b2_181 "Can quantum break Bitcoin?" -> negative; b2_081, b2_265 -> negative |
| R4 | **New initiatives and favourable policy are positive.** A company announcing a build-out or product initiative, or a policy removing a burden, leans positive; naming favourite holdings is positive. | b2_119 Google Cloud blockchain -> positive; b2_039 tariff removal -> positive; b2_277 "favorite names" -> positive |
| R5 | **Mixed analyst or rotation lists are neutral**, and a bare product-launch headline with no stated benefit is neutral. A list takes a direction only if one side clearly dominates. | b2_045, b2_096 PT lists with cuts -> neutral; b2_228 launch headline -> neutral; b2_268 fund dumps AMD, rotates into IREN -> neutral |
| R6 | **Factual filler is neutral, not excluded** (decided 20 Oct). Fund-holdings lists, earnings-call transcripts, scheduled-event notices, generic how-to articles and roundups with no stated direction are neutral. Tag them `low_information = true` so results can be reported with and without them. | b2_019, b2_037, b2_072, b2_077, b2_093, b2_099, b2_124, b2_155, b2_192, b2_197, b2_287 (all neutral; Dom had excluded them) |
| R7 | **Exclude only** spam/advert/SEO market-research boilerplate, a bare mention list or thread header, non-English text, or text too fragmentary to judge. | unchanged from v1 |

## Retained from v1
- Positive/negative/neutral definitions; mixed items are neutral unless one side clearly dominates; questions are neutral unless they state a lean (but see R3).
- Labels are assigned blind to FinBERT's output and (for the relabel) blind to Dom's labels.
- Labeller identity, rubric version and date are stored in every label file.

## Note on R1
Dom labelled the two position-gain items positive. R1 follows the rubric's definition (value of the asset), so Dom's
independent labels on this kind of item will disagree with v2 by construction if his own instinct is the author's outcome.
Disagreements of this type are reported as such in the paper rather than hidden.
----- ITEMS (50: c2_1600 .. c2_1649) -----
[c2_1600] (reddit, AMZN) Is too much money in a HYSA a waste of capital? I have $72.5k in an Amex HYSA, $47.5k in stocks (GOOGL, AVGO, AMZN, NVDA, TCEHY), $7.5k in a Roth IRA (VOO, VXUS), and a few thousand in crypto (BTC). I try to keep around 10k min in my bank account between paying off loans and expenses. I realize my investments are all in tech and therefore risky. Would it make more sense to diversify and invest more into stock or have a safety net with higher risk stocks?
[c2_1601] (reddit, PZZA) PZZA-Papa Johns Buyout Investment firm Irth Capital is working with Papa John's International's [(PZZA.O), opens new tab]( largest U.S. franchisee, who controls ​around 10% of the pizza chain's domestic restaurants, to take the company private, three sources told Reuters.
[c2_1602] (twitter, AAPL) $AAPL Apple Inc.'s latest iPad Pro, renowned for its thin design, has sparked discussions within the tech community since its unveiling by Tim Cook in May. However, despite Apple's efforts to enhance its structural integrity, the thin profile of the device has raised concerns [Mr Chart Norris tweeted about AAPL]
[c2_1603] (twitter, IBM) $IBM Weekly chart still looks good but took too much time and killed my options... I need a big spike :) [Shardi B Trading #teamshardi tweeted about IBM]
[c2_1604] (twitter, SPCE) $SPCE Interesting sub $10 name with a high short float at 20%. Tightening up on decreasing volume following an earnings gap. Could get fun over 2.35 [Albertrays tweeted about SPCE]
[c2_1605] (twitter, SPY) $SPY 1H Chart Breakout & Retest , lets see if the retest holds - > price will form the right shoulder of IH&S [Mr Chart Norris tweeted about SPY]
[c2_1606] (news, A) Agilent Introduces CrossLab Connect Services for Lab-wide Asset Monitoring
[c2_1607] (twitter, RIG) $RIG Daily Chart Bullish flag breakout on daily candle chart [Mr Chart Norris tweeted about RIG]
[c2_1608] (twitter, GRPN) $GRPN weekly Pretty obvious HTF structure change Nice bounce on the weekly clouds/previous lows Weekly RSI reset Looks good for some uppies [StonkTrump tweeted about GRPN]
[c2_1609] (twitter, SHOP) $SHOP breaking 200sma in the PM [rachels_44 tweeted about SHOP]
[c2_1610] (reddit, XRP) BTC crosses 117 000 USD Okay, I definitely didnt expect this but a few minutes ago BTC ACTUALLY crossed the 117K line. What are the reasons for this price movement since yesterday? Is it because the dollar is going down only? Is it because Elon Musk quit the gov? Is it because the tariff situation is still a mess? Bitcoin hit $117,080, up over 60% since April, while Ether hit $3,000 and XRP jumped 5% to $2.6. [
[c2_1611] (reddit, TSLA) Hitting the jackpot of Xiaomi Yu7, earn a small fortune Not big bucks, but still some gain. Has been playing around in stock market with 50k investment for a while. Knowing that Xiaomi is about to release the Yu7 electric SUV which is supposed to be popular. Getting the stock around 34 and sold out at 39.6. First time getting some real money. Not a big shot, but still my first time gain.
[c2_1612] (reddit, BTC) Buy Bitcoin I will be in Perth in the next few months. Can you tell me where I could buy BTC without KYC? I'm coming from outside Australia and will be staying for a few years. Thanks to all community!
[c2_1613] (twitter, HOOD) $HOOD Daily Chart ⭐️ Bullish flag - daily candle closed right above the lower boundary of the flag in the form of morning star doji [Mr Chart Norris tweeted about HOOD]
[c2_1614] (twitter, U) $U Daily Candle Chart Testing the upper boundary of mini falling wedge pattern [Mr Chart Norris retweeted Mr Chart Norris about U]
[c2_1615] (twitter, BAH) $BAH Mega bullish looking chart but maybe one to monitor for a breakdown to short. [Nebraskangooner tweeted about BAH]
[c2_1616] (twitter, NFLX) $HOOD - Have room for one more DCA and will save that if we go lower. People seem to forget Netflix subscribers dropped and they did a 100% 6 months later. This is an app I use daily and there isn't anything else like it. Fundamental play for me and happy to hold for a year. [IncomeSharks tweeted about HOOD]
[c2_1617] (twitter, COST) $COST relative strength this morning. Nearing a trendline resistance on daily [rachels_44 tweeted about COST]
[c2_1618] (reddit, NVDA) $13,000 in 1.5 hours. THANK YOU NVDA GAINS. on my way to $100,000
[c2_1619] (reddit, AAPL) NVDA lifetime return 456,705% I’m sure there’s someone out there that bought day one and is still holding. Even a tiny amount at the beginning would be a fortune today. How much have you made on NVDA over the years? I know it’s been stagnant for nearly a year now. Are we about ready to make another leap forward?
[c2_1620] (reddit, QQQ) Turned $700 into $70,000 in 3 days I’m a compulsive gambler. Since March of last year I’ve basically lost all my money sports betting. I had over $50k. I decided to throw my last $700 into 0dte SPY and QQQ calls. I made my money back and I hope to never gamble again.
[c2_1621] (twitter, OI) $AAPL Seeing some chatter in regards to this flow from today. IMO this is indicative of bullish put writing; two below bid orders to start, yellow so we know it exceeded OI, spot stays the same and price drops (between first two fills). Everything traded in under a minute. [ThetaWarrior tweeted about AAPL]
[c2_1622] (news, AB) AXA Equitable creates new subsidiary to hold AllianceBernstein interests
[c2_1623] (reddit, MU) Chat, i did it. I did it. Almost. Damn whats with this moomoo app bugging, made my day when i woke up for a split second 💀 i guess i can be happy for showing a screenshot of my dream return lol. \+$51.9K gain is in my local currency and that would be my 100% gain for my portfolio. Gotta love the MU news. SNDK up another 50% for me premarket today.
[c2_1624] (twitter, MSFT) $MSFT Daily Candle Chart 🔨 Approaching the upper boundary of bullish flag Breakout is matter of time 425 is processing........... [Mr Chart Norris retweeted Mr Chart Norris about MSFT]
[c2_1625] (twitter, QQQ) Something to watch on $QQQ Has a 1hr level inside a larger 4hr level which we have been trading in [JMA_Trades tweeted about QQQ]
[c2_1626] (reddit, BTC) Could crypto currencies like BTC becomes geopolitical hedge assets? What thought you are having on this above question.
[c2_1627] (reddit, CVX) Will Occidental Petroleum $OXY keep it's gains after the Iran war stops? The stock's price range was $35-45 before the war. There are too many so called analysts putting price targets of $63-70 on $OXY, but if oil per barrel goes back to $50-70, is this realistic? Have been a long term disappointed shareholder due to it's inability to get out debt, though it's now below $15 versus $55 billion in 2019- over priced Anadarko acquisition. $OXY has never come close to the returns of $CVX and XOM.
[c2_1628] (twitter, CVX) $TSLA - Upgraded at UBS, PT $1,100 $FCX - Upgraded at Credit Suisse, PT $38 from $32 $TWLO - Initiated at Scotiabank, PT $215 $CVX - PT $202 from $190 at Credit Suisse $COP - PT $127 from $99 at Societe Generale $MDB - PT $375 from $325 at RBC $RBLX - PT $30 from $24 at Jefferies [rachels_44 tweeted about TSLA, FCX, TWLO, CVX, COP, MDB, RBLX]
[c2_1629] (twitter, PYPL) In the midst of the 2023 tech rally, it's pretty crazy $PYPL hasn't caught more of a bid yet. Looks like it's on the verge of breaking out of the base consolidation range it's been stuck in since mid 2022, and nearing a break above the 200 day EMA for the first time since late… [K A L E O tweeted about PYPL]
[c2_1630] (reddit, DUOL) DUOL recovery Longing DUOL calls into earnings as AI solves translation but doesnt solve man’s addiction to feeling productive/learning. DUOL is positioned to leverage AI to improve teaching outcomes as language fluency becomes a new social class indicator.
[c2_1631] (twitter, SOFI) $SOFI Retested its base breakout at $8. Now tightening up in a weekly flag. [Albertrays tweeted about SOFI]
[c2_1632] (reddit, BULL) Is the $71K Pump a Bull Trap? Why Analysts Are Calling for a $50K Bitcoin Crash
[c2_1633] (reddit, META) 31k painful gain on META This one took my margin call virginity when it was down over 80% in April. Made it out with 100% gain after many months of hopelessly bagholding. Is it really gambling if I win?
[c2_1634] (reddit, COIN) Bitcoin addresses with over 1,000 coins surge during recent market dip
[c2_1635] (twitter, SPY) Just noticed the 200dma lines up perfectly with my demand zone on $ES_F $SPY [JMA_Trades tweeted about SPY, ES=F]
[c2_1636] (twitter, AAPL) $AAPL #AAPL roadmap for the next few weeks / months. [_bread911 tweeted about AAPL]
[c2_1637] (twitter, SPOT) The collapse in most growth stocks has had ZERO to do with rates, and all to do with a collapse in their biz & profit outlook. $SPOT is a prime example. In February, market expected +$1.9 EPS in ‘23. Now? A $2 loss! Music streaming is fine, yet shareholders get nothing. Zilch. [CryptoParadyme 🔁 ecommerceshares about SPOT]
[c2_1638] (twitter, PLTR) $pltr up 2.2% nice start 🤝 > [@StonkTrump]( > Nice spot to bid $PLTR [StonkTrump quote tweeted StonkTrump about PLTR]
[c2_1639] (twitter, CAT) $CAT 11.50 points move to the upside since Friday [Mr Chart Norris tweeted about CAT]
[c2_1640] (twitter, RIVN) $RIVN (weekly) breaking below box range following rejection at downtrend line and channel resistance. [kpak tweeted about RIVN]
[c2_1641] (twitter, NL) $FI On track to revisit $0,020 and going higher after that. Strong project, will run big in coming months IMO. [Ed_NL tweeted about FI]
[c2_1642] (twitter, GOLD) Confirmation of daily bullish divergence on Gold, Silver as well as on Miners and junior miners. It’s getting ready for the perfect storm.. > [@graddhybpc]( > That is one very interesting, larger bullish reversal candle at present for $GOLD, after having undercut May 1786 low. And the volume for both #gold and #silver today so far is massive. [JJcycles 🔁 graddhybpc about GOLD]
[c2_1643] (twitter, TSLA) $TSLA Tesla has received approval from the Canadian government to deploy its steer-by-wire system, paving the way for Cybertruck sales in Canada. [Mr Chart Norris tweeted about TSLA]
[c2_1644] (reddit, BULL) Gold will keep blowing so many accounts for a long time until the market settles down. For a M5 candle that continuously prints 500pips…
[c2_1645] (twitter, META) $META Channel worth watching. [Nebraskangooner tweeted about META]
[c2_1646] (reddit, MSFT) Last week , I saw that MSFT was undervalued and tried a 420c lotto, now it's carrying the port 🙉🎉 Eventually I trimmed down to only 10Cs left, just to let runner runs! #TakeProfits
[c2_1647] (twitter, AAPL) Taking some profits off the table on the open on this $AAPL short. Still looks great! [@Amdtrades]( I've added to my $AAPL short this morning. I will continue to destroy this market. [Amdtrades 🔁 Amdtrades about AAPL, AAPL]
[c2_1648] (reddit, NFLX) Netflix's next attempt at growth takes on TikTok and Instagram Reels
[c2_1649] (reddit, UNH) Are there any undervalued sectors out there now? I don't see any. I'm sure there are undervalued individual stocks. But I prefer to invest in a basket of stocks in an undervalued sector/area. The last one was with the Medicare Medicaid policy providers (i.e. UNH, MOH) which were out of favor due to policy changes, but they've now recovered a good 10/20%.
