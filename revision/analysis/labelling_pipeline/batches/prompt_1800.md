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
----- ITEMS (50: c2_1800 .. c2_1849) -----
[c2_1800] (twitter, SPY) $SPY still making lower highs and lower lows... bearish below 390-392 (lower high), looking to short low time frame bounces🐻 [BTstocktrading tweeted about SPY]
[c2_1801] (twitter, ROKU) $ROKU everything about this chart was bullish and gave you the bottom > [@kholov23]( > $ROKU Daily Chart Update 💥 > > Bullish divergence in conjunction of beautiful hammer at the bottom of tightening falling wedge pattern = Falling Wedge BREAKOUT and target 🎯 > > 2 points move to the upside if you caught the move [Mr Chart Norris quote tweeted Mr Chart Norris about ROKU]
[c2_1802] (reddit, AMZN) Meta Core Reinvestment Strategies for Social Network Profits
[c2_1803] (reddit, HSBC) Tether hired top HSBC gold traders, then cut them weeks before auditors arrive
[c2_1804] (twitter, SPY) $SPY Price retested the breakout post FOMC and tightened up on low volume while the 8 day caught up. Now breaking out through a small flag... Gap to fill above $420. [Albert_trades tweeted about SPY]
[c2_1805] (reddit, NOW) I want my gains $NOW Someone tell me to sell😭 everything just happened so fast in two days. Using a lot of margin
[c2_1806] (reddit, GOOG) Copper etf as an 18 year old I’m 18 and currently building my investment portfolio, but most of it is pretty tech-heavy (QQQ, NVDA, GOOG, etc.). I’m looking to diversify a bit into other sectors. Lately I’ve been reading about copper demand from EVs, electrification, and AI/data centers, and it seems like copper could have strong long-term demand. I was looking at COPX (Global X Copper Miners ETF) as a way to get exposure to copper through mining companies. Do you think COPX is a good buy?
[c2_1807] (reddit, BTC) Is Solana a good buy and hold? I'm new to Crypto and bought some of BTC and SOL. As per my study, I think SOL has good long term potential and can be a real game changer as adoption and acceptance increases. What's your take?
[c2_1808] (twitter, SNAP) $SNAP chart looks bearish. Here's my next support for it. Er tonight too. Flows are bearish. You neve know with ER though. 2/4 $15p and $20p targeting this week has high volume. [Honeydew0101 tweeted about SNAP]
[c2_1809] (twitter, PM) $PM 🟪🟪 Philip Morris Intl Q1 2024 Adj EPS $1.50 Beats $1.41 Estimate Sales $8.793B Beat $8.468B Estimate [Mr Chart Norris tweeted about PM]
[c2_1810] (twitter, GS) Taking profits on many long positions on the open :) Amazing way to start the week $GS $HD $FB [Amdtrades tweeted about GS, HD, FB]
[c2_1811] (twitter, CVS) $CVS Would want to see this support hold for upside continuation. [Nebraskangooner tweeted about CVS]
[c2_1812] (twitter, DJT) $DJT On July 3, Trump Media and Yorkville finalized a SEPA agreement allowing Trump Media to direct Yorkville to buy up to $2.5 billion in shares. [Mr Chart Norris tweeted about DJT]
[c2_1813] (twitter, TSN) $TSN 4H morning star potential [Nebraskangooner tweeted about TSN]
[c2_1814] (twitter, LULU) $LULU It's just going sideways. My conviction is very high that it will break down but it can chop around for a bit here. I don't have puts now, but I'm going to soon if it continues to look bad. [Amdtrades tweeted about LULU]
[c2_1815] (reddit, CC) Selling CSP and CC Options Only I wanna start options trading and only want to start with Cash Secure Puts and Cover calls. What are the most valuable lessons you have learned in this journey where you think you should have understood far earlier in your options trading. I’m in the stock market since 2022 and was only buying and holding shares.
[c2_1816] (twitter, TSLA) $TSLA Top Goon X sell approaching alert given on 2H time frame. Buy approaching was given back in mid-January and price went up 70%+ from there. Sell approaching here could be signaling a 2H shift in trend. [Nebraskangooner tweeted about TSLA]
[c2_1817] (reddit, TSLA) Tesla Q2 Earnings Call Live Transcript This post contains content not supported on old Reddit. [Click here to view the full post](
[c2_1818] (twitter, CHWY) RT @kholov23: $CHWY Monthly Chart 💥💵 🔨 The problem with Chewy is that, you can't ignore the MONTHLY BULLISH DIVERGENCE PLUS HAMMER 🔨 Ca… [Mr Chuck Norris retweeted Mr Chuck Norris about CHWY]
[c2_1819] (twitter, DE) $DE Oppenheimer analyst Kristen Owen maintains an Outperform rating on Deere and lowers the price target from $465 to $448. [Mr Chart Norris tweeted about DE]
[c2_1820] (twitter, ES) $ES_F so far double topped at overnight highs [kpak82 tweeted about ES_F]
[c2_1821] (reddit, DTE) WHAT A BAILOUT I just so happen to be holding 270 OTM 2 DTE NVDA calls when Trump tweets this super corrupt shit. A win is a win!!!
[c2_1822] (reddit, AI) Should I stay in SoundHound AI? Basically the title. I’ve got some shares in the company at an average of $9.30, and have held off on buying at their lowest points because they consistently seem to be doing pretty bad. Missed my chance to sell at $22ish last October. Is there any good reason to think their stock might soar in the future, or should I wait for the next decent peak and get out? At what share price might it be worth buying more? Thank you!
[c2_1823] (twitter, AMZN) $AMZN Breakout / Retest / Bouncing ……. > [@kholov23]( > $AMZN Monthly Chart Update 0.4 ☕️ > > BREAKOUT [Mr Chart Norris quote tweeted Mr Chart Norris about AMZN]
[c2_1824] (reddit, BTC) Bitcoin ETFs bought $1.42B in BTC last week - Strongest inflows since October Source: [
[c2_1825] (reddit, BULL) Thanks to $CAR, I finally made gains big enough to share them on WSB. After $CAR had been fully juiced and proceeded to dump, I bought 2 OTM puts at $250 and $200 strikes on the CHEAP and sold for a 1009% gain and an 861% gain, respectively.
[c2_1826] (twitter, AAPL) 🐳WL 1/14/22 $TSM $143c above $142.40.$135p below $136.35 $F $26c above $25.86. $24p below $24.80. $AAPL $175c above $173.31.$170p below 171.60 $FCX $46c above $45.67. $43p below $44.16 Happy Friday! [Honeydew0101 tweeted about TSM, F, AAPL, FCX]
[c2_1827] (twitter, META) $FB Meta Platforms Q4 EPS $3.67 Misses $3.84 Estimate, Sales $33.67B Beat $33.38B Estimate [Honeydew0101 🔁 RyanRozbiani about FB]
[c2_1828] (twitter, DJT) Imagine he mentions $DJT > [@ICODrops]( > Former President Donald Trump is in talks to speak at crypto conference Bitcoin 2024, according to Axios. > > [TraderSZ quote tweeted ICO Drops about DJT]
[c2_1829] (news, AAP) Earnings missed, margins slide and outlook tightened for Advance Auto Parts
[c2_1830] (reddit, COIN) The Mechanism Most people think buying is what pushes the price higher. It isn’t. What matters is whether coins return to the market —and increasingly, they don’t. Liquidity disappears not because demand spikes, but because conviction removes supply. *The Ownership Shift* — **HODLHero ₿**
[c2_1831] (reddit, BULL) Covered Calls gone wrong 😭 I’ve been making good pocket money last couple of months selling CC for NBIS. Welp, I guess I’ll be switching to Covered Puts to reacquire the stock back.
[c2_1832] (reddit, GME) A funny COHENcidence! The vote regarding Ryan Cohen's compensation package is scheduled for June 8th, which also happens to be Keith Gill's birthday. Just saying, I will vote yeeeeeeeeeeeeeeeeeeeeeees GameStop reported quarterly earnings of $0.49 per share which beat the analyst consensus estimate of $0.37 by 32.43 percent. This is a 63.33 % increase over earnings of $0.30 per share from the same period last year.
[c2_1833] (reddit, TSLA) Record month for option income Wow, August was a record-breaking month for options trading income! Should I consider this to be an outlier month, or are we going to have this trend going into September. How was your August overall? Share your thoughts on August and predictions for September! I only sell PUTS and CALL not a buyer. 90% of my trades are CSP and CCs.
[c2_1834] (twitter, BLNK) $BLNK Pushing out of support zone. I never did take a position but looks solid. [Nebraskangooner tweeted about BLNK]
[c2_1835] (reddit, GOLD) Oil tumbles after Trump postpones U.S. strikes against Iran energy infrastructure for five days Oil prices tumbled Monday after President Donald Trump said the U.S. and Iran had productive talks, leading him to order a halt on strikes against key energy infrastructure in the country. West Texas Intermediate futures were down 8% at $90.10 per barrel. Brent crude lost around 8% as well to trade at $103.91.
[c2_1836] (reddit, ATR) What indicator is best for identifying choppy vs trending markets? I need to switch between my chop and trending signal generators. I've tried ATR, ADX, bollinger bands, volatility & donchain lines but they don't have the improvements I'd hoped for. I ended up using my own system's alerts with a 72hr rolling average and this works well for my chop sniper but not for trending and is a homemade solution so unlikely the best I can use. Just wondering what you guys use?
[c2_1837] (reddit, GOOGL) Michael Burry Says Apple Should Buy OpenAI Rival – ‘Apple Can Afford It For Now’ Big Short investor Michael Burry believes that tech titan Apple should acquire a Google and OpenAI rival before it turns profitable.
[c2_1838] (twitter, LAND) $LAND Strong fundamental RWA projects will experience a significant price increase once Bitcoin stabilizes. @Landshareio has a limited supply of only 3.8M tokens ~ all in circulation. Their real-estate-backed RWA token pool value on the Ds swap shows a significant move during [Prof.Noan.Ai 𓃵 tweeted about LAND]
[c2_1839] (reddit, ORCL) The AI circle of money... 1. $NVDA sells GPUs to $ORCL. Oracle uses to power its data centers 2. Oracle builds cloud on those GPUs 3. Nvidia rents compute back via signed deals from Oracle Nvidia gets recurring revenue from hardware sales and cloud services without owning all the data centers. Oracle turns capex (buying GPUs) into opex revenue by renting compute to Nvidia, OpenAI, and others. More AI Stocks Watchlist: $MRVL $AVGO $NBIS $AIFU $IREN $CRWV
[c2_1840] (twitter, SPY) Closed $SPY Calls. Probably done for the day on stocks unless we gap fill but we'll see > [@JMA_Trades]( > $SPY 4hr demand held like a boss💪 [JMA_Trades 🔁 JMA_Trades about SPY]
[c2_1841] (reddit, FLWS) Flws squeeze. Beat earnings by 39.5% Here we go! Not sure how accurate fintel is, I get mixed responses. But FLWS is up about 21% as we speak. According to fintel it has 101% short interest with 25 days to cover. Even if its half wrong, thats still high. Im in at 1800 shares with $4.37 cost.
[c2_1842] (twitter, INTC) Intel beats by $0.18, beats on revs; guides Q1 EPS below consensus, revs above consensus; raises dividend by 5% $INTC [StonkTrump 🔁 The_Real_Fly about INTC]
[c2_1843] (reddit, TSLA) With AI stocks at highs, is anyone else getting more selective with options? Feels like NVDA/AVGO/TSLA are strong, but one headline or macro shift can flip momentum fast. I’ve been tightening entries and exits on calls and bull spreads instead of just riding them because “the trend is up.” Staying aggressive or playing it more tactical? Which one will be appropriate?
[c2_1844] (twitter, MA) Hard intraday rejection on the retest of the bottom of the range. 200d MA continues to roll over and AVWAPs from previous highs continue to act as resistance as well... $IWM $RUT $RTY_F [the_chart_life tweeted about IWM, RUT, RTY_F]
[c2_1845] (twitter, PFE) $PFE - Adding more to my shorts here. OBV at daily resistance, gap filled, and more bearish news seems to be coming out. Looking to add some more PUTS as well. [IncomeSharks tweeted about PFE]
[c2_1846] (twitter, CRWD) $CRWD Weekly Chart Price presently rejecting at supply zone Noting RSI entered the OVERBOUGHT zone on weekly chart Pullback is the best scenario in this case [Mr Chuck Norris tweeted about CRWD]
[c2_1847] (twitter, NVDA) $NVDA Losing a very important level at 207.4 after a bear flag breakdown. [Albert_trades tweeted about NVDA]
[c2_1848] (reddit, TSLA) I have no idea what I’m doing I bought in 2014, and I’ve sold a decent amount of the years. There are probably better investments I can make with this $10k at this point. How should I decide when/if to sell TSLA, realize the gains, and re-invest?
[c2_1849] (twitter, ZS) $ZS 🟦🟦 Zscaler Q3 2024 Adj EPS $0.88 Beats $0.66 Estimate Sales $553.201M Beat $535.906M Estimate [Mr Chart Norris tweeted about ZS]
