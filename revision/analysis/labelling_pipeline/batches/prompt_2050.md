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
----- ITEMS (50: c2_2050 .. c2_2099) -----
[c2_2050] (reddit, BULL) Never pays to be a gay bear Keep selling this POS Sandick yet it never stops pumping
[c2_2051] (news, A) Agilent Technologies (A) Surpasses Q4 Earnings and Revenue Estimates
[c2_2052] (twitter, FCX) $fcx ARE YOU NOT ENTERTAINED ! THE COPPER IN MY BLOOD IS FLOWING. $TGB next, so bullish it hurts. [HackermanAce tweeted about FCX, TGB]
[c2_2053] (twitter, BABA) $BABA Daily Chart Price moving within falling wedge pattern They say, use Fear and Greed Properly [Mr Chuck Norris tweeted about BABA]
[c2_2054] (twitter, GOLD) $GOLD, W. Without new high in % or $, Gold will bottom here. #gold [JJcycles 🔁 Jhinster1 about GOLD]
[c2_2055] (reddit, NVDA) How to take advantage of possible end of China Trade War? What should we put in pre-market BUY LIMIT ORDERS for to take advantage of the China Trade War possiblt being over?? TSLA? AMZN? APPL? NVDA? TSMC? SOXL? TSLL? Stocks badly impacted by China embargo will erupt if a good deal is reached. I figure a stock like PLTR would not be a good choice since they rallied big this year because they had no China exposure.
[c2_2056] (twitter, ABBV) $ABBV 🟦🟦 AbbVie Q2 2024 Adj EPS $2.65 Beats $2.57 Estimate Sales $14.462B Beat $14.027B Estimate [Mr Chart Norris tweeted about ABBV]
[c2_2057] (reddit, ETH) Bitcoin Is Dropping Because Nobody Knows Which Trump Shows Up Tomorrow.
[c2_2058] (twitter, DBI) Really finding a lot of these types of very attractive setups over the last few sessions. On an additional note, this meets the "didn't revisit its respective July base breakout on the latest correction" criteria that I mentioned last night. > [@G_krupins]( > $DBI Held the old high. If it can move up just a bit (through AVWAP from high) might add w/ stop no lower than today's low [the_chart_life 🔁 G_krupins about DBI]
[c2_2059] (twitter, AAPL) $TSLA vs $AAPL some similarities in the weekly charts 👀 is $AAPL's move to the downside just getting started? [BTstocktrading tweeted about AAPL, TSLA]
[c2_2060] (twitter, AAPL) $AAPL sell off probably overdone per Dan Ives > [@DivesTech]( > On Apple reports from China we continue to focus on demand remains firm in this key region based on our latest iPhone checks. Some of the shadow China government bans will remain but does not move the needle that much in our view. Overall firm demand into 24. 🍎📱🐂 [Shardi B Trading #teamshardi quote tweeted Dan Ives about AAPL]
[c2_2061] (news, ABR) Arbor Realty Trust, Inc. Announces Pricing of $–'… million of 5. ...
[c2_2062] (reddit, AMD) The S&P 500 broke 9-day winning streak. Tensions in the Middle East have risen again and oil prices will complete 3-day winning streak. Weekly U.S. crude oil inventories were expected -2.9M, but it fell by -7.974M. It's the biggest drop since February. On the other hand, S&P 500 is still up more than 10% since start of the year.
[c2_2063] (reddit, RITE) RITE AID ( RAD ) AFTER HOURS ON EVE OF Q1 EARNINGS ANNOUCNEMENT HEADLINE READS: "Rite Aid CEO - Our vaccine rollout was much bigger than expected". [Stock is rising after hours on comments by CEO. Earnings announcement due out before market open Thursday, June 24, 2021.]
[c2_2064] (twitter, BIDU) $BIDU > [@kholov23]( > $BIDU 4H Chart Update > > Lower boundary of falling wedge in conjunction of bullish divergence = a sexy bounce we got [Mr Chart Norris quote tweeted Mr Chart Norris about BIDU]
[c2_2065] (reddit, GOOGL) What is your definition of a value stock? I've posted several times in this sub that I compare a company's Enterprise Value/Operating Profit multiple with the sum of its Operating Margin plus Projected Revenue Growth to calculate what I call a Value Score. I consider a score above 2.0 as a value candidate. Here is a list of stocks with a market cap above $100 billion sorted by Value Score: [Mega Cap Value Stocks]( How do you define value stocks?
[c2_2066] (reddit, DTE) 10k 1DTE MU Yolo 💫 This is easily the most degenerate shit I’ve ever done. Up 18k off of 3k this week. Not holding over the weekend. MU is insane
[c2_2067] (reddit, NVDA) How do you feel about AEVA? Title, they recently got a deal with NVDA that nearly double their stock in a week. According to some its greatly undervalued and to others it's high risk due to cash burning among other things. I feel like NVDA just pulled this company off the ground assuming they don't mess it up. I just haven't seen anyone talk about it or maybe I haven't looked hard enough. Thoughts?
[c2_2068] (reddit, ETH) Oil at $100, war in the Middle East, dollar strengthening. Yet people still surprised crypto payments haven't gone mainstream. The infrastructure has been there for years.
[c2_2069] (twitter, BTC) Possible chance I cut at the low of the day but I'd rather just re-enter with additional confirmation so it's okay. Waiting now. #BTC [Omz tweeted about BTC]
[c2_2070] (reddit, AMZN) Is it better to wait for green days days to sell covered call? I have Amazon stocks with 221$ average and been selling calls from last 2 months. But as the stock is quite down, is it worth starting new one now, I generally try to do monthly ones as the stock can rise a bit and might be become tricky and premium might not be that great. How do you guys do? Wait?
[c2_2071] (twitter, CCJ) There's your top Uranium name starting the week off with a push to new highs... $CCJ [the_chart_life tweeted about CCJ]
[c2_2072] (twitter, AAPL) $AAPL Failed to hold the 200ema today & sold into the 50ema. To me it's still weaker looking & if it can't hold the $145, it looks like it can move lower. For any upside move, AAPL would need to recover the $150 to start to move up. [optionsprochick tweeted about AAPL]
[c2_2073] (twitter, SPY) $spy with the gap up! Alot of u played the fta bounce. You should take profit early! [HackermanAce tweeted about SPY]
[c2_2074] (twitter, MO) Its been a great week! Just a reminder Instead of spending your profits on material things that you are only going to use once or twice, buy some shares of your favorite stock or a high dividend stock. $SCHD $PG $MO $INTC $ABBV etc. Trust me, your future self will thank you. [JMA_Trades tweeted about INTC, ABBV, SCHD, PG, MO]
[c2_2075] (twitter, SPY) Gonna flip some calls into today's event $spy [HackermanAce tweeted about SPY]
[c2_2076] (reddit, GME) GME Polymarket odds just fell off a cliff! Dropped from 90% to 62% in hours. What is going on?! 📉 [6H CHART]( So earlier, the prediction market was pricing in a massive 90% chance for GameStop to beat its quarterly earnings. Now, looking at the chart, it just took a sudden, massive nosedive straight down to 62%. How is a sudden drop like this even possible on a prediction market? Would love to hear your theories on what could cause such a violent swing out of nowhere.
[c2_2077] (twitter, SPY) $SPY Continues to rally off the bottom Bollinger Band (-2 Std. Dev). > [@ThetaWarrior]( > $SPY DTL breakout and continuation higher, price now pushing above the median range. [ThetaWarrior 🔁 ThetaWarrior about SPY, SPY]
[c2_2078] (twitter, QQQ) LCG and SCV tilts going in opposite directions here to start the week. $QQQ $IWM [the_chart_life tweeted about QQQ, IWM]
[c2_2079] (twitter, SN) $SN 🟦🟦 SharkNinja Q1 2024 Adj EPS $1.06 Beats $0.99 Estimate Sales $1.066B Beat $949.881M Estimate [Mr Chart Norris tweeted about SN]
[c2_2080] (twitter, CMT) $GTES nice breakout [Ian McMillan, CMT retweeted kkern about GTES]
[c2_2081] (twitter, NFLX) oof > [@TrendSpider]( > $NFLX Netflix down -8% AH following Q1 earnings report! 🩸 > > ~EPS: $2.88 vs $2.86 est > ~SALES: $8.16B vs $8.18B est [CryptoParadyme 🔁 TrendSpider about NFLX]
[c2_2082] (twitter, RSI) $AAPL basic monthly time frame technicals to take note of: 1)+44% rally in only 4 months. 2) Extensive 4 consecutive close higher with 4 straight gap ups. 3) Progressively smaller candle bodies ending with current hanging man at ATHs. 4) RSI most overbought in 2 years. [kpak tweeted about AAPL]
[c2_2083] (twitter, BLK) Very weak indeed... $BLK [@verrone_chris]( From today's research... don't ignore the message from $BLK - the Russell 1000 constituent with the highest correlation to the market itself over the past decade (.82). Breaking down in both absolute & relative terms. @StrategasRP @strategasasset [the_chart_life 🔁 verrone_chris about BLK, BLK]
[c2_2084] (twitter, META) good fucc’n luck 😂 “checks long exposure…” > [@ROTKofstonks]( > $GOOGL $MSFT $META $AAPL [hyperbitcoins 🔁 ROTKofstonks about META, GOOGL, MSFT, AAPL]
[c2_2085] (reddit, MS) SpaceX revenue could hit $3.4T by 2040 - Source: The same folks taking the company public: Morgan Stanley - (Trust me bro...) I can't believe this is actually news... "Much of the cash will come from SpaceX's AI business" 😂
[c2_2086] (twitter, MS) @Ultra_Calls Definitely bullish $KTRA 🔥 Institutional Ownership • Institutional Owners-38 total • Shares Outstanding- 66M • Institutional Shares - 4M-6% • Institutional Value- $2M USD • There are many major companies invested such as Morgan Stanley, Fidelity, Wells Fargo @Ultra_Calls [Ultra_Calls 🔁 MrMikeInvesting about KTRA]
[c2_2087] (twitter, ES) $ES_F (4H) shooting star backtest rejection. bears in full control [kpak82 tweeted about ES_F]
[c2_2088] (twitter, EXPE) $EXPE I still really like the higher time frames on this chart and think it can do well long term so I'm going to pick up some for my HODL portfolio here Pretty low risk with awesome reward potential Would cut below 99sma which is just beneath [Nebraskangooner tweeted about EXPE]
[c2_2089] (twitter, NKE) $NKE Finally breaking resistance. Expecting a push to at least the 99sma above. Want to see it hold the breakout for upside bias. [Nebraskangooner tweeted about NKE]
[c2_2090] (twitter, SPY) $ES The spooz looks good, but getting overextended and it's probably time to punish fake bulls I'm still looking to long a confirmed range flip with a retest of range highs at the 4408-4425 region This would correspond with the heavy dark pool accumulation on $spy at 441.84 [StonkTrump tweeted about ES, SPY]
[c2_2091] (news, A) Tracking William Von Mueffling's Cantillon Capital Management Portfolio - Q1 2019 Update
[c2_2092] (reddit, NVDA) What are your moves for 2026? So as (hopefully) all of you know, we are already on 2026 and this has brought the stock market many both good and bad news for certain people and companies, but there’s also many things that will most probably arrive this year, things like GTA 6, or google and NVIDIA advancements in AI. Feel free to post your thoughts, your opinions or what you are thinking of buying/selling in the next days, weeks or months. This also helps me btw so yeah, comment
[c2_2093] (news, SAM) Boston Beer (SAM) Rises on Q1 Earnings Beat, Strong Shipments
[c2_2094] (reddit, NFE) Good luck to all $NFE holders entering end of March. This is the week? Hope yes, nothing tehnical, nothing mathematical. Upcoming catalysts: \-restructuring \-Brazil auction \-just survive and not bk
[c2_2095] (twitter, TRNR) $TRNR Solid 50%+ move so far, watch to see if it can break $2 leading into open for continuation > [@Bullish_Trades]( > $TRNR Bang bang. News > > $TRNR Interactive Strength Inc. (Nasdaq: TRNR) Secures Exclusive Distribution for CLMBR in Fast-Growing Gulf Region With MEFITPRO and Receives Six-Figure Initial Order [BullishTrade quote tweeted BullishTrade about TRNR]
[c2_2096] (reddit, MSFT) Astera Labs ($ALAB) is up 15% AH. From the earnings commentary, this is the core of the upside. Relative stocks: $AMD $MSFT $NVDA $BGM $NBIS $CRWV
[c2_2097] (twitter, ES) $ES Big trigger area at 3909.75-3911.50 (fluid) Avwaps from top and bottom converging in same place Both have already acted as support or resistance Will be a big area to watch to the upside, flip and look for longs, reject there can look for shorts [StonkTrump tweeted about ES]
[c2_2098] (twitter, MSFT) $MSFT What do I say about Megas that break All Time Highs?? I added more 50 more yesterday (posted) while we did an s/r test and added yet another 100 short term calls today (Dec 1 375) [Don't follow Shardi B If You Hate Money tweeted about MSFT]
[c2_2099] (twitter, FSLR) $FSLR RBC Capital analyst Chris Dendrinos reiterates an Outperform rating on First Solar and maintains a $315 price target. [Mr Chart Norris tweeted about FSLR]
