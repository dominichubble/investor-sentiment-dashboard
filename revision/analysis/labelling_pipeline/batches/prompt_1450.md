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
----- ITEMS (50: c2_1450 .. c2_1499) -----
[c2_1450] (reddit, BABA) "DISRUPTIVE TECHNOLOGY" Would this be a good investment if I'm looking for a somewhat aggressive investment to hold 10+ years, accumulation, with not too high fees? Thanks: [
[c2_1451] (twitter, EXPE) $BA - Upgraded at Citigroup, PT $209 from $219 $DDOG - Initiated at CFRA, PT $143 $EXPE - PT $118 from $200 at Citigroup $LULU - PT $315 from $303 at MS $RBLX - PT $25 from $27 at MS $GOOGL - PT $2650 from $3600 at UBS $META - PT $215 from $310 at UBS [rachels_44 tweeted about BA, DDOG, EXPE, LULU, RBLX, GOOGL, META]
[c2_1452] (reddit, AON) $AON Been watching the insurance/reinsurance brokers lately. Activity seems stronger than expected this quarter, while hiring looks more disciplined. Feels like the focus is shifting toward improving revenue per headcount. I trimmed some and would look to add back around low 320s if it pulls back. If you can get in @320 levels, I would, earnings 5/1 and I expect a break upwards 🤑
[c2_1453] (twitter, SPY) $SPY 1H Chart Breakout / Retest Scenario Presently price is testing back the previous breakout zone / bull flag trend line / [Mr Chuck Norris tweeted about SPY]
[c2_1454] (twitter, TLSA) Great day for me trading with @BlackBoxStocks @swan1427 and @ThetaWarriorPro. Caught a great trade idea on $GME thanks to Theta. Found my own set up on $TLSA thanks to Swan and his knowledge bombs from a webinar. Good day! [ThetaWarrior 🔁 McmanusWalter about GME, TLSA]
[c2_1455] (twitter, BTC) Are both the US10 year yield and the CRB commodity index telling us where inflation (and thus the dollar) are heading on the medium/short term? #DXY #CRB #US10y If that's happening, then that would be the most bullish thing for #Bitcoin. [JJcycles tweeted about]
[c2_1456] (reddit, ETH) Is ETH's Long-Term Bull Run Just Getting Started? This Data Says So
[c2_1457] (twitter, DKNG) $DKNG > [@kholov23]( > $DKNG this post was the bottom 🤷‍♂️ > 4H Chart - dragonfly doji 🐉 at the bottom [Mr Chart Norris quote tweeted Mr Chart Norris about DKNG]
[c2_1458] (reddit, CSV) Managing the "do not mail" contacts. Is there a program where people who receive mail can "opt out", I can upload my CSV list, it scrubs those "do not mail contacts" and spits out a clean CSV list without those contacts? How do you all handle these people who don't want to receive mail anymore? I know I can setup a custom setup myself, but it's not something I'm looking to do at this time. Thank you!
[c2_1459] (reddit, MSFT) MSFT showing relative strength during market weakness in beat up software sector if it holds above 20 day MA stop 401, target 420 not financial advice
[c2_1460] (twitter, NVDA) $NVDA was a sell the event today, but it did hold most of its gains from the last 3 day very well. I am watching this > $271 now for a move higher if it can break the supply. 275c 📈 Resistance $276.21 🎯 $284.83 🎯 [optionsprochick tweeted about NVDA]
[c2_1461] (twitter, WFC) $WFC Got the breakout I wanted. Now hitting the 22ema. Trade closed. Nice quick profit. [Nebraskangooner tweeted about WFC]
[c2_1462] (twitter, FF) #FF + #ShoutOutSaturday @TrendSpider @AdamMancini4 @Mayhem4Markets @AyeshaTariq @ripster47 @ThetaWarrior @unusual_whales @coiledspringcap @MacroCharts @cfromhertz @korinek_trades @RealJuicyTradez @TraderAmogh @ThePupOfWallSt @snorlax_uw @BlueMoonTrades @traderstewie [ReciKnows tweeted about SHOUTOUTSATURDAY, FF]
[c2_1463] (twitter, AAPL) $AAPL - Once everyone is done obsessing over a do nothing FOMC day we got a lot of gaps above to fill. Don't let a little red day change your bias. [IncomeSharks tweeted about AAPL]
[c2_1464] (twitter, ILMN) $ILMN 🟦🟦 Illumina Q1 2024 Adj EPS $0.09 Beats $0.04 Estimate Sales $1.06B Beat $1.05B Estimate [Mr Chart Norris tweeted about ILMN]
[c2_1465] (reddit, AMC) Here we goes again… Just a reminder, Since the end of 2020, AMC’s share count has grown from roughly 23 million to over 500 million , meaning roughly \~490 million new shares have been added over that period through issuance, conversion, and other equity actions. I’ll buy the stock again when Adam Aron does. And you should too. [
[c2_1466] (reddit, UK) Brown Forman - too cheap to ignore? Yes it’s challenged company in a challenged industry. Young kids drink less wine and spirits nowadays. They also have the tariff issue. But recent earnings numbers were good. Q3 organic sales growth overall was only 1% but emerging market grew 25%. If you model out a high single digit top line growth, the current valuation is too cheap to ignore. The chart is also showing decreased selling momentum, so I’m thinking about adding.
[c2_1467] (reddit, MCHP) MCHP $300M Whale MCHP reported earnings on May 7th with strong results. It looks like some institution or large fund dropped around $300M on \~100,000 09/18/26 ITM MCHP calls a couple days ago as well. For a trade of this magnitude I would expect to see more info/coverage on this so I might be reading it wrong but f it we ball Position: 20x 09/18/26 100C cause i’m greedy
[c2_1468] (twitter, PAA) $PAA Heading towards the upper resistance. I think it breaks through this time for bullish trend reversal [Nebraskangooner tweeted about PAA]
[c2_1469] (reddit, GOOGL) Calculating potential profits How do you calculate how much you can potentially make before purchasing an option contract. I don't want to purchase a contract and not know how much I can make(or potentially lose) on a contract. I use Charles Schwab and trade on the think or swim platform.
[c2_1470] (twitter, ES) $ES_F $SPY $SPX Still plenty to unpack here… but possible bearish case. (I saw tons of feb sub $400 put flow today..) [@ATMSnipes]( $ES_F #SPY $SPX Today was different. This price action is not bullish But the bears current onslaught is almost over (for now) Look back to Oct. 18’.. even 16’ Also Minor W3 @ $380 is a perfect 1.618 of W1 There is a gap at $400 $460 & $480 both Major resistance now.. [ATMSnipes 🔁 ATMSnipes about ES_F, SPY, SPX, ES_F, SPX]
[c2_1471] (reddit, HOOD) Is robinhood’s ira match and gold card worth using robinhood for? Same as title. Can’t decide if it’s worth using robinhood over other brokers for the Ira match and gold credit card. I know that over time that small Ira match growing tax free will be a huge benefit but at the same time you have to stick with robinhood for that. Is it worth swapping over for?
[c2_1472] (twitter, SPY) $SPY Win streak continues. No green to red telegram fam. Raise profit stops [kpak82 tweeted about SPY]
[c2_1473] (reddit, COIN) I have invested in crypto ENA coin Does anyone have a prediction when will it go up actually im in 50 percent loss i am not sure when should i be expecting some increase
[c2_1474] (reddit, META) Mag 7 ranked by TTM Cap Ex spending 1. AMZN - $120.1BN 2. GOOG - $77.9BN 3. MSFT - $69.0BN 4. META - $62.7BN 5. AAPL - $12.7BN 6. TSLA - $8.9BN 7. NVDA - $5.8BN Who is getting the most bang for their buck? Information was pulled directly from the company filings. I'm surprised AMZN spends so much. Is this for their data centers?
[c2_1475] (twitter, AAPL) $AAPL Weekly shaped up well, if market has legs to this rally would like to see $AAPL through $158. [Albert_trades tweeted about AAPL]
[c2_1476] (twitter, SPY) Still trying to figure out how $Spy monthly is bearish at all. Or Bear market period. Bear markets to me start when we have consecutive closes below the monthly lower kelt. We’ve only breached once then recovered. So again tell me because fundamentals says so. Chart says no [HackermanAce 🔁 TradeWithNinja about SPY]
[c2_1477] (reddit, BTC) Best broker to trade BTC, I felt Delta exchange has high charges But can anyone please break down the charges Delta and compare it to other best broker Looking forward to getting started with BTC with best and safe broker after complete understanding so suggest me with example if I have an account of 50K then how much charges I spend for risking 1-2%
[c2_1478] (twitter, SPY) $SPY $QQQ (1H) both with failed breakouts to start the new year. Not bullish. [kpak82 tweeted about SPY, QQQ]
[c2_1479] (twitter, TSLA) $TSLA and $AMZN both showing up on the Doubles scan. Not shabby, not shabby at all. [the_chart_life tweeted about TSLA, AMZN]
[c2_1480] (reddit, NVDA) AMD Gains Pron For Ya Lads Accumulated throughout February buying 5-10k at a time. Timing the market > Time in the market … JK pure luck
[c2_1481] (reddit, AAPL) Is aapl call gonna print About $10,000 205 July call. Figured they promised $50billion in US investmenr and were exempt last time. Plus i'm down 50k from highs on owning the shares too
[c2_1482] (reddit, ADX) keep losing on setups that look perfect - finally figured out why not a strategy problem. was a regime problem the whole time was trading valid setups in ranging markets and getting chopped to death. ADX was sitting under 20 on half my entries and i wasn't checking it at all. once i added that one filter my false signal rate dropped a lot. probably obvious to most people here but it took me way too long to make this connection. anyone else had a similar lightbulb moment?
[c2_1483] (twitter, ES) $ES_F 4H Solid 80pt drop so far, but the structure isn't great. Bears want to see this slice through the fib zone or there's a chance it gets even more complex [rbc_trading tweeted about ES_F]
[c2_1484] (reddit, COIN) Spy rebalancing today. Does it create volatility for the market? Does this rebalancing really affect the market? The third Friday of March, June, September, and December. An high AUM ETF with major daily fund flows. Power hour pump or dump? Or no movement at all? Volatility beginning, during, or right at the end of the close?
[c2_1485] (reddit, TSLA) Michael Burry can't be wrong! $26k worth of TSLA put options for 2026
[c2_1486] (twitter, ROKU) $ROKU Benchmark analyst Daniel Kurnos reaffirms a Buy rating on Roku and maintains the price target at $105. [Mr Chart Norris tweeted about ROKU]
[c2_1487] (twitter, CVNA) $CVNA Daily Chart Descending triangle breakdown and holding above Swing Low ( April 16 ) AVWAP Yet doesn't make sense above 100 🤷‍♂️ [Mr Chart Norris tweeted about CVNA]
[c2_1488] (twitter, TSLA) $ARKK Weekly. #ARKK broke out from the falling wedge, near-term resistance levels 42/45. $TSLA $SPY $QQQ $IWM $DIA [reciknows 🔁 reciknows about ARKK, IWM, TSLA, SPY, QQQ, DIA]
[c2_1489] (twitter, COIN) $COIN - Stocks moving like #Crypto again. 18% gain in a day almost hitting the first target. Hoping for another pullback to add more. > [@IncomeSharks]( > $COIN - Stock looking promising. Finally some large volume coming into this one. [IncomeSharks 🔁 IncomeSharks about COIN, COIN]
[c2_1490] (twitter, RSI) RT @Banana3Stocks: $PANW $QQQ $SPY PANW 4hr chart analysis ❤️ Bullish divergences in MACD and RSI 🤔 Broadening formation as volume incr… [Ace retweeted about QQQ, PANW, SPY]
[c2_1491] (twitter, SPY) Finally took some of my own advice and shorted the open lol $spy $sqqq $spxu [@StonkTrump]( $ES Nice bounce off the 200ema, congrats if you grabbed calls yesterday However, throwback into previous lows and the 20/50ema region seems like a bearish retest to me, might be worth a short and look for the downside niner tomorrow Reclaim 20ema and ill look for upside [StonkTrump 🔁 StonkTrump about SPY, SQQQ, SPXU, ES]
[c2_1492] (twitter, NVDA) $NVDA Bull sweepers active. [ThetaWarrior tweeted about NVDA]
[c2_1493] (reddit, BULL) Bought the LUNR dip 🫡 Shit went from $45 to $11. Bought and slowly climbing. Rocket companies go to the moon right
[c2_1494] (twitter, NL) $FI On track to revisit $0,20 and going higher after that. Strong project, will run big in coming months IMO. [Ed_NL tweeted about FI]
[c2_1495] (twitter, SPY) $SPY Gap filled. It's very close to the 200 ema again @ $457.51 , could bounce here now [@optionsprochick]( $SPY a little bit of gap filling action happening. Nothing exciting on the upside unless $SPY $447 comes. [optionsprochick 🔁 optionsprochick about SPY, SPY, SPY]
[c2_1496] (reddit, AMZN) AMZN AWS YOLO Tariffs don't mean much for Amazon with more and more of their profits coming from AWS I feel they are gonna set a new ATH by the end of the year! I want more contracts but don't want to over leverage myself again haha. Do you think my strike price is too low for how high it can go this year?
[c2_1497] (twitter, ES) Here we go…. > [@MFHoz]( > This is what I see for S&P 500($ES, $SPX) > > Pullback to 3400, then a possible bear market rally up to 3900 again. [MFHoz 🔁 MFHoz about SPX]
[c2_1498] (twitter, TSLA) $TSLA #TSLA Volume-backed breakdown today. "Balloon" breakouts/breakdowns generally have continuation in the direction of the break for at least 1-3 days. [Nebraskangooner 🔁 Jake__Wujastyk about TSLA]
[c2_1499] (reddit, MU) Wife’s Account MU$ average flex Bought about 1k$ worth of MU$ sometime during Covid for wifey. She told me good job, and that she will allow me a 100$ buy-in for me and her to play online roulette tonight together before she heads over to her bull’s house. Feeling grateful.
