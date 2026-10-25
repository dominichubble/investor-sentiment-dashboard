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
----- ITEMS (50: c2_1850 .. c2_1899) -----
[c2_1850] (twitter, HD) $HD #HD This is very strong bull setup as price holds above the volume shelf & lags the broad markets. Target: $318-$321 [Nebraskangooner 🔁 Jake__Wujastyk about HD]
[c2_1851] (twitter, PLTR) $PLTR Daily. #PLTR relative strength breaking down, pullback incoming. Would be a nice buying opp if it re-tests the breakout from downtrend $AI [reciknows 🔁 reciknows about AI, PLTR]
[c2_1852] (twitter, RBLX) $RBLX Tightening up under the $38 level. $38 break should get this name some momentum. Targeting $45. [Albert_trades tweeted about RBLX]
[c2_1853] (twitter, RBLX) $RBLX triple or double but the outcome is the same 📈🐢 > [@kholov23]( > $RBLX Monthly Chart Update > > Double Bottom Formation / W [Mr Chuck Norris quote tweeted Mr Chuck Norris about RBLX]
[c2_1854] (reddit, BULL) $115K to $288K - 150% Gain Buying $CAR Puts Missed the run-up, but timed the drop reasonably well. I've gotten a few scars over the years, but this single trade more than erases them all. Now, I'm just sitting here waiting for SPY to drop so I can cash out some puts I bought over the past couple weeks.
[c2_1855] (reddit, CC) Is 0% purchase CC worth it for general expenditure? I get using 0% CC for a upfront big purchase and avoiding a "normal" loan, but Is there really any value in just 'trickle' spending normal expenses on a 0% card, and putting the money in a HYSA/ISA? Inflation would pretty much negate any interest earned over time, right?
[c2_1856] (twitter, SPY) $QQQ Weekly. #QQQ strongest of all indices, bull flag in play here after retesting 50/200sma. Needs to hold > 285 $NDX $NQ_F $XLK #Nasdaq $AAPL $MSFT $AMZN $SPY $CS $SVIB $SBNY [reciknows tweeted about XLK, NQ=F, QQQ, AMZN, NASDAQ, MSFT, NDX, SBNY, SVIB, SPY, CS, AAPL]
[c2_1857] (reddit, TSLA) JPMorgan warns Tesla could drop 60% as deliveries miss, EV tax credit ends, competition rises
[c2_1858] (reddit, BULL) Softening the blow from bad longs So I sometimes buy OTM LEAPS and unfortunately some of them have turned sour (I don’t expect them to reverse). What are my best options to recoup some of the losses? I was thinking selling a further OTM call to create a debit call spread, but is there a better strategy?
[c2_1859] (reddit, MU) Micron stock, cut my losses or keep buying? I invested in MU a couple months ago, and I'm now down by 20%. I could keep dollar cost averaging, or I can cut my losses now, sell my shares and reinvest them into something safer like SPY. I'm usually quite risk averse, but just this once I thought I'd take a chance by buying into the memory stock hype and then immediately regretted it lol. I invested quite a significant amount into MU too (foolish of me, I know). But what should I do now?
[c2_1860] (reddit, BULL) $72 --> $21k in 24 hours Bought 24 contracts on a whim before close on Monday. Paper handed those within the first hour of trade and fomoed back in with most of it shortly after. I'll probably regret selling it today, but better to take profits and blow it all in a few days rather than just the one.
[c2_1861] (twitter, TSLA) $TSLA Trying to clear 265 Sell Orders Courtesy @bookmap_pro [Ace retweeted Ripster📈📉 about TSLA]
[c2_1862] (twitter, SPY) *beep *bop *boop measured move complete > [@StonkTrump]( > $SPY 15min > > mandatory bullish post at resistance [StonkTrump 🔁 StonkTrump about SPY]
[c2_1863] (twitter, FF) #FF amazing traders @Jake__Wujastyk @XclusiveTrading @Trader_Joe_LPP @Simply_Sim1 @SwingPony @MelStone31 @iluvstocks1 @Elitewarriorpro @LanaTheInvestor @ElaineBenes99 @alex_robartXT @TradesKey @Hunter_407 @alec_zamora @CAMR_Group [Honeydew0101 tweeted about]
[c2_1864] (reddit, CC) Next Investment Advice (40m) 40 - Single - 95k W2 Looking to potential start a dividend section of investments in my taxable account…. This is extra money I would be starting with, not taking away from any current contributions. All Investments are in Growth ETFs 401k - 90.5k - 16k contribution Roth IRA - 188k - 7k contribution Trad IRA - 47k Crypto - 1.1 BTC Taxable - 27k - Use as HYSA currently Checkings/Savings - 25k Home - 198k CC Debt - 0 Car Debt - 0 Let me know if other info is necessary
[c2_1865] (news, AAPL) Remote Working Boosts Peripherals, PC Sales: 4 Stocks to Buy
[c2_1866] (twitter, AMZN) $AMZN saved the market. Woohoo. I'm still not holding overnight but tomorrow will be a fun lotto Friday. 😊 I'm ready for it, are you? [Honeydew0101 tweeted about AMZN]
[c2_1867] (reddit, DTE) SPX 0DTE or 0 brain cells — turned CPI report into +30k Hot CPI this morning → bought SPX 0DTE 6695 puts at open No hedge, no plan, just vibes. Go with the flow moment! Market chopped for a bit then dumped hard. Contracts went stupid fast. Was up big, held longer than I should’ve, then closed for a bit over +30k before lunch. Yeah I know this could’ve gone to zero just as easily. had to get out! Today we eat. Tomorrow I probably donate it back.
[c2_1868] (twitter, TSLA) $TSLA Definitely the most easy & obvious short lately showing the strongest trend Usually the easiest and most obvious ones are the hardest to trade because you wait for a bounce to get short entry and it never comes then by the time it bounces a little you don't want to chase [Nebraskangooner tweeted about TSLA]
[c2_1869] (twitter, BYND) $BYND looks beyond dead. Pretty, pretty crazy how quickly one of the most hyped names in 2019 has completely died. Down 95% from the peak. Chart looks awful. Best case scenario if you're holding this is hoping for exiting on a dead cat bounce. > [@byDevCreates]( > @CryptoKaleo Give me hope for $BYND [CryptoKaleo 🔁 byDevCreates about BYND]
[c2_1870] (reddit, BULL) The US Just Lost 92,000 Jobs. Bitcoin Should Be Rallying. Instead It’s Falling
[c2_1871] (twitter, TSLA) $TSLA [Don't follow Shardi B if you hate trading tweeted about TSLA]
[c2_1872] (twitter, MCK) $MCK 🟧🟧 McKesson Q4 2024 Adj EPS $6.18 Misses $6.34 Estimate Sales $76.40B Miss $79.32B Estimate [Mr Chart Norris tweeted about MCK]
[c2_1873] (reddit, META) Meta Stock Drops as Free Cash Flow Collapses 91% Despite 28% Revenue Growth
[c2_1874] (reddit, BULL) SLS - Been jumping in and out As title states have been jumping in and out of SLS since reading about it here on WSB. I'm a regard, but after reading DD from much smarter people, looking at the math, the ongoing trial, and now the key leaders compensation restructuring to favor a BO I'm more convinced then ever. I haven't ever traded a bio tech company but have heeded the warnings of all the past companies referenced on each SLS post. So holding 5k shares and letting it ride. Fuck cancer.
[c2_1875] (twitter, SPY) $SPY $SPX Two strong possiblities here. Black path is a more complex correction that would take shape as a Daily WXY (3-3-3) The alternate red is that we complete an impulse down and make a simple Daily ABC correction (5-3-5) Which camp do you sit in? [Avery tweeted about SPY, SPX]
[c2_1876] (twitter, SPY) $SPY looking good after holding 412 level this week, flip and hold of 415 resistance was the long trigger today Still has room above towards 426-429ish if we get a positive reaction from PCE and Jackson Hole event on Friday [BTstocktrading tweeted about SPY]
[c2_1877] (reddit, TSLA) Tesla long game Trying to build a position in Tesla. Bought drop attempts sub 300 so far and averaging. In for long term hold.
[c2_1878] (twitter, TSLA) What a stupid fucking cunt. $TSLA [@SawyerMerritt]( Joe Kernen: If he (@elonmusk) had never been born, do you think that the $11B (taxes in 2021) is a zero sum game? Someone else would have paid it? Because he created that company (Tesla) here & it’s this successful, why not celebrate the $11B he’s paying? [CryptoCX1 🔁 SawyerMerritt about TSLA]
[c2_1879] (twitter, TSLA) $tsla Daily isnt bearish yet, make sure if you play puts you take profit and get out until the next setup! Currently looking to see if the tenkan hold which would lead to a nice push up. [HackermanAce tweeted about TSLA]
[c2_1880] (twitter, DBGI) $DBGI got the gapper look - see yall tomorrow 😎 > [@thankskuya]( > $DBGI revs 740%, profit margin 94%, EPS improvement 671% [Ultra_Calls 🔁 thankskuya about DBGI, DBGI]
[c2_1881] (reddit, BULL) Here's How Much Bitcoin Has Surged Past the S&P 500 Since the 2020 Covid Crash TLDR: If you invested $10,000 in Bitcoin back on March 23, 2020, your investment today would be in six figures, at just over $100,000. By comparison, if you invested the same amount in the S&P 500 via index funds, then you'd be up to around $29,000. While both investments would have been highly profitable for you, there's no question that going with Bitcoin has proven to be much better.
[c2_1882] (twitter, AMC) $amc & $gme to the moon🚀 fucc cramer > [@jimcramer]( > It's time for all the stonkers and the GMEsters and the AMCers to move on. You are for the most part, bitter gamblers who know one thing: to blame me for your losses. I know you will keep targeting me, but it is time you go back to a real job and tell your mommies!!! [hyperbitcoins 🔁 jimcramer about AMC, GME]
[c2_1883] (reddit, CD) Rebuilding an emergency fund. Are CD Trees still viable? I was laid off and it took a while to find a new job. During my time out of the workforce I liquidated CDs to fund the household. Im wiling to do the work to find the right rates. Im thinking 1500 per month and 5k per quarter until I have 12 monthly and 4 quarterly and then add and roll over at the same amounts. Can anyone suggest a better plan?
[c2_1884] (reddit, ETH) Ethereum Price Forecast: ETH jumps over 20%, outperforms top cryptos following US-UK trade agreement
[c2_1885] (reddit, MSFT) Wow! Earnings drop sends Microsoft into a slide. Market overreacting? $MSFT is dragging down markets after earnings, yet fundamentals look strong. Revenue growth, cloud dominance, and massive cash reserves are still there. It’s still a $3 trillion company. Are investors overreacting, or is this a real red flag? Thoughts? Source: Blossom Social
[c2_1886] (reddit, CIO) FLWS true short interest?! 🤔 thoughts? Okay everyone, here's an update. Found this gem when looking up FLWS on X. Cash flow positive, Q2 2026 earnings beat. The difference between this and other shit? The fundamentals are changing. New CEO, new CIO, new CMO, and i might be missing some. If you change the treasury shares from buybacks to a positive and add to shareholders equity like Buffett did, the new debt to equity is 1.46 with positive free cash flow. All we need is volume
[c2_1887] (twitter, RR) 1/19/2022 Daily RRGs: Europe ETFs $EPOL $GREK $EWU [the_chart_life 🔁 adaptiv about EPOL, GREK, EWU]
[c2_1888] (twitter, QQQ) $QQQ Daily Candle Chart 🔨 Falling wedge breakout within the right shoulder of inverse head and shoulders [Mr Chart Norris tweeted about QQQ]
[c2_1889] (reddit, META) Today Meta was 537$ and I didn’t cut my losses now I’m in deep ocean 🌊.. Got in , out , in on meta I didn’t realize about the antitrust case on mark Zuckerberg. I had a chance to cut the losses guess not any more.
[c2_1890] (reddit, RS) $AEHL super high CTB , fresh RS , Zero Borrow .. they bought $1M Bitcoin and moving forward with Crypto. Spiked to Equiv. $10+ on Feb 28 . It's very low right now ... something likely coming .
[c2_1891] (twitter, QQQ) $SPY Weekly. #SPY looking like a fake breakout from the downtrend, potential bounce zone between 377 - 371 (~200sma). $SPX $ES_F #SPX #ES_F $XLK $IWM $QQQ [reciknows tweeted about XLK, ES=F, QQQ, IWM, SPY, SPX]
[c2_1892] (twitter, SMA) (2/2) $DIA Daily Complex triple 3 to meet the lower trendline at the October low. Impulsive 5 waves up at (1) and a double 3 correction (2) to perfectly back test the diagonal from the ATH. Needs to quickly recover the 200 day SMA (end of week target). [reciknows 🔁 PikesPeakTrades about DIA]
[c2_1893] (twitter, TSLA) Something to ponder… > [@FrankCappelleri]( Cappelleri): > When was the last time that the $NDX had +90% breadth with $AAPL, $GOOGL, $GOOG, $AMZN, $TSLA $MSFT and $NVDA all down on the day? > > Ever? [Ian McMillan, CMT 🔁 Frank Cappelleri about AMZN, MSFT, AAPL, GOOG, TSLA, NDX, NVDA, GOOGL]
[c2_1894] (twitter, RBLX) $RBLX JP Morgan analyst Cory Carpenter has upgraded Roblox from Neutral to Overweight and raised the price target from $41 to $48. [Mr Chart Norris tweeted about RBLX]
[c2_1895] (twitter, SPY) Hey, my stock market friends , do you see this wild stuff ? Lol Atlas group charges dropped. $spy > [@HackermanAce]( > Ay yo! > > They dismissed Zak Morris n friends charges ???? [Ace quote tweeted Ace about SPY]
[c2_1896] (twitter, BILI) $BILI 2022-07-15 C $50 Underlying: $38.86, % Diff: 28.67% 🗓️🐻🦴🇨🇳 Bid-Ask: $4 - $4.3 Interest: 61 Volume: 316 IV: 67.44% Daily $ Volume: $131,140 Sector: Technology Details + emoji legend: [unusual_whales tweeted about BILI]
[c2_1897] (twitter, SPY) 🤔 > [@AndrewThrasher]( > Last week's bounce in the Indices actually saw breadth weaken further, with more stocks dropping under the 200-day moving average. > $SPX $SPY [the_chart_life 🔁 AndrewThrasher about SPX, SPY]
[c2_1898] (twitter, AAPL) $SPY Daily. #SPY testing 50sma and testing the breakout from the downtrend line from ATH's. To keep bullish upside in play, needs to hold > 390 $SPX $ES_F #SPX #ES_F $XLK $IWM $QQQ $AAPL $TSLA [reciknows tweeted about SPY, TSLA, IWM, XLK, AAPL, ES=F, QQQ, SPX]
[c2_1899] (reddit, COIN) ECB cuts rates again to help economy weather erratic U.S. trade policy
