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
----- ITEMS (50: c2_0150 .. c2_0199) -----
[c2_0150] (twitter, NVDA) $NVDA Nice breakout today > $243 Now @ $250 👀 > [@optionsprochick]( > $NVDA Flag breakout possible. I'm watching this >$243 > Resistance: $244.89, $247.8, $251.25 > Eyeing the 245c [optionsprochick 🔁 optionsprochick about NVDA]
[c2_0151] (twitter, WTI) The global economy is rolling over and oil demand is only going to weaken so the supply/demand imbalance will continue and in fact worsen. I would avoid the oil sector here. $WTI will likely visit $65 soon. [MFHoz tweeted about WTI]
[c2_0152] (reddit, HUT) HUT 8 Corp jumps 30%+ with announcement of signing a 15 year AI lease worth $9.8B HUT 8 Corp announced today a 15-year lease agreement valued at $9.8 billion for 352 megawatts of IT capacity at its Beacon Point data centre campus in Nueces County, Texas, according to press conference release
[c2_0153] (twitter, COIN) $COIN - Calls closed, stock sold. Enjoyed the ride up, now we move onto an asset that doesn't have so many accusations and issues! > [@IncomeSharks]( > $COIN - Calls printing this morning on the 3% move. Media saying to sell as I'm buying. Love countertrading them. [IncomeSharks 🔁 IncomeSharks about COIN, COIN]
[c2_0154] (twitter, LMND) $LMND 🍋 Lemonade Q1 2024 GAAP EPS $(0.67) Beats $(0.80) Estimate Sales $119.100M Beat $114.319M Estimate [Mr Chart Norris tweeted about LMND]
[c2_0155] (news, ABBV) Claro Advisors LLC Buys First Trust Low Duration Mortgage Opportunities ET, First Trust VL ...
[c2_0156] (reddit, GOLD) Data centers need power. Power systems need transmission. Transmission needs copper. The further back you trace the chain, the less attention the companies receive. The largest names in the sector already have analyst coverage everywhere. Then there are explorers still running surveys and defining drill targets. Different levels of risk. Different timelines. Same supply chain.
[c2_0157] (twitter, ES) $ES Been trading light focusing on some real estate moves, but this is how I see things rn ES looks weak, rejected at vwaps and previous low Thinking we'll see monthly lower bband If it can flip 3807.50 I'll scalp long for gap fill > 3897.75 is dated long territory [StonkTrump tweeted about ES]
[c2_0158] (twitter, AAPL) $AAPL Continuing to edge markets higher, as its shown incredible strength all year. Likely to test into its all time highs (182) in the following weeks. [Albert_trades tweeted about AAPL]
[c2_0159] (twitter, AMD) I know to some, risk is over-rated, but once you make it priority. Priority turns to profit. $QQQ -35% small step back $AMD +64% step in the right direction A loss is only bad if you don't control it. A loss is only temporary if you let it be. This is what progress looks like. [optionsprochick tweeted about QQQ, AMD]
[c2_0160] (reddit, AMD) Stock futures rise on renewed hopes for Iran deal; AMD gains after earnings
[c2_0161] (twitter, NFLX) $NFLX Shrugged off its earnings & is now back into the 50/20emas. If it can get continued momentum >$328 it can move higher towards $333.49, $339.97, $344.02 resistances. Eyeing the 330c More of a pullback look below $320 [optionsprochick tweeted about NFLX]
[c2_0162] (twitter, VZ) $VZ Verizon - continued relative strength 6.35% divvy [BigCheds tweeted about VZ]
[c2_0163] (reddit, OS) Dogecoin MemeTracker fully functional on Dogebox OS :D success! Dogecoin MemeTracker (mempool), fully functional on [Dogebox]( OS, tracking any dogecoin address you sent to him and will start tracking any payment sent in seconds (\~3 seconds) if any is sent will reply to you with all transactions ID's and amounts :)
[c2_0164] (twitter, QQQ) $SQQQ Daily. #SQQQ bullish divergence building, long here = short $QQQ #QQQ $NDX $NQ_F $XLK #Nasdaq $SPY [HackermanAce 🔁 reciknows about SQQQ, NQ=F, NASDAQ, XLK, SPY, NDX, QQQ]
[c2_0165] (twitter, AMD) $AMD Needs > the $118/ 50ema next to start to make a bigger move [optionsprochick tweeted about AMD]
[c2_0166] (twitter, TSLA) $TSLA GAAP Operating Margins down ~1000bps YoY which means the stock will be up 10% tomorrow of course. [CryptoParadyme 🔁 qcapital2020 about TSLA]
[c2_0167] (twitter, TRT) $COST The consumer is getting weak but $COST usually tends to hold....getting close to a buy zone.... TRT combo at the top [Don't follow Shardi B if you hate trading tweeted about COST]
[c2_0168] (twitter, PDSB) $PDSB 🟦 PDS Biotechnology Q1 2024 GAAP EPS $(0.30) Beats $(0.37) Estimate Cash $66.6M [Mr Chart Norris tweeted about PDSB]
[c2_0169] (reddit, CE) P&l 23/100 (update) I took another position after my equity trade, so posting that update as well — sold 23400 CE (ATM) and bought 24000 CE (OTM) as a bear call spread. Target hit 🎯 Trade closed. Net profit after all charges. Disclaimer: Personal trade. Not financial advice.
[c2_0170] (reddit, SMCI) Advice on SMCI wheel I'm new to the Wheel. SMCI just crashed due to co founder being caught smuggling . i just sold a 28$ put a few days ago on SMCI. with about 3weeks to expiry So.... any advice appreciated. what would you do? get out now or ride it out? i feel like any news that hits is going to hit the stock price.
[c2_0171] (twitter, META) $META getting close to confirming this 4hr top candle It's the last day of the quarter/month so some of this action can probably be discounted Think markets rallied nicely first day of the quarter last time.. [Don't follow Shardi B if you hate Crypto tweeted about META]
[c2_0172] (twitter, WFC) #earnings scheduled to end the week with the #banks kicking off earnings season tomorrow $JPM $WFC $C $BLK $FRC $WAFD $GAME [eWhispers tweeted about JPM, WFC, C, BLK, FRC, WAFD, GAME]
[c2_0173] (twitter, NVDA) $NVDA Testing a big prior support level-now resistance at 159... a break over would set up a move to 170. [Albert_trades tweeted about NVDA]
[c2_0174] (reddit, ATER) 9/28/21 ATER DD: ATER is a bigger problem than we realized and why they need the shares. Plus the Entire Ortex Short Interest screen showing 60.19% of Free Float on Loan and 17.53 Million shares on Loan.
[c2_0175] (twitter, SPY) $SPX / $SPY When you have something that's as up only as the stonk market is, everyone is always looking for the next great crash / COVID type of event that will "pop the bubble." The most likely scenario however is that it just keeps melting higher. Before the last election [K A L E O tweeted about SPX, SPY]
[c2_0176] (news, ABR) John And Jane's September Dividend Increases And Income Tracker - Taxable Account
[c2_0177] (reddit, GOOGL) Anyone writing puts in the oil/gas sector? With the oil/gas sector pretty beat up right now, anyone writing puts on any of the companies? Many of them are at or near 52 week lows and so the premiums are nice. I have an inherent, conservative bias of writing puts with strikes below 52 week lows so yea
[c2_0178] (twitter, FSLR) $FSLR Calls up decent already Few gaps above lets see what the market gives us [StonkTrump tweeted about FSLR]
[c2_0179] (reddit, TSLA) SpaceX has now lost the equivalent of a full Tesla in market capitalization
[c2_0180] (twitter, SBUX) @StonkTrump [@faststocknewss]( *SBUX SEES EPS RETURNING TO DOUBLE-DIGIT GROWTH IN FY23 $SBUX [StonkTrump 🔁 faststocknewss about SBUX]
[c2_0181] (reddit, AMD) FUCKING POS fucking pos ruined me. sold it before it tanks to 350. everyone is going to say oh now it will go up and thanks for your sacrifice. not this pos.
[c2_0182] (twitter, CCJ) $ccj placed more bids for the week. Long and strong! [HackermanAce tweeted about CCJ]
[c2_0183] (twitter, HD) $HD - Stocks and crypto filling some gaps, now the upside is mostly up. [IncomeSharks tweeted about HD]
[c2_0184] (reddit, NVDA) Just Need advice on $NVDA holding So , I bought 13 shares of $NVDA at 100$ average price. With the stock being #1 market cap and not much of a gain. I was thinking maybe selling most of it and investing into the newer A.I. stocks such as RGTI, SOUN or APLD. What do you think ? Should I just keep it in NVDA since it’s almost a guarantee to raise or risk it for other assets.
[c2_0185] (reddit, MU) 23.6k to 577k for 2344% Gain in $MU 🚀 No master plan here — buy undervalued stonks and ride the wave. Bought MU Jun 2026 $150 calls back in July 2025. 30 contracts total. Cost basis: ~$23.6k. Closed today at ~$192 per contract. Proceeds: ~$577k. Return: +2,344%. Congrats to anyone else who caught this move. On to the next one 🚀
[c2_0186] (news, ABR) The Retirees' Dividend Portfolio - John And Jane's March Taxable Account Update - Tax Season
[c2_0187] (twitter, HOOD) $HOOD 2022-02-25 C $17 Underlying: $15.88, % Diff: 7.05% 🗓️🐻🦴👴🏻 Bid-Ask: $1.64 - $1.78 Interest: 15 Volume: 325 IV: 92.11% Daily $ Volume: $55,575 Sector: Technology Details + emoji legend: [unusual_whales tweeted about HOOD]
[c2_0188] (twitter, GOOG) $GOOG Daily Chart Bullish pennant breakout , currently price is testing October 12 highs at 142.40 [Mr Chart Norris tweeted about GOOG]
[c2_0189] (reddit, GOLD) SanDisk or GainDisk Ate Wendy's last week, today I got Chipotle double protein, guac, and a drink. Probably sold early again but I'm impulsive and rotating into another sector.
[c2_0190] (twitter, AAPL) $AAPL Trade Idea: $185 calls above $182.91. $180 puts below $181. Flow data w/ 🐳 posted this morning. [Honeydew0101 tweeted about AAPL]
[c2_0191] (reddit, GOLD) Is it worth it I am into buying momentum stocks making decent returns, but started learning options but it takes too much effort for me to too little.. same energy invested in buying stocks would have better returns. Anyone felt the same?
[c2_0192] (reddit, LLY) LLY just dropped a $149/month weight loss pill shipping today - analysts project $101B peak revenue
[c2_0193] (reddit, VXX) $SPX $SPY $VIX $VXX $VXST $UVXY $TVIX Closing CBOE SPX and VIX Index summary for October 12th
[c2_0194] (twitter, SPY) $SPY into resistance, scaling into shorts near 410 [BTstocktrading tweeted about SPY]
[c2_0195] (twitter, HKD) $HKD from what I found, they are primarily in the metaverse sector. $AIU $2+ coming > [@Bullish_Trades]( > $AIU Since their name change is now a Chinese metaverse play, should not be down here after $HKD [Bullish_Trades 🔁 Bullish_Trades about HKD, AIU, AIU, HKD]
[c2_0196] (reddit, VST) CRWD beats and raises. Also announces 4 for 1 split. Yet the stock is down over 10% in the after-hours. Why is that? Are you guys long this? I have a small position as part of my “AI stack” (along with GOOG, AMZN, HPE, ANET, GLW, and VST). I know it’s a volatile market and today was down across a lot of equities as oil prices rose. Curious what your sentiment is on this company and stock for the long term?
[c2_0197] (news, V) The CEO who was told he’d never run American Express has made Amex cool again—and is beating JPMorgan, Visa, and the S&P 500
[c2_0198] (reddit, BIP) Bitcoin News of the Day – February 14, 2026 1. Bitcoin Rebounds Above $70,000 After Cooler US Inflation Data 2. Bitcoin ETFs See Mixed Flows with Recent Outflows and Rebound Signals 3. BIP 360 Advances to Shield Bitcoin from Quantum Computing Threats 4. Plan B Network Strengthens as Global Bitcoin Education Hub 5. Brazil Proposes Sovereign Bitcoin Reserve Targeting 1 Million BTC in 5 Years
[c2_0199] (twitter, AI) Watching $COF $CRWD $CVNA $AI today to name a few! 👀 [optionsprochick tweeted about CRWD, CVNA, AI, COF]
