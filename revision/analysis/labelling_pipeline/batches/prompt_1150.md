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
----- ITEMS (50: c2_1150 .. c2_1199) -----
[c2_1150] (reddit, GME) WAKE UP KENNY! Q4 earnings today Wake up fellow apes. It’s Q4 earnings day and you don’t want to miss this one. Kenny and SHFs about to be bent over. This day will go down in history as a pivotal point for GameStop. RC is preparing to drop a bomb over the next few months. Q4 earning HYPE is real. Buy, DRS, and hold your GME.
[c2_1151] (twitter, UNG) Cut UNG for a loss Could have kept it and continued to scale in a build a position, but I always cut losers if the port is up big for the day on other positions > [@StonkTrump]( > $UNG flow this morning, I nibbled on some of these calls [StonkTrump 🔁 StonkTrump about UNG]
[c2_1152] (reddit, AI) Finance worth it in 2026 and AI Is majoring in finance with a commerce degree worth it in our current year and the future. Ive heard that AI may take over given that ive heard a lot of finance careers are just on excel spread sheets. Im interested in majoring in finance but Im a bit worried hearing this.
[c2_1153] (twitter, BMBL) $BMBL 🟦🟦 Bumble Q1 EPS $0.19 Beats $0.07 Estimate Sales $267.77M Beat $265.47M Estimate [Mr Chart Norris tweeted about BMBL]
[c2_1154] (twitter, SPY) $SPY Opportunities on both sides of the tape today thanks to the Trapped Trader study. 🪤 [Theta Warrior tweeted about SPY]
[c2_1155] (twitter, PSX) no interest here for the market ... being away from the charts I remain on the sidelines ... wait $PSX open ... in livestream we will take stock of the situation 👊🏻 [eliz883 tweeted about PSX]
[c2_1156] (twitter, RSI) RT @kholov23: $AAPL Daily Chart 💥👇 RSI - Entered the OVERBOUGHT Zone Price reached the previous highs / September 05 / and hanging below… [Mr Chuck Norris retweeted Mr Chuck Norris about AAPL]
[c2_1157] (twitter, NVDA) $NVDA NVIDIA shares are on the rise ahead of their upcoming earnings report. Rosenblatt analyst Hans Mosesmann, who maintains a Buy rating on the stock, has set a price target of $1,400. Mosesmann anticipates that Nvidia will surpass expectations and revise upwards for the April [Mr Chart Norris tweeted about NVDA]
[c2_1158] (twitter, XOM) $SPY Poots 💸 [@ItsSqueezy]( ORB Watchlist 3/15 $PTON - Sizeable gap up with decent volume $XOM - Gap down, volume and eyes on oil $SPY - Solid range from 8-9 am [ItsSqueezy 🔁 ItsSqueezy about SPY, PTON, XOM, SPY]
[c2_1159] (reddit, AMZN) Understanding Wheeling I’m struggling to understand wheeling strategy despite reading a few times. I sold an Amazon put at 205 and collected a small premium of 175 which expires November. It’s trading at 220 which is way higher than my strike but what if closer to November it becomes 207 or 199? How to wheel this? Buy back the put and make a loss and resell or how does it work? Thanks so much
[c2_1160] (twitter, SNDX) $SNDX Syndax Pharma ..."A clinical-stage biopharmaceutical company, develops therapies for the treatment of cancer". [the_chart_life tweeted about SNDX]
[c2_1161] (reddit, ETH) ETH outperforming btc in gains. ETH is out performing BTC recently is it alt season or just ETH adoption what you guys think ? and if its alt season what project you guys think it will skyrocket and go to the moon atleast for short time ? can eth finally break the 5k after so many years?
[c2_1162] (twitter, OKTA) $OKTA Okta Q1 2025 Adj EPS $0.65 Beats $0.55 Estimate Sales $617.000M Beat $604.339M Estimate [Mr Chart Norris tweeted about OKTA]
[c2_1163] (twitter, ETSY) $ETSY held trendline support on daily. Watching 200sma next if market holds 📈 [rachels_44 tweeted about ETSY]
[c2_1164] (reddit, MSFT) MICROSOFT $MSFT JUST REPORTED EARNINGS * EPS of $3.65 beating expectations of $3.37 * Revenue of $76.4B beating expectations of $73.8B
[c2_1165] (reddit, GS) Goldman Sachs cuts Nifty target by 14%, sees more downside for India equities Goldman Sachs downgrades Indian equities to marketweight. The brokerage has reduced its 12-month Nifty target to around 25,300–25,900, down from earlier projections near 29,300–29,500, a cut of roughly 14 percent. Oil price rise and weak foreign flows weigh on market outlook [
[c2_1166] (reddit, BULL) Should I default on unsecured personal loans? Homeowner with 180k left (3.5%) on mortgage at 1400$ per month (including escrow). Take home each month is 3500$. Also have 70k-plus in personal loans. As a result I’m in the red. Have a refinance offer for 265k (7.5 %) at 2300$ per month. So long as I keep paying the mortgage, can I default on the unsecured loans and arrange payment plans? Edit: no car, very few expenses aside from essentials
[c2_1167] (reddit, MSTR) The dip is part of the cycle The dip is part of the cycle, the big players are stacking up, they are quietly accumulating. Position yourself wisely now, before the market bounces back.
[c2_1168] (reddit, TSLA) What stock is the absolute most overpriced?? Give me names and tickers people. What is the most absolutely overpriced, reeking, dogshit stock that EVERY talking head on wall street is over-hyping to the moon?? I mean retched financials, dwindling business model, but for some reason the share price is soaring??
[c2_1169] (twitter, COIN) $COIN - And there's your bullish divergence playing out. Super easy trade with 40% profit! [IncomeSharks tweeted about COIN]
[c2_1170] (reddit, PL) Superficial loss rule for futures Does the superficial loss rule apply when day trading futures or is it only for stocks? Edit: chatgpt suggests not because with stocks you own the underlying but futures is just a contract and realized PL can be used instead. Basically it's not a capital property
[c2_1171] (twitter, FDX) $FDX Looks a good short but i'm not opening up any swing positions today with vacation coming tomorrow. [Nebraskangooner tweeted about FDX]
[c2_1172] (twitter, MSFT) $MSFT Flagging on the daily above all short & Medium term EMAS. This needs to get some momentum above $290 to move higher towards $293, $296 Resistances. Earnings on 4/25 [optionsprochick tweeted about MSFT]
[c2_1173] (twitter, SPY) $SPY seeing some similarities to April 22’ & Jan 22’ If true… sellers want to break $385 with volume From there I expect stops to get ran at $369 before a favorable rally can be in play [ATMSnipes tweeted about SPY]
[c2_1174] (twitter, SPY) $spy will flip to puts soon if we get the pump to the upper cloud . Also almost any plays on spy unless i specify is going to be weeklies or at the most 2 weeks out. I havent bough/held any spy contracts for longer than a month cuz conditions suck for leaps. [HackermanAce tweeted about SPY]
[c2_1175] (twitter, META) $META We were all talking about how good a buy it was when it was below $90... I snagged a bunch of calls but we are getting to an interesting area here to watch or TP [ShardiB2 tweeted about META]
[c2_1176] (reddit, BTC) Is 0.01 a good target to aim for with BTC? I started accumulating BTC at the end of last year. I live in Brazil and buy 200 to 500 BRL (40 to 95 USD) worth of BTC per month. My initial goal is to reach 0.01 BTC, which I believe is a good goal and I intend to achieve it by the end of the year.
[c2_1177] (twitter, COIN) $COIN - I wouldn't be a buyer again until this lower gap. Glad I sold when I did, and have been able to accumulate so much by swing trading instead of sitting and holding and being break even. The perfect swing trading $COIN streak continues [IncomeSharks tweeted about COIN]
[c2_1178] (twitter, LCID) $LCID nice hold above trendline resistance. MACD crossing📈 One to keep on watch for 2nd day follow through [rachels_44 tweeted about LCID]
[c2_1179] (twitter, ROKU) $ROKU closed back above its downtrend & the 50ema. I'm watching for a continuation >$62.60 if it can get momentum. Resistance: $66.63, $69.77, $72.67 Eyeing the 62-65c if this continues [optionsprochick tweeted about ROKU]
[c2_1180] (twitter, JMIA) $JMIA Jumia Technologies reports Q1 revenue of $49 million, up 19% year-over-year, with Gross Merchandise Volume (GMV) reaching $181 million, up 5%. [Mr Chart Norris tweeted about JMIA]
[c2_1181] (reddit, NVDA) BofA flagged 26 companies at risk because of AI. This is the Bloomberg chart showing their performance relative to the S&P. I could only find 9 on the list using ChatGPT. I don't have a Bloomberg terminal, so can't get to the full list.
[c2_1182] (twitter, QQQ) $QQQ Price finding support at $260. Ranging under $270 in a tight base... Over 270 and tech names can start getting some momentum off the lows. [Albert_trades tweeted about QQQ]
[c2_1183] (twitter, TSLA) $TSLA $ARKK $ARKF Cathie Wood of Ark Investment Management LLC believes Tesla Inc. could experience a substantial increase in its stock price as it enters the autonomous taxi market. Wood estimates this move could unlock a global revenue opportunity of $8 trillion to $10 [Mr Chart Norris tweeted about TSLA, ARKF, ARKK]
[c2_1184] (reddit, SMC) Nifty 50 & Bank Nifty Analysis & Prediction 6th April | Live Chart Breakdown & Trading Strategy SMC
[c2_1185] (twitter, TGT) $TGT Inching closer to a breakout as the bull RSI divergence works its magic 🪄 [HackermanAce 🔁 TrendSpider about TGT]
[c2_1186] (twitter, ETSY) $ETSY > [@kholov23]( > $ETSY Weekly Chart / Before & After > > Price is testing the upper boundary of tightening falling wedge [Mr Chart Norris quote tweeted Mr Chart Norris about ETSY]
[c2_1187] (reddit, ETH) Sometimes it’s really just this simple. You don’t need 10 indicators or some complex model to understand what’s going on. Lower highs, lower lows. Clean trendline. Key levels getting respected. That’s the whole picture right there. A lot of times we overcomplicate things trying to “confirm” the move, when price has already been telling the story the entire time. Simple doesn’t mean easy, but it’s clear. Stay patient, wait for your levels, and let price do the talking.
[c2_1188] (reddit, BTC) 99.85% of Bitcoin (BTC) Addresses Are Now In Profit Source: [
[c2_1189] (reddit, RCAT) skipped selling covered calls and sold straight shares to escape RCAT after getting assigned last week…not making that mistake again 😅
[c2_1190] (twitter, SLV) Closed out the remainder of my position > [@Nebraskangooner]( > $SLV > > Weekly gap filled. Main first target reached here. I already took some profit along the way so I may hold for now. Kind of want to hold long term on the remainder. [Nebraskangooner 🔁 Nebraskangooner about SLV]
[c2_1191] (reddit, BULL) thoughts on this etf allocation can the community give me a thorough evaluation of my etf allocations at the moment. i want to build long-term wealth with around medium risk tolerance. from aus hence why the australian equivalent of known us etfs are used to minimise fees. thanks in advance
[c2_1192] (reddit, COIN) Kinda confused about wallet transfers I bought 95$ worth on solana from coinbase (100$ after fees) after that i transfered it to phantom. Why does phantom show me wallet as 70$ ???
[c2_1193] (twitter, TSLA) $TSLA Tesla has temporarily halted production at its Grünheide plant for four days as a precautionary measure ahead of announced demonstrations against the factory expansion. All employees have been instructed to work from home during this period. [Mr Chart Norris tweeted about TSLA]
[c2_1194] (twitter, ALTS) Pretty high PPI number and $SPX weakness. Be careful out there. I'm personally just chilling, planning out potential entries on $ALTS. [DaanCrypto tweeted about ALTS, SPX]
[c2_1195] (twitter, EMA) $BLDR Sitting at a key support. Should at least retest the gooner EMA's but has potential to break new highs from here [Nebraskangooner tweeted about BLDR]
[c2_1196] (reddit, BULL) Scanning opportunities What's your process for finding the best short vol opportunities? Genuinely curious how advanced traders here scan for opportunities. Do you manually check each ticker? Use a screener? Have a watchlist? I've been trying to systematize my process and would love to hear different approaches.
[c2_1197] (twitter, QQQ) $Grom First PT is the blue line (ma 50). Did you miss my $Poai call? Don’t miss this one! $IONQ $bynd $nnox $pton $zura $hepa $pbr $dis $cetx $bb $spy $muln $pacw $btc.x $qqq $tsla $dwac $trka $amc $pola $boil $tqqq $sqqq $top $tirx $pypl $jagx $aud $bbig $myo [Bullish_Trades 🔁 Trad1ngMadeEasy about DIS, PBR, BBIG, TRKA, AMC, POAI, NNOX, PACW, BOIL, SQQQ, HEPA, TOP, QQQ, ZURA, IONQ, POLA, MYO, PYPL, JAGX, CETX, AUD, BYND, TQQQ, PTON]
[c2_1198] (twitter, SPY) Done for the day. $SPY puts 145%. $SPY calls 140%. What more can a girl ask for? Going to stop trading to keep these gains. Let's do it again tomorrow! 🥂 [Honeydew0101 tweeted about SPY, SPY]
[c2_1199] (twitter, RIOT) #MidtermRally - Majority will still miss making life changing money on this move up the next 5 months. Look at what $COIN and $RIOT are doing already. [IncomeSharks tweeted about COIN, RIOT]
