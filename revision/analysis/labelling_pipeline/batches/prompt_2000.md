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
----- ITEMS (50: c2_2000 .. c2_2049) -----
[c2_2000] (reddit, BTC) 187 BTC at Age 43: Cash Out for Peace or Hold for Glory? Status: I’ve accumulated 187.21 BTC and $600k+ in USDT. I also earn $12k per month from my job. I’m torn between selling now to become a comfortably wealthy “fiat guy,” or continuing the Bitcoin ride. Consideration: I’m 43, and I can feel my heart isn’t as strong as five years ago. Advice sought: Should I sell now, or hold to the very end — for better or worse?
[c2_2001] (reddit, META) My little niece was orange-pilled in utero. My brother got me in the game back in 2021. Someday I’ll get to have conversations with her about how much her dad’s advice helped me, I’m grateful for that.
[c2_2002] (reddit, ETH) BitMine deepens Ethereum bet with 41,788 ETH buy, ramps up staking
[c2_2003] (twitter, AMD) $AMD Nasty looking retrace on weekly. If it closes below this support and earnings are bad this month it can easily drop back close to $100 Also a Top Goon X dump marker (white mark) [Nebraskangooner tweeted about AMD]
[c2_2004] (twitter, TGT) $TGT Daily Candle Chart 🔨 🐉 🔨 🐉 Falling wedge is getting TIGHT [Mr Chart Norris tweeted about TGT]
[c2_2005] (twitter, EXPE) $EXPE 🟦🟦 Expedia Group Q1 EPS $(0.21) Beats $(0.24) Estimate Sales $2.89B Beat $2.81B Estimate [Mr Chart Norris tweeted about EXPE]
[c2_2006] (twitter, RSI) $IWM Daily. Inverse h/s breakout negated the bearish descending triangle. No signs of slowing down despite RSI being overbought, may need to hit 200/205 supply zone before a meaningful pullback #IWM $RTY_F #RTY_F #smallcaps $SPY $QQQ $DIA [reciknows 🔁 reciknows about IWM, SMALLCAPS, SPY, RTY_F, QQQ, DIA]
[c2_2007] (twitter, MSFT) $AAPL Big earnings next week with $MSFT and $GOOGL on Tuesday.... but I added some today [Don't follow Shardi B if you hate trading tweeted about GOOGL, AAPL, MSFT]
[c2_2008] (twitter, GOOGL) Fib extensions I'm watching on $GOOGL as it broke to new highs. It's respecting that 187.8 fib support today. Nice reversal zone [VS Options Trading tweeted about GOOGL]
[c2_2009] (reddit, NVDA) Nvidia becomes first company to touch $4 trillion market cap Nvidia ([NVDA]( stock gained more than an intraday record high on Wednesday, becoming the first company to ever hit a [market cap of $4 trillion]( The AI chip maker's valuation has now surpassed Apple's ([AAPL]( record market cap of $3.91 trillion reached in December 2024. At roughly 9:40 a.m ET, Nvidia shares hovered near a record of $164, building on its closing high of $160 in the prior session.
[c2_2010] (twitter, GOOGL) The bear flag pattern on $GOOGL that has been forming since October 2022 is now showing clear signs of completion. The next move for the stock is likely to be a sharp downward leg, which will likely take it below the COVID-19 lows. [MFHoz tweeted about GOOGL]
[c2_2011] (reddit, BTC) What's better? Hi everyone, I was wondering if its better for privacy and security, to p2p(although you run the risk of buying tainted btc right?) Thank you! Edit: thank you everyone for your responses, appreciate you:)
[c2_2012] (reddit, BTC) Strategy has acquired 705 BTC for ~$75.1 million at ~$106,495 per Bitcoin. As of 6/1/2025, Strategy hodl 580,922 BTC acquired for ~$40.68 billion at ~$70,001 per Bitcoin.
[c2_2013] (reddit, MRVL) Load up the chopper! MRVL I don't run to the stock because of premiums but this is something even (if the worst case scenario happens - assignment) worth holding on at this price for a very long time. I currently own 100 shares plus these positions now.
[c2_2014] (reddit, AMD) Amd I bought this in April when my portfolio was getting trashed and I was in total despair.
[c2_2015] (twitter, NVDA) $NVDA Hit new all time highs then rejected at the backtest of prior channel support, now resistance. [kpak tweeted about NVDA]
[c2_2016] (twitter, NVDA) $NVDA 10 MIN Chart - bearish flag [Mr Chart Norris tweeted about NVDA]
[c2_2017] (reddit, TSLA) Tesla’s 3% Spike Is a Smoke Screen for a Government AI Deal TSLA pops after the Samsung chip news, but let’s be real, this isn’t about chips. Word is Tesla’s secretly tied to defense AI contracts. Autonomous weapons testing? Surveillance tech? The spike is just cover.
[c2_2018] (twitter, CLSK) $CLSK > [@kholov23]( > $CLSK Daily Candle Chart 🔨 [Mr Chart Norris quote tweeted Mr Chart Norris about CLSK]
[c2_2019] (reddit, TSLA) Tesla stock climbs on Q4 earnings beat, Optimus robots on track for end-of-year production
[c2_2020] (reddit, PLTR) My PLTR gains (so far) Fairly happy with the returns so far. Still hodling and have no plans of selling yet.
[c2_2021] (reddit, AMZN) 2x etfs wheel strategy. Hi guys, ehat do you think about selling 2x etf puts, or wheeling in general? For example Amazon price is over 200$ and amzu 2x etf only 28$ so in theory i need much less capital to wheel. Please tell me what you think about this.
[c2_2022] (twitter, META) $META is hugging the 20ema with minimal range today and tight price action. If it recovers the $116 id watch for a bounce back towards the $18.03-122 range. There are gaps all over on this chart to be filled. META under $112.76 & it can move towards the $110.55-$108.18 range [optionsprochick tweeted about META]
[c2_2023] (twitter, NFLX) $HOOD moving nicely. $NFLX stagnant. Tightened $NFLX stop loss to 0.6% to reduce risk. > [@Nebraskangooner]( > Bought some $HOOD and $NFLX on this opening dip but I'm going to keep a 1.5% stop loss on each position so it'll be tight stops with low risk. [Nebraskangooner 🔁 Nebraskangooner about HOOD, NFLX, NFLX, HOOD, NFLX]
[c2_2024] (twitter, RUN) $RUN - Accumulation sub $25 might be over. As weather warms solar demand rises. Expecting $40 to $50 targets by EOY. > [@IncomeSharks]( > $RUN - Long term play based on solar panel narrative. Slowly building a position over the next few months. Dips are for buying, especially if we go under $25. [IncomeSharks 🔁 IncomeSharks about RUN]
[c2_2025] (twitter, AMD) $AMD Weekly bear flag / wedge confirmed as price gets rejected at prior trendline support. The prior demand zone ~73 and rising 200 week SMA may offer some support, if not there's a large gap in price volume back down to ~57 [ThetaWarrior tweeted about AMD]
[c2_2026] (reddit, AI) Not sure if SOXQ was the right choice I don’t think AI is gonna slow down anytime soon, but man, invested in SOXQ a couple of weeks ago and I am down almost 13% all-time. I’m trying to wait it out to see if it turns around. Not to mention, I don’t want to sell at loss. It has been tough to witness though.
[c2_2027] (reddit, BTC) Which one of you, regards, did this technical analysis of Bitcoin?
[c2_2028] (reddit, BTC) How many Bitcoins to be considered a Whale in 2025? I see a whale as over 1000BTC. What do you guys think? edit: also another question, do you think that there are any female BTC whales out there? Your answers might be used for content in a presenation at the Bitcoin conference in Amsterdam. (November 2025)
[c2_2029] (twitter, TSLA) Added more $TSLA today also...going all Cathie Wood.. [ShardiB2 tweeted about TSLA]
[c2_2030] (twitter, SPY) $SPY Whale is trimming and rolling down with a profit, again. [ThetaWarrior tweeted about SPY]
[c2_2031] (reddit, TSLA) $TSLA RSI at 35: Is the Selloff Overdone or Just Getting Started? Tesla ($TSLA) is down 18% this month, and the RSI is now at 35, just above oversold territory. The MACD is also showing a bearish crossover. Is this a buying opportunity, or should I wait for a clearer trend reversal?
[c2_2032] (reddit, FANG) Upstox vs Vested for buying Indian-listed global ETFs - which saves more for investments? Planning to invest in Motilal Nasdaq Q 50 (0.47% ER) Mirae S&P 500 Top 50 (0.60% ER) Mirae NYSE FANG+ (0.65% ER) Did the math and Upstox came cheaper over 2 years despite USD appreciation. Anyone with actual experience comparing both? What am I missing?
[c2_2033] (reddit, BULL) 1dte es 18k long put yolo (4mm notional) thesis: they have been taking risk off pretty much every Friday since the war started for the gap risk, why should tomorrow be different? (also I lost money shorting and trying to get it back) averaged in to these all day today
[c2_2034] (twitter, AMZN) $AMZN - flag 4H RBR demand at $111.60 If bulls defend this a push through $116.53 supply (not fresh) is likely to follow. I’m anticipating a larger degree flat to play out. [ATMSnipes tweeted about AMZN]
[c2_2035] (reddit, MCD) OPEN gain porn 69k to 1 million As a "not complicated retail investor" who simply believed in Opendoor's mission (to simplifies the home buying and selling process), 4 years of patiently buying every dip with all of my 2 jobs' salaries while eating McDonald happy meals everyday. Being laughed at for holding the huge bag of down nearly 70% and STILL keep buying the dip all the way to $0.51.... Just turned that 69k to 1 million today...
[c2_2036] (twitter, SOUN) You asked me what’s going on with $SOUN , I tell you it’s doing fantastic [Mr Chart Norris tweeted about SOUN]
[c2_2037] (reddit, MTCH) What do you think about MTCH What do u think about match group (Tinder owner)? The PE is low (12x) and i think the business is quite niche and maybe recession proof. Though i dont think there will be huge growth coming to the business but i think the needs for this type of business will always be there. What is your view?
[c2_2038] (twitter, DELL) Everyone globally getting laid off and you think they putting their entire redundancy package into your dog coin….you’d be absolutely correct, that’s what they all gna do for a quick flip. Exit liquidity is coming 🤝 > [@gurgavin]( > 🚨BREAKING 🚨 > > DELL TO FIRE OVER 6,500 EMPLOYEES $DELL [trader1sz 🔁 gurgavin about DELL]
[c2_2039] (reddit, SPY) Last Minute bulk buy of $SPY I genuinely want to know why there's a strategy like this to buy a boat-load of stocks (about 5M) at the closing bell today. It was a blood bath all day. Is this just a perception that the market has already reached the bottom? Does someone know what's going to happen on Monday's opening bell? Or is this just what traders do at the end of the week in a major down turn?
[c2_2040] (reddit, CYCN) CYCN for the win Did anyone else get in early on CYCN and rode that momentum up from around $2 to over $3?
[c2_2041] (reddit, GOOGL) Does time in the market apply to individual stocks or etfs? I guess I’m just wondering, if an individual stock rips does it really make sense to buy more of it even if it is at a ATH? I do mostly ETFs and just hold, but I have 10 stocks that I like to play around with But like for example, I got into Google when it was 150 and now it’s 338 a share, does it really make sense to add more right now even if I believe in Google long term? Or would it be best to just hold and put more into a dip?
[c2_2042] (twitter, CLSK) $CLSK CleanSpark has announced that it mined 445 Bitcoin in June. Additionally, the company reported that its operating hashrate has exceeded 20 EH/s, which represents a doubling since December. [Mr Chart Norris tweeted about CLSK]
[c2_2043] (twitter, AAPL) $AAPL getting really close to that ascending triangle breakout target [Nebraskangooner tweeted about AAPL]
[c2_2044] (reddit, AMZN) shorted the market today. being a contrarian feels right at this point shorted the market today. it feels like everyone is too busy being bullish and forget nothing has really changed. i might be wrong. however i feel that this contrarian trade is good. after hours the outlook by amzn confirms my bias. although timing is not something i can pinpoint. only time will determine if i was correct or wrong also was my timing correct. thoughts?
[c2_2045] (twitter, M) $M Macy's weekly chart below major supports. [kpak tweeted about M]
[c2_2046] (twitter, DLO) $DLO Weekly Chart Price moving within falling wedge pattern , where its aiming for a test of lower boundary / support / of the wedge [Mr Chuck Norris tweeted about DLO]
[c2_2047] (news, META) Meta cuts 8,000 jobs, Microsoft offers buyouts amid AI spending push
[c2_2048] (twitter, QQQ) Look at that sell on close volume on $QQQ [Shardi B Trading #teamshardi tweeted about QQQ]
[c2_2049] (twitter, PYPL) $PYPL Monthly Chart Inverse head and shoulders, price is in the process of forming the right shoulder Neckline at 75 - 76 August 01 / 22 AVWAP at 70 [Mr Chart Norris tweeted about PYPL]
