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
----- ITEMS (50: c2_0350 .. c2_0399) -----
[c2_0350] (twitter, ISRG) If there are any certainties in life, it's that you can't fade medicine. $ISRG down 30% on a single fucking downgrade. Beat EPS and maintained guidance on earnings Incredibly oversold. Great value buy here. [ThetaTendies tweeted about ISRG]
[c2_0351] (twitter, TSLA) $TSLA Breakout continuation over 950! As long as it can stay above 950 and build should see higher prices. [@Albert_trades]( $TSLA Stopped in its tracks around $950.. Will continue to watch for a move over 950-where this can start heading towards 1000. Ideally some digestion and consolidation below 950 before a break above. [Albert_trades 🔁 Albert_trades about TSLA, TSLA]
[c2_0352] (twitter, SNOW) $SNOW Dip buy in the gap per the plan/ idea I had. 140c +230% so far. It's all about holding the $139.88 now.😀 [optionsprochick tweeted about SNOW]
[c2_0353] (twitter, REGN) $REGN - SVB Leerink initiates at outperform, PT $738 [rachels_44 tweeted about REGN]
[c2_0354] (twitter, PYPL) $PYPL Down 3% after earnings [Nebraskangooner tweeted about PYPL]
[c2_0355] (reddit, BTC) Bought some parts for my bike with BTC Today I bought some parts for my bike using BTC. The seller is in another country but as we know BTC has no frontiers so it did not matter at all. This marks my fist purchase with BTC, up until now I was only holding. Hope to use it more and more in the future.
[c2_0356] (twitter, MA) $SPX completing bearish outside weekly candle, after and advance (MA 20 underside rejection as well) Suggestive of mean reversion completing, trend continuation [BigCheds tweeted about SPX]
[c2_0357] (reddit, BB) BlackBerry completes transformation just as demand grows for secure, sovereign defence communications Trump about to help revive BBs roots thanks to his bullshit. BB used to be the king in Enterprise Security and Risk arena back in the day before iPhone completely changed the mobile landscape. Remember BlackBerry messenger. Hopefully this gives other countries more leverage against future US bullying as they won't be reliant on US tech.
[c2_0358] (reddit, BULL) Elon has ruined me My prior bullish post He was supposed to take us to Mars and burn short sellers. Still holding 7300 shares. Please Elon fly again.
[c2_0359] (reddit, NVDA) AAPL PUTS. Almost 100k in gains Got lucky with all this volatility. Started to accumulate puts when the stock was consolidating above $220, betting it was a flag leading to another leg down. I think this volatility will last till Q3. I’ll wait till it’s over and I’ll go all in NVDA. In the meanwhile I’m just gonna chill and get myself a little treat with this money. Cheers and good luck to y’all 🥂
[c2_0360] (reddit, RBC) When Central Banks Fall: The Cost of Losing Monetary Independence in the U.S. and Beyond (RBC Wealth Management, August 08, 2025)
[c2_0361] (twitter, TDOC) $TDOC Daily Chart Tightening falling wedge within the descending channel But the real question is that , what the hell is wrong with this stock ? RSI 5 points below the OVERSOLD zone ( short term reversal ) [Mr Chart Norris tweeted about TDOC]
[c2_0362] (twitter, CMT) ATHs for Semis. Put it in the bullish bucket. $SMH [Ian McMillan, CMT tweeted about SMH]
[c2_0363] (twitter, MRNA) $MRNA right to the 50sma. Needs through for next break out📈 [rachels_44 tweeted about MRNA]
[c2_0364] (twitter, PYPL) $PYPL Don't have a position anymore but looks interesting [Nebraskangooner tweeted about PYPL]
[c2_0365] (reddit, BULL) Bitcoin Long-Term Holders Retain 79% of Supply as Exchange Reserves Decline
[c2_0366] (twitter, TLRY) $TLRY Possibly creating a base here. Holding Gooner EMA's so far on the daily. Can assume more upside unless it loses the EMA support. [Nebraskangooner tweeted about TLRY]
[c2_0367] (twitter, AMZN) $AMZN Key weekly OBV breakdown I'm monitoring. Earnings on Thursday. [Nebraskangooner tweeted about AMZN]
[c2_0368] (reddit, ETH) Where can I buy tiny amounts of ETH for gas funds I'm extremely new to crypto, I bought USDT on Binance and that I then transferred to localcoinswap using erc-20 without realizing that I need ETH on the account to actually do any transfer in or out of the platform or to actually do swaps between coins on the platform itself, are there any services that allow buying small amounts of ETH for these transfers?
[c2_0369] (twitter, SPY) $SPY Net Options Delta = Hot Mess! [Theta Warrior tweeted about SPY]
[c2_0370] (twitter, JNJ) $JNJ Much clearer set up now after Fridays breakout candle [Nebraskangooner tweeted about JNJ]
[c2_0371] (twitter, NFLX) $NFLX 😍 [@0xd0n]( $NFLX (Netflix) [0xd0n 🔁 0xd0n about NFLX, NFLX]
[c2_0372] (news, LEA) Lear (LEA) Beats Q3 Earnings Estimates, Lowers 2018 Guidance
[c2_0373] (reddit, BTC) Biggest Bitcoin Treasury Teases ‘Bigger Orange’ After $1.25B BTC Buy — Another $2B Coming?
[c2_0374] (twitter, KEY) $ES 2H -- (Jan 03) Today close comes at a KEY location Rejection here leads to a larger WXY towards 4730 Breakout overnight confirms the end of the correction We should have our answer by the morning 🎯 [rbc_trading tweeted about ES]
[c2_0375] (reddit, BTC) I sucked the past year The past year, I’ve been working like a slave. Just to lose most of it investing. I’ve gotten myself in such a huge whole with credit card debt, lawyer fees and I’ve lost 2 cars in the past month and have to save 2k for a new one. I wish I could just win lol
[c2_0376] (reddit, BULL) Bitcoin looking Juicy, thinking about making an entry Open to feedback. Feel like the bottom still isn't quite in yet Thinking about making an entry when initial transfer clears, then plan to add at 60k, 55k, 50k, 45k, and 40k Edit: I guess I should add that I'm not looking to be long on bitcoin, looking to take advantage of this opportunity to invest a sizeable amount of liquid cash with the intention of 2x or 3xing it
[c2_0377] (reddit, WKSP) $WKSP on Watch . Chart could be ready for reversal soon . On watch . Low float gem Solar truck bed covers + modular power systems . ✅ Launching ZeroFrost Heat Pumps in 2026. Record revenue months, gross margins exploding, targeting $45M+ in 2026. Analysts love / very high targets . New Heat Pump Partner *Disclaimer in linktree*
[c2_0378] (reddit, COIN) October's crypto crash left market makers stuffed with coins, slowing trading: BitMEX
[c2_0379] (reddit, ETH) Why Not Just Buy QQQI and Chill? The more I learn about CCs the more I think parking cash in these income generating ETFs is the way to go. Don't need to actively manage anything and the returns are good enough. Let's hear some thoughts
[c2_0380] (twitter, SMCI) $SMCI Recovered the 50ma on Friday. I'm looking for a supply break & momo above $282.09 so this can continue higher towards $288.49, $296.36, $305.40 Eyeing the 285-300c 👀 [Pro Trader - Kay tweeted about SMCI]
[c2_0381] (news, SAH) Auto Stock Roundup: SAH, CPRT, MGA Beat on Earnings, AN Lags Estimates
[c2_0382] (twitter, AMZN) $AMZN high and higher 🍃 > [@kholov23]( > $AMZN [Mr Chart Norris quote tweeted Mr Chart Norris about AMZN]
[c2_0383] (reddit, GOLD) Buying new house. Is this monthly budget reasonable? |Mortgage (plus insurance, taxes)|8365| |:-|:-| |Internet|50| |Electricity|200| |Oil|600| |Car|515| |Personal expenses (food, gas, etc.)|3000| |HOA|225| |HOA Water/sewer|200| |total|13155| ||| |income|14000| my income listed is post-tax and is after I maximize my retirement plans (401 and IRA). im afraid that i'm left with so little each month to contribute to general savings
[c2_0384] (reddit, BTC) Ancient Bitcoin Whale Comes Alive After Lying Low for Over 14 Years, Moves $469,811,057 in BTC at 30,00,000%+ Profit: On-Chain Data
[c2_0385] (reddit, GOLD) Usoil news trading Traders bet millions on oil minutes before Trump’s ‘Iran talks’ post crushed prices. Classic news trading with perfect timing… or a leak? Wild.
[c2_0386] (twitter, ALTS) Gm. Lots of #alts on discount. Specialy the onchain ones. Volume is lacking a bit there but will be back eventually. Onchain season 2.0 late summer. Whats your most recent buy ? Post below [WIZZ🥷 ( beware scammers ) tweeted about ALTS]
[c2_0387] (twitter, WTI) $WTI Big breakout on monthly [Nebraskangooner tweeted about WTI]
[c2_0388] (twitter, AMZN) $AMZN > [@BTstocktrading]( > $AMZN weekly chart > > After losing major support and breaking down in April, Amazon dumped right to the level where the previous uptrend started near 102 > > Now consolidating at the top of the 2018-2020 range, $AMZN looks to be forming another range between ~100-130 [BTstocktrading 🔁 BTstocktrading about AMZN, AMZN, AMZN]
[c2_0389] (twitter, ROKU) $ROKU > [@kholov23]( > $ROKU Weekly Candle Chart > > Descending broadening wedge [Mr Chart Norris quote tweeted Mr Chart Norris about ROKU]
[c2_0390] (twitter, PAA) $PAA finally having a good day. Break $12 please so we can confirm bullish trend breakout. [Nebraskangooner tweeted about PAA]
[c2_0391] (twitter, SPY) 15min $spy Nothing today so maybe we get some vol around 3 and can grab something tho the real move will be tomorrow around noon so im likely just going to observe today and play tomorrow. Should be no problem to bang out a 100% this week, stay tuned ill share as I do it. [Ace tweeted about SPY]
[c2_0392] (twitter, ORCL) WOW, don't know how I missed this of course they are going to be huge #AI winners This thing is about to rip through ALL TIME HIGHS Stocks, that do that, tend to go higher I need to get long some leaps here $ORCL [ShardiB2 tweeted about AI, ORCL]
[c2_0393] (reddit, BULL) My second realized gain for a long time. I might not kill myself Just keep going you can do it too
[c2_0394] (news, AAP) IBD Stock Of The Day Nears Buy Point Days Ahead Of Earnings Report
[c2_0395] (twitter, MSFT) $QQQ Weekly. #QQQ fake bear flag breakdown turned into double bottom. Now breaking out from falling wedge + reclaimed 200sma. Near-term resistance 300 (50sma), will be an interesting next week with lots of tech reporting also #FOMC $NDX $NQ_F $XLK #SPY #Nasdaq $AAPL $MSFT $SPY [reciknows 🔁 reciknows about XLK, AAPL, MSFT, NASDAQ, NDX, NQ=F, SPY, FOMC, QQQ]
[c2_0396] (twitter, FF) #FF @Crypto_McKenna Had the priviledge of meeting him in Crypto Insiders several years ago and have watched him turn into a bottom buying/top selling chad lately. [Captain_Kole1 tweeted about]
[c2_0397] (twitter, NFLX) $NFLX - Some stocks offering some insane returns if we rally even partially to where some were. [IncomeSharks tweeted about NFLX]
[c2_0398] (reddit, GOOGL) New Im looking to learn. Any ideas of where to start? Books podcasts yt channels etc? im sure i could google search but i think jumping to a group would help me weed out the unnecessary stuff in the search and learning the hard way. kinda migate my risk.
[c2_0399] (twitter, PLTR) $PLTR 2D After a 2 year decline, PLTR makes a bullish move through the cloud. W3 trying to reach the 1.618........🎯 [Zig_ZagTrades tweeted about PLTR]
