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
----- ITEMS (50: c2_0600 .. c2_0649) -----
[c2_0600] (reddit, TSLA) After underperforming most of the SP500 in recent months, is it time for Visa ($V) to chart new highs?
[c2_0601] (news, ABBV) Why AbbVie (ABBV) is a Top Dividend Stock for Your Portfolio
[c2_0602] (twitter, GOOG) Closed $GOOG short for profit at $95.20 Taking the win. Conditions still choppy. Going to keep locking in profit when I have it. > [@Nebraskangooner]( > $GOOG > > Up about $2k on this one. > > - Closed half > - Tightened stop loss > - Risk free trade now 👍 [Nebraskangooner 🔁 Nebraskangooner about GOOG]
[c2_0603] (twitter, QQQ) 9.3 milli order up 1.3mil so far with a over a month left to expiry > [@StonkTrump]( > Rolling in at the lows here on $qqq > > @CryptoParadyme [StonkTrump 🔁 StonkTrump about QQQ]
[c2_0604] (twitter, QQQ) $QQQ #QQQ Currently just an S/R flip retest from last week's breakout. [reciknows 🔁 Jake__Wujastyk about QQQ]
[c2_0605] (reddit, BTC) Please review my Mutual Funds Current Age : 25 Investment timeline : 25 years Appetite: Moderate Risk Returns expected : \~16% Please help in diversifying and removing overlapping mutual funds.
[c2_0606] (twitter, MSTR) Bearmarket things #MSTR > [@kerooke]( > MicroStrategy is raising up to $500M to buy more Bitcoin. > > Probably nothing. [JJcycles 🔁 kerooke about]
[c2_0607] (twitter, TSLA) $TSLA Weekly; W2 of 5 If I’m going to bet on anything being bullish into EOY. It’s Tesla. Potential W2 complete explained further below. @TrendSpider [ATMSnipes tweeted about TSLA]
[c2_0608] (twitter, T) $T As @BigCheds noted this thing is lower than it's been in decades.. Massive breakdown or good buying opportunity? Monthly close below this support surely seems like it would send it much much much lower. Of note - earnings this month could be the next directional catalyst [Nebraskangooner tweeted about T]
[c2_0609] (reddit, BULL) Being a perma-bear is a bad habit I should really stop before I blow my account (again)
[c2_0610] (reddit, GME) WSJ - GameStop Sales Fall Amid Continued Retail Troubles, Bitcoin Value Decline GameStop Sales Fall Amid Continued Retail Troubles, Bitcoin Value Decline Another fun hit piece against GameStop has splashed across the screens. I’m sure the algos are internalizing this information to push the stock to the “price it should be trading at.”
[c2_0611] (reddit, YDKG) 🚨YDKG Alert: Trading at a massive discount? \\\* Market Cap: 4.5M \\\* crypto treasury: 160M+ (Heavy SOL, BTC, ETH) \\\*Upcoming Catalyst Estimated Earnings and 20F expected soon. I am trying to figure out if the market is missing the math here? Even as a Chinese company this should be trading much higher. Anyone have some insight?
[c2_0612] (twitter, SPY) $SPY Wall Street veteran investor Ed Yardeni predicts a "classic 10% correction" for the S&P 500 in the coming months amid rising yields and a resilient U.S. economy. Despite concerns, recent economic indicators, including robust retail sales in March, suggest sustained economic… [Mr Chart Norris tweeted about SPY]
[c2_0613] (reddit, BTC) Need Advice, Should I trade? Hi everyone, I have quite substantial amount of BTC accumulated, around 0.8. I have never tried trading, playing with other coins, etc.. what would u suggest? Should I try that or let this BTC sit and hold?
[c2_0614] (twitter, ES) $ES "Another One" *DJ Khalid voice Did you fall for the 20ema throwback again anon? [StonkTrump tweeted about ES]
[c2_0615] (twitter, CMT) While most were worried about the "TechWreck" we did see $XLV push up to new all-time highs on Thursday and joined Equal-weighted Sectors like Industrials & Financials which also moved to new highs last week @IBDinvestors @marketsurge chart below- Have a great weekend All [Ian McMillan, CMT retweeted Mark Newton CMT about XLV]
[c2_0616] (twitter, SPY) $SPY @CryptoParadyme seems to think these are just horny breras and flow is kinda meaningless, lets see how it plays out I'm still kind of operating under this thesis: -Fed printing money?... Buy every dip Inverse bruh -Fed no printy and rates risey risey? ...Sell every rip [StonkTrump tweeted about SPY]
[c2_0617] (twitter, EMA) $RIVN Starting to look ready for continuation. High and tight flag at the 9 EMA after a weekly breakout. Needs through $26, for 30+ [Albertrays tweeted about RIVN]
[c2_0618] (twitter, NUE) $NUE Beautiful continuation in steel and commodity names. Second target hit here at $150... Some continuation into 160+ possible. [@Albert_trades]( $NUE Breakout continuation into new highs as steel names continue to show lots of strength. Second target at $150. [Albert_trades 🔁 Albert_trades about NUE, NUE]
[c2_0619] (twitter, SPY) $SPY Fed's Goolsbee views the addition of 175,000 new jobs as a very solid report, noting a bump in inflation at the start of the year. He emphasizes that the closer jobs reports resemble pre-COVID levels, the more confidence there is that the economy isn't overheating. [Mr Chart Norris tweeted about SPY]
[c2_0620] (reddit, BULL) My first realized gain for a long time. Feeling blessed. Just keep going you can do it too
[c2_0621] (twitter, MA) $MA Looks solid for intra-day short. Not full size, low risk. [Nebraskangooner tweeted about MA]
[c2_0622] (reddit, COIN) Coinbase Bitcoin Premium Index Turns Positive for Two Consecutive Days
[c2_0623] (twitter, FDX) $FDX 232🎯 > [@rachels_44]( > $FDX big jump in the PM. Watching 226, 232 (200sma) as resistance levels 📈Support 215, 209📉 [rachels_44 🔁 rachels_44 about FDX, FDX]
[c2_0624] (reddit, CME) No Rate Cuts This Year CME Group Fed Watch went from showing very high probability of rate cut later in the year, especially September timeframe. Now it's completely reversed, shows no longer any probability for rate cuts even into 2027, and shows rate hikes now far more likely with high probability even back to 4% again. This is going to not be good for the market. Higher interest rates, less growth, less jobs, etc.. This war really was so frivolous, I'm scared to see what 2027 will look like.
[c2_0625] (twitter, KITT) $KITT for @TheRoaringKitty? [BullishTrade tweeted about KITT]
[c2_0626] (twitter, SOFI) $SOFI Weekly Chart Mini bearish flag within the falling wedge pattern , where price is testing the lower boundary of the flag [Mr Chart Norris tweeted about SOFI]
[c2_0627] (news, A) Agilent Technologies, Inc. 2019 Q2 - Results - Earnings Call Slides
[c2_0628] (reddit, AIFU) When everyone’s bullish, contrarians take notice This chart shows the percentage of consumers who believe the stock market will decline. That share has dropped to its lowest level in years, meaning very few expect a downturn. Historically, such extreme optimism has been a contrarian indicator. For example, back in April, the percentage was among the highest on record. While there could be some stocks worth noting, following NVDA, PLTR, AIFU, RAPP, CRCL
[c2_0629] (twitter, SPY) FRIENDLY REMINDER ⚠️ Fed Chairman Jerome Powell will be speaking to the Stanford Business, Government & Society Forum tod… [Nebraskangooner retweeted TrendSpider about SPY]
[c2_0630] (reddit, AMZN) Own 86 shares of Amazon - should I continue to buy?? Chats thoughts?? I started buying Amazon 7-10 years ago a few shares every few month. I feel like Amazon will be a large beast in 20 years- it’s going no where. Would you continue to buy? I believe they are going to be a beast in years to come but want to know what you guys think….. Btw I am 28 years old
[c2_0631] (news, ABEV) Brazil stocks, currency rally as far-right presidential candidate extends lead
[c2_0632] (twitter, NVDA) $NVDA 4H Chart Bullish Flag [Mr Chart Norris tweeted about NVDA]
[c2_0633] (twitter, ES) $ES_F This is the stat from my tradovate small account for this week. See max drawdown > [@Junaidk36013131]( > @kpak82 Plz can you tell me how many used honda civic or fraction of a used honda civic you would have lost if your SL was hit? [kpak82 🔁 Junaidk36013131 about ES=F]
[c2_0634] (twitter, LAC) $LAC Guess I'll have a go off this trendline breakout. [Nebraskangooner tweeted about LAC]
[c2_0635] (twitter, MWA) $MWA Weekly Chart ☕️ 🔨 From cup and handle breakout to descending broadening wedge and a perfect hammer at the bottom [Mr Chart Norris tweeted about MWA]
[c2_0636] (twitter, OXY) Pump & dump, without the pump part. > [@UnlocksCalendar]( > $OXY unlock in Dec. 2021 > > "Oxygen will be the most sophisticated risk management tool in DeFi — with a strong team, growing liquid ecosystem and large potential user base we believe this is the next step in mass adoption" > > @AlamedaResearch dunno, looks like a pump and dump to us [HsakaTrades 🔁 UnlocksCalendar about OXY]
[c2_0637] (twitter, INFY) As strong as Indian stocks have been on a relative basis, this one will hurt if it can't get back above ~$18... $INFY [the_chart_life tweeted about INFY]
[c2_0638] (reddit, AMZN) the time is $NOW pt.2 i am once again full porting $now at these low levels. scared money don’t make money. just going off of vibes
[c2_0639] (twitter, NKE) $NKE Daily Candle Chart 🔨 🔨 1. Falling wedge is getting tight 2. Bullish divergence on daily chart 3. RSI at 18 / RSI & MACD overextended to the downside 4. Hammer candlesticks above the lower boundary of the wedge All combined = 🪄 [Mr Chart Norris tweeted about NKE]
[c2_0640] (reddit, TSLA) Khel khatam beta khel khatam💔 Don't know much about Tommorow but it will impact, also Tommorow going to be again gapdown with bearish move. So be ready. Pura portfolio laal pe hai
[c2_0641] (reddit, MSFT) What is a better bullish strategy if the stock eventually goes against you: vertical put spreads or diagonal spreads (buy 60-90 DTE, sell 45 DTE)? For example, my vertical put spreads are deep ITM on tech stocks like MSFT. I was wondering if I had done diagonal spreads on them instead, if rolling them would be easier when the stock price went against me. Would love to hear your thoughts.
[c2_0642] (twitter, TXRH) Texas Roadhouse is a great example of the recent weakness in Restaurant names... $TXRH $EATZ [the_chart_life tweeted about EATZ, TXRH]
[c2_0643] (twitter, FF) thx you sir 🙌 [@TheCryptoData]( #FF Some of the best people I know in crypto 💯👇 @TheCryptoDog @HsakaTrades @GiganticRebirth @Pentosh1 @koreanjewcrypto @hyperbitcoins @AltcoinPsycho @CredibleCrypto @CryptoKaleo @Trader_XO @pierre_crypt0 @Crypto_Chase @cryptodude999 You won't be disappointed 👌 [hyperbitcoins 🔁 TheCryptoData about]
[c2_0644] (reddit, IONQ) Quantum Computing is a bubble. Quantum computing stocks are so overvalued - no commercial use case, computers dont work and revenue is fake. This is exactly like 3D printing in 2014, Cannabis in 2019 & NFTs in 2022. Once all the fraud is over stocks like IONQ, RGTI, QUBT, QBTS will all drop 85%+.
[c2_0645] (twitter, SBUX) Bullish for $sbux lol Starbucks earnings after the bell tomorrow… predictions? [StonkTrump tweeted about SBUX]
[c2_0646] (twitter, ROST) I see a lot of stocks like $ROST that get hit and there had been clear signs to sell the stock beforehand. Not so in this case. A very unfortunate, 4 day drubbing. Hard for long-term holders, given that from a technical perspective, there really hadn't been a ton of damage. [the_chart_life tweeted about ROST]
[c2_0647] (twitter, RSI) $RCRT 23 RSI. MACD getting ready to cross [Bullish_Trades tweeted about RCRT]
[c2_0648] (reddit, STRC) STRC is buying 800 bitcoin per day So pay atention to this new information about the bitcoin supply
[c2_0649] (twitter, ACMR) $ACMR Weekly Candle Chart Falling wedge breakout / Simple S / R flip 25 ? why not ? [Mr Chart Norris tweeted about ACMR]
