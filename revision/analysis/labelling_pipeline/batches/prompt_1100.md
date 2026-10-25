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
----- ITEMS (50: c2_1100 .. c2_1149) -----
[c2_1100] (reddit, BETA) Google Finance Beta is terrible, lost my watch lists Whenever I go to Google Finance as of today it is the BETA version. I've seen notes about in the top right I can revert to classic, but I see no ability to do so. It looks like all my watch lists are gone and overall I very much don't gain anything from this new view. Wondering if anyone knows how to revert to the classic view?
[c2_1101] (twitter, GOLD) #GOLD held support and continued its bullish march as expected Big test now is the ATHs range Break out above that and miners are gonna go bananas I'de expect the big bank paper shorts to try to tamp it down there tho Be cautious of a scam wick above b4 we really breakout [StonkTrump tweeted about]
[c2_1102] (twitter, PALI) $PALI NHOD from $5.80 dip adds > [@Bullish_Trades]( > $PALI added [BullishTrade quote tweeted BullishTrade about PALI]
[c2_1103] (twitter, QQQ) $TLT Daily. #TLT reclaimed 200sma, but still has work to do as it's still contained within this ascending triangle $SPY $QQQ $IWM $DIA $XLF $GLD $VIX $KRE $DXY $SVIB $SBNY $CS #StockMarket [reciknows tweeted about GLD, STOCKMARKET, KRE, DXY, CS, XLF, DIA, VIX, SVIB, SPY, TLT, SBNY, IWM, QQQ]
[c2_1104] (reddit, MU) ORCL vs. RAM/GPUs Has anybody paid attention to the fact that stock prices of data center providers (ORCL, CRWV) are falling while RAM prices are rising, and NVDA is slashing gaming GPU orders? Short (or limit) MU and NVDA?
[c2_1105] (twitter, TSLA) $TSLA (daily) after 3 rejections at 20ema, now false breakout reversal with close below 5/9ema (Yes, that's bearish) [kpak82 tweeted about TSLA]
[c2_1106] (twitter, BA) $BA Boeing and Avolon have announced their intent to purchase an additional 40 "737-8" airplanes, expanding Avolon's 737 Max portfolio. The order comprises both the 737-8 and 737-10 variants. [Mr Chuck Norris tweeted about BA]
[c2_1107] (reddit, BULL) When do we buy the dip? What are the things you watch or do you you just DCA? My framework is pretty simple… I buy half of intrinsic value and then watch the VIX closely to enter. Im keeping an eye on the close tonight as its the 3rd Friday of the month.
[c2_1108] (reddit, BTC) I feel so late. Late 20s So this year i started buying Bitcoin. Im on my late 20s. Currently i have around .4 BTC i do my best to DCA monthly but i have to be honest i feel that im late to the party. I want to retire by 45. Is it still possible? I also have powder just in case of a crash.
[c2_1109] (reddit, NPV) Bought a commercial building last year - how do I analyze coworking vs long-term lease Bought a commercial building last year, now deciding between two strategies: (1) operate as coworking business or (2) rent long-term to single tenant. I want to measure which strategy creates more value going forward, with a 5-6 year exit horizon. What analytical framework would you use to compare these two strategies? What metrics matter most? (IRR, cash-on-cash, NPV, payback period?)
[c2_1110] (reddit, GOLD) Stagflation [ Westpac report suggests if war continues with a close of the strait of hormuz then inflation will rise and GDP will decline....we are headed into recession and would this ultimately break the unprecedented 17 year bull market? Inflation and GDP decline = stagflation This is suprisingly similiar to 1973 oil embargo that also restricted oil supply [
[c2_1111] (twitter, F) Important ⚓️VWAP concept Buy the pullback to a VWAP which has been support in the past and hope it will become again, or wait for buyers to regain control and buy strength? 2 me its clear, WAIT 4 buyer to regain control so u r in position of strength from start $F #VWAPtips [BTstocktrading 🔁 alphatrends about F]
[c2_1112] (reddit, CVX) Oil stocks- Late in the game - Swing or short term What are some good oil stocks to swing trade or buy short term to get some profit from delta. I know its late in the game but I am looking for some short term delta to profit. Any suggestions. Some stocks like XOM and CVX are just too high
[c2_1113] (twitter, TW) Extremally excited to introduce: @TWEliteAlgo We have worked very hard to develop a proprietary algorithm that can consistently provide profitable buy/sell signals for: $ES $SPY $QQQ $NQ $VIX $VVIX The Elite Algo is free for all Elite members. [ThetaWarrior tweeted about ES, SPY, QQQ, NQ, VIX, VVIX]
[c2_1114] (twitter, TSLA) A word of warning to those short $TSLA... A huge ~$8bn in put deltas expire in January, which could fuel a pretty big rally. Plus, @elonmusk knows options, he could spark early put destruction with a well timed headline... [HackermanAce 🔁 spotgamma about TSLA]
[c2_1115] (twitter, HOOD) $HOOD - An important SuperTrend level to hold and support. Hold this and I'm liking the chances of upside going into earnings. [IncomeSharks tweeted about HOOD]
[c2_1116] (twitter, SSYS) $SSYS Weekly Chart Falling wedge issues getting TIGHT Bullish divergence is worth noting [Mr Chart Norris tweeted about SSYS]
[c2_1117] (twitter, BBW) $BBW reported earnings of $0.51, consensus was $0.38 via @eWhispers #epsbeat [EPSGUID tweeted about BBW, EPSBEAT]
[c2_1118] (reddit, BTC) Why is there a heavy dump ? Why is there a sudden fall in the crypto market. What's the reason for the institution and whales to sell ? Simple answer would be liquidate and have cash in hand.. but what for.. Even XAU rose above 5000 and then a steep downward trend.. Heard US bonds are going to be out next wee. Read somewhere, th he big whales are selling BTC. But would like to know the real behind it. Educate me . Thanks
[c2_1119] (twitter, TLRY) $TLRY Looks so primed to bounce at major support with a possible double bottom here too [Nebraskangooner tweeted about TLRY]
[c2_1120] (reddit, META) Micron quadruples revenue, +14% after hours. Revenue: $41.46 billion versus $35.84 billion estimated EPS: $25.11, adjusted, versus $20.78 estimated Gross Margin: Adjusted gross margins hit a record-breaking 84.9%, topping tech leaders like Nvidia and Meta. Data Center Strength: Driven by skyrocketing AI memory demand, data-center sales soared 415% year-over-year to $25 billion (61% of total sales). Guides $49-$51B. +14% AH move. Reddit neva right.
[c2_1121] (reddit, BABA) Rising insurance costs push Fannie, Freddie to adjust mortgage rules Mortgage giants Fannie Mae and Freddie Mac are loosening homeowners insurance requirements in an effort to lower one of the ["hidden costs" of homeownership]( The Federal Housing Finance Agency, which regulates Fannie and Freddie, said the mortgage giants would now accept mortgages from homeowners with a leaner type of insurance that covers only the current value of a roof, rather than its full replacement cost. [
[c2_1122] (twitter, AAPL) $AAPL Everyone's been plotting the demise of Apple, but it still lives lol Peep da bounce from the stock split avwap after oversold levels not seen since covid Also reclaiming some previous highs, structure looks great for a move up Big call flow hitting the tape recently too [StonkTrump tweeted about AAPL]
[c2_1123] (reddit, GOLD) Oil stocks Wishing I had some cash flipped back to buy everything that is on discount right now with the war.
[c2_1124] (twitter, COIN) $COIN ERs tomorrow Send it to Valhalla Still hanging onto my calls, I simply didn’t log into my broker during the chop lol > [@blknoiz06]( > also think $190-200 is good spot for dip buys to swing $COIN to all time highs if you're still not in [StonkTrump quote tweeted Ansem 🐂🀄️ about COIN]
[c2_1125] (twitter, SPY) 🚦Consumer Sentiment Coming in Higher Consumer Sentiment: ACTUAL 63.5 .. Expecting 62 🚦Michigan Inflation Coming In higher Expectations: ACTUAL 4.6% .. Expecting 3.5% 🚦Businees Inventories Coming in Below ACTUAL 0.2% .. Expecting 0.3% $SPY $QQQ [reciknows 🔁 ripster47 about SPY, QQQ]
[c2_1126] (twitter, AMPX) RT @kholov23: $AMPX Weekly Chart 💥👇 Falling wedge breakout, currently price is in the process of forming the right shoulder of inverse hea… [Mr Chuck Norris retweeted Mr Chuck Norris about AMPX]
[c2_1127] (reddit, NVDA) Bought NVDA in 2022 because I liked gaming. 870% gain for $168k later
[c2_1128] (reddit, BULL) Shorting Bumble? Anyone looking to short bumble? Their stock price has been going downhill for as far as I can remember. Especially now with potential rising inflation and general uncertainty, investors might be less likely to invest in a stock still relying a lot on growth and investments to keep growing.
[c2_1129] (twitter, ES) $ES Blue levels indicate areas of resistance ill be watching where a bounce could end, if we start to see exhaustion there I'll be taking profit and managing risk on longs I think we may be in store for another 20ema throwback again this week (pink) above the 20 it gets fun [StonkTrump tweeted about ES]
[c2_1130] (twitter, NVDA) NGL, A little worried about this daily TOP candle confirming on $NVDA Track line around $121 Like I said Triple Witching today so a lot of vol [Don't follow Shardi B if you hate Crypto tweeted about NVDA]
[c2_1131] (reddit, IREN) Why did I buy IREN calls? I honestly thought it was going to spike. I'm an idiot and am going to lose all my money 😭. If anyone needs me, I'll be out back behind the dumpster.
[c2_1132] (twitter, MCD) $MCD 2D/W - Intermediate W4 🔑 RBR Supply $260 - $263 🔑RBR Demand $236 - $239 I’ll be looking forward to this breakout. It either happens soon or mid to late fall. Worth Watching until then [ATMSnipes tweeted about MCD]
[c2_1133] (twitter, NVAX) $NVAX Daily Candle Chart Falling wedge is getting TIGHT [Mr Chart Norris retweeted Mr Chart Norris about NVAX]
[c2_1134] (twitter, CE) CEOs, CFOs, and directors of banks buying their stock: Lakeland Financial $LKFN Metropolitan Bank $MCB Customers Bancorp $CUBI Cullen/Frost $CFR PacWest $PACW East West Bancorp $EWBC Heartland Financial $HTLF First Citizens $FCNCA Glacier Bancorp $GBCI [reciknows 🔁 allstarcharts about CFR, LKFN, FCNCA, HTLF, MCB, GBCI, CUBI, PACW, EWBC]
[c2_1135] (twitter, PANW) $PINS - Upgraded at GS, PT $31 from $24 $TTWO - Upgraded at GS, PT $165 from $131 $VZ - Upgraded at Oppenheimer, PT $50 $SPLK - Downgraded at UBS, PT $86 from $125 $OKTA - Initiated at Evercore ISI, PT $45 $PANW - Initiated at Evercore ISI, PT $207 [rachels_44 tweeted about PANW, TTWO, PINS, SPLK, VZ, OKTA]
[c2_1136] (twitter, SPY) Bid here $SPY $ES_F [HackermanAce 🔁 JMA_Trades about ES=F, SPY]
[c2_1137] (news, NVDA) Supermicro soared because of $4 trillion Nvidia—but Jensen Huang can walk away any time he wants
[c2_1138] (twitter, TSLA) $TSLA Textbook example of previous support acting as new resistance 📚 Thin volume just below in the $160 area... ⚠️ [reciknows 🔁 TrendSpider about TSLA]
[c2_1139] (reddit, GOLD) Iraq Oil Output Plunges About 60% as Iran War Blocks Tankers > Iraq’s oil production has collapsed by about 60% as the Iran war means there aren’t enough tankers to load the nation’s crude. > The country is now pumping about 1.7 million to 1.8 million barrels of oil every day, down from about 4.3 million a day before the conflict began, according to people with knowledge of the matter, who asked not to be identified because the output levels aren’t public.
[c2_1140] (twitter, ES) S&P500 #SPX #ES1 Key Level Must Hold!! Otherwise we will drop to 3800 and maybe even revisit to the 3500 low [CryptoBullet1 tweeted about SPX, ES1]
[c2_1141] (twitter, COIN) $COIN Coinbase approaching area of interest on daily (throwback) [Cheds (Trading Quotes) tweeted about COIN]
[c2_1142] (twitter, MC) $F Sentiment Alert: $F Increasing bullish pressure with an imbalance in net option delta of 618 K. [HackermanAce 🔁 MC_OptionTrades about F]
[c2_1143] (reddit, COIN) Hawk Tuah Girl: SEC ends meme coin investigation, 'work is complete'
[c2_1144] (reddit, ETH) Cryptocurrency Vs. Stock Market Will crypto be a thing in the future? I know it’s certainly more risky than stocks but could crypto have More upside than stocks over the next 20-30 years? I have about $5k in crypto (Eth and metaverse coins) and wondering if I should pull out and go all in stocks. I know the current administration is pushing crypto and I see more ads for companies accepting as a form of payment. Opinions?
[c2_1145] (twitter, FF) #FF @wantonwallet - I learned most of I know about astrology from the clues he left here and there over the years > [@wantonwallet]( > So the current astrology is the planet of big (Jupiter) conjunct the planet of crazy (Uranus) in the sign of the Bull (Taurus). Make of that what you will and let's all check back in a month to determine if we indeed live in a simulation. [Awawat quote tweeted wantonwallet about FF]
[c2_1146] (reddit, NVDA) NVDA gains 20k-230k Ive been told to sell since 2023. But as a 26 year old it’s nice to see numbers go up. The rest is in QQQ
[c2_1147] (reddit, BTC) Why gold and silver is falling? As a noob I thought war at this scale will lead to rise in gold and silver prices. Currently my portfolio has been reduced to a fraction of original as stop losses triggered in equity holding and I am sitting on cash but didn't keep stop losses in gold and silver ETFs.
[c2_1148] (twitter, S) $S Good spot for me to pick up a small position to have some AI narrative exposure. Not going crazy but a couple small diversified positions could run nicely. High risk high reward when trading narratives. But the chart looks good as well. Small allocation. [Nebraskangooner tweeted about S]
[c2_1149] (twitter, NFLX) $NFLX Wish I'd have seen this one yesterday. Can't chase a short with any size here but definitely looks ready to nuke. [Nebraskangooner tweeted about NFLX]
