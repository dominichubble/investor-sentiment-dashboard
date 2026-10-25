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
----- ITEMS (50: c2_1650 .. c2_1699) -----
[c2_1650] (reddit, SI) ICICI prudential technology growth fund This was my first Fund when i started SIPs 3.3 years back so please don't troll me 🥺 (i know how people despise sectoral funds in this sub) I know timing is the key in sectoral funds and i think probably did timed it right, Before the anthropic AI crash i was sitting on steady 20% profit in this fund with peak of 33% Anyone else invested in this fund? What will be your advice average or sell?
[c2_1651] (twitter, PLTR) $PLTR hope this made sense before the earnings [Mr Chart Norris tweeted about PLTR]
[c2_1652] (reddit, GOLD) Good news for Dogecoin Meme "Like ashes in the fall" Is this life even real Fam?? Look at the shadows!!! You can't even optimize the angles because the loads are too heavy. They aren't even in sync Fam...
[c2_1653] (news, RS) Stocks Showing Rising Market Leadership: Agilent Technologies Earns 82 RS Rating
[c2_1654] (twitter, LIN) Relative Strength Rankings - European ADRs $RELX $NVO $OEC $NVGS $LIN $CCEP $CAAP $BP $AZN $AON $ALKS [the_chart_life tweeted about ALKS, RELX, LIN, CCEP, OEC, CAAP, AON, NVGS, NVO, AZN, BP]
[c2_1655] (reddit, MSFT) CVNA continues its Downside with the NASDAQ. As of 10:25 my portfolio has been helped by my short positions. CVNA down 11.38. Quoting SamShames of Simpler Trading. “ I think it’s a 200 stock. The QQQs are down 11.5 GOOGL down 8 new low WFC down 1.5 IGV the software ETF off.25 GLD -14 META crushed -44 Hope to update later
[c2_1656] (twitter, ROKU) Someone DM'd about $roku n its gap. Ill post a chart n explain ur message. > [@HackermanAce]( > Gap tickers tips: > > Lets say something gaps up or down, it doesnt matter if its a random event or a result of ER's. > > A setup forms several LTFS candles later. > > Meaning you'll see a bbands extension followed by consolidation before a valid move. > > Dont chase gaps, chase setups! [HackermanAce 🔁 HackermanAce about ROKU]
[c2_1657] (twitter, META) $META emerging from the 20ema with notable strength today 👀 [optionsprochick tweeted about META]
[c2_1658] (reddit, HOOD) Robinhood: Extended Hours OPTIONS Now 8:15PM-9:25AM I just got a pop-up for Extended Options trading: 8:15P-9:25A EST SPX, XSP, VIX, RUT I received it by searching for ticker SPX. Edit: Seems to also be on Schwab (ToS). Please post if it is available on other platforms. Does not seem to be on Tasty yet. Also I’m fairly confident if you open a position at 8:15 and sell before 8:15 the following trading day it will count as a Day-Trade.
[c2_1659] (twitter, NVDA) $NVDA Net Options Delta: +1.3M shares Blue = Call Delta Orange = Put Delta White = Total Delta Options market showing a highly bullish correlation with price. [Theta Warrior tweeted about NVDA]
[c2_1660] (twitter, BILI) $BABA $BILI Alibaba offloads ADRs in streaming site Bilibili for $357.8 million, reports Bloomberg. [Mr Chart Norris tweeted about BILI, BABA]
[c2_1661] (reddit, PMI) Geopolitical Shocks Rattle Markets, PMI Rebound Signals BTC Liquidity Boost
[c2_1662] (twitter, QQQ) The trend continues to move in the wrong direction for Growth- and Tech-oriented investors... $QQQ vs. $IWM [@the_chart_life]( $QQQ vs. $IWM Losing its rate of trend from the March 2021 lows... [the_chart_life 🔁 the_chart_life about QQQ, IWM, QQQ, IWM]
[c2_1663] (reddit, RENT) Arguments FOR trading? I feel like trading is like gambling. The house wins, the house being robots for the big guys with HFT capabilities. Even if you are really good, it's hard to beat a robot with access to information and hardware you don't have. What are some arguments FOR trading as a single individual? EDIT: I'm not trying to strike a nerve here, it's quite literally a question.
[c2_1664] (twitter, UNG) $UNG Going for a LTF breakout here. [Nebraskangooner tweeted about UNG]
[c2_1665] (twitter, QQQ) $PYPL Weekly. #PYPL setting up for a possible wedge breakout w/ bullish divergence $SQ $QQQ [reciknows tweeted about PYPL, SQ, QQQ]
[c2_1666] (twitter, BTC) Equities still look bullish short term, trend is up & price coiling in a bull flag Smash below 3840 ish on $ES to nuke market structure & local top probably in but until then pain trade might still be higher... which would help keep BTC bid [MacroCRG tweeted about ES]
[c2_1667] (twitter, ES) $SPX $ES_F grateful to the market. +5.9k Locked in 3/4. Riding my $SPY swing position. [kpak82 tweeted about SPX, ES_F, SPY]
[c2_1668] (reddit, BULL) If I buy mutual fund tomorrow. Will it reflect by 31st March,as in be my investment for FY 25-26. how long does it take ?? Please answer ASAP. I need to make some investment in FD or Mutual fund. And the bank are closed for the next 4days. Thank you. .
[c2_1669] (twitter, MRNA) $MRNA is up about $7 dollars more from the $210 break when I posted this. What a strong mover today! Definition of strength & momo 👀 > [@optionsprochick]( > $MRNA what a weekly breakout candle 👀 [optionsprochick 🔁 optionsprochick about MRNA]
[c2_1670] (twitter, QQQ) The largest drops in RS over the last two weeks have come from Tech, Software, High Beta, Nasdaq 100, Mobile Payments, Cloud Computing, Homebuilders, Mega Cap Growth, and Data Sharing $XLK $XSW $SPHB $QQQ $IPAY $XLOU $XHB $MGK $BLOK [the_chart_life tweeted about XLK, XSW, SPHB, QQQ, IPAY, XLOU, XHB, MGK, BLOK]
[c2_1671] (news, AB) AllianceBernstein (AB) Gains But Lags Market: What You Should Know
[c2_1672] (reddit, META) Assuming you have $1 million, which of the following stocks do you think would maximize your returns over the next 10 years? Assuming you have $1 million, which of the following stocks do you think would maximize your returns over the next 10 years? Please list examples of the stocks you believe are the most worthwhile investments. ☺ 1. Nvidia 2. SpaceX 3. AMD 4. Micron Technology 5. SanDisk 6. ASML 7. Broadcom 8. Apple 9. Google 10. Microsoft
[c2_1673] (reddit, ETH) Invest on ETF at my 39yrs for 20 ye plan hey to everyone, I am 39yrs old male and I am trying to invest on ETF for my pension. I am already give 500euros per month on a family plan in a greek etf through my job and because of that I have a big txa reduction. I have about 200e per month to invest and I am thinking to do it through revolut. I am looking for a 20yr plan. it is enough to help my kids to pay their studies. what do you recommend to invest?
[c2_1674] (twitter, ES) S&P 500 all-time highs before year end? $SPX $SPY $ES_F [Ian McMillan, CMT tweeted about SPY, SPX, ES=F]
[c2_1675] (twitter, HUT) Getting Ready ……… > [@kholov23]( > #Bitcoin $BTC $MARA $CLSK $RIOT $CAN $WULF $HUT - > Getting Ready [Mr Chart Norris quote tweeted Mr Chart Norris about MARA, BTC, CAN, RIOT, CLSK, BTC, HUT, WULF]
[c2_1676] (twitter, ES) $DIA Daily. #DIA was the strongest of all indices from the low's, but as of late has been a laggard. Looks like it's setting up for a move higher though with a wave triangle (invalid below 329.35) $DJIA $SPY $QQQ $IWM $ES_F $SDOW #DowJones [reciknows 🔁 reciknows about DJIA, IWM, SDOW, SPY, ES=F, QQQ, DIA, DOWJONES]
[c2_1677] (reddit, GOOGL) 0% Apr Bitcoin calculator Seeing posts about using 0% Apr promos on credit cards to buy bitcoin and hold or reverse DCA. Thought it was an interesting strategy so I built a google sheet calculator to test the numbers. I wanted to share for others interested or feedback. Let me know what you think!
[c2_1678] (twitter, DPZ) $DPZ Currently at 99sma resistance retest. Could reject here or could deviate. Deviation play would be a good one to get in a short. Meaning if it breaks back below the trendline and red resistance would be a more high confidence play. At monthly resistance now too [Nebraskangooner tweeted about DPZ]
[c2_1679] (reddit, GOOGL) Is there a website that provides the FUTURE GROWTH RATE? I have gone through at least 20 popular sites and like another 20 sites that came up on Google search. None of them seems to show the future growth rate. For context, I am creating my own rule one investing sheet. I have my own growth rate based on the calculation in the book. I want to just check against what others have as the growth rate for the same company. I can't find it on any sites.
[c2_1680] (twitter, COST) $COST Key resistance backtest on monthly time frame [Nebraskangooner tweeted about COST]
[c2_1681] (reddit, BTC) Need help with portfolio I am investing for my kid. I want to build a dividend portfolio so that I can use for his education and other expenses if he wants to start his own business. Not sure what career options will be available after 18-20 years because of AI. What ETF should I be investing in? My monthly budget is $100-$200. TIA
[c2_1682] (reddit, AI) NEC stock surges 5% on AI strategy and Anthropic partnership By Investing.com
[c2_1683] (twitter, LAC) $LAC pulled into the 20/ 50emas. To me it looks like it needs to recover the $24.30 to move higher. Its trading in a tighter range below the 200 ema. It looks like the emas are trying to realign but has some work to do still. More of a pullback look below $23 [optionsprochick tweeted about LAC]
[c2_1684] (twitter, SPY) $SPY things are looking ugly and extended but it could get worse unless we rally into close over today's highs. Major support breakdown today. [kpak82 tweeted about SPY]
[c2_1685] (twitter, CRM) $CRM daily downtrend break above the 50sma 📈 [rachels_44 tweeted about CRM]
[c2_1686] (twitter, CMT) New YTD highs for $USB... [Ian McMillan, CMT tweeted about USB]
[c2_1687] (twitter, MU) $MU Micron Technology has shipped the industry’s first ninth-generation TLC NAND in SSDs. The G9 NAND achieves a transfer speed of 3.6 GB/s, enhancing performance for AI and data-intensive applications, and offers 73% higher density compared to competitors. [Mr Chart Norris tweeted about MU]
[c2_1688] (twitter, SNAP) Feeling Monday morning euphoria after a big win in 30 minutes on a $SNAP short. What a sweet trade. Entry $36.3 75% exited $34.9 and maybe one more drop to $34.60 area. I think I deserve the rest of the day off now. [Nebraskangooner tweeted about SNAP]
[c2_1689] (reddit, COIN) Meme Coin analysis I think we can all agree that all meme coins are a 0 sum gain, so if you make money someone else's is loosing that money. So you are either a thief or someone getting money stolen. Therefore it is immoral to buy meme coins. prove me wrong.
[c2_1690] (twitter, BABA) $BABA yes that was hammer 🔨 / falling wedge breakout 💥 > [@kholov23]( > $BABA 1H Chart > > Falling wedge pattern / price approaching the upper boundary / resistance / of the wedge [Mr Chart Norris quote tweeted Mr Chart Norris about BABA]
[c2_1691] (twitter, AAPL) $AAPL Continues to look amazing, blue skies on deck IMO Big AI upgrade cycle coming [Don't follow Shardi B if you hate Crypto tweeted about AAPL]
[c2_1692] (reddit, PGR) What financial stocks are worth investing in? I am 95% in Big Tech and I'd like to venture out so I decided to add some finance. What do you guys suggest? My list right now is MS, GS, JPM, AXP, COF, and PGR. Anything bad about these picks? I tried to pick well known names that have seen good returns the past 5 years.
[c2_1693] (reddit, AMZN) US Stocks are Reddish..What’s Your Go-To Short-Term Pick Since US stocks are currently in the red, I’m considering putting some money into a stock for a 2–3 month short-term investment and then selling. What stocks would be your go-to picks for this kind of timeframe right now? Looking for ideas based on fundamentals + potential near-term recovery, not just hype. Would love to hear what you’re watching and why. My Longterm is MSFT, Palo Alto and Nvidia.
[c2_1694] (reddit, NVDA) Leveraged ETF's vs. traditional ETF I tend to find my investment style a little more active and aggressive.. Even for long term stuff, everyone loves to suggest VOO. Except, if this is the default and people are "almost" banking on 10% year over year long term.. Then why not UPRO (x2) over VOO? Or.. QLD (x2) over QQQQ? I don't own NVDA.. but I have owned NVDL (x2) since the spring dip.
[c2_1695] (reddit, ETH) 5k-66k in 3 days. Life finds a way This shit is not for the faint of heart. Literally just scalping SPY 0dtes in and out in less than 30 seconds.
[c2_1696] (twitter, OXY) BIIIG fan of $OXY at this level. Buffett's stake now at 26.8% as he added another 6.8% in recent reports. The technicals suggest a surge in price is eminent. Daily expanding Triangle + 1,2 wave count in place 🎯$76, $82.59, $89.66 [ATMSnipes tweeted about OXY]
[c2_1697] (reddit, AAPL) Michael Saylor’s Bitcoin Strategy Drives 2,466% MSTR Surge, Outperforming Nvidia, Tesla, Google, Apple, and Microsoft
[c2_1698] (news, A) Agilent (A) Beats on Q4 Earnings & Revenues, Lowers Guidance
[c2_1699] (twitter, COIN) At what point do people realize that volatility brings more trading, not less Good for exchanges like $COIN and $HOOD [Shardi B Trading #teamshardi tweeted about COIN, HOOD]
