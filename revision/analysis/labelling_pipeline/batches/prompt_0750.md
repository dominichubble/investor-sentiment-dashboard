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
----- ITEMS (50: c2_0750 .. c2_0799) -----
[c2_0750] (twitter, TSLA) Had to cut $TSLA while still profitable earlier will be back but my calls were Dec 29 so could not take risk [Shardi B Trading #teamshardi tweeted about TSLA]
[c2_0751] (twitter, SPY) $SPY +$15 since this tweet… Volume increases. [@ATMSnipes]( $SPY Shout out @Dougie_dee for sniping this. Fresh demand zone test. This is the most volume we have seen on the 4 hour since March 4th 2021. Not calling bottom but volume speaks [ATMSnipes 🔁 ATMSnipes about SPY, SPY]
[c2_0752] (twitter, TIVC) You sell, I buy. BING BONG!!! $TIVC [Bullish_Trades 🔁 AD4024 about TIVC]
[c2_0753] (twitter, QQQ) $QQQ The weekly is bearish but the 4hr jumped off support to validate the reversal [Don't follow Shardi B if you hate trading tweeted about QQQ]
[c2_0754] (twitter, TGT) 12. $TGT struggled the previous few years has turned it around and become a household favorite amongst moms TGT has a 29% CAGR since 2015 and also offers a 1.6% dividend yield that has been raised for 50+ years in a row. A good bet on customer loyalty here. [ThetaTendies tweeted about TGT]
[c2_0755] (twitter, CMT) great chart, love the breadth data point > [@DavidCoxRJ]( > SPDR Technology $XLK > sitting 55.4% stocks in #Stage2 uptrends... [Ian McMillan, CMT quote tweeted David Cox, CMT, CFA about STAGE2, XLK]
[c2_0756] (twitter, RIG) $RIG Daily Candle Chart S / R flip at diagonal downtrend line [Mr Chart Norris tweeted about RIG]
[c2_0757] (reddit, GOLD) Portfolio diversification Jeremy Grantham was in the Diary of a CEO podcast and, besides predicting yet another bubble, suggests disinvesting in the USA by investing in index funds that exclude the USA. Also a bit of gold/silver, plus bonds. What do you think of this portfolio distribution? [
[c2_0758] (twitter, MRNA) $BNTX Redburn Partners upgraded from sell to neutral. $MRNA Redburn Partners upgraded from sell to neutral $WFC Atlantic Equities upgraded from Neutral to Overweight. $60 pt [Honeydew0101 tweeted about BNTX, MRNA, WFC]
[c2_0759] (reddit, BULL) Bears are back in town Screenshots are mostly of gains today (June 9), but the June 8 puts were from last Friday.
[c2_0760] (twitter, CMT) $CG with a similar look... [Ian McMillan, CMT tweeted about CG]
[c2_0761] (reddit, ETH) Cathie Wood Sees No More 85% Bitcoin Price Drawdowns Versus All-Time Highs
[c2_0762] (twitter, AMZN) That beat $AMZN was nothing slight Sheesh [Avery tweeted about AMZN]
[c2_0763] (twitter, ITP) $ITP added. Premarket high of .324 [Bullish_Trades tweeted about ITP]
[c2_0764] (news, CSCO) Bronson Point Management LLC Buys Cisco Systems Inc, Albemarle Corp, Xilinx Inc, Sells Lumentum ...
[c2_0765] (twitter, BOOM) $NFLX BOOM! Target two achieved. Front running slightly and locking most of my gains. [Amdtrades tweeted about NFLX]
[c2_0766] (twitter, DPZ) Been building a little short exposure on stocks incase they pull back today. Wish I got more exposure on $DPZ that's still my favorite monthly time frame position. Just trading momentum this morning [Nebraskangooner tweeted about DPZ]
[c2_0767] (twitter, ALTS) #Altcoins - Not saying this is a bottom, but loading up on $ALTS after a red daily wick has been a very profitable day. Time that with a Tuesday pump and I'm shopping heavy today. Lower time frame trades, selling on weakness. [IncomeSharks tweeted about ALTS]
[c2_0768] (reddit, ETH) ETH is dipping right now — good time to buy or should I wait? Ethereum has been falling over the past few days and the price looks tempting. I’m thinking about buying some for the long term, but I’m not sure if I should enter now or wait in case it drops more. What would you do in this situation, buy the dip slowly (DCA) or hold off for a bigger pullback? Would appreciate some advice from people with more experience.
[c2_0769] (twitter, BA) Will never buy $BA in the age of social media. Any time there's a plane crash their stock tanks. Used to not be as big of a deal, until the entire world was aware of every crash every time it happens. > [@avalex]( > @CryptoParadyme Have you looked to $BA lately ? Looks like it might be a good buy [CryptoParadyme 🔁 avalex about BA, BA]
[c2_0770] (twitter, PFE) $PFE A new study shows that Pfizer's RSV vaccine, Abrysvo, is safe for pregnant women and does not increase the risk of preterm birth. [Mr Chart Norris tweeted about PFE]
[c2_0771] (twitter, CMCSA) Came down until about 11 and then it pumped up. Plan worked from stream, gonna ride these for now and see what we do closer to power hour. Also still in comcast. $spy $cmcsa [HackermanAce tweeted about SPY, CMCSA]
[c2_0772] (reddit, ETH) How can I buy a crypto without a Bid-ask-spread? Can someone help me out I've been trying to buy any crypto without a Spread because when I do, I am already in a lossing position. Is there any brokers that don't do Spreads for crypto?
[c2_0773] (reddit, VELO) VELO up almost 14% today, squeezing Seeing if momentum holds, could be nice NFA
[c2_0774] (twitter, PINS) $pins +24% off the buy zone after hours …👀 [hyperbitcoins tweeted about PINS]
[c2_0775] (twitter, MSFT) #1 $INTC > $56.16 Supply/ flag break Resistance: $57.42, $58.79, $59.98 JAN 21 56c #2 $MSFT > $311.50 Range re-break Resistance: $314.48, $320.70, $328.12 JAN 21 315c Two setups I am watching this week to see if they materialize! 📈 😀 [optionsprochick tweeted about INTC, MSFT]
[c2_0776] (twitter, EMA) $SPY $SPX $ES Little blow off the top action this morning to trap the bulls. Rejection of ORH should set for Downside continuation. 🎯$456, $452.50, $450, $448, $443-$444 This could get choppy, Time to play defense and protect your capital. [@ATMSnipes]( $SPY $SPX $ES Red Pill or Blue Pill and why? Both counts make complete sense. For me the structure looks to be there for an impulse to $471. 200EMA should meet price near the .382 as well. [ATMSnipes 🔁 ATMSnipes about SPY, SPX, ES, SPY, SPX, ES]
[c2_0777] (reddit, AAPL) How can you pass up DCA into AAPL? So, at the 207, how can this ultimately be a bad way to go. Will it really dip well under $200? And if so, then buy more. This will certainly pass, and when it does, how do you justify not buying into this dip? What are some of the other reasons that you would stay away from Apple? They seem to be too smart of a company to not be able to weather this storm. If this is not the time, then when? Thoughts?
[c2_0778] (twitter, ETHE) $ETHE confirmed that bottom candle nicely right at support before the trend flip [Shardi B Trading #teamshardi tweeted about ETHE]
[c2_0779] (reddit, GOOGL) Blockchain.com "Wallets & Addresses > Manage > Recover Funds" | The entire "Manage" section is removed from GUI - how do proceed? How do I proceed with this? I have an unorthodox setup which requires me to access my funds this way. This options doesnt seem to exist anymore and they banned/suspended my zendesk support account.
[c2_0780] (reddit, COIN) The Quiet Lock-Up Supply doesn’t tighten because of events. It tightens when coins quietly move into custody and stop circulating. Volatility compresses first. Repricing comes later. Scarcity isn’t loud. It’s structural. — HODLHero ₿ reconrender.store
[c2_0781] (twitter, VNO) Feeling a lot better continuing to day trade current conditions rather than try to swing Some positions I've closed early but others have completely retraced I'll hold $VNO & keep with my lottery plays but still focus on day trading Decent day for stock longs on the gap down👍 [Nebraskangooner tweeted about VNO]
[c2_0782] (news, A) Agilent Technologies (A) Earnings Expected to Grow: Should You Buy?
[c2_0783] (news, A) Agilent Technologies Announces Cash Dividend of –8 Cents Per Share
[c2_0784] (twitter, INTU) $INTU New short position taken today that's already in profit. [Nebraskangooner 🔁 Amdtrades about INTU]
[c2_0785] (reddit, QQQ) Investing in QQQ(M) in 2026? It seems like tech is currently extremely saturating the S&P, but is it still worth separately investing in QQQ? (Even if I have 80% of my portfolio in the S&P?) I also am a little worried that this year tech may not do that well or even beat the market but want to hear others thoughts.
[c2_0786] (twitter, ADBE) Some relative strength names from today may be setting up for a downtrend break on daily if market holds 📈 To name a few $ADBE $DDOG $COST $AVGO $NOW $TEAM $LULU $HD [rachels_44 tweeted about ADBE, DDOG, COST, AVGO, NOW, TEAM, LULU, HD]
[c2_0787] (news, AB) Global Deferiprone Market is Set for a Moderate Growth and is Expected To Reach USD '7. ...
[c2_0788] (reddit, QQQ) TOS sold my QQQ C615 "The execution was the result of a margin sell-out" **I DONT HAVE A MARGIN ACCOUNT**, this is a cash account and Im on a computer (re "Phone Sell" I was in the money bought at .73, "sold" at 2.24 Account was not negative settled funds showed I had enough for 1 last trade of the day I always close the position before the end of the day, this is the first time they ever did it for me
[c2_0789] (reddit, AIXI) $AIXI Sued Apple for $1.3B ... AIXI Patent has been reconfirmed .. if they settle for $500M . That's enormous for a $3M market cap Co. this hasn't been fixed yet ... at some point they will settle for something and go for a buyout may be ... let's see .. ✅ Keep this company on watch 👀
[c2_0790] (twitter, PT) Flagging on the daily. Above $191.95 is new ATH that means clear skies. [@MelStuart9]( #honeydewlist @Honeydew0101 @TheStockKnight1 @StockSwingAlert Thoughts on $expe PT ? 👇 [Honeydew0101 🔁 MelStuart9 about EXPE]
[c2_0791] (twitter, NVDA) $NVDA Keybanc analyst John Vinh maintains an Overweight rating on NVIDIA and raises the price target from $130 to $180. [Mr Chart Norris tweeted about NVDA]
[c2_0792] (reddit, GOOGL) Closed positions before holidays Greece, here I go! Thanks Google. May a dip happen so I can hop on again 🙌🏻💥 One year after I gave back all my gains because of NVDA. Finally recovered my profits.
[c2_0793] (twitter, AMZN) $AMZN there she goes Targets over flag highs: $3349-$3355, $3400 [@ATMSnipes]( $AMZN nice consolidation this past week. Break north of resistance I’ll be looking for $3317 next. I wouldn’t be surprised to see $3400 trade. [ATMSnipes 🔁 ATMSnipes about AMZN, AMZN]
[c2_0794] (reddit, RC) $GME about to go up & Warrant Interest still @ ~31% Stochastic(20,12,12) shows short-mid term reversal as of today. NFA, the stock looks like we are going up for the next few weeks. What's RC cooking? GME warrant interest rate is still sitting at \~31% (fidelity, getting paid out to me) as of today. What's going on with $GME while other stocks are weakening?
[c2_0795] (reddit, RSI) AMD Puts?? Isn’t this stock along with the rest of the circle-jerk-prone sector overrated?? RSI approaching overbought, MACD looks like it’s topping, Volume diminishing… From the blue vertical line (since mega pump), sales have actually outweighed buys, 530M buys vs 775M sells… dating back to last April, sell power still slightly outweighs buying power, at a clip of 3.37B buys vs 3.51B sales I’m thinking it retests the bottom of the FVG, which I have roughly around the $200/sh mark…. thoughts??
[c2_0796] (reddit, BULL) SpaceX. Anyone else planning to buy puts and hedge nasdaq. I'm extremely bearish on SpaceX due to its horrible fundamentals, media coverage, and my belief that holders of locked-up stock options will sell once their shares unlock. I plan to buy puts and short SpcveX at a weighting of about 2% of the Nasdaq, since that's roughly its share of the index's market capitalization, so it should hedge it out.
[c2_0797] (reddit, BTC) When do you all hypothesize BTC will drop under $100k again I’m curious what you all think, Will BTC dip below $100k again soon, or are we past that level for good? I’m interested in your thoughts, or any theories/ hypotheses
[c2_0798] (reddit, BABA) [The Economist] China’s hereditary elite is taking shape: The Communist Party is afraid to tax inherited wealth
[c2_0799] (twitter, ZTS) $ZTS gud 🤝 [@StonkTrump]( $ZTS at a big level here Cloud entry with daily momentum indicators pointing up Option flow also bullish betting on a breakout cc: the cloud god @HackermanAce [StonkTrump 🔁 StonkTrump about ZTS, ZTS]
