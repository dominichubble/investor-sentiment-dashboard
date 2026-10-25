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
----- ITEMS (50: c2_2100 .. c2_2149) -----
[c2_2100] (reddit, MSFT) What is your method of keeping up with economic and market news? Besides reddit, how do you consistently track key financial / economic news to cut through the noise? Both in terms of broad macroeconomic trends and also developments about the companies/markets you’re interested in? What sources / methods are your go to?
[c2_2101] (news, BFS) BFS Products Market Surge at 5.8% CAGR, Decline in Growth Likely Due to Coronavirus Outbreak, ...
[c2_2102] (reddit, GME) The most mentioned stocks on Reddit last week - GME is still #1 GME once again was the most mentioned stock on Reddit with 2,156 mentions, up 11.5% week over week, while the stock gained 3.2%. The most popular posts focused on hopes for a seventh straight positive earnings print, a broker move removing margin on the stock in Canada, and strength in GameStop warrants after hitting a new after-hours high. Source: [
[c2_2103] (twitter, INTU) $INTU Close to breaking down on weekly. [Nebraskangooner tweeted about INTU]
[c2_2104] (reddit, ETH) Ditching gold for Bitcoin Obviously opinions are divided here, that’s the nature of being outside the formal system. Can I throw this open to evidence based grounded in reality responses to the following question: What are the best arguments for moving past the idea of a diversified portfolio, gold and Bitcoin and selling your gold for Bitcoin? Thanks
[c2_2105] (reddit, AI) Block just cut 40% of its workforce for AI - now traders think it could trigger the next Bitcoin rally
[c2_2106] (reddit, ETH) Buy now or Dca? I just liquidated 10k of stocks to put into bitcoin. Would it be a better decision to dca over the next year/years or so with it or just put the 10k in right now?
[c2_2107] (twitter, JPM) $FDX - PT $279 from $282 at JPM $ULTA - PT $485 from $475 at Credit Suisse $SQ - PT $170 from $180 at Credit Suisse [rachels_44 tweeted about FDX, ULTA, SQ]
[c2_2108] (twitter, TSLA) I think there will be two factors putting some form of buying pressure onto the markets early Q1. 1. Position rebuying from tax loss harvesting. (Most hit stocks should see the biggest effect from this e.g. $TSLA.) 2. Funds that unwound which re-open and rebuy their positions. [DaanCrypto tweeted about TSLA]
[c2_2109] (reddit, CVX) Oil price and Producer & operators Ratio (XOM, DVN., DOW, APA etc) If Oil goes up $150- Does stocks of producers, service operators -- XOM, DVN APA DOW and All others go up with same ratio? I mean does it hit a ceiling. Reason for asking is because XOM CVX are already at all time high.
[c2_2110] (reddit, BULL) Macro analysis tips of the day! Volatility VIX = High High VIX means: • risk aversion • unstable liquidity • trend failure probability increases So holding positions too long becomes dangerous.
[c2_2111] (reddit, COIN) Which projects do you think are next to collapse? Curious what exchanges, coins, or platforms people think are most at risk if this bear market continues. Are there any you’re actively avoiding right now?
[c2_2112] (reddit, DELL) Why no one posted DELL gains? With the recent earnings I’m surprised no one posted any Dell gains I moved to the hood earlier this year and got my bonus tendies threw it all into Dell calls and some more. Will continue to hold cuz this thing undervalued, trust me bro
[c2_2113] (reddit, BLK) Bitcoin Daily News – February 18, 2026 1. Strategy buys 2,486 BTC for $168 million 2. Bitcoin ETFs hold $85B despite price crash 3. Nakamoto Inc. acquires BTC Inc. & UTXO for $107M 4. Abu Dhabi sovereign funds boost IBIT positions 5. Walmart heirs invest $200M in BlackRock Bitcoin ETF
[c2_2114] (twitter, COIN) Cathy Wood must be leaning on $COIN today No reason for it to be down Has to be a buy IMO [Shardi B Trading #teamshardi tweeted about COIN]
[c2_2115] (twitter, SIFY) $SIFY v2 is live Full fucking send > [@StakifyFinance]( > $SIFY v2 is now live on @Uniswap > > Contract Address: 0xb92da1111d193e5b1f11e6a131b3f5dcaf432af5 > > With a revitalized Repellent and a stronger community backing, where will #Stakify lead us next? > > #AutoStaking #DeFi [CRG quote tweeted Stakify.Finance about AUTOSTAKING, DEFIPERP, SIFY, STAKIFY]
[c2_2116] (twitter, SPY) $QQQ Daily. #QQQ double bottom + falling wedge breakout 🤔 – #FOMC this upcoming week $NDX $NQ_F $XLK #Nasdaq $AAPL $MSFT $SPY [reciknows 🔁 reciknows about XLK, AAPL, MSFT, NASDAQ, NDX, NQ=F, SPY, FOMC, QQQ]
[c2_2117] (twitter, SPY) Seen someone buying puts and losing consistently on these moves up with $spy ? Just wait for them to finally buy calls and go short lol. [HackermanAce tweeted about SPY]
[c2_2118] (twitter, BTC) #FF MY FAV #CRYPTO ACCOUNTS 👑 @ShardiB2 @Pentosh1 @MuroCrypto @CryptoWizardd @eliz883 @CryptoTony__ @casperdefi @Hayess5178 @FatManWithGems @Blockchainsanta @DaanCrypto @FeraSY1 @j0hnnyw00 @CryptoNewton @BTC_Revolution @Phoenix_Ash3s @tehMoonwalkeR @ManilaMax1 @MiddleChildPabk [ShardiB2 🔁 SatoshiOwl about]
[c2_2119] (reddit, TSLA) S&P 500 Bitcoin holdings tracker I vibecoded a site that tracks public companies' Bitcoin holdings according to Coingecko with a focus on companies in the S&P 500 Index. You can find it at Which other features should I add?
[c2_2120] (reddit, FBTC) Don’t want to hold physical gold. What Gold stocks not miners, to buy in trustworthy brokerage I have a small amount of physical gold/silver coins but I don’t want to any more. What trustworthy gold can I buy in schwab or fidelity like buying FBTC for my mom because she doesn’t/can’t hold understand btc. What is the equivalent for gold for her. EDIT: COSTCO might be an option, never done it but they seem to make it easy and 2% of spot price seems not great but ok. Anyone do that?
[c2_2121] (twitter, PINS) $ARKK $ARKF $PINS Cathie Wood's Ark Invest added to its Pinterest Inc holdings by purchasing 26,925 shares through its Fintech Innovation ETF on Wednesday. This move, valued at approximately $949,914 based on the closing price of $35.28, reflects Ark Invest's continued confidence… [Mr Chart Norris tweeted about ARKF, PINS, ARKK]
[c2_2122] (news, ABBV) Will Rescue Deal Impact Last? 4 Coronavirus-Proof Dividend Picks
[c2_2123] (twitter, SPY) $SPY Powell: "these expectations have been disappointing" (inflation expectations) [kpak82 tweeted about SPY]
[c2_2124] (twitter, SI) $SI Rejected the $20 level & the 50ema. If it can get momentum above the $20 level, then it looks like it can get moving towards the $20.93, $21.90, $22.64 resistance. If the $17.18 support doesn't hold, then it looks like it could pullback in the short term towards the $13.89 [optionsprochick tweeted about SI]
[c2_2125] (twitter, VFF) $VFF LONG since $0,95. Stop Loss: Close sub trendlines [crypto_scar tweeted about VFF]
[c2_2126] (reddit, NVDA) Nvidia shares rise 8% as Jensen Huang says $660 billion capex buildout is sustainable
[c2_2127] (reddit, MSTR) Bitcoin in an Ira I wish i could buy bitcoin in my Ira but am only able to either buy a bitcoin etf or something like mstr. What would u guys do if u want the most exposure to bitcoin in ur retirement account that wont be touched for 18 years. Think they’ll be better return for a Bitcoin etf or mstr?
[c2_2128] (reddit, ACB) Does the hassle of tracking ACB affect investor behaviour? I always hear, keep the number of transactions low or only have a few ETFs, don’t buy / sell too often etc… I use the adjustedcostbase website to keep track of everything, but why isn’t this process something that can be improved? Maybe an open standard framework where all financial institutions uploads to, and voila, it’s all done for you.
[c2_2129] (news, ABBV) Another Good Week for Biopharma and Diagnostics Stocks: Innovation is Driving Gains
[c2_2130] (twitter, MRVL) 𝐓𝐫𝐚𝐝𝐞 𝐩𝐥𝐚𝐧 7/9 $AVGO Break over 1757, 1800c Jul 19--to target: $1800+ $CRWD Break over 393, 400c Jul 19-- to target: $397+ $MRVL Break over 75.7, 77.5c Jul 19--to target: $77+ $UBER Break over 71.8, 73c Jul 19--to target: $73+ YOU WANT MORE TRADE PLANS ❤️ [Sean trades tweeted about UBER, AVGO, CRWD, MRVL]
[c2_2131] (twitter, TSLA) $TSLA flow: - Put/call ratio: 0.854, call premium: 2.32 billion - Avg 3d call volume: 940K, Wed: 816K - Avg 3d put volume: 596K, Wed: 697K - 5K 🐳: 59% 🐂 - 15K 🐳: 55% 🐂 - 30K 🐳: 48% 🐂 - 🐂 targeting 01-21 expiry, $1125 strike [unusual_whales tweeted about TSLA]
[c2_2132] (twitter, PG) $PG Setting up for another leg higher. [ThetaWarrior tweeted about PG]
[c2_2133] (reddit, BTC) Bitcoin Down to $65K as Heavy Selling Drives a Sharp BTC Price Drop
[c2_2134] (reddit, BTC) A crypto dashboard focused on high-density data visualization and low-latency updates. The goal was to create a high performance and a ad-free user experience
[c2_2135] (reddit, GOLD) Oil company shares soar to all-time highs as Middle East war turbocharges price per barrel
[c2_2136] (twitter, SPY) $SPY - 1 2 and 3 fail to break 455 and VWAP. 4 breaks 455 and VWAP with volume. You can use top level as trim or take profit because it is resistance [@ItsSqueezy]( Keep an eye on $SPY 455 (our bottom ORB level). VWAP right around that area as well [ItsSqueezy 🔁 ItsSqueezy about SPY, SPY]
[c2_2137] (reddit, GOOGL) Daily Discussion Thread for January 08, 2026 This post contains content not supported on old Reddit. [Click here to view the full post](
[c2_2138] (news, BP) BP’s profits more than doubled at the war in Iran pushed gas prices to new multiyear highs
[c2_2139] (reddit, BTC) US market open It's interesting - I sell options on BTC so I regularly check the price especially market open US time. Seems quite common to have a dump of BTC about 30 mins before, or after market open. I wonder what causes that, or am I just misreading it as increased volatility as people prepare their positions / options.
[c2_2140] (twitter, SPY) $ES_F $SPY $SPX V shape recovery loading in beautiful fashion 🛡$470 🎯$475, $478, $480 [@ATMSnipes]( $ES_F $SPY $SPX Double bottom or V-shape likely as long as $456 lows hold Still some volatility events this week to clear Mega cap tech ER are right around the corner😙 🎯$471, $477, $480, $488, $498 (They might penny tick $500) 🛡$460, $457, $456.50 🐻 Under $450 [ATMSnipes 🔁 ATMSnipes about ES_F, SPY, SPX, ES_F, SPY, SPX]
[c2_2141] (twitter, CLSK) $CLSK Daily Candle Chart 🔨 [Mr Chart Norris retweeted Mr Chart Norris about CLSK]
[c2_2142] (reddit, BULL) Would you buy or sell? I picked buy and lost. Again. Short disclaimer, this is not a live marked its a hindsight marked. So i thought with this big green candle there is no way price will close lower in 10 candles. But i got banged again. But still up all time PNL.
[c2_2143] (reddit, AMZN) When is META going to be done with these lawsuits? [ So many lawsuits, so many states/plantiffs trying to get on board with these claims... when are we going to see this being resolved good or bad? This is the single biggest headwind capping Meta's stock price. They can generate over 30% growth next quarter and sell compute up the wazzu and investors wouldn't care, they want some clarity whether Meta will be able to settle/win or what the true financial impacts will be with these cases many
[c2_2144] (reddit, AI) Leopold Aschenbrenner’s Situational Awareness seeks to raise capital after AI rout
[c2_2145] (twitter, CLSK) $CLSK Weekly Candle Chart Falling wedge pattern [Mr Chart Norris tweeted about CLSK]
[c2_2146] (twitter, MSFT) $MSFT ONLY DOWN $6 $20+ gain from last night's purchase another big score [Don't follow Shardi B if you hate Money tweeted about MSFT]
[c2_2147] (twitter, TSLA) $SPY Daily. #SPY after a few weeks of consolidation, looks like it's breaking out to the upside. Notable bullish divergence on momentum indicators, caution as it can all be erased with #CPI data next week $SPX $ES_F #SPX #ES_F $XLK $IWM $QQQ $AAPL $TSLA [reciknows tweeted about ES=F, AAPL, IWM, SPX, TSLA, QQQ, CPI, SPY, XLK]
[c2_2148] (twitter, SHOP) $SHOP Full gap fill. If it's going to do a LTF bounce it'll be soon.. Otherwise this thing will just fall off a cliff. [Nebraskangooner tweeted about SHOP]
[c2_2149] (twitter, F) $F #F weekly, losing $16 not good Looking at $14 area below next bid zone Target back to $16 [_bread911 tweeted about F]
