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
----- ITEMS (50: c2_1750 .. c2_1799) -----
[c2_1750] (twitter, SHOP) Pardon ? $SHOP > [@arianrah]( > @kholov23 $58 before the end of July [Mr Chart Norris quote tweeted Arian Rahbari about SHOP]
[c2_1751] (twitter, MP) $MP not a long time ago > [@kholov23]( > $MP Daily Chart ☕️ > > Cup and handle formation, presently price is testing the neckline [Mr Chart Norris quote tweeted Mr Chart Norris about MP]
[c2_1752] (twitter, GOLD) Plunge on the way. Target 1715-1725 > [@MFHoz]( > Big drop coming for $Gold. [MFHoz 🔁 MFHoz about GOLD]
[c2_1753] (twitter, AI) It‘s safe to say that there are by far less $ARC tokens available than 200 mill, but let’s do this: 500k #ReactorAI users can only buy 400 $ARC each until supply is bought up. Do you think an #AI with world leading benchmarks will have only 500k people using it? @ARCreactorAI 👀 [WIZZ🥷 ( beware scammers ) retweeted McToken about AI, REACTORAI, ARC]
[c2_1754] (twitter, SPY) Lol I picked up one $SPY 2/7 450p contract overnight at 3:59 pm EST simply based on this supply zone after a weird V shape recovery in a weak market... I expected weakness but not a rug pull... [ItsSqueezy tweeted about SPY]
[c2_1755] (twitter, SPY) Yes 😂 engulfing only on $SPY > [@574capital]( > @kpak82 Just spy right [kpak82 🔁 574capital about SPY]
[c2_1756] (twitter, NCNO) $NCNO reported a loss of $0.01, consensus was ($0.02), Earnings Whisper was ($0.02) via @eWhispers #whisperbeat [EPSGUID tweeted about WHISPERBEAT, NCNO]
[c2_1757] (twitter, LLY) $LLY 🟪🟧 Eli Lilly's Q1 2024: Adjusted EPS: $2.58 (Beats estimate of $2.46) Sales: $8.768 billion (Misses estimate of $8.922 billion) [Mr Chart Norris tweeted about LLY]
[c2_1758] (reddit, GOOGL) GOOGL call options What do you all think about GOOGL hitting $160 by 5/30? It was at $165 earlier this week and just got beat up with Apple News and it seems overblown. They are still dominant in Search and have over 52% of digital ads market share. Seems like it has resistance at $165 but at least should make its way to $160-$161. Curious on other’s thoughts on this play.
[c2_1759] (reddit, SPRC) $SPRC bouncing off 52-week lows while MSOS momentum heats up
[c2_1760] (reddit, NNE) Nuclear energy - good investment? What do you guys think about nuclear energy stocks like OKLO, NNE and others? Goes without saying, energy is in short supply for AI and nuclear energy looks like a really good solution. The current administration just recently announced an investment. However, it doesn't look like the technology is there yet, the stock charts look like a crazy rollercoaster. I am a strong believer in the long term though - are you?
[c2_1761] (twitter, BA) $BA hits the middle channel, shut down at resistance and volatile with horrible plane crash news. I'll be careful with this one for the time being. Too much going on news-wise [ItsSqueezy tweeted about BA]
[c2_1762] (reddit, ETH) Vitalik doing his impression of the ETH chart since EF announced they were selling
[c2_1763] (twitter, LULU) $LULU Below 99sma Nice drop already starting today [Nebraskangooner tweeted about LULU]
[c2_1764] (twitter, QQQ) $SPY Daily. #SPY broke out of the broadening formation + negated previous bearish divergence. Next areas of interest for potential resistance 429, 431, 436 $SPX $ES_F #SPX #ES_F $XLK $IWM $QQQ $XLF $KRE $TLT [reciknows 🔁 reciknows about QQQ, XLK, SPY, IWM, SPX, TLT, ES=F, XLF, KRE]
[c2_1765] (twitter, TSLA) $ARKK. #ARKK signs of life and breaking out of a multi-month falling wedge $TSLA $SPY $QQQ $IWM $DIA [reciknows tweeted about ARKK, IWM, TSLA, SPY, QQQ, DIA]
[c2_1766] (reddit, ETH) Why is the crypto market sideways currently ? Not up or down Are we ever going to get out of this bear market in the near future ? It seems like Bitcoin and ETH will pump a little bit then fall back down constantly…..you see all the news that tons of record buying across the board recently but the prices just stay stagnet….i thought volume controls the prices ….does it not ?
[c2_1767] (twitter, CLSK) $CLSK Daily ( Candle ) Chart Keep an eye on the price as it nears a critical and robust support level at 15. This level has historically served as a reliable bounce point for the price Additionally, daily bullish divergence is worth 👀 Moreover, RSI & MACD OVEREXTENDED to the… [Mr Chart Norris tweeted about CLSK]
[c2_1768] (reddit, QQQ) If you're thinking of buying puts, wait I posted this few days back, and it turned out to be the right call (pun intended). Just a note of caution, now the VIX is back up, put premiums are expensive. I would not suggest buying puts (on SPY /QQQ or any individual stocks) at this time. Wait for a 'bull trap'. Edit: (4/22) This may be the bull trap / dead cat bounce. Watch for a reversal, and then buy the put. >[VIX has dropped. Time to buy puts if you're bearish]( by [u/New-Ad-9629]( in [options](
[c2_1769] (twitter, FUBO) $FUBO Daily Chart If had a choice to name it , would've call it Sexy Tightening Falling Wedge with a Daily bullish divergence ( the Loooooong one ) [Mr Chart Norris tweeted about FUBO]
[c2_1770] (twitter, BTC) Eyes on a reclaim of $64,500. Above there, all bets are off. #Bitcoin [Jelle tweeted about BTC]
[c2_1771] (twitter, SPY) Bulls did the thing. If you didn't swing long I would be careful chasing here > [@ATMSnipes]( > $SPY I’m seeing a 3-3-3-3-3 channel forming into $420 (Cc @MauroBianchi24) > > Hold channel support early into tomorrows session and a rally into $416.45 should follow > > Failure to breakout over $416.45-$417.56 supply and new highs will not happen > > 📈Target $420.47 > > 📉 Target $406.78 [ATMSnipes 🔁 ATMSnipes about SPY]
[c2_1772] (reddit, RA) Many consumer electronics manufacturers 'will go bankrupt' by the end of 2026 thanks to the RAMpocalypse, Phison CEO reportedly says
[c2_1773] (twitter, SLV) My $SLV $18 puts actually went into profit today. Been holding them for a month. > [@rsrighty]( > @CryptoParadyme gm ser can you say metals are dead so they pump for once [CryptoParadyme 🔁 rsrighty about SLV]
[c2_1774] (twitter, PT) $GBS Zacks Small Cap Research initiating with $5 PT! 250%+ upside on this $20m mkt cap biotech! [Ultra_Calls tweeted about GBS]
[c2_1775] (twitter, SPY) $IWM Daily. Bank exposure in #IWM making it one of the weakest US indices, needs to hold 170 for a potential bounce to be in play. 177-180 (and above) will prove to be tough resistance $RTY_F #RTY_F #smallcaps $SPY $QQQ $DIA $SVIB $SBNY $CS $XLF #FOMC [reciknows 🔁 reciknows about QQQ, DIA, SBNY, IWM, SVIB, SMALLCAPS, SPY, CS, RTY_F, XLF, FOMC]
[c2_1776] (twitter, AMD) $AMD Net Options Delta: +1.3M shares Early bulls aggressive. [Theta Warrior tweeted about AMD]
[c2_1777] (twitter, SBUX) $SBUX Sweeper locks in massive loss on wrote puts.. 😬 [Theta Warrior tweeted about SBUX]
[c2_1778] (twitter, PM) #LottoFriday 1/14 High risk, High Reward🚨 $WMT 147c >146 | 144p &lt;144.5 $BA 230c >227 | 222.5p &lt;223 $MU 98c >97 | 95p &lt;95.3 $PM 103c >102.5 | 101p &lt;101.5 Enjoy the extended weekend! [Albert_trades 🔁 SpyHatesMe about WMT, BA, MU, PM]
[c2_1779] (reddit, ETH) Cash to Crypto Transfer in United Kingdom Hi everyone, I’m looking to buy USDT using a bank transfer. Is this possible in the UK? Ideally, I’d prefer dealing with someone reputable—maybe a physical business like a jeweler or shop that handles crypto transactions. If anyone knows trusted people or places in London or Birmingham where I can do this safely, please let me know. Thanks!
[c2_1780] (reddit, IBIT) Degen MSTR Options play Probably not the only degen that's thought of doing this -- Sell the 700Put option on MSTR for Dec 2027 and collect $42K in premium up front Look to double that $42k either through more MSTR/IBIT/BTC before 2027 and hope to God MSTR stays above 500+ in 2027 XD Thoughts? Worst case scenario I get assigned early if MSTR dips big again or crypto is in a bear cycle by then and MSTR is trading under 150 and I'm holding a major bag?
[c2_1781] (reddit, AMD) Took 10 years but I did it I remember buying Nvidia in 2016 and riding it up to $350K gains in 2021 before it had the massive crash and lost 70% of its value...almost gave up and then the AI revolution happened. Im now up $2.8+ million on 14,000 shares.
[c2_1782] (reddit, V) Visa and Banco do Brasil (Brazil State Bank) are testing purchases made by AI Agents (without human assistance)
[c2_1783] (news, AAPL) Apple to sue Corellium over ‘illegal replication’ of iOS “Corellium's true goal is profiting off its blatant infringement…Corellium encourages its users to sell any discovered information on the open market to the highest bidder." Read more here. Read more... More about Tech, Apple, Mashable Video, Copyright Infringement, and Corellium
[c2_1784] (twitter, BAC) $BAC (weekly) H&S neckline backtest rejection followed by channel breakout failure. [kpak82 tweeted about BAC]
[c2_1785] (reddit, UK) UK reopens Teesside CO2 plant with £100m support amid supply fears
[c2_1786] (twitter, OPEN) $OPEN 5C 17 January 2025 ( .08 -> 0.32 ) [Mr Chart Norris tweeted about OPEN]
[c2_1787] (twitter, GRI) RT @TheTradeXchange: $GRI - GRI Bio Announces Publication of Comprehensive Invariant NKT (iNKT) Cell Review in Frontiers in Immunology Demo… [BullishTrade retweeted about GRI]
[c2_1788] (twitter, CVNA) $CVNA Setting up for a disgusting move on the weekly.. Over $58 the real fireworks begin. [Albertrays tweeted about CVNA]
[c2_1789] (reddit, SF) Small Business in Houston - To Hire a Tenant Rep Broker? Hi everyone, I own a small business in houston and the business is expanding and I'd like to start exploring options for a small retail space. I'm looking at 1,200 - 2,000 SF space and for 2-3 years term. My question is, is it worth the time to hire a tenant rep broker and will any take this on?
[c2_1790] (twitter, COIN) $COIN - This is why I ignore the media. Maximum fear at the bottom, euphoria at the top. Baseless FUD claims on staking unregistered securities. And now that dip about to get erased, recent buys both in profit. Earnings next week which I expect to be #bullish. [IncomeSharks tweeted about COIN, BULLISH]
[c2_1791] (twitter, STNG) $STNG Downtrend breakout Weekly momentum indicators look primed Little retest this morning to provide entry Option flow been steady Time for new bull cycle? Bought 7/15 20C [StonkTrump tweeted about STNG]
[c2_1792] (twitter, CI) $CI Dropping nicely again so far today > [@Nebraskangooner]( > $CI > > Worth the R:R to have a go if this is top. [Nebraskangooner 🔁 Nebraskangooner about CI, CI]
[c2_1793] (twitter, NVDA) RT @kholov23: $NVDA Weekly Chart ☕️ Breakout / Rejection / Retest Scenario As long as price holds the retest of previous breakout zone, t… [Mr Chuck Norris retweeted Mr Chuck Norris about NVDA]
[c2_1794] (reddit, AAPL) Anyone went to open an account with Futu moomoo and deposit for the free AAPL stocks? Recently opened an account with Futu Moomoo just to get the free AAPL stock, and surprisingly the process was fast. Deposited S$2700 through FAST and 5 mins later the amount was showed in my Moomoo app. Recommending to someone who wants to claim a free AAPL stock, may follow this article (just my own experience, not sponsored by any brokers or platform)
[c2_1795] (reddit, BTC) Is this the right time to buy BTC and Eth? I've got a small amount, 350usd to put into BTC and Eth and I want to hold for a few months. But i heard BTC will fall alot in next 2-3 weeks, is this the right time to buy? I'm new to this so any advice is appearing.
[c2_1796] (reddit, LUNR) Good buy the dip opportunities: Tempus AI & LUNR? Tempus AI dropped after good earnings, no bad signs, looks oversold and undervalued right now to me. LUNR dropped after a dilution and people panic selling/stop losses being hit? Also looks oversold to me What are your thoughts?
[c2_1797] (twitter, F) RT @alazo92: @drippy2hard This is where I ended up. Thankful for these gains via @drippy2hard &amp; @unusual_whales $F [unusual_whales tweeted about F]
[c2_1798] (twitter, TSLA) 🍋Weekly Watchlist $CCL 1/7 20p attacked - watching for more downside $LCID 1/7 40c catching my attention - sympathy off TSLA delivery news $MRNA flow = bearish, chart leaning bullish. Keep an eye out $SNOW 1/7 365c &amp; 375c 👀 - potential b/o Flow from @unusual_whales 🐳🍋 [ItsSqueezy tweeted about CCL, LCID]
[c2_1799] (twitter, MRNA) Do you want to see which insiders are which company are selling or buying? Check For example, the CEO of $MRNA sold loads on 12/30! The stock dipped 4% that day. [unusual_whales tweeted about MRNA]
