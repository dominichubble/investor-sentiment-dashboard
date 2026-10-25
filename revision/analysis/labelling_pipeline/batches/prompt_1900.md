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
----- ITEMS (50: c2_1900 .. c2_1949) -----
[c2_1900] (twitter, SPY) RT @ChandlerTrading: $ES_F $SPX $SPY 1H -- Oct6 • This is NOT the only possibility, so stay nimble... • However, if the expanded flat is… [ChandlerTrading 🔁 ChandlerTrading about ES_F, SPX, SPY]
[c2_1901] (reddit, ADBE) Adobe’s (NASDAQ:ADBE) Q2 CY2026 Sales Top Estimates But Stock Drops
[c2_1902] (twitter, BTC) The new wave of ETF demand is underway! Over 200 million dollars in inflows for the ETFs yesterday, nearly all of that through BlackRock's ETF. Don't let the suits take your coins. #Bitcoin [Jelle tweeted about BTC]
[c2_1903] (twitter, OXY) Watching $GOOGL $MSFT $OXY today to name a few! 😀 [optionsprochick tweeted about OXY, GOOGL, MSFT]
[c2_1904] (reddit, RENX) RENX Enterprises Corporation has a huge short interest formerly Safe & Green RENX has a monstrous short interest larger than the float, high percentage of insider ownership and a long enough runway to not get delisted. I think this one checks all the boxes. Let me know!!!!
[c2_1905] (twitter, ES) $SPY $ES_F $SPX I can’t promise doomsday after the $449-$450 but structure is pointing R:R is to the downside. Plenty of Fed bombs can still be dropped this week. Short term demand is $436-$437; There is limited support below there. [@ATMSnipes]( $SPY $ES_F $SPX Euphoria is in the air again; ⚠️Levels to lookout for: $440, $444 Bulls have room to run IF price can claim and confirm the $446 level. [ATMSnipes 🔁 ATMSnipes about SPY, ES_F, SPX, SPY, ES_F, SPX]
[c2_1906] (twitter, SPY) RT @optionflys: $SPX is up five days in a row now. Last time, it went 6 days, which ended up correcting lower by -2.6%. $SPY $QQQ @Optuma… [ReciKnows retweeted about SPX, QQQ, SPY]
[c2_1907] (twitter, CGC) don't think it can't happen to your stocks simply because the "story is just too good"... > [@the_chart_life]( > May get the chance to buy Canopy at the same price I bought it in 2015 (sub $2), back when it was still Tweed and traded on the pink sheets. Wild to think about. But will we get to sell it for $40+ again after that? That's the real question. $CGC $TWMJF [the_chart_life 🔁 the_chart_life about CGC, TWMJF]
[c2_1908] (twitter, SPY) RT @JPATrades: My thanksgiving dinner sponsored by $SPY puts 🥲 [StonkTrump retweeted John Trades MBA about SPY]
[c2_1909] (twitter, TTD) DISCLAIMER: HIGH RISK #LottoFriday Keep bets small. Lock in gains. $CHWY 40C >39.76 | 38P &lt;37.73 $PLTR 12C >12.02 | 11P &lt;11.27 $TSLA 880C > 866.14 | 800P &lt;810.36 $TTD 68C >67.4 | 61P &lt;60.61 Good luck everyone! 🎲 [ThetaWarrior tweeted about CHWY, PLTR, TSLA, TTD]
[c2_1910] (twitter, NFLX) $NFLX +10% after hours *NETFLIX 2Q STREAMING PAID NET CHANGE -970,000, EST. -2M Only lost 1m subscribers lmfao LFG [MacroCRG tweeted about NFLX]
[c2_1911] (reddit, AMZN) Amazon signs $11.57 billion deal for satellite firm Globalstar to challenge Musk's Starlink
[c2_1912] (news, AB) AllianceBernstein Holding L.P. 2019 Q1 - Results - Earnings Call Slides
[c2_1913] (twitter, AAPL) $AAPL interesting spot, retested the trend break on the hourly. Would keep an eye on it > $ 145 👀 [optionsprochick tweeted about AAPL]
[c2_1914] (twitter, CF) $CF Looking like yesterday's buy back was just noise. Testing yesterdays lows and unlikely to hold. If this breaks, lot of room to fall [Nebraskangooner tweeted about CF]
[c2_1915] (reddit, EMA) Interesting week on Gold and Oil Oil had a great day on Friday and gold had decent day too. The weekly close on both the instruments are quite similar however elevated oil prices could hurt gold. We did just retest the 200 EMA on the daily timeframe and we got a strong bounce off of that but this could just be a pull back for the sellers to continue selling next week.
[c2_1916] (twitter, SPY) $spy red line focus. Cloud structure still strong n intacts, as long as it hold. Pop this higher! [HackermanAce tweeted about SPY]
[c2_1917] (twitter, NVDA) $AMD This is starting to look really good, with $NVDA hitting new highs daily this is starting to get some traction.. swapped my $NIO for this on open [Don't follow Shardi B if you hate trading tweeted about AMD, NVDA, NIO]
[c2_1918] (twitter, MP) $MP 👀 on 15.50 right ? > [@kholov23]( > $MP Weekly Chart > > Inverse head and shoulders - pullback to 15.50 would be your friend [Mr Chart Norris quote tweeted Mr Chart Norris about MP]
[c2_1919] (reddit, ETH) The rise of Ethereum We are currently in a period where the momentum has shifted from BTC to ETH. Wouldn’t it be much more profitable to keep all our money only in ETH rather than in a coin basket (ETH, XAIBT, LDO, SUI, ONDO)? After all, ETH is the second most solid coin, so there’s no risk of it crashing suddenly, and we already have our stop-losses in place. *This is not financial advice.*
[c2_1920] (reddit, DTE) SPY QuantSignals V3 0DTE 2026-01-14 SPY QuantSignals V3 0DTE 2026-01-14 📊 Premium Signal - Full analysis available to subscribers only. Click to learn more! 🔗 [ 🔥 Unlock full content: [
[c2_1921] (reddit, GOLD) March 17, 2026 - De-Dollarization: US Pays China in Gold as Treasury Hits Record Buyback
[c2_1922] (reddit, ETH) ETH in Jaxx, where to transfer? I inherited some ETH and have never had crypto before. The ETH is currently in a Jaxx wallet (?) that has since been discontinued. I am looking for advice on which wallet to transfer the ETH to that will also allow me to sell it at some point. I would be very grateful for any help!
[c2_1923] (reddit, GOLD) What’s Actually Working in Crude Oil Trading Right Now? The crude oil market has been moving aggressively lately, and I’m trying to figure out what strategies are still effective. Are you guys finding success with trend-following, or is mean reversion working better right now? Also, do you rely more on fundamentals (like inventory reports) or pure price action?
[c2_1924] (reddit, HOOD) Exxon and Chevron Follow-up Hi guys! Took everyone’s advice and bought shares of xom and cvx. I just got like 5 notifications from Robinhood saying both keep hitting their 52 week highs. Do I sell tn or hold still? Lmk!! Literally when he bombed Venezuela the other night my shares were up like 10 and 13 dollars respectively. Now they keep fluctuating between returns of like $40-$68. I just don’t want to wait too long.
[c2_1925] (twitter, SPY) $SPY now that the complicated secret strategy is revealed, it might not be long before everyone finds out and starts to buy even more leading us to 600 target in 2 months. [kpak tweeted about SPY]
[c2_1926] (twitter, SGML) $SGML Also a solid Brazil play as $EWZ and $BRLUSD both continue to perk up... [the_chart_life tweeted about SGML, EWZ, BRLUSD]
[c2_1927] (twitter, AAPL) Intresting retest spot for $aapl here [StonkTrump tweeted about AAPL]
[c2_1928] (news, ABR) Buy 5 High-Yielding Stocks Below $15 Amid Extreme Volatility
[c2_1929] (twitter, SNY) 7-year highs and another attempt to push through ~$55... $SNY [the_chart_life tweeted about SNY]
[c2_1930] (reddit, AAPL) AAPL QuantSignals V3 Earnings 2026-01-13 AAPL QuantSignals V3 Earnings 2026-01-13 📊 Premium Signal - Full analysis available to subscribers only. Click to learn more! 🔗 [ 🔥 Unlock full content: [
[c2_1931] (twitter, NET) I really think I need to add more... just don't know what to sell! $NET [Shardi B Trading #teamshardi tweeted about NET]
[c2_1932] (twitter, EOD) $LVS Dated calls coming through EOD at monthly support [StonkTrump tweeted about LVS]
[c2_1933] (twitter, CC) $STX (stock) Nice cup and handle breakout here. CC: @BigCheds [Nebraskangooner tweeted about STX]
[c2_1934] (reddit, MARA) Empty mempool = bad for bitcoin? I've never seen mempool this empty, nearly nobody uses the main chain. How can this not be bad for bitcoin? It seems like only few people are using the network. I get it, it's a great time for UTXO management, but that's the only advantage right now. What are your opinions why there is nearly no usage of the network (main chain)?
[c2_1935] (reddit, BULL) Apple seeks to buy memory chips from blacklisted Chinese company
[c2_1936] (reddit, SPY) Dumb illogical hedge worked out As a hedge for my armageddon puts, I bought 10 SPY 673 calls yesterday morning for $27 each on the off chance we wouldn't all die. Sold 9 of them at market open for $468 each. Thought about letting the 10th ride a few hours to see if the market would keep pumping, but changed my mind 3 minutes later and sold at $372. Profit of $4,314. My armageddon puts: I lost $3,569 on 4/8 652, 650, and 625 puts bought on 4/6 and 4/7 🫡
[c2_1937] (reddit, META) Realized Gains - Need Advice I have realized gains YTD but bought META Leap options at strike 670 and 680 expiring Dec 2026 and now incurring large unrealized losses. For tax purposes, should I sell these LEAP options at a loss before year end and buy the same strike prices expiring Jan 2027? To my understanding, this won't be a wash and it will allow me to take a realized loss for this year.
[c2_1938] (news, ABR) The Rose 93 Stock July Portfolio Update: Delightful Spicy Dividends Up 60.2% From 2017
[c2_1939] (twitter, SB) Caught the outperformer $SB again for round 2! 📈✔️ I'm flipping memes too and it's profitable these days, but I love to see hard working teams getting rewarded. Pioneering work on Solana, love to see it. > [@StepFinance_]( > Top onchain movers of today via > > @Solbankfi the only non meme coin? [ANBESSA quote tweeted Step Finance | Solana's Portfolio Dashboard about SB]
[c2_1940] (reddit, NVDA) Nvidia sideways from 195. F all puts and calls Lol that price action to keep Nvidia from breaking out is crazy, even after record earnings beat. Odds are soooo stacked against commoners in this casino lol. MMs gods strike again.
[c2_1941] (reddit, COIN) Tomorrow's play?! Just wanna know what people are going to do. Calls, puts, straddles, 0DTES, anything! I just really wanna know what people are going to do tomorrow the nanosecond the market opens!
[c2_1942] (twitter, DJT) Like I was saying, best short ever $DJT is a doomed symbol > [@TheStalwart]( > Trump's media company has now plunged 66% since its peak in late March. > > (Now need to go back and find all the savvy people on fintwit who said it was insane to short a stock like this) [Shardi B Trading #teamshardi quote tweeted Joe Weisenthal about DJT]
[c2_1943] (twitter, QQQ) $QQQ Daily. #QQQ inside day after hitting resistance from the broadening formation. Hot jobs report may bring near-term downside, but thinking this is just pullback to re-test the double bottom breakout before a move higher $NDX $NQ_F $XLK #Nasdaq $AAPL $MSFT $AMZN $SPY [reciknows 🔁 reciknows about XLK, AAPL, MSFT, AMZN, NASDAQ, NDX, NQ=F, SPY, QQQ]
[c2_1944] (reddit, NVDA) Playing with market gods. Intentional SPX puts vol. 3 So far every time I post my hedge tactical play, market gods fall for it, inverse and calls go shrekking . DW guys I am a bull undercover, let’s hope our calls go brrrrrr until at least NVDA earnings week Positions: SPX 9 x 6700P exp March 31th
[c2_1945] (twitter, PFE) $PFE Daily flag and it ain’t bullish Once this flag breaks down I expect plenty of downside to follow on this pig 🎯 $43.70, $39 > [@eyesonyoupeeps]( > @ATMSnipes @tfinancialcloud $PFE [ATMSnipes 🔁 eyesonyoupeeps about PFE, PFE]
[c2_1946] (reddit, ETH) Is Ethereum worth investing in today? Holding ETH since $1.3k avg entry. Love the tech, trust the future. But at $3.9k… **Is it still worth?** I also keep L1 and L2 coins in my portfolio, moreover, I even transferred some of my ETH to L1/L2... but still, every month when I replenish my wallet, I think about whether I should buy again some more Eth.
[c2_1947] (twitter, AREB) $AREB $2 break will flood it with buyers [BullishTrade tweeted about AREB]
[c2_1948] (twitter, AAPL) $AAPL Other companies should take note: Talk about AI and your company's stock can also pump to new ATH [Nebraskangooner tweeted about AAPL]
[c2_1949] (twitter, QQQ) RT @Marlin_Capital: There is a growing divide between optimism on Main Street and optimism on Wall Street. Despite being closely aligned… [StonkTrump retweeted about IWM, SPX, QQQ]
