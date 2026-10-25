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
----- ITEMS (50: c2_1400 .. c2_1449) -----
[c2_1400] (reddit, GME) GameStop Trade Value? GameStop would have to at least include GTA6 for this. No lowball offers from the staff, I know what I have. Anybody else out there ever see one of these before? I picked it up a few years ago from a customer of mine, along with a lot of promotional gear from USA shows when they were doing Mr. Robot and others.
[c2_1401] (twitter, QCOM) Love to see this! > [@TrendSpider]( > $QCOM Beautiful falling wedge breakout, retest and bounce ✅ Price now sits just below the volume POC 👀 > > [HackermanAce 🔁 TrendSpider about QCOM]
[c2_1402] (reddit, BURU) $BURU - UP almost 19% @$0.20 on 123M volume, HOD @$0.204... The program is aligned with Phase 1 revenue objectives in the €5–10 million range and provides visibility toward scaled deployment as production ramps. $BURU - UP almost 19% @$0.20 on 123M volume, HOD @$0.204... The program is aligned with Phase 1 revenue objectives in the €5–10 million range and provides visibility toward scaled deployment as production ramps.
[c2_1403] (reddit, GOOGL) Commerce Secretary Lutnick says he is confident US will secure EU trade deal — but warns 30% tariffs hit August 1 if talks fail
[c2_1404] (reddit, COIN) Trouble with transaction Hi, I really need help with transaction hash: 0x04c132fb63d8cd6a295c27e3916682c8dbce39d280e92b915ee8a6c1a2927296 Sender sent me 11234 usdt, but I didn’t receive my money though stasis is successful…
[c2_1405] (reddit, UWMC) Thoughts on UWMC? United Wholesale Mortgage (UWM) is a wholesale mortgage lender headquartered in Michigan. UWM underwrites loans for independent brokers, and is the largest mortgage lender in the United States. The stock is beaten down to $4 in recent days, and the dividend is $.10 per quarter giving annual yield of about 10%. Is it part of your dividend portfolio already or is it worth considering? Please share your thoughts on why and why not.
[c2_1406] (twitter, MMM) $MMM Backtesting this breakout. Could get a 0.5%+ upside bounce from here for intra-day traders. [Nebraskangooner tweeted about MMM]
[c2_1407] (twitter, CLSK) $CLSK down over 60% from April highs and not looking good asymmetric returns they said.... lmao [Don't follow Shardi B if you hate Money tweeted about CLSK]
[c2_1408] (reddit, ETH) ETH - huge unstaking queue + further profit taking going to tank it soon or do you see further upside for the rest of the year? And what - if anything - do you think this means for alt-season? Like a lot of people I’m trying to figure out if a more “traditional” BTC > ETH > Other alts cycle is happening (and if so, when) or if we are in a new paradigm now. The altcoin index is at 77 today but it doesn’t feel like we’re mid-alt season at all. How are you guys seeing this playing out?
[c2_1409] (twitter, CMT) Update on the Weekly HA candles... $SPY $SPX $QQQ $NDX $IWM $RUT $DIA $DJIA [Ian McMillan, CMT tweeted about SPX, NDX, DJIA, RUT, QQQ, DIA, SPY, IWM]
[c2_1410] (twitter, PAA) $PAA Wish I'd have seen this earlier. 99sma resistance retested. Could be a significant level for rejection and a nice R:R opportunity considering there's not much support below [Nebraskangooner tweeted about PAA]
[c2_1411] (twitter, NFLX) $NFLX low time frames. High sell volume coming in. Possible momentum shift. I'm not going to enter a trade on this but will observe. Should expect sellers to continue momentum here short term. [Nebraskangooner tweeted about NFLX]
[c2_1412] (reddit, AIM) AIM good pre-market House payments were made today with AIM but after the bell rang those circuit breakers were a killer if you were still in it. Anyone else working AIM?
[c2_1413] (twitter, SPY) $SPY dark pool print just rolling in at the highs Possible hod? [StonkTrump tweeted about SPY]
[c2_1414] (twitter, SPY) $SPY If you are struggling to find consistency, remember that there are some things you can do thatll guarantee improvement: Trade less. Focus on less tickers or just one. Increased entry criteria. Multiple timeframe confluence. Try not trading for a day. Block out all noise. [kpak82 tweeted about SPY]
[c2_1415] (twitter, AAPL) Here is a easy couple hundred percenter. $aapl Watch and wait for the 1d cloud entry for a bearish play. Weekly very extended as well, no need to rush into puts but observe for the signal and then short it. Too extended, it will punish late longers sooner or later. [Ace tweeted about AAPL]
[c2_1416] (reddit, COIN) Sudden price spikes on USDC coin pairs Am I the only one getting annoyed (and getting my stop losses unnecessarily triggered) by these sudden price spikes? You dont see this on USDT pairs, I guess due to the higher volume?
[c2_1417] (twitter, CANE) $CANE Recent headlines of refusal to export sugar have helped CANE to buck its cool off period and break out, if this breakout holds, lots of blue sky above What a beautiful trend off the covid lows [StonkTrump tweeted about CANE]
[c2_1418] (twitter, AAPL) Picked up 350 $AAPL calls today I'll buy the FUD and the iWatch news that everyone will forget tomorrow [Shardi B Trading #teamshardi tweeted about AAPL]
[c2_1419] (reddit, BULL) Chinese filing implies DeepSeek valuation of around $52 billion
[c2_1420] (twitter, TSLA) $TSLA 45min Likely setting up for a move higher. A 61% drop from the open would be a sweet sub-w2 entry. Originally labeled 3,4 but 1,2 also counts. [Paul tweeted about TSLA]
[c2_1421] (reddit, GME) All-In GME. Just wanted to share my all-in with you. I've never posted a screenshot of my portfolio before. I have 100 warrants Since I got them, I bought more of the stock. I never thought I'd reach that much stock. In the last two years, I've bought more and more every time I had extra cash. Right now, this money is ALL of my savings. I believe in GME, I believe in Ryan Cohen, and I believe in all of you. Plus, I think Keith Gill is active in this forum, so I wanted him to see this..
[c2_1422] (reddit, RS) $AEHL could squeeze soon .. Zero Borrow , recent RS . Could News and Boom . One to watch . As been accumulated since RS
[c2_1423] (twitter, TDOC) This is just getting stupid. Cathie Wood strikes - or I should say, strikes out - AGAIN!! Adding to her big $TDOC position the day before ANOTHER collapse. No respect for risk and clearly doesn't have the "best in the world" analysts (as she professed) to justify bottom fishing. [hyperbitcoins 🔁 markminervini about TDOC]
[c2_1424] (reddit, NVDA) Biggest all time winners and losers Been actively trading for about 7 years with an all time gain of 200k, obviously thanks mostly to Sofi.
[c2_1425] (twitter, XPEV) In stock land tho $nio paid out and im holding $xpev was filled n sold n I re-entered at .19 Closing out my $gld n $slv plays, will add back later on a pullback. [HackermanAce tweeted about NIO, XPEV, GLD, SLV]
[c2_1426] (reddit, ETH) What advice would you give to yourself as a beginner? Im just dipping my toes into trading and im already in the rabbit hole , there’s so much to learn and so many paths to take so im wondering what advice would you give to yourself of the past to avoid or maybe shorten the path to knowing what you know now ? (Besides buy bitcoin in the 90s hahah) Also any courses books or YT videos that you recommend or that helped you ?
[c2_1427] (reddit, UNH) Target is to get to $1000/week in premium on 50k capital. Have been selling CC on $UNH and CSP on $HOOD recently.
[c2_1428] (twitter, AMD) $AMD Such a beast, I watch it go up everyday and am not in....don't know why I waited....grr... [ShardiB2 tweeted about AMD]
[c2_1429] (twitter, NIO) $nio Wait 6$ area for buy for long term then wait for 60$ - above 100$ to sell in long term [mhdzx1 tweeted about NIO]
[c2_1430] (twitter, VS) $AAPL strong week💥 Fibs above are 215, 222, 226. Support 209, 204, 201. [VS Options Trading tweeted about AAPL]
[c2_1431] (reddit, XRP) BTC and XRP are my choice. What are yours? These are my only two investments in Crypto, I like bitcoin more, however I don't like the idea of losing exposure in What cryptos do you like and which ones are you investing in? If possible, you can explain the project in a brief description. It's all interesting and all your opinions are welcome, including those of the maxis.
[c2_1432] (twitter, MRVL) $MRVL 🟨🟨 Marvell Tech Q1 Adj $0.24 Misses $0.25 Estimate Sales $1.16B Miss $1.18B Estimate [Mr Chart Norris tweeted about MRVL]
[c2_1433] (twitter, AMZN) $AMZN My largest position over the last few months is reaching its conclusion. Officially poking into the top of my take profit zone. [NebraskanCrypto tweeted about AMZN]
[c2_1434] (twitter, WFC) Closed $WFC and $BAC positions at basically break even. They're my only laggards and I'm just cutting the risk on them at nothing. [Nebraskangooner tweeted about WFC, BAC]
[c2_1435] (news, A) Stock Upgrades: Agilent Technologies Shows Rising Relative Strength
[c2_1436] (twitter, RSI) $CVNA Weekly Chart Noting hangman candlestick pattern RSI sitting below the OVERBOUGHT zone Bearish divergence on weekly spotted Double Top Formation? [Mr Chart Norris tweeted about CVNA]
[c2_1437] (twitter, AAPL) Yessiiiirrrrr He's rootttaaatttinnggg > [@reciknows]( > $QQQ Weekly. #QQQ bullish reversal + reclaim of 200sma – expecting a test of the upper portion of the falling wedge 👀 > > $NDX $NQ_F $XLK #SPY #Nasdaq $AAPL $MSFT $SPY [HackermanAce 🔁 reciknows about AAPL, NDX, XLK, NQ=F, MSFT, SPY, QQQ, NASDAQ]
[c2_1438] (reddit, POET) Ain’t nothing to it but to do it baby Took all mu gains from $poet puts and putting it all on this bad boy let’s see how it plays out
[c2_1439] (twitter, DVN) $DVN Bull flag breakout and continuation. [ThetaWarrior tweeted about DVN]
[c2_1440] (reddit, BULL) To all the shorts out there: Numbers Don't Lie! Hey Shorts: Does it offend you that the 2024 domestic box office is also beaten with 5 days remaining in this month? Does it offend you that the U-shaped recovery has completed, and is on an upswing? If it does offend you, Be offended.
[c2_1441] (twitter, SPY) Basically we need to fill today's gap n then push the one we should of before $spy With $dxy on the verge of a bounce, seems without a push downside is more likely. Looking to see if $tsla n $amzn bounce or if we gotta short that too. [HackermanAce tweeted about SPY, TSLA, AMZN, DXY]
[c2_1442] (twitter, SPY) Good morning🌞 The levels are the same until they are not! Watching for chop before continuation higher this week > [@ATMSnipes]( > (2/2) $SPY - 15m > > Key levels to start the week: > > ⬆️ potential above $389.42 > > 🎯392.36, $394.77, $396.79, $398.32, $402.20 > > ⬇️ potential below $389.42 > > 🎯$387.47, $385.09, $382.94, $380.84, $378.87 > > Sell bias > Buy bias > $394.77 > > If we reclaim $386.81 Monday there is an… [ATMSnipes 🔁 ATMSnipes about SPY]
[c2_1443] (reddit, AMZN) Amazon pledges up to $50 billion to expand AI, supercomputing for US government
[c2_1444] (reddit, BULL) First time Boglehead Im moving about $75K in a brokerage account from a managed account with Schwab to Vanguard and looking for input on what I should invest in. I’m assuming VT and chill but looking for long time Boglehead input. Would you recommend all VT? Should I consider anything else like VOO or international exposure?
[c2_1445] (twitter, GOOGL) $GOOGL Triangle setup with strong resistance at 2385. Over 2385, and price can retest 2500 breakdown area. [Albert_trades tweeted about GOOGL]
[c2_1446] (reddit, RENT) Apartment rents weaken further as war and job cuts weigh on demand
[c2_1447] (twitter, DJT) Bought $DJT made by grifter Martin Shkreli? Buy Boden instead, our dev is dead [Dyme retweeted DedCatson about DJT]
[c2_1448] (twitter, RBLX) +$2500 for the day Growth names all summer long, CPI tomorrow let’s see what flows we get. Traded: $RBLX [Albertrays tweeted about RBLX]
[c2_1449] (twitter, ES) @CryptoParadyme Lol it’s almost like you taught me a thing or two about trading dymo :) [@StonkTrump]( $ES Here we are at the bottom of the range Big decicion time Like I said before above 4295 I get concernicus lol Gg so far bullas, but cpi tomorrow so take profits [StonkTrump 🔁 StonkTrump about ES]
