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
----- ITEMS (50: c2_1350 .. c2_1399) -----
[c2_1350] (news, BYD) BYD, Geely see more EV demand as oil prices climb from Iran War
[c2_1351] (twitter, PYPL) $PYPL doublebottom.exe ? [CryptoParadyme tweeted about PYPL]
[c2_1352] (twitter, HOOD) Also had to buy 25k of $hood for now…. [yatesinvesting tweeted about HOOD]
[c2_1353] (twitter, SPY) Well. For the first time in awhile.. I was wrong and I am paying for it. Update on $SPY coming this weekend. [Avery tweeted about SPY]
[c2_1354] (twitter, AAPL) $AAPL Low time frame double bottom breakout [Nebraskangooner tweeted about AAPL]
[c2_1355] (twitter, SMCI) $SMCI 10 points 💥 > [@kholov23]( > $SMCI 1H Candle Chart > > Second round of 550 ? [Mr Chart Norris quote tweeted Mr Chart Norris about SMCI]
[c2_1356] (twitter, DDD) Here is a super simple example Look for price compression and then look for orderflow confluent in the direction you wanna bet. > [@HackermanAce]( > No leaps but some dated flow. Its at the end of price compression and ER risk is off. $DDD [HackermanAce 🔁 HackermanAce about DDD]
[c2_1357] (twitter, WK) Seems about right tbh > [@alphatrends]( > WEEKLY $SPY CHART > > 20 (R), 30 (B), 40 (G) week MAs in steep decline = guilty til proven innocent > > 350ish seems inevitable, it is location of > - 200WK MA (black) > - ⚓️VWAP 2018 low > - 50% retrace Covid low to ATH [StonkTrump 🔁 alphatrends about SPY]
[c2_1358] (twitter, SNOW) $SNOW ❄️ > [@kholov23]( > $SNOW Weekly Candle Chart > > Breakdown / Bounce / Retest Scenario [Mr Chart Norris quote tweeted Mr Chart Norris about SNOW]
[c2_1359] (twitter, SPY) Lets take a poll This was posted 4 months ago. Damn near spot on. Do we leg up from here or dump into an abyss. Leave your thoughts below⤵️ > [@ATMSnipes]( > $ES_F #SPY $SPX > > Today was different. > > This price action is not bullish > > But the bears current onslaught is almost over (for now) > > Look back to Oct. 18’.. even 16’ > > Also Minor W3 @ $380 is a perfect 1.618 of W1 > > There is a gap at $400 > > $460 & $480 both Major resistance now.. [ATMSnipes 🔁 ATMSnipes about ES_F, SPX]
[c2_1360] (twitter, AAPL) $AAPL Couldn't bring myself to short this yesterday. May have to consider today though. Clear invalidation above yesterday's high and at my key resistance [Nebraskangooner tweeted about AAPL]
[c2_1361] (reddit, UK) Europe Warned It Must Break 'Downward Coping Syndrome' to Counter Trump's America
[c2_1362] (twitter, LYFT) $LYFT little bit more 💼 [Mr Chart Norris tweeted about LYFT]
[c2_1363] (twitter, JPM) $NFLX - Upgraded at Daiwa Capital, PT $330 from $226 $TSLA - PT $330 from $350 at MS $ETSY - PT $75 from $83 at MS $SHOP - PT $35 from $36 at Citi $LULU - PT $413 from $464 at JPM $FDX - Initiated at Jefferies, PT $170 $UPS - Initiated at Jefferies, PT $190 [rachels_44 tweeted about UPS, NFLX, TSLA, SHOP, ETSY, LULU, FDX]
[c2_1364] (reddit, AMZN) The Bitcoin "Steel Plate" Stress Test: Will Your Backups Survive a Fire? You secured your generational wealth with flawless cryptography, but trusted the physical backup to a $15 Amazon purchase. Here is why most 'fireproof' wallets melt when you need them most.
[c2_1365] (twitter, BTC) $BTC 65k [Don't follow Shardi B if you hate Money tweeted about BTC]
[c2_1366] (twitter, PINS) $KO - Upgraded at Guggenheim, PT $66 from $61 $PINS - Downgraded at Guggenheim, PT $39 from $46 $RIOT - Initiated at Cantor Fitzgerald, PT $45 $TSLA - PT $1,150 from $1,050 at Daiwa Capital Markets [rachels_44 tweeted about KO, PINS, RIOT]
[c2_1367] (twitter, ES) $SPX $ES_F (30min) what was this setup called? its my bread and butter. Hope u guys caught it. [kpak82 tweeted about ES_F, SPX]
[c2_1368] (reddit, CD) Whats the smartest way to save Federal income tax. i started a new job middle of last year. My employer is not taxing federal income taxes out of my paycheck for some reason. luckily, I paid extra from my prior job so I only owed $2000 this year. i dont want the bill to get too high and me not have the money, so plan to set money aside. what is the best way to set it aside for next year? CD? open a second savings account?
[c2_1369] (reddit, AAOI) How high can $LITE, $COHR,$AAOI, $AXTI go? How often are photonic products replaced within the AI infrastructure? Once the data centers are built, are these companies relevant? $LITE forward P/E- 86 $COHR forward P/E- 36 $AAOI forward P/E- 21 $AXTI forward P/E- 24
[c2_1370] (reddit, BTC) BTC Can Rip Because Everyone Expects Nine Months of Pain The people that have been around for multiple cycles have experienced buying Bitcoin's dips too soon. Thus, they are holding back from unloading their clips in the current correction. What Bitcoin likes to do, is make the greatest number of people regret their buying and selling decisions. it's this reason that Bitcoin is more likely to take off than continue a long slow bleed to $40k.
[c2_1371] (twitter, APT) $APT - #Aptos bullish scnario > [@MacnBTC]( > • Tokenomics scenario 1: All investors are locked on launch. Marketmaker is the only one having tokens together with the community airdrops and incentives. [MacnBTC 🔁 MacnBTC about APT]
[c2_1372] (reddit, DUOL) Duolingo earnings: Prioritizes user growth over monetization, forecasts softer bookings. Plummets -22% in after hours. > Feb 25 (Reuters) - Duolingo (DUOL.O) forecast first-quarter and 2026 bookings below expectations on Thursday as the language-learning app shifts strategy toward faster user growth, a move it said will weigh on bookings growth and profitability this year. Shares fell nearly 20% in extended trading.
[c2_1373] (reddit, TSLA) Lost $20K on TSLA Puts but made $42K on OPEN Calls So I thought Elon taking on TACO was going to result in more pain for TSLA. I was wrong. I took a $20K loss but then shifted my focus to OPEN. Caught the wave when shares were trading at $1.04. Cashed out of my August $1 Calls yesterday but letting my November $1 Calls ride. I don’t know about $82 (which is what Eric Jackson thinks) but $10 seems very reasonable. I almost feel bad for those that are short OPEN right now. Almost.
[c2_1374] (reddit, DTE) Buying LEAPS near all time high? I’m really bullish on Google but they are only a few dollars away from all time highs. I’m looking at 500 DTE, seemingly plenty of time to recover from a dip Is anyone else buying options at this time?
[c2_1375] (twitter, NVDA) $NVDA To illustrate again how s/r tests work [ShardiB2 tweeted about NVDA]
[c2_1376] (reddit, ACHR) Tesla “Announcement” I wasn’t regarded enough to think it was going to be anything to do with ACHR and I wasn’t even expecting a Roadster update… But to tease an announcement for it to just be base trim 3 and Y that they have no excuse to have not been selling for years? That didn’t even make up for the tax credit, let alone get under $35k? Ridiculous. I’m glad this was a minor FOMO instead of a YOLO. Anyone else underwhelmed by Tesla’s “announcement”? Q4 is gonna be rough!
[c2_1377] (reddit, ETH) ETH to fall more ?? I’ve been watching ETH closely and it looks like **$2000–$2020 is a strong resistance zone right now**. If price pushes into that area and shows rejection, I’m considering a **short trade**. Trade idea: Entry: $2000–$2020 Stop Loss: $2055 Targets: • $1950 • $1920 The [Ethereum Price Prediction]( structure on the 1H chart still looks slightly bearish, so a rejection here could bring a quick move down.
[c2_1378] (reddit, TSLA) What Are Your Moves Tomorrow, December 23, 2025 This post contains content not supported on old Reddit. [Click here to view the full post](
[c2_1379] (twitter, CL) $CL - that yearly doesnt look too nice if it closes like this [TraderSZ tweeted about CL]
[c2_1380] (twitter, NET) There goes $NET after invalidating the bear flag. I should've kept a closer eye on this one. Nice set up. Earnings today I think. [Nebraskangooner tweeted about NET]
[c2_1381] (twitter, TSLA) $TSLA 15m Top Goon X sell approaching alert. Tends to bounce then potential higher low and sell back off [Nebraskangooner tweeted about TSLA]
[c2_1382] (twitter, NVDA) $AMD $NVDA two names on my watch list today. $AMD 83c web t0 115% today so far! $NVDA above $234 made a nice move also. Have to watch what's strong when the indices are weak, then take action when the trade comes to you! 😀 > [@optionsprochick]( > Watching $RBLX $TSLA $AAPL $AMD $NVDA today to name a few! 👀 [optionsprochick 🔁 optionsprochick about RBLX, AAPL, NVDA, AMD, TSLA]
[c2_1383] (reddit, BTC) Option Trading success rate I have lost a lot from trading same day, same week, or a couple week expiration day options, even while being part of a trading discord. Is anyone successful with it or is it only buying options that expire 1 year or more. I never bought options that far in the distance before.
[c2_1384] (twitter, SPY) $SPY All time closing highs. Happy 4th of July! 🇺🇸 [Theta Warrior tweeted about SPY]
[c2_1385] (reddit, MSTR) Bitcoin treasury bear market tipped to end as short seller backs off MSTR
[c2_1386] (twitter, TSLA) $OPTI > [@cb_doge]( > "Tesla Optimus Bot is gonna be bigger than the car." > > 一 Elon Musk > > [WIZZ🥷 ( beware scammers ) quote tweeted DogeDesigner about OPTI]
[c2_1387] (twitter, SPY) I've seen that $SPY daily bear flag posted more this weekend than: - Jesus Christ - the Easter bunny - the masters [ATMSnipes tweeted about SPY]
[c2_1388] (reddit, NVDA) Nebius Is Dropping £1.7B on UK NVIDIA AI Infrastructure While Everyone Keeps Calling AI a Bubble Source - [
[c2_1389] (reddit, DJT) Options on NewsMax So we have all seen this before. There’s DJT and, for the crypto space, we’ve seen $TRUMP. Now, there’s NewsMax that after just two days went from $10 to over $230 per share. This puts this new offering at a valuation of almost $30 billion, which is 20% higher than Fox News. There should be a dramatic fall that follows in the not too distant future.
[c2_1390] (reddit, GOLD) How do investors think about physical gold supply businesses versus gold stocks? Hello, I’m trying to understand how investors evaluate gold exposure from a business perspective. When you compare a physical gold supply business with gold mining equities, what are the main factors that matter most to you? I’m thinking about margins, transparency, counterparty risk, regulatory risk, and scalability. I’m trying to understand how serious investors think about the space.
[c2_1391] (twitter, KSCP) $KSCP news out Knightscope Announces All-New K1 Laser Product > [@Bullish_Trades]( > $KSCP Forgotten security penny that has my attention today for a move back to .33/.34, strong break of that and we could see this head towards .40c+ [BullishTrade quote tweeted BullishTrade about KSCP]
[c2_1392] (twitter, AMD) $AMD held weekly 99-100 support zone. Can see follow through if above 105.4, under 99 weak. Resistance📈105.4, 109.2, 112.8, 118.8, 125 Support📉 101.9, 99.4, 95.4, 91.7, 89.1 [rachels_44 tweeted about AMD]
[c2_1393] (twitter, BC) $BC - 4 days old and the new cash grab "Bitcoin" is already down 60% . Weird how everyone shilling this is already on to the next project. If you are going to play this game of ponzi musical chairs at least sell when the parabola of hype breaks. [IncomeSharks tweeted about BC]
[c2_1394] (twitter, SLV) $SLV If this is truly an accumulation breakout then this is a breakout retest and the bounce would likely be explosive from here. [Nebraskangooner tweeted about SLV]
[c2_1395] (twitter, SPY) $SPY No I am not deleting this tweet like a nerd. I play probabilities. Locked in gains, price consolidated, then continued with the next leg up. Nothing wrong with playing the PA and staying green [@ItsSqueezy]( Weakness beginning now on $SPY 15 min with a reversal candle. Now is when I close runners on anything left open from ORBs. Doesn't even matter if it continues to run after this candle. I always lock in gains [ItsSqueezy 🔁 ItsSqueezy about SPY, SPY]
[c2_1396] (twitter, GME) [@CryptoParadyme]( $GME approaching 1 year since the ATH it's killed everyone holding options. Option sellers paradise. would not be surprised if everyone got rinsed before new highs. [CryptoParadyme 🔁 CryptoParadyme about GME]
[c2_1397] (twitter, RIVN) $RIVN Still no bottom confirmation. I initially regretted selling but I can see I made the right choice since I followed my system and still didn't have confirmation. [Nebraskangooner tweeted about RIVN]
[c2_1398] (twitter, GM) $GM General Motors is recalling approximately 7,000 Chevrolet Bolt EVs (model years 2017-2023) due to fire risks associated with improperly repaired seat belt pre-tensioners. In case of a crash, the pre-tensioners' exhaust may ignite carpet fibers, posing a fire hazard. GM will… [Mr Chart Norris tweeted about GM]
[c2_1399] (twitter, MSFT) $MSFT Symmetrical triangle breakout, MACD bullish cross. [ThetaWarrior tweeted about MSFT]
