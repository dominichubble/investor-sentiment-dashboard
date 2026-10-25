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
----- ITEMS (50: c2_1200 .. c2_1249) -----
[c2_1200] (twitter, SPY) $SPY $ES_F 30min false breakdown reversal (channel) off daily demand [kpak82 tweeted about ES_F, SPY]
[c2_1201] (twitter, NIO) $NIO recovering the 50ema ahead of earnings. Looks like $20 is the magnet from here [optionsprochick tweeted about NIO]
[c2_1202] (reddit, GME) Reminder GME hit ATL of $0.64 before moon shotting 5000%+ But I’m stupid and eat crayon. How do you guys like yer corndogs?
[c2_1203] (news, AB) Fiber Bragg Grating Market Projected to Display a Robust Growth with a CAGR of –5. ...
[c2_1204] (reddit, BULL) Assignment on Long term holdings I’m just curious if people are tracking the equity loss on contracts that get assigned. Then you end up buying back your position straight out or via CSP at higher price then assigned ??
[c2_1205] (reddit, BTC) They Said You Can't Swap Bitcoin for ADA Without a Bridge — They Were Wrong - Learn Cardano
[c2_1206] (twitter, MSFT) $MSFT & $GOOG both missed? Welp let’s hope it’s priced in [MacroCRG tweeted about MSFT, GOOG]
[c2_1207] (twitter, CAT) $CAT 4H Candle Chart Falling wedge is getting TIGHT / Golden Zone [Mr Chart Norris tweeted about CAT]
[c2_1208] (twitter, CMT) $COF 18-month highs and also attempting to turn the corner on a relative basis... [Ian McMillan, CMT tweeted about COF]
[c2_1209] (twitter, DMA) $SPX approaching DMA 200 where it rejected last two times > [@BigCheds]( > $SPX back over 4000 [BigCheds 🔁 BigCheds about SPX]
[c2_1210] (twitter, SNOW) $SNOW $160 supply is the next level I'm watching to see if it can break. $164.29, $170.09 🎯 if so 👀 [optionsprochick tweeted about SNOW]
[c2_1211] (twitter, SPY) $ES_F +60 points. Locked in profit. Ride runners and $SPY swing [kpak82 tweeted about ES_F, SPY]
[c2_1212] (twitter, SOFI) $SOFI > [@kholov23]( > $SOFI Weekly Candle Chart > > Looks like the price is living within the tightening falling wedge pattern for free, that's why not willing to leave it 😆 > > Breakout is near [Mr Chart Norris quote tweeted Mr Chart Norris about SOFI]
[c2_1213] (reddit, ETH) Just sold my ETH bag. So, be sure to thank me once it breaks the ATH, and keeps going up. According to my years trading crypto, the moment I sell, is the time the ETH will break free and moon. You are welcome.
[c2_1214] (twitter, WAFU) Wants to go para! [@teerackzzzz]( $WAFU.....byeeeeeeee... [Ultra_Calls 🔁 teerackzzzz about WAFU]
[c2_1215] (twitter, HOOD) 🚨BROKEN NEWS🚨 $HOOD closes at an all time low (again). Link: [unusual_whales tweeted about HOOD]
[c2_1216] (reddit, COIN) You know a coin is great when its value drops almost 90% in a few months Source: [
[c2_1217] (reddit, BTC) BTC stays strong above $80K Bitcoin is currently holding steady above 80,000 dollars, what do you all think about that?Does anyone have a different perspective on Bitcoin?
[c2_1218] (twitter, TSLA) Deviated just a little below. > [@Nebraskangooner]( > $TSLA support around $186 > > Watch for bounce opportunity [Nebraskangooner 🔁 Nebraskangooner about TSLA]
[c2_1219] (twitter, NVDA) $NVDA down almost 12% total on the day in after hours. 😱😱 [Nebraskangooner tweeted about NVDA]
[c2_1220] (news, AB) Has AllianceBernstein Holding L.P. (AB) Outpaced Other Finance Stocks This Year?
[c2_1221] (twitter, SPY) Tech actually finding itself as the only sector hitting a YTD high for relative lows, and its highest reading since January 2022... $XLK $SPY [the_chart_life tweeted about SPY, XLK]
[c2_1222] (twitter, DIA) $IWM Daily. #IWM buyers coming in the last few sessions, however need to be mindful of this potential bear flag. Will need to break above 195 to negate potential bearish plot #IWM $RTY_F #RTY_F #smallcaps $SPY $QQQ $DIA [reciknows tweeted about RTY_F, SPY, SMALLCAPS, DIA, IWM, QQQ]
[c2_1223] (reddit, META) GOOGL - 270$ to 10k in 24 hours Bought 1dte 30 contracts of 350$ strike calls expiring today for about 0.09 cents apiece yesterday morning. Sold over the course of the day for about 10k total, with an average gain of 3600%. Had I timed the top perfectly, I could have made over 20k. Not a lot of money in the grand scheme of things, but definitely my most successful trade ever from a purely gain % standpoint.
[c2_1224] (reddit, DTE) Long dated leap options Yesterday during the dip I decided to go for the long term and bought intc 651DTE and aapl 288DTE. Of course given the current situation I think the value is going to fall further. However since it is a leap option and not short dated, should I be selling it to cut the losses (around 30% down) or holding onto it. It's my first time going for leap options and would like to hear some opinion how these options are managed.
[c2_1225] (twitter, CMT) Poor Mike. He finally capitulated. The bear market can begin now. [ReciKnows retweeted CoiledSpringCapital, CMT about STOCKMARKET, SPX]
[c2_1226] (twitter, COIN) Added some $COIN calls on that dip, these are short term vs. the $100 leaps for 2024, need to sell these in the next week or so.... [ShardiB2 tweeted about COIN]
[c2_1227] (reddit, ETH) Please Help!!!! Please say how to turn this loss into profit.. I mean, should I hold and wait.. Or should I sell and invest in others... Please give a tip. I'm a newbie in this
[c2_1228] (reddit, BULL) Need advice on cashing a check Hello everyone. I am currently homeless without a state issued ID. I was recently able to find a short term job opportunity, the problem is I will be paid by check. I have a couple other forms of ID, can anyone give me any advice?
[c2_1229] (reddit, BULL) Stock > warrant volume 03/18/26 Stock wins again with a nice rally right before close. The score is now 109/2 in favor of the stock The warrant doing Jack shiiii but doesn't matter cuz free money baby Todays song of the dayyyyy: CASUALTY By MOTHICA
[c2_1230] (twitter, SNAP) $Snap 📈 Another 10% makes 20% in two days Higher High after breakout. ✔️ [Anbessa100 tweeted about SNAP]
[c2_1231] (twitter, PEP) $PEP (per request) Pretty nasty looking chart - however - sitting right at 99sma on weekly and nice candle today so maybe can put together a small bounce here. [Nebraskangooner tweeted about PEP]
[c2_1232] (twitter, AAPL) $AAPL $170 Demand & 50 ema breakdown Friday. If AAPL is unable to recover the $170.30 on a retest. I will be eyeing the 165p this week. Support: $168.22, $167.01, $165.61 [optionsprochick tweeted about AAPL]
[c2_1233] (reddit, BTC) When should I buy BTC I actually have a portfolio of 500euro and i want to buy BTC but I dont know when, I feel like I need to wait for the cycles to go on and the prices will go down but Idk. Yall got any tips ?
[c2_1234] (twitter, AAPL) $FB barely missed earnings. Omega overreaction because of AAPL privacy shit and boomers angry about metaverse transition. It's underpriced now. Patience will pay in this one. It might chop for 6 months. [ThetaTendies tweeted about FB]
[c2_1235] (twitter, SHOP) $SHOP Taking a modest position here. Nice double bottom with some bullish candles at support. Not sure how much this can push with some resistance levels above but a nice R:R for a small potential push here. [Nebraskangooner tweeted about SHOP]
[c2_1236] (reddit, AI) Retail Euphoria Turns Six-Week Fund Into Record-Busting AI Trade >Brian Emes manages a retail store in Lethbridge, Alberta, a small city on the Canadian prairie about a two-hour drive from the US border. On the morning of May 11, before opening the shop, the 43-year-old launched his brokerage app and bought 55 shares of an exchange-traded fund \[DRAM\] that hadn’t existed until early April. That's the type of article I used to read months before the dotcom bubble deflated.
[c2_1237] (twitter, XOM) $XOM Weekly Chart ⛽️ Falling wedge pattern / price holding above the crucial zone x psych level 100 Double Bottom Formation ? Why Not? September 25 anchored VWAP at 105.66 [Mr Chuck Norris tweeted about XOM]
[c2_1238] (reddit, BULL) First withdrawals… then blew 4 accounts in 3 days Finally managed to withdraw profits for the first time (twice actually), felt amazing… and then reality hit me hard. Blew 4 accounts in the next 3 days. Yeah… humbling as hell. Guess this is just part of the journey. Clearly still got a long way to go with discipline, risk management, and not getting carried away after a win. Trying not to beat myself up too much, just taking it as a lesson and moving forward. Anyone else been through this phase?
[c2_1239] (twitter, NVDA) $NVDA close to ATH's Pretty bullish when this cycle's darling breaks... market going to go nuts IMO [Don't follow Shardi B if you hate Crypto tweeted about NVDA]
[c2_1240] (twitter, ES) $ES - i dont have a clean read on this today. possibly squeeze higher into box. if we swing fail that high will look to short it back down to range lows. above the box we could have a low in [trader1sz tweeted about ES]
[c2_1241] (twitter, NVDA) $NVDA Sweepers today making the bet prior demand remains current demand. [@ThetaWarrior]( $NVDA Bull sweepers active. [BTstocktrading 🔁 ThetaWarrior about NVDA, NVDA]
[c2_1242] (reddit, USO) Is there a point in hedging vs just holding cash? I have about 50% cash right now. I bought some USO at \~105 and holding, but not much (5% of portfolio). Would you do anything in this position beyond holding cash, or maybe selling for more cash? VIX, short, options, ETFs, I'd be open to any option.
[c2_1243] (reddit, GME) What do the green box and line mean in RK charts? Hi everyone, I've made it my mission to check RK charts every day to learn what to look for, what potential indicators are, etc. However, I have to admit I still don't understand the green box and line. Does anyone know more about it? Lets's go GME!
[c2_1244] (twitter, DLTR) $DLTR Back above the 99sma on daily and a lot of open space above. [Nebraskangooner tweeted about DLTR]
[c2_1245] (twitter, AMD) $AMD Weekly Candle Chart ☕️ Bullish flag within the handle of cup and handle formation [Mr Chart Norris tweeted about AMD]
[c2_1246] (reddit, BULL) Am I doing the right thing as a 19 year old I'm planning to put 800 dollars a month into just VT for the long term. Is this a smart choice and are there any underlying risks behind it. Should I be investing into vti and vxus instead?
[c2_1247] (reddit, MSTR) MSTR is currently sitting on a $12.5B unrealized loss, exceeding the largest trading loss in history (Bill Hwang/Archegos) by $2.5B
[c2_1248] (twitter, COIN) Well she definitely sold $COIN and $HOOD into some strength today thank god vs. when they are weak and she nails them > [@ArkkDaily]( > Cathie Wood and Ark Invest's trade activity from today 12/19 [Shardi B Trading #teamshardi quote tweeted Ark Invest Daily about HOOD, COIN]
[c2_1249] (twitter, GOLD) $GOLD Looking for a LTF gap fill Nice R:R for intra-day trade. [Nebraskangooner tweeted about GOLD]
