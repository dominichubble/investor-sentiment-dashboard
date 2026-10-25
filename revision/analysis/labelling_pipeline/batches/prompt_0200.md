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
----- ITEMS (50: c2_0200 .. c2_0249) -----
[c2_0200] (twitter, NVDA) $NVDA #NVDA new year, new trend? 👀 The first weekly high gained since the downtrend started in March ‘22. Expect profit taking from long swings throughout this range. [_bread911 tweeted about NVDA]
[c2_0201] (twitter, NTR) $NTR Short looking great. Weekly bearish engulfing candle is in the works > [@Nebraskangooner]( > $NTR > > Similar to CF. Got filled above $114 [Nebraskangooner 🔁 Nebraskangooner about NTR, NTR]
[c2_0202] (reddit, META) Meta sell off is overdone I am taking a bullish call option expiring 11/14 expecting meta to be at least 690$ in 2 weeks. Results were outstanding, Z is ahead of the curve spending that capex for AI.
[c2_0203] (twitter, AMZN) $AMZN Nice. 😎 If Amazon trades 3300 tomorrow these contracts will be worth $525+ each (purchased for $120). $7.6M gain overnight. [ThetaWarrior tweeted about AMZN]
[c2_0204] (twitter, PT) $DIS - Downgrade at Guggenheim PT $165 $AAPL -$200 from $175 at Piper,$210 from $165 at Loop Capital $TSLA - PT $1045 from $1005 at RBC $SQ - PT $225 from $300 at Keybanc $TWLO - PT $283 from $392 at Keybanc $ZM - PT $253 from $344 at Keybanc $BABA - PT $180 from $230 at Needham [rachels_44 tweeted about DIS, AAPL, TSLA, SQ, TWLO, ZM, BABA]
[c2_0205] (reddit, AVAV) High Quality Small Caps? (20M) The vast majority of my net worth is in wide moat quality large caps. I am trying to build small-mid caps portfolio with $6K Here are some picks: \- Sterling Infrastructure (STRL) \- SoFi Technologies (SOFI) \- AeroVironment (AVAV) \- Robinhood (HOOD) \- Crowdstrike (CRWD) \- ARM Holdings (ARM - I love trajectory but so freaking expensive)
[c2_0206] (twitter, PENN) $PENN Truist Securities analyst Barry Jonas upgrades PENN Entertainment from Hold to Buy and maintains the price target at $23 [Mr Chart Norris tweeted about PENN]
[c2_0207] (reddit, MSTR) Has anyone actually recovered from a massive drawdown? Serious question has anyone here ever been down 70–90% on a position and still managed to come back to green? Not talking about quick flips. I mean those nasty bags that sit in your account for months or even years after a huge drop… and then somehow crawl back because the company finally fixes itself or the sector rotates. Just curious if anyone has actually lived through it, or if once you're down that bad it's basically game over.
[c2_0208] (twitter, UNH) $UNH with the bullish flip of the 20/50 emas [StonkTrump tweeted about UNH]
[c2_0209] (twitter, TSLA) 🚨NEW #BREAKFASTRADES MORNING SCALP VIDEO: In this video I have a PERFECTLY executed pattern trade on a $TSLA short Entry trigger: Pattern Breakdown Confirmation Total position size: $125,000 Trade duration: 3 minutes Profit: $625 Watch it here 👇 [Nebraskangooner 🔁 Nebraskangooner about BREAKFASTRADES, TSLA]
[c2_0210] (twitter, SNAP) $SNAP You know my feelings on this one with a Tik-Tok ban looming.. a ton of active users, I think this benefits from the potential ban but is also a takeover target at these levels [ShardiB2 tweeted about SNAP]
[c2_0211] (twitter, NVDA) $NVDA (daily) Spinning top at .618fib and potential bear flag channel resistance. [kpak tweeted about NVDA]
[c2_0212] (reddit, GOLD) Nikkei 225 declines more than 6% as oil breaks $100 a barrel
[c2_0213] (twitter, SNAP) $SNAP Daily Chart Bullish flag - above 15.20 -> welcome 16 [Mr Chart Norris tweeted about SNAP]
[c2_0214] (reddit, TSLA) SpaceX to buy Cursor AI coding agent operator Anysphere for $60 billion &#x200B; SpaceX already making moves. Know theyve talked about this before &#x200B; Calls?
[c2_0215] (twitter, APLM) $APLM .26’s printing already, let’s see .30c by close 🫡 > [@Bullish_Trades]( > $APLM added some .22’s with $APVO profits > > Chart range very similar [BullishTrade quote tweeted BullishTrade about APVO, APLM]
[c2_0216] (reddit, META) Thanks Tim Apple 🫡 After losing 15k on NOW calls last week, and watching it climb back 20% in a few days, I went full tilt and spun the wheel one more time on AAPL puts and AMZN calls. 2/2 was in for roughly 10k each.
[c2_0217] (twitter, MA) $JPM JP Morgan MA 200 daily underside tag. Same spot it rejected back in February [BigCheds tweeted about JPM]
[c2_0218] (reddit, ATO) My friend has drawn out their super using SMSF, made a fake agreement with another guy as though he's lending money to him on an interest, but using the money for trips and cars. Why don't the ATO fine them? Title
[c2_0219] (twitter, SPY) Well there’s $436 Relief rally into next week.. should be shallow and choppy. $NVDA ER or Jackson hole will send us into $430 > [@ATMSnipes]( > $SPY $SPX - From this mornings update > > The downside target remains the same with that being $430 > > We are just moving there far faster than I forecasted initially. > > $436 - $436.70 likely by the end of the week > > $430 by the EOM as we are still targeting weekly demand. > > This is… [Avery quote tweeted about NVDA, SPX, SPY]
[c2_0220] (reddit, LNG) Iran Strike to Cut 17% of Qatari LNG for 3-5 Years Iran’s latest attack on Qatar damaged facilities that produce about 17% of its liquefied natural gas export capacity, Reuters reported, citing QatarEnergy’s chief executive officer. Repairs will take three to five years, CEO Saad al-Kaabi said, according to the news service.
[c2_0221] (twitter, SLV) $SLV Fly little birdy, flyyy 🕊️ [Nebraskangooner tweeted about SLV]
[c2_0222] (reddit, MSTR) Please Explain the Appeal of BTC Treasury Companies to Me Like I'm 12. I'm talking about firms like MSTR. Why would I pay 2x for BTC when I can just buy it myself?? I'm confused.
[c2_0223] (twitter, ES) $ES > [@trader1sz]( > $ES - arrow is where im looking for trigger IF it follows plan [trader1sz 🔁 trader1sz about ES, ES]
[c2_0224] (twitter, IMPP) $IMPP 50%+ runner to start the day followed by a small loss trade 2.19 to 3.45 [Ultra_Calls 🔁 Ultra_Calls about IMPP]
[c2_0225] (reddit, AI) Prompting AI for the trading strategy Does anyone know how to make Claude Opus give correct answers to the questions like - "Find stocks with 20-day historical volatility in the bottom 10th percentile, current price within 2% of the Year-to-Date Volume Profile Point of Control (PoC), where Net GEX has flipped from negative to positive in the last 48 hours."?
[c2_0226] (twitter, DRCT) $DRCT Another 100% for the books $3.65 highs [Ultra_Calls tweeted about DRCT]
[c2_0227] (reddit, BTC) BTC and alt season is not at the top yet I see so many posts saying should i sell and take gains is it too late? Nobody knows obviously. But considering the instituional adaption and the new laws for crypto etc. We are still early.
[c2_0228] (reddit, AMZN) Cooked this week? What are yalls thoughts for Monday and this week? I picked these up this past week during what I thought was a good dip. Im a couple months from exp but the more Im reading about the Israel and Iran ordeal, Im fearing this could be a loss
[c2_0229] (twitter, XOM) $XOM makes sense now ? > [@kholov23]( > $XOM Daily Chart > > Falling wedge within descending broadening wedge pattern > > RSI 3 points above the OVERSOLD zone > > REVERSAL is near [Mr Chart Norris quote tweeted Mr Chart Norris about XOM]
[c2_0230] (twitter, TW) $VVIX TW Pivot confirmed buy signal on the daily. VVIX = Expected Volatility of the $VIX [ThetaWarrior tweeted about VVIX, VIX]
[c2_0231] (twitter, ES) 3953 needs to hold as support now or we could round trip quite a lot of this move > [@StonkTrump]( > $ES > > 3953 is the level to hold above to confirm bulla, and beyond that the vwap from past low that previously acted as support > > Fridays close was pretty bullish, r/r likely to the long side here. But I would look for pullbacks if you wanna fresh entry. Like a gap fill for example [StonkTrump 🔁 StonkTrump about ES]
[c2_0232] (twitter, AMZN) $AMZN rallied over 20% in 3 days. Doji at major resistance after huge gap up. [kpak82 tweeted about AMZN]
[c2_0233] (twitter, NOV) $NOV Another oil ticker relatively un pumped Just now breaking out with a covid crash gap above to fill It's been toying with this breakout for over a year [StonkTrump tweeted about NOV]
[c2_0234] (reddit, ETH) My wsb inspired strategy of gambling and praying I get lucky Not really that wealthy so I’m doing what I can but so far I’ve gotten lucky. I also have a Roth IRA entirely composed of mutual funds and ETFs that has performed worse in the same timespan. (Thank you to whoever in here told me to buy palantir at $23 a while ago.)
[c2_0235] (reddit, RDDT) Thoughts on the RDDT stock? I'm curious to know what other people think about it. I bought almost ten shares of it when it first came out in 2024 at 34/share, and now it's up to mid-250s. Do you think it has potential for substantial long-term growth? Or will it be a victim of the eventual AI bubble burst? (I do think it is coming)
[c2_0236] (twitter, CMT) 1 positive Sector on the week so far. And to no ones surprise, it's Tech. $XLK $SPY $QQQ [Ian McMillan, CMT tweeted about XLK, SPY, QQQ]
[c2_0237] (reddit, LASE) SOLID $LASE gains - Sold half at end of day, swinging the remaining 50%. 100%+ gains. Made half the position free. Let’s see where we go tomorrow…
[c2_0238] (reddit, COIN) My best day in market thus far :) [90K gain] I was trading $COIN options the whole time. Calls and puts, let’s just say I’m SO proud of myself for not having paper hand because at one point I was down 110k!!! I am done for the week, possibly the month going enjoy my money :) Win for the good guys!
[c2_0239] (news, AAP) Advance Auto Parts' (AAP) CEO Tom Greco on Q1 2018 Results - Earnings Call Transcript
[c2_0240] (twitter, SI) [@saylor]( MacroStrategy, a subsidiary of @MicroStrategy, has closed a $205 million bitcoin-collateralized loan with Silvergate Bank to purchase #bitcoin. $MSTR $SI [damskotrades 🔁 saylor about MSTR, SI]
[c2_0241] (reddit, AD) QuantSupport: a pricing and risk analytics library written in Rust Hi guys, I'm sharing a project I've been building for a while: QuantSupport is a pricing and risk analytics library that aims to take advantage of all nice features of Rust. It features AD for sensitivities and many different products that can be priced and analyzed with different pricers. If anyone is interested or has any feedback is highly appreciated!
[c2_0242] (twitter, WFC) $WFC Looking at something like this. Although earnings next Friday to look out for. [Nebraskangooner tweeted about WFC]
[c2_0243] (twitter, DDOG) $DDOG Cup base setup-post earnings. Targeting 130, on a break over 116. [Albert_trades tweeted about DDOG]
[c2_0244] (twitter, FTNT) #earnings after the close $AMZN $F $SNAP $PINS $ATVI $U $FTNT $SWKS $GPRO $CLX $BILL $SKX $MCHP $DECK $BYD $COLM $PRU $WWE $SYNA $ATEX $NLOK $MIME $NOV $BECN $SKYW $BRBR $QVCO $DLB $UTI $USAK $NFG $AVTR $WERN $POST $LESL $PCTY $NWSA $OTEX $BHE $MWA [eWhispers tweeted about AMZN, F, SNAP, PINS, ATVI, U, FTNT, SWKS, GPRO, CLX, BILL, SKX, MCHP, DECK, BYD, COLM, PRU, WWE, SYNA, ATEX, NLOK, MIME, NOV, BECN, SKYW, BRBR, QVCO, DLB, UTI, USAK, NFG, AVTR, WERN, POST, LESL, PCTY, NWSA, OTEX, BHE, MWA]
[c2_0245] (reddit, TACO) Stocks are down a little When "investing" has come to the point where everyone's main bull thesis is that a politician is going to "TACO", we have reached a point where the only way to go is down. Theres a reason no legendary investor gives the following advice: "buy stocks which require political decisions made in your favor , then pray" So if you have bought stocks which require a political decision to go up, and then decided to pray, its probably a BAD IDEA.
[c2_0246] (twitter, QQQ) $QQQ #QQQ Weekly Chart - bullish engulfing for week - inside '23 Value Area [reciknows 🔁 cfromhertz about QQQ]
[c2_0247] (reddit, GOLD) U.S., Israel preparing to bombard energy-related targets in Iran
[c2_0248] (twitter, SPY) I'm going to make it simple We're likely heading to $440, $435 then $430 for W4 on $SPY If we fail to make new highs into $470 (I think we do) it's chalked.. see you at $410 - $415 For now enjoy the opportunity to short & if anyone is screaming the world is ending; ignore it [Avery tweeted about SPY]
[c2_0249] (twitter, GLD) $GLD ready? 👀 [@BTstocktrading]( $GLD close to a long-term breakout [BTstocktrading 🔁 BTstocktrading about GLD, GLD]
