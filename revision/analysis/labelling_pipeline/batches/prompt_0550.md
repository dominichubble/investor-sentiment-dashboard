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
----- ITEMS (50: c2_0550 .. c2_0599) -----
[c2_0550] (twitter, NFLX) $NFLX This ones been bullish lately. Maybe it bounces to fill the gap. [Nebraskangooner tweeted about NFLX]
[c2_0551] (twitter, MSFT) @TrendSpider $MSFT Weekly continues to put in lower highs and downside BRB's (blue arrows). Main target below is still the .618 Fib Retracement level. [ThetaWarrior tweeted about MSFT]
[c2_0552] (reddit, LUNR) Where my LUNR losers at? Bag holding 3200 shares at a -20k loss now. 12 $42 calls for Friday that are -95%. Great company, love the leadership, great future, other bag holder sayings.
[c2_0553] (reddit, DNP) Issue with Snowball Analytics? Hi there: Is anyone else noticing that the dividend amounts as displayed in the Portfolio --> Holdings tab are off? For example DNP's annual dividend (paid monthly) should be $0.78 but Snowball is showing it as $0.72....
[c2_0554] (twitter, ETH) $ETH has been steadily weakening against $BTC, as reflected in the ETH/BTC chart. However, a bottom may be near as the pair approaches two key Fibonacci levels, which also coincide with an unfilled monthly Fair Value Gap (FVG). This could mark a potential turning point for ETH. [BATMAN ⚡ tweeted about BTC, ETH]
[c2_0555] (news, MSCI) Fairpointe Capital LLC Buys Agilent Technologies Inc, Magna International Inc, iShares MSCI ...
[c2_0556] (reddit, SA) Willy Woo now FUDding Bitcoin He just dropped a post on X stating why the quantum threat (with it's attendant potential loss of the Sathoshi 1 million coin and other lost coins) will hang over Bitcoin for the next 5-15 years and that Bitcoin may not make ATH until after that time period Like to know your thoughts For me I read and moved on to stack more SATs. Just another day in the Bitcoin lane
[c2_0557] (reddit, BTC) BTC is in an "accumulation" phase rn. Based on my technical analysis, which I do have a few years of experience in, BTC has just entered an accumulation phase that will probably last for a few months and then we could see the price climb. I know you can't base much off of TA but I've been accurate with BTC in the past. Been stocking up my bag for the next bull run. NFA
[c2_0558] (twitter, TSLA) @3ncrax If we’re playing this game lets not forget the $TSLA rapid puts that called the top yesterday at $287🫖 [ATMSnipes tweeted about TSLA]
[c2_0559] (twitter, FEMY) $FEMY Big buyer just hit. Chart close to monster breakout soon [Ultra_Calls tweeted about FEMY]
[c2_0560] (reddit, MSFT) Up 250k in the last month from 17k initial… INTC AMD 90% of what I played. Robinhood I put $1550 in three weeks ago, Fidelity IRA had $15.8. I have AMD and INTC heavy a good bit of NVDA calls plus BB HIMX MRAM DRAM UMC SMCI RXT. I’ve also been buying spy calls 3-4 days out $10-12 above current when it tanks hard and cashing them out for 100% gains or more too.
[c2_0561] (reddit, NYC) How is it that no one has yet asked Zhoran Mamdani, a leading candidate for mayor of NYC, what his stance is on Bitcoin, and that he is not on the record about it? No google search has yielded anything, and Grok has specifically said he has not made his position known yet. That just leaves me to believe no one — no reporters, and I assume none of the citizens he’s come into contact with while campaigning — has yet asked him. Odd. Have any of you heard if he has made his stance known?
[c2_0562] (twitter, DIA) $IWM Daily. Bank exposure in #IWM making it one of the weakest US indices, needs to hold 170 for a potential bounce to be in play. 177-180 (and above) will prove to be tough resistance $RTY_F #RTY_F #smallcaps $SPY $QQQ $DIA $SVIB $SBNY $CS $XLF #FOMC [reciknows tweeted about QQQ, DIA, SBNY, IWM, SVIB, SMALLCAPS, SPY, CS, RTY_F, XLF, FOMC]
[c2_0563] (twitter, MMM) $MMM Weekly bearish rectangle breaking lower. [ThetaWarrior tweeted about MMM]
[c2_0564] (reddit, IBRX) Lets go WSB! Time is now for the next wave! 👋 let’s go buy IBRX!!!
[c2_0565] (reddit, BTC) DCA or lump sum Just received some money I’ve been owed approximately 2.2k USD; just wondering if I should lump or dca ? Personally thinking lump sum as BTC may have bottomed now ? Or is there more to come ?
[c2_0566] (twitter, AAPL) $AAPL +3.3% move off daily demand on friday [kpak82 tweeted about AAPL]
[c2_0567] (twitter, SPY) $SPY hourly chart looks good for upside as long as $410.88 continues to hold $416.97, $420.57 🎯 [optionsprochick tweeted about SPY]
[c2_0568] (twitter, CMT) Okay, let's play the 'What Does the Data Tell Us?' game... We have: 1 Decreasing Absolute Highs 2 A recent spike in Relative Highs 3 A recent spike in Absolute Lows 4 Decreasing Relative Lows 5 Broad index has underperformed $SPX for ~2 months #2 and #3 are the most intriguing [Ian McMillan, CMT tweeted about SPX]
[c2_0569] (reddit, PDT) the PDT rule going away just means mom dad your brother can buy more 0dte SPX calls this summer. when millions of traders get the green light to yolo spx calls, the market will be difficult to rugpull. agree or disagree?
[c2_0570] (reddit, DVN) $DVN Devon Energy call volume above normal and directionally bullish
[c2_0571] (twitter, COIN) $COIN Coinbase rallying on news that Jim Cramer doesn't like it at the current price [BigCheds tweeted about COIN]
[c2_0572] (reddit, BTC) Macro Forensic Report: Retail Capitulation (7/100 Fear) vs. BTC Whale Accumulation at $60,000
[c2_0573] (twitter, GOLD) Go rt this post and ill post a thread of 24 tickers with a little analysis and general target. I think maybe like 10 or 20 RTs on the main post and ill do it? > [@HackermanAce]( > Thread of $gold related Stonk plays EZ WINS. [Ace quote tweeted Ace about GOLD]
[c2_0574] (reddit, ISRG) ISRG - BUY PUTS ISRG Reports this week. They will have to talk about the new HUGO Medtronic system that is superior to the Da Vinci system. HUGO is approved in the US. Hospitals and surgeons love it.
[c2_0575] (reddit, BTC) 100 BTC gold bar bought for $12,000, 12 years ago. A $10,000,000 HODL ✨
[c2_0576] (twitter, COST) Costco, $COST, hikes dividend by 13.3% to $1.02 quarterly. [ThetaTendies 🔁 unusual_whales about COST]
[c2_0577] (twitter, VS) $ARM still holding above 50sma and trendline. 126 next break out [VS Options Trading tweeted about ARM]
[c2_0578] (twitter, SPY) $SPY Broke the uptrend & formed an inside day today. If $SPY Can't hold the $390 next, it can move towards the $387.46-384.13 & start to fill the gap below. If it can recover the 20ema @ $396 then it should bounce back towards $400 [optionsprochick tweeted about SPY]
[c2_0579] (twitter, ES) $ES / $SPX Strong momentum - another reason I'm not eager to short btc or eth or alts for that matter just yet. Heavy short deltas on btc,eth and alts. [Trader_XO tweeted about ES=F, SPX]
[c2_0580] (twitter, CAT) @MacnBTC the constant fud and downplaying of how big $CAT is making me incredibly more bullish tbh $1b is programmed God willing 🙏 [Mac 🐺 retweeted Unipcs (aka 'Bonk Guy') 🎒 about CAT]
[c2_0581] (twitter, QQQ) $QQQ Retested the post FOMC breakout level at 297, and now tightening in a bullflag. Targeting back into 314. [Albert_trades tweeted about QQQ]
[c2_0582] (twitter, MCD) $MCD Macdonalds weekly chart - Bullish with throwback [BigCheds tweeted about MCD]
[c2_0583] (reddit, NVDA) NVDA covered calls I have 100 shares of nvda. Last year I did a few covered calls. It didn't go too well. I still have the shares. I stopped as I didn't want the tax bill if the shares get called. What if I buy another 30 shares. If my covered call is at a loss I can sell the 30 shares and cover a good part of the loss. Is there something I am missing here. Is there a wash sale etc.
[c2_0584] (reddit, GME) Dear Erma Gerd, plz accept this GME Dubbed Parody GME: I really wanted this song to represent how it feels and should feel to be a GME holder. It's hard to be optimistic at times, so hopefully this reminds us why we are here, and what we are waiting for. Recorded/edited on mixpad with a paid singer from Fivver, and video edited on cyberlink.
[c2_0585] (reddit, V) Buying bitcoin with giftcards?What websites can I use to buy bitcoin via my apple, visa, Mastercard giftcards? Buying bitcoin with giftcards?What websites can I use to buy bitcoin via my apple, visa, Mastercard giftcards
[c2_0586] (news, ABG) Here is Why Growth Investors Should Buy Asbury Automotive (ABG) Now
[c2_0587] (news, ABR) Arbor Realty Trust Announces Details of Special Dividend Payable on January '–, —…–9
[c2_0588] (reddit, INTC) Intel cuts 15% of workforce amid $2.9B Q2 loss, pushes for turnaround
[c2_0589] (twitter, BENF) $ZAPP did a split just about the same time as $BENF. Definitely keep on watch with floats being the same also [BullishTrade tweeted about BENF, ZAPP]
[c2_0590] (twitter, WMT) $WMT halves paid leave for COVID-positive workers, extends hybrid work guidelines. [unusual_whales tweeted about WMT]
[c2_0591] (reddit, GOOGL) Google won't be forced to sell Chrome after judge rules divestment a 'poor fit' in landmark antitrust case Google (GOOG, GOOGL) won't be forced to sell Chrome after a federal district judge ruled divestment a "poor fit" in a landmark antitrust case, but it will have to share data that helped it hold onto its search monopoly. The ruling from District of Columbia judge Amit Mehta sent Google's stock soaring by more than 8% in after-hours trading.
[c2_0592] (twitter, IBM) $IBM Remember this?? Here we go again... [Don't follow Shardi B If You Hate Money tweeted about IBM]
[c2_0593] (twitter, SHOP) $DASH - Upgraded at Evercore ISI, PT $256 $SHOP - Upgraded to Outperform at William Blair $SNOW - Upgraded at Barclays, PT $393 $PYPL - Downgraded at Jefferies, PT $200 from $255 $COST - PT $585 from $575 at Evercore ISI $HD - PT $435 from $445 at Evercore ISI [rachels_44 tweeted about DASH, SHOP, SNOW, PYPL, COST, HD]
[c2_0594] (reddit, PEP) A Million Ain't Enough - Earnings Day Hype Hype Video For Your Viewing Pleasure!! Still hear, still ain't selling shit. No cell, no sell. A million ain't enough. #IT'S PHONE NUMBERS OR GO FUCK YOUR MOTHER!!! Let's celebrate with an ice cold, delicious Pepsi. Pepsi is the beverage of champions.
[c2_0595] (twitter, SNAP) $MSFT $SNAP $META Microsoft Corp's LinkedIn is deploying artificial intelligence to provide career advice and has introduced games to the platform. These initiatives aim to attract users more frequently and stimulate growth in the competitive social media landscape. [Mr Chart Norris tweeted about META, MSFT, SNAP]
[c2_0596] (twitter, ES) $ES Switch on some basic 4hr MA's and things seem pretty clear I'm not a perma bear, there's a lot of r/r to the upside down at these levels, but trying to snipe a bottom for clout is unnecessary, especially if your a swing trader Wait for a little trend flip confirmation [StonkTrump tweeted about ES]
[c2_0597] (twitter, BJDX) @Ultra_Calls Couldn’t resist got some more $BJDX this morning. [Ultra_Calls 🔁 StewartJudd5 about BJDX]
[c2_0598] (reddit, BULL) Made a small bet for next week after liquidating my portfolio I sold off much of my portfolio, atleast I could do Tax loss harvesting. I've bought triple leveraged Oil etf, and Short on S&P. The markets have closed a few mins ago, let's see how the next week or two go.
[c2_0599] (reddit, ASX) ASX 200 LIVE:ASX sinks 2.9pc in $95b wipeout; oil soars 15pc to $US107
