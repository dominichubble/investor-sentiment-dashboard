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
----- ITEMS (50: c2_1050 .. c2_1099) -----
[c2_1050] (twitter, APT) but also pump my $APT bags lol [StonkTrump tweeted about APT]
[c2_1051] (twitter, ATAI) $ATAI Deviation and go. Almost thought about getting back in. Would consider buying dips if it maintains above the support level then it's bullish bias [Nebraskangooner tweeted about ATAI]
[c2_1052] (reddit, AP) Blindly fear dilution? You miss crucial FACTS. In Jan ‘21 AMC issued shares, AMC share price soared. May/June ‘21 more AMC shares, again our stock soared. Stopped selling shares July ‘21, $AMC started a long fall. APEs announced last week, $AMC up again.
[c2_1053] (twitter, LYFT) $LYFT pushing up out of this base [Nebraskangooner tweeted about LYFT]
[c2_1054] (twitter, XPOF) $XPOF Xponential Fitness..."Operates as a boutique fitness franchisor in the United States and internationally". Recent IPO with a base/breakout [the_chart_life tweeted about XPOF]
[c2_1055] (reddit, BABA) How does overseas student loan repayment work with payments being taken out of your after-tax income, rather than gross income? Australia's repayment threshold is barely above the UKs which is hilariously wrong given the cost of living difference. I wonder if this is done to effectively "punish" people who leave the UK Edit: Sorry, I misunderstood. I just learnt that student loan repayments even in the UK are paid with after-tax income. So I guess the question is invalid
[c2_1056] (reddit, BULL) 900 shares SQQQ $72k YOLO I figure if I liquidated everything (which I have except for some mutual funds I promised to keep) I should look for returns on the ride down. This is what I came up with so I don’t get jerked around by a specific stock. My price target is $110 or if I suddenly retrace 50% of max gains (currently $7k). Or if peace looks viable, but that will probably trigger the second condition anyway. Good luck out there.
[c2_1057] (twitter, LCID) $LCID Weekly falling wedge breakout > [@kholov23]( > $LCID Weekly Chart Update > > Nothing new, price testing the upper boundary of falling wedge pattern > > Bull Case: Price above 4.50 [Mr Chuck Norris quote tweeted Mr Chuck Norris about LCID]
[c2_1058] (reddit, ETH) bitmine just revealed they hold 1.86 million eth. that's 1.5% of ethereum's entire supply. the institutional takeover is happening
[c2_1059] (twitter, AMZN) $AMZN down $10 [Don't follow Shardi B if you hate Money tweeted about AMZN]
[c2_1060] (reddit, AMZN) Is Jade Wallet available on Amazon US? just saw it and want to buy it but there is not any official announcement on Blockstream website
[c2_1061] (news, ABG) Asbury Automotive Group, Inc. (ABG) CEO David Hult on Q3 2019 Results - Earnings Call Transcript
[c2_1062] (reddit, RENT) $MU 20k -> 75k (so far) in 4 trading days Bought the perfect dip in March 12th for March 20th calls 435 TP x 20 (\~21k cost) So far have sold the peaks and still have 9 x 435C for March 20th. House money locked with +7k profit. The remaining 9 calls will propably open around 42k - 50k market value. Would you sell or let it rip/die after market closure?
[c2_1063] (reddit, BTC) Bitcoin falls under $101K: Analysts say BTC is ‘underpriced’ based on fundamentals
[c2_1064] (twitter, META) $META impressive strength today. All about this $125 level now [optionsprochick tweeted about META]
[c2_1065] (news, AA) Alumina oversupply concerns outweigh short-term gain for Alcoa, analyst says
[c2_1066] (reddit, SSNLF) why is ewy only up 2%? bought this to get exposure to sk-hynix and samsung...they are both up 10% + today and make up 40+% of the fund. yet it is only up 2%...i checked kia (the 3rd largest holding) and it is up 6%. did other stocks crash in Korea, or do I not understand how this fund works?
[c2_1067] (twitter, ES) No changes to this so far, choppy price action. I'm on the sidelines waiting for setups to develop on lower timeframes. $ES $SPX [@rbc_trading]( $ES 8H--(Feb01) W4 counts losing likelihood, but bulls may be celebrating early. I see this move as corrective still Bearish divergence @ the .618 + 1.272 - not convinced the low is in. Looking for a rejection, possibly more complexity to follow I'm not short, just waiting [rbc_trading 🔁 rbc_trading about ES, SPX, ES]
[c2_1068] (twitter, CING) $CING We got a $9.00 target recently and a $8.50 target this morning. You already know the rally is coming soon on our swing. We get minimum $5 here [Ultra_Calls tweeted about CING]
[c2_1069] (reddit, AMD) AMD announces $6 billion buyback; shares climb 8% AMD said on Wednesday that its board of directors approved $6 billion in share buybacks. The stock climbed 8%. The authorization is in addition to $4 billion in existing approved share repurchases, the company said. “Our expanded share repurchase program reflects the Board’s confidence in AMD’s strategic direction, growth prospects, and ability to consistently generate strong free cash flow,” AMD CEO Lisa Su said in a statement. Source: [
[c2_1070] (twitter, MIN) $CLSK 10 MIN Chart Price is testing the upper boundary of falling wedge Do we see a breakout ? [Mr Chart Norris tweeted about CLSK]
[c2_1071] (twitter, TSLA) $TSLA (daily) [@kpak82]( $TSLA (daily) after 3 rejections at 20ema, now false breakout reversal with close below 5/9ema (Yes, that's bearish) [kpak82 🔁 kpak82 about TSLA, TSLA]
[c2_1072] (reddit, DHT) Iran's revolutionary guards tell ships passage through Strait of Hormuz 'not allowed', EU naval mission official says DHT calls will print. godspeed
[c2_1073] (twitter, JMIA) $JMIA 10 matter of time 🪄 [Mr Chart Norris tweeted about JMIA]
[c2_1074] (twitter, CMT) $XLV and $XLF slipping back below those late 2021/ early 2022 highs... [Ian McMillan, CMT tweeted about XLF, XLV]
[c2_1075] (twitter, NKE) $NKE relative strength post earnings. Next break out is 119.18 📈 [rachels_44 tweeted about NKE]
[c2_1076] (twitter, SPY) I don’t short penny stocks; my brokers actually don’t even allow me to. Only thing I’ve shorted was $SPY $412.76 and $HUDI $23 last 30 days which my broker force covered me at $22.16 because they had to buy the shares back. I would never share an idea and short my fintwit family. [Ultra_Calls tweeted about SPY, HUDI]
[c2_1077] (twitter, BABA) $BABA This is low risk so I gotta have a go. Could end up closing out early but we'll see how it plays [Nebraskangooner tweeted about BABA]
[c2_1078] (twitter, RIVN) -8.5% from my SZ going into open🤝 $RIVN > [@ATMSnipes]( > $RIVN Textbook so far.. 2 trade opportunities on a silver platter with this one. > > I don’t see quality demand until $31.42 where it should catch a strong bid for a potential W3 > > Check the thread to learn more on this projection [ATMSnipes 🔁 ATMSnipes about RIVN]
[c2_1079] (reddit, BULL) Just started little of a month ago, I’m 37 years old Just started a little over a month ago my holdings for long term. I do have a 401K this is just build up for month passive income growth.
[c2_1080] (twitter, SPY) $SPY time for a bounce? > [@BTstocktrading]( > $SPY 435.50 level lost today, watching 428 for a relief bounce > > Still making lower highs and lower lows, looking to short rips next week > > If $SPY regains 435.50 level (also near AVWAP) will look for longs into 443.50 and 449 lower high [BTstocktrading 🔁 BTstocktrading about SPY, SPY, SPY]
[c2_1081] (twitter, AMZN) Added a little bit back to my $AMZN short this morning. Not a whole lot, will look to fill more higher if we continue upwards. (Previously exited about 70% of position at low $2,700's) [NebraskanCrypto tweeted about AMZN]
[c2_1082] (twitter, ES) $ES Bounce at 50ema... calls to 4509-4534 Lose the 50ema... puts too the 200ema-4380 [StonkTrump tweeted about ES]
[c2_1083] (news, QCOM) Qualcomm stock surges 20% after earnings tease a mystery chip deal
[c2_1084] (reddit, SLS) Stocks that feel Inevitable Looking to find stocks that “feel” inevitable in their success. Yes, I know nothing is a given but some companies are set up better for the future than others. For example, ASTS has the tech to create near monopoly on satellite cellular service but they’ve been held back by delays and funding concerns. Or SLS has a phase 3 study that is in progress that is already well past the expected end date of the trial. What stock do you feel has a strong thesis and why?
[c2_1085] (reddit, QQQ) 35k to 643K all I do is buy 0-3 DTE QQQ calls and puts. for instance I made $250k just yesterday lmao
[c2_1086] (twitter, AAPL) $AMZN - Upgrades at Edgewater Research, PT $150 from $130 at Piper Sandler $AAPL - PT $220 from $205 at Wedbush $NFLX - PT $470 from $380 at JPM, PT $500 from $400 at Wells Fargo $TTD - PT $80 from $75 at Needham $ROKU - Reiterated at Rosenblatt, PT $61 $COIN - Reiterated… [rachels_44 tweeted about AAPL, COIN, NFLX, ROKU, AMZN, TTD]
[c2_1087] (twitter, COIN) I have all day meetings today so put some thoughts in the Discord that @ZeroHedge_ should publish shortly. $DXY looks strong and trend changing over resistance. I am hoping for a dip today to stack some $COIN and other crypto in advance of the FOMC. I think we rally on 25bps [ShardiB2 tweeted about COIN, DXY]
[c2_1088] (reddit, DOW) These are the times that try men's souls (also women) Trying times we're in, I made the mistake of seeing a story on the DOW close today. 😟 But, I've listened to the sage advice of all the Bogleheads who tread the path I'm on before me. I made my plan, then I made my just in case plan, then I made my Oh crap this happened plan. I think I can ride out the storms and not have to work until I drop dead. Thank you to everyone who has shared their wise wisdom over the years.
[c2_1089] (twitter, ES) $ES - above blue box then bear market rally resumes imo then will look for btc longs...below last week monday high, we back to range lows [trader1sz tweeted about ES]
[c2_1090] (reddit, BTC) Bitcoin ETF & Coin Supply So I have a question If I buy say for easy math 1 BTC worth of an ETF in USD Now the ETF company has to buy and hold 1 BTC for me So when I buy BTC ETF am I (and everyone else using etf) lowering the total supply of BTC Sorry if this is a dumb question , just trying to clarify because it’s much simpler for me in my situation to use ETF
[c2_1091] (reddit, BABA) Starbucks closes China deal with Boyu, plans to expand stores.
[c2_1092] (twitter, COIN) Looking into $coin n $mster puts as a hedge for bitcoin downside [HackermanAce tweeted about MSTER, COIN]
[c2_1093] (reddit, AMZN) THE $2400 HAILER MARY INTO $180K Was doing some up downs over the weekend and ended up around 2400, -50% port. Bought 40 spy calls for 1500\~ on Friday for Monday expire. Sold for 4k\~ in the morning on Monday. Decided to buy a bunch of OTM 0dte spy calls right after cause I felt like I should have held. Proceeded to sell them for 16k. Loaded up 2000 spy 767 calls for Tuesday. Sold them early afternoon missing out on 300k.​ I just took random screenshots of history hopefully it shows everything.
[c2_1094] (reddit, MSTR) Is it still smart to DCA with bitcoin nearing 100k? I’ve been consistently DCA’ing into Bitcoin, but now that the price is pushing close to $100K, I’m starting to wonder if it’s worth continuing. Wouldn’t it make more sense to pause and wait for the price to come down after the bull market cools off?
[c2_1095] (twitter, COIN) yes > [@TrendSpider]( > $COIN Highest weekly volume since early March. 💣 [Dyme quote tweeted TrendSpider about COIN]
[c2_1096] (reddit, AMD) +27k last week HOOD AMD Thanks you regards for always teaching me
[c2_1097] (twitter, CMT) cc: @_TBone_Pickens What do you think? Can we get it added? [@EricBalchunas]( $ARKK w/ serious bounce today, up 9.3%, fourth best one-day return ever and now up 13.5% since the Cramer Signal (which should prob be part of the CMT exam at this point). Context imp here tho, ARKK still down 20% YTD and 45% over 1yr altho up 240% over 5yr [the_chart_life 🔁 EricBalchunas about ARKK]
[c2_1098] (reddit, COIN) BREAKING: Silver Crashes 38% From Record High, Gold Falls 16% From Record High — Largest Single-Day Drop in Precious Metals History
[c2_1099] (reddit, BABA) Don’t bully me for this idea I normally don’t trade on news days and like to get small wins on GBP/JPY and EUR/USD, but I just had this idea and I wanna know it’s dumb. I want to put a long and a short on the US 500 tomorrow as the news comes out with a fairly tight stop loss on both but a huge tp. Is there a chance it works? Or is it going to lose both due to volatile movements in both directions just as the news is being released? Just wondering it’s 3am uk time and I can’t sleep😂
