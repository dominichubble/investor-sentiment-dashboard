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
----- ITEMS (50: c2_1250 .. c2_1299) -----
[c2_1250] (reddit, LLY) LLY thoughts LLY has its earnings tomorrow morning and judging from the past performance of this stock it has enormous potential going up or down. Regardless if you get a option correct there is huge gain. Has anyone been watching the stock? Would love some opinions. Im planning to go in big tomorrow right after market open just need to flip a coin on call or put.
[c2_1251] (reddit, GOOGL) What Are Your Moves Tomorrow, September 25, 2025 This post contains content not supported on old Reddit. [Click here to view the full post](
[c2_1252] (twitter, PLTR) $PLTR Daily Chart Selling pressure is decreasing Welcome 22 - 23 ? [Mr Chart Norris tweeted about PLTR]
[c2_1253] (twitter, SPY) Look for short ideas? 👀 May we present to you the 'Cramer Short Scanner' 📉 This scanner combs the news and finds any articles that mention both "Cramer" & "Bullish" $SPY [BTstocktrading 🔁 TrendSpider about SPY]
[c2_1254] (news, AA) Alcoa reports Q3 loss but investors like portfolio review to drive lower costs
[c2_1255] (reddit, META) Mark Zuckerberg Tops Billion-Dollar CEO Sell-Off Before Tech Tumbles Smart move. He probably wishes he sold a lot more than he did.
[c2_1256] (twitter, GOLD) $GOLD looking good. Will be watching the 3 day close. [@AltcoinPepe]( $GOLD Very interesting play for me this. It looks really nice. It's been using the high of the Covid crash as a demand zone. Bullish imo. It's also above the yearly open and the 21 weekly MA. Again, good signs. [AltcoinPepe 🔁 AltcoinPepe about GOLD, GOLD]
[c2_1257] (twitter, DIA) Team ATM buying the dip all 3 days👀 > [@SethCL]( > 3-consecutive days in the life of Dumb vs. Smart Money > > "Somebody knows something", as the saying goes? > > $SPX $SPY $QQQ $DIA $NDX [Avery quote tweeted Seth Golden about SPY, QQQ, NDX, DIA, SPX]
[c2_1258] (twitter, GRI) $GRI taking out this 15k wall at $1.05 [BullishTrade tweeted about GRI]
[c2_1259] (twitter, SPY) $SPY Whale rolls down on solid profits +$5.6M gain. Now in the 368P [ThetaWarrior tweeted about SPY]
[c2_1260] (twitter, ES) Really nice base forming here into FOMC minutes.. Just need positive news for lift off now $SPY $SPX [@ATMSnipes]( $SPY $SPX $ES $444(.382) Bullish 🎯 hit If bulls defend here May 🎯 is $471 (1.618) of W4 R: $449.75, $451, $454, $457, $463, $467, $471.95 Blowoff Target: $475, $477 Red Pill is your bearish alt noted below [ATMSnipes 🔁 ATMSnipes about SPY, SPX, SPY, SPX, ES]
[c2_1261] (twitter, SPY) Everyone wants to be the hero calling the top.. Eager bears ran over > [@ATMSnipes]( > I invite everyone who is incredibly bullish to look at the $SPY weekly and visit: > > 3/18/2022 > > You aren't even remotely safe yet. > > Eager bears can get cooked trying to short too early too [Avery quote tweeted about SPY]
[c2_1262] (twitter, KAI) $KAI is up 75%, but I actually just was out and too tired to share at night. Some already guessed it right. Alpha incoming... > [@Anbessa100]( > I shared so many flying memes the last days that I came to the conclusion it's better to ride the next one alone. > > I hope you going to celebrate with me though. [ANBESSA quote tweeted ANBESSA about KAI]
[c2_1263] (twitter, SPY) After $SPY gave us a new high out of that downtrend, the pullback to 618 was perfect. I was focusing so much on the weekly broadening formation, but this was a great opportunity for solid R/R [ItsSqueezy tweeted about SPY]
[c2_1264] (reddit, BTC) Do you think this Bitcoin crash is temporary? Could BTC drop back near $50,000?
[c2_1265] (reddit, NFLX) The most recession-proof stock? Not to be morbid or anything, but I can think of no better recession-proof sector than end-of-life services. Funeral services, coffins, cemeteries, hearses, etc., etc. Is there a big, nationwide funeral company? Are there death ETFs? Can I park my money in the before-the-afterlife segment of society? Not really joking. Death and taxes, but I can't invest in taxes. What companies or ETFs should I be researching to move forward with this strategy?
[c2_1266] (twitter, ES) $ES Structure change looks to be confirmed with another close over the last lower high before range We also broke out of the upward channel today Also broke through the ATH avwap which to me always signals a downtrend reversal Next stair step up is the 4581-5603 region [StonkTrump tweeted about ES]
[c2_1267] (reddit, ETH) JPST/Corporate Bond Abnormal Movement Lately What is going on with Corporate Bond ETF? Normally you would see a daily +0.0x% and then big drop at month end due to dividend payout. But JPST has been relatively flat for the past month. Meanwhile treasury ETFs like SGOV or SHV still look like normal cyclical movement. So that rules out fluctuation in interest rate for the cause. Are we anticipating default in corporate bond or something? [ [
[c2_1268] (twitter, COIN) Buying 525 weekly calls on $COIN yesterday on that dip was such a strong move [Shardi B Trading #teamshardi tweeted about COIN]
[c2_1269] (reddit, MSFT) worst investment i ever made even AMZN is better fucking investment than this crap. analysts are so full of shit, like please downgrade this piece of shit to the 450s or less your fucking so many people over with your bs predictions
[c2_1270] (reddit, NFLX) Netflix Discussion Have been looking at it for a minute, with this new split and the fight next month I feel like there will be a lot of marketing hype. Over the last year the stock did well, the last three months not so much. Is this a catalyst for a good November? Has anyone else being sitting on the sidelines or in on it? What’s your opinions? Tryna get some good discussion started, looking at it I felt positive about the hype for this one
[c2_1271] (twitter, CAT) Eyes on $CAT BNB flagship memecoin, feels super undervalued. So strong even when market is down, I believe it leads the market when market gets strong again and CZ comes back. [EliZ tweeted about CAT]
[c2_1272] (reddit, PDT) I love the new no PDT rule Big gain was spy puts after switching back and forth between puts and calls all day not knowing what to do, but luckily I wasn’t restricted in # of trades I could do
[c2_1273] (twitter, TSLA) $TSLA Classic falling wedge breakout trade. 🙌 [ThetaWarrior tweeted about TSLA]
[c2_1274] (reddit, NVDA) PUTS NVDA Tomorrow is the last day of the week .There are three days of rest ahead.Everyone will be closing positiones.Especially since nothing good is expected.
[c2_1275] (reddit, BEAT) APPLE $AAPL JUST REPORTED EARNINGS. DOUBLE BEAT! * EPS of $1.57 beating expectations of $1.42 * Revenue of $94B beating expectations of $88.9B
[c2_1276] (twitter, SPY) A slow start to the year for the 'inflation' theme... $INFL $SPY [the_chart_life tweeted about INFL, SPY]
[c2_1277] (reddit, MSTR) Only because avg price of Saylor is 67k Butcoiners think below 67k MSTR will bankrupt lol
[c2_1278] (twitter, AMZN) $AMZN - MS maintains overweight, PT $3800 to $3500 $FB - MS maintains overweight, PT $330 to $300 $GOOG - MS maintains overnight, PT $3270 to $3000 [rachels_44 tweeted about AMZN, FB, GOOG]
[c2_1279] (twitter, PHIO) $SMFL nice 30% overnight move as well here. $AZTR igniting the nanofloats $PHIO watching > [@Bullish_Trades]( > $SMFL been eyeing it also [BullishTrade quote tweeted BullishTrade about SMFL, AZTR, PHIO]
[c2_1280] (reddit, ETH) Ether outpaces Bitcoin’s trend change: Is ETH on track for a 20% rally?
[c2_1281] (reddit, COIN) Freeze quantum-vulnerable addresses that refuse to move their coins to new addresses?
[c2_1282] (reddit, BTM) BitCoin ATM Operator Pain Im currently working on a project that i think will solve a number of pain points for BTM operators in the AML / KYC space for a fraction of the cost that is currently being charged by TRM / chainalysis, or really any of this big players. Although im curious if you operate any number of BTMs, what are your current pain points that suck a lot of your time and what are your costs for your current set up?
[c2_1283] (twitter, UNH) Not a great look for the largest holding in the Dow. Goldman (the second largest holding) also getting hit hard yesterday at a logical spot. Things to keep an eye on. $UNH $GS > [@DJwrath]( > @the_chart_life $UNH could be going from tailwind to headwind for the Dow [the_chart_life 🔁 DJwrath about GS, UNH]
[c2_1284] (twitter, PPC) $ppc hmmmmmm > [@trndtrader]( > REPEAT BULLISH POSITIONING DETECTED IN $PPC, SIZE BULL CALL SPREADS .. [HackermanAce 🔁 trndtrader about PPC]
[c2_1285] (twitter, BIDU) $BIDU Weekly Candle Chart With patience comes a BEAUTIFUL bounce off the lower boundary of falling wedge pattern [Mr Chart Norris tweeted about BIDU]
[c2_1286] (twitter, BTC) RT @Jokerparabol: Here go 40K $BTC #BITCOIN Breakout🔥 $SLNH & $BETS LFG🃏 [BullishTrade retweeted Paraboljoker about SLNH, BTC, BETS]
[c2_1287] (reddit, MSCI) Portafoglio a 21 anni Ho 21 anni, devo investire 50k + piano di accumulo da 500€ al mese. Vorrei investire in : IShares MSCI world Xtracker emerging marcket ETF in bit Coin Vorrei investire anche nell’Avantis Global Small CAP, solo che su Fineco ha delle commissioni di acquisto di 2,95€. Ha senso comprarlo o oggettivamente non aggiunge niente , sono molto indeciso
[c2_1288] (reddit, BTC) Nflx yolo clutching up rn Praying they beat their earnings estimate tmrw 🙏
[c2_1289] (reddit, AAPL) Is Qcom still Considered a Good Hold?? Many people consider qcom as a strong buy because of its potential in the AI sector, however, I feel like they have been lagging behind AMD and other chip companies for a while now. One of Qcom's biggest customers is also Apple, and Apple is notorious for switching to in house components. If they switch to in house, then I don't see how qcom could compete at all in this super competitive sector.
[c2_1290] (reddit, HTZ) CAR (Avis) Put $1200 -> $35,000 Trade closed. \~$35k in profit. In my Roth IRA btw. Bought 8x $300p 4/24 CAR yesterday at 9:48am. Sold 3 yesterday for $4.70 to cover cost and just sold the remaining 5x just now for $69. LFG. Biggest trade of my life. See you next week. Also, buy HTZ -> it will crush earnings. Currently holding 5000 shares.
[c2_1291] (news, AA) Alcoa (AA) Outpaces Stock Market Gains: What You Should Know
[c2_1292] (twitter, SPY) $SPY couldn't get momo above the $390 last week. If SPY can break above $390.06 Ill be watching for momo through $392.56 & the 395c Resistance: $395.88, $400.61 If it cant hold $384.13, id be more cautious for a pullback toward $381.51, $378.73, The 381p could be an option [optionsprochick tweeted about SPY]
[c2_1293] (twitter, PULM) $PULM back to back 13G’s in the past 24 hours. 5.6% and 4.36% Data any day now: PUR1800 Phase 1b top-line data are expected in Q1 2022 and is anticipated to enable a Ph2 efficacy study in AECOPD. [Bullish_Trades tweeted about PULM]
[c2_1294] (twitter, FF) > [@HackermanAce]( > Preecciooouuussss meettaaallss Only person I know was checking them out around the same time is @stoxkcharts big #ff . Also great simple charts. [HackermanAce tweeted about FF]
[c2_1295] (reddit, AMD) Need help on investment, thank u hi guys, im new here, but im looking to invest 6k a month in stocks, 3k from me and 3k from my wife, my goal is like invest for like 5 years +, what is the best stock to buy if i dont want too high risk with a moderate return? like 10% a year is good enough should i just buy sp500, qqq? or stocks like novo, amd, etc?
[c2_1296] (reddit, NFLX) Netflix (NFLX) is still a great buy Netflix is going to be bullish until the 1240s. Still a great buy until November 21st.
[c2_1297] (reddit, SI) Thoughts on Benficient (BENF) for a potential squeeze or is it already over? BENF released mildly positive earnings yesterday and the stock popped from $4 to $8.80 in the after hours/overnight market. The price dropped back $4 at market open. The float is 14m. The SI is about 13%. The CTB jumped from 78% in the morning to 214%. Is this over over before it started?
[c2_1298] (twitter, NFLX) $NFLX 242, 249 next big levels above. Strong day [rachels_44 tweeted about NFLX]
[c2_1299] (twitter, BABA) $BABA Looking bullish here. Close to breakout [Nebraskangooner tweeted about BABA]
