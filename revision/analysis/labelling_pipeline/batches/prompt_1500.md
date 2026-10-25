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
----- ITEMS (50: c2_1500 .. c2_1549) -----
[c2_1500] (twitter, SHOP) $SHOP Critical loss of support. [ThetaWarrior tweeted about SHOP]
[c2_1501] (twitter, BITB) Stopped out on the 2x when we fell under the line Live to fight another day Held the $BITB 1x ETF [Don't follow Shardi B if you hate Crypto tweeted about BITB]
[c2_1502] (reddit, BTC) Crypto market soars: BTC hits $96k, Ethereum tops $3,300, altcoins follow
[c2_1503] (reddit, GLW) Finally beating the S&P 500 thanks to GLW Bought these leaps when GLW first hit $85 in 10/25, 12/27 expiry, $100 strike because I read that GLW makes fiber optics for data centers, among other things. It barely moved for three months, then they went ITM when GLW announced their $6 billion deal with Meta early January. Rolled to a higher ($125) strike but longer expiration when GLW was at at $110. Up over $30k since December. I think it still has room to run.
[c2_1504] (twitter, SPY) $SPY Fun fact: The TW Pivot produced a monthly buy signal at the bottom of the GFC bear market. It has not produced another since. [ThetaWarrior tweeted about SPY]
[c2_1505] (twitter, WWR) $WWR Nice profit on the pop and just closed the rest of what I was holding.. I just assume this breaks down. [Nebraskangooner tweeted about WWR]
[c2_1506] (twitter, GOLD) Disbelief intensifies in the metals market as Gold nears new All Time Highs. Mining exposure is going to be extremely lucrative while hugely underinvested. #Gold [JJcycles tweeted about GOLD]
[c2_1507] (reddit, BULL) BofA warns investors to take profits as 70% of the bank’s bear market signals flash red
[c2_1508] (reddit, DRS) 🟣 Reverse Repo 04/01 2.107B - BUY, HODL, DRS, Pure BOOK, SHOP, VOTE 🟣
[c2_1509] (news, ABR) Rose Portfolio Options: Covered Puts And Calls - Nothing Naked
[c2_1510] (twitter, ISRG) 15. $ISRG medical tech growth name that has essentially establish itself as the leader in non invasive surgery. Big growth potential and still providing 49% CAGR since 2015 [ThetaTendies tweeted about ISRG]
[c2_1511] (reddit, MSFT) Nebius signs $17.4 billion AI infrastructure deal with Microsoft Sept 8 (Reuters) - Nebius Group (NBIS.O), opens new tab said on Monday it will provide Microsoft (MSFT.O), opens new tab with GPU infrastructure capacity, in a deal worth $17.4 billion, over a five-year term. Shares of AI infrastructure firm Nebius rose 33% after the bell. The demand for data centers has surged as businesses are locked in a race to develop more sophisticated generative AI technology.
[c2_1512] (reddit, BULL) I build the system which snowball the wealth Every smart dip you catch doesn’t just add returns it compounds momentum. DipTip’s Snowball Tracker shows how small, disciplined investments grow into powerful long-term outcomes. Track every opportunity. Visualize your compounding. See how consistency turns ₹5,000 into ₹1,40,000. Open for discussion / suggestions
[c2_1513] (twitter, GLD) $GLD Weekly. #GLD nearing an inflection point now, downtrend line from ATH's $GDX #GDX #GOLD $GOLD $NEM [reciknows 🔁 reciknows about GOLD, GLD, GDX, NEM]
[c2_1514] (twitter, EW) The largest jumps in RS over the last 2 weeks have come from Capital Markets, Banks, Genomic Revolution, EW Nasdaq, Water Resources, and Small Cap Value $KCE $KBE $ARKG $QQEW $PHO $IWN [Ian McMillan, CMT tweeted about KBE, QQEW, IWN, PHO, ARKG, KCE]
[c2_1515] (reddit, BULL) Sold before the dump I didn't expect it to dump so hard even though I did expect it to drop because much of the push up was premarket. Luckily I got out at the top before the value got decimated. The calls were bought around the dip from the day before.
[c2_1516] (reddit, BULL) There is no capital gains tax in my country. I am new to stock investment and didn't know some of you pay tax when selling at a profit. That’s why I was so confused about why many of you just hold on to stocks forever. Of course, I cannot write off losses either; however, it is such an advantage because I am freer to sell with profit and re-enter with dip.
[c2_1517] (twitter, BYND) $BCAN $BYND BYND Cannasoft Enterprises Inc. has received a positive opinion from the Israel Patent Office (ISA), providing the ability to file a patent application under the Patent Prosecution Highway (PPH) in the U.S [Mr Chuck Norris tweeted about BYND, BCAN]
[c2_1518] (twitter, MA) -DeMark 13's -200d MA rejection -Trend line/Rate of trend resistance -Underside of June high range at 4080-4100 it's a lot to get through not saying it can't happen or won't happen, but it's a lot. $SPX $ES_F $SPY [the_chart_life tweeted about SPY, ES=F, SPX]
[c2_1519] (news, INTC) Here's how much the US government has made on its Intel investment after the latest big stock rally
[c2_1520] (twitter, EMA) $AAPL Weekly falling wedge setup off the 50 EMA. Apple should give clues if market is going to follow through. [Albert_trades tweeted about AAPL]
[c2_1521] (twitter, TSLA) $TSLA weekly and monthly view - so far rejecting higher time frame resistance and the weekly lower-high level @ 950 750 is an interesting monthly level to watch below if it continues lower Grabbed a few Sept. puts today near 900 📉 > [@Swilkes88]( > @BTstocktrading Could we possible see tsla? [BTstocktrading 🔁 Swilkes88 about TSLA]
[c2_1522] (reddit, AMZN) Bitcoin Returns for AAPL : Getting Rekt Most investors think they’ve made massive gains in stocks like $AAPL, $MSFT, $NVDA, $GOOG, and $AMZN. But what if you had just held #Bitcoin instead? Let’s compare their returns in USD vs in BTC 📉📈
[c2_1523] (twitter, NFLX) $NFLX Flow from today since I know we're all wondering. Best payday is going to be the line highlighted. If price opens here (~$265) contracts will trade $95 plus premium. Total gain ~$1.3M at a minimum. [ThetaWarrior tweeted about NFLX]
[c2_1524] (reddit, AGI) Roth IRA Contribution Error I over contributed to my Vanguard Roth IRA in 2025 because of an unforeseen influx in income. My MAGI/AGI is above the limit allowed by the IRS. I am aware of at least two possible routes to undo this. The first is to sell all of the 2025 contribution and pay a 10% on the gains. The second is to convert it to a traditional IRA. What is the best route to undo this? Are there other routes? Ideally, I want to pay as little tax as possible.
[c2_1525] (reddit, BULL) Wallet which one Been away for too long and wondered what wallet is best to use now? I'd like to access dapps if that changes anything?
[c2_1526] (twitter, SPY) $SPY is trying to make its first higher high since the market topped in Jan'22 also pushed up against AVWAP from ATH, the ~410 level on $SPY is pivotal - can bulls lead a break out over this level and start a new uptrend, or does the market reject here and look for new lows? 👀 [btstocktrading tweeted about SPY]
[c2_1527] (reddit, MSFT) Most Risk On Market Since 2021 Despite all the doomers in this sub, it's not all representative of the general sentiment out there. Risk-on is at the highest level in years. Important to not take too much weight from here (or any other single source of perspectives)
[c2_1528] (twitter, SPY) $DIA Weekly. #DIA continuation to the downside after retesting the break down from triangle. May see a bounce here off the downtrend line, but momentum not looking good for the bull plot $DJIA $SPY $QQQ $IWM $ES_F $SDOW #DowJones $SVIB $SBNY $UDOW [reciknows tweeted about QQQ, ES=F, DIA, SBNY, DJIA, IWM, SVIB, SDOW, SPY, UDOW, DOWJONES]
[c2_1529] (reddit, HOOD) my portfolio at 25 years young i would like to get some thoughts on my current portfolio total investments + savings ~ $81k hysa ~ $15k s&p 500 ~ $19k total market index fund ~ $6K $qqq ~ $6K $hood ~ $19k $btc ~ $12k $nbis ~ $2k $crwv $2k is this too risky or just right for someone around my age?
[c2_1530] (reddit, BTC) Babylon and Ledger Integration Expands Access to Trustless Bitcoin Vaults Hardware wallet support for Bitcoin vault strategies could be an important step for BTCFi. Babylon and Ledger are working toward that integration. Do you think it will open the door for more people to Bitcoin DeFi?🤔
[c2_1531] (twitter, QQQ) RT @reciknows: $SPY Weekly. #SPY bearish week w/ above average volume to back it up. Expecting 444/441 to provide some support, but could s… [ReciKnows retweeted about XLF, SPX, SPY, ES=F, QQQ, IWM, XLK]
[c2_1532] (twitter, PYPL) #PYPL Some stock information to supplement the Crypto content [CryptoNTez tweeted about PYPL]
[c2_1533] (twitter, GOLD) With gold at ATHs do not fade the laggards in this thread. > [@HackermanAce]( > Thread of $gold related Stonk plays EZ WINS. [Ace quote tweeted Ace about GOLD]
[c2_1534] (twitter, QQQ) $SPX Next update : 03/09/2023 $SPY $QQQ [reciknows 🔁 optionflys about SPY, QQQ, SPX]
[c2_1535] (twitter, TSLA) $TSLA #TSLA loosing $910? The next untested support for me is all the way down at $720 🥶 [_bread911 tweeted about TSLA]
[c2_1536] (twitter, QQQ) $IWM Daily. Looking for a re-test of the breakout from descending triangle, but given Fri's inside day can still head higher before re-testing. 200 level a very important level #IWM $RTY_F #RTY_F #smallcaps $SPY $QQQ $DIA [reciknows 🔁 reciknows about IWM, SMALLCAPS, SPY, RTY_F, QQQ, DIA]
[c2_1537] (twitter, AMZN) $AMZN Strong bounce off the 20/50emas last week. I'm watching to see if it can continue its momo above the $103.44. Resistance $104.76, $106.92, $109.70 Eyeing the 103/104c More of a pullback look below $100 [optionsprochick tweeted about AMZN]
[c2_1538] (reddit, AI) Sold it at the pump. Thank you to the people who told me to sell at the pump today. I also tried out covered call contracts and wanted somebody to buy my shares at a less loss at 210 but then I quickly backed out and decided for my mind's sake to just sell. Good learning experience and never touching these investments again. Thanks Wallstreetbets and have a good one! 🙏❤️
[c2_1539] (twitter, ES) $ES looks like another deviation and play back to range low. Reclaim range high then im wrong [trader1sz tweeted about ES]
[c2_1540] (twitter, CAT) $CAT Daily Chart 🐉 Falling wedge is getting TIGHT [Mr Chart Norris retweeted Mr Chart Norris about CAT]
[c2_1541] (twitter, SOFI) Recently got $SOFI Checking & Savings account, really liking it thus far. Including referral details below You’ll earn up to 4.60% APY and pay no account fees. Use my link to sign up and you’ll get a $25 bonus & up to $300 when you set up direct deposit. [ReciKnows tweeted about SOFI]
[c2_1542] (twitter, TTWO) $AAPL - Upgraded at Credit Suisse, PT $201, PT $220 from $200 at Wedbush $DKNG - Upgraded at Roth Capital, PT $25 (from $18 $FUBO - Downgraded at Wedbush, PT $6 from $5 $SE - Downgraded at Daiwa Securities, PT $90 $TTWO - Downgraded at Deutsche Bank, PT $160 from $200 [rachels_44 tweeted about AAPL, DKNG, FUBO, SE, TTWO]
[c2_1543] (twitter, GLD) Each one in this thread popping like popcorn. Also look at $gld $silj and related. All popping up. We don't get a chance like this in gold often but currently the cycle is in play. > [@HackermanAce]( > Thread of $gold related Stonk plays EZ WINS. [Ace quote tweeted Ace about GLD, SILJ, GOLD]
[c2_1544] (twitter, SMA) $ETSY Bear pennant continuation and loss of the 200 week SMA. > [@ThetaWarrior]( > $ETSY Bear pennant breakdown, hanging on by a thread at the 200 week SMA. [ThetaWarrior 🔁 ThetaWarrior about ETSY, ETSY]
[c2_1545] (reddit, GOLD) Oil price at two-year high after Qatar minister warns all Gulf production could stop
[c2_1546] (reddit, GOOGL) Google searches for "Buy Bitcoin" have spiked - It's happening... Source: [
[c2_1547] (twitter, SPY) If we fail to close above $440 I will need to restructure my counts as this is far more impulse than anticipated🫡 > [@ATMSnipes]( > $SPY $SPX - Have the dip buyers quit yet? > > Hope not because we are nearly there > > $442 target and gap fill tagged today > > It would be the cherry on top to gap down into $440.21 - $441.50 demand tonight > > followed by a strong rebid of the 50D EMA to kick off the relief rally [Avery quote tweeted about SPX, SPY]
[c2_1548] (twitter, GM) $GM GM's President envisions an electric return for the Chevy Camaro, aiming for a $35K starting price, but distinct from a Mustang Mach-E crossover. [Mr Chart Norris tweeted about GM]
[c2_1549] (twitter, CF) $CF Filled as much as I could above $106 [Nebraskangooner tweeted about CF]
