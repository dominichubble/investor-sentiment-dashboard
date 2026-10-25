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
----- ITEMS (50: c2_1700 .. c2_1749) -----
[c2_1700] (twitter, ALB) $ALB Have to consider the possibility of a fake out above ATH here [Nebraskangooner tweeted about ALB]
[c2_1701] (twitter, RSI) $AMD Monthly shooting star at ATH with negative RSI divergence. [kpak tweeted about AMD]
[c2_1702] (twitter, SPY) Bought $SPY 1/6 $383 Calls Levels holding with possible Bulldog pattern in play🐶 [JMA_Trades tweeted about SPY]
[c2_1703] (twitter, FG) Don't like this on $FG but we'll see how it plays out. Working on strong monthly and yearly closes [Nebraskangooner tweeted about FG]
[c2_1704] (twitter, UEC) $UEC Currently my biggest % gainer. Always feels nice to snipe a killer entry. Got in sub $5.30 on that quick dip a couple days ago Looks to keep running [@Nebraskangooner]( $UEC Mega strong daily close... Maybe I'll take a tiny position if there's a dip tomorrow morning. [Nebraskangooner 🔁 Nebraskangooner about UEC, UEC]
[c2_1705] (twitter, DBD) $SPY $SPX Took a couple of days but 🎯 hit. If there was a time to go short. Now is the time. ABC up | A=C @ 1.618 | .382 retrace W1🎯$410 > [@ATMSnipes]( > $SPY $SPX Soon… > > Wedging into 2HR supply. > > 10m DBD at $4253 (425) was the cause for the PM rejection. > > Break $425 and $427(1.618 Ext./.382 retrace) can trade next. > > Fail to do so and thats a double top. > > Would hate to see someone important miss ER tonight☕️ > > W1 🎯 Target $410 [ATMSnipes 🔁 ATMSnipes about SPY, SPX, SPY, SPX]
[c2_1706] (twitter, SPY) $SPY PARTIAL DECLINE > [@kholov23]( > $SPY Daily Chart Update 0.2 > > Broadening tops formation , price tested the upper boundary and got rejected [Mr Chart Norris quote tweeted Mr Chart Norris about SPY]
[c2_1707] (twitter, ES) $ES_F needs to hold here for 5min bullish inverse H&S [kpak82 tweeted about ES_F]
[c2_1708] (reddit, TSLA) Markets are down to say the least…what are you buying? Buy low they say. I remember at the beginning of COVID I invested in some stocks and was wishing I had purchased more after the market recovered. Hopefully, the market recovers, but of course we’re in a different situation. Are you going to invest in any stocks? If so, what are you purchasing?
[c2_1709] (twitter, SPY) $SPY Opened down into trapped sellers zone (blue) and immediately found support. Then bounced right into trapped buyers zone (red) and hit resistance. Level to level. [Theta Warrior tweeted about SPY]
[c2_1710] (twitter, QQQ) Looking forward to next week's market. I will be doing another Market Analysis video like last weekend's. Aiming to release tmrw/Sunday. $SPY $QQQ $BTC SUBSCRIBE!: [Honeydew0101 🔁 RyanRozbiani about SPY, QQQ, BTC]
[c2_1711] (twitter, SPY) $SPY WHALE SIGHTING 🐋 CLOSES 370P & OPENS 374P [ThetaWarrior tweeted about SPY]
[c2_1712] (reddit, AMZN) Question of Amazon’s Consistent Oustand Share Count Increase As the title implies, I’m curious if anyone has insight into why Amazon is seemingly increasing their shares outstanding every year and diluting shareholder’s equity? Is it due to acquisitions, stock compensation, or other reasons? It’s not too concerning given Amazon’s EPS and growth rates but is a little because I’d rather be seeing a slight decrease. Thank you.
[c2_1713] (news, ABBV) MinichMacGregor Wealth Management, LLC Buys United Parcel Service Inc, iShares –-' Year ...
[c2_1714] (reddit, GOLD) Lost all my savings, they drank my milkshake. World sucks, everyone sucks. Why are you gonna apease war mongers and allow strategic reserves??? Thanks a lot G7 for taking all my money. Tell me how to make it back. This is a discussion. Might rope
[c2_1715] (reddit, COIN) Missed the golden dip and still crying — how do you guys never miss these rallies?? 😭💀 What do you guys do so you don’t miss rallies like yesterday’s dip? I missed a golden opportunity and regret it. There was literally 50% off on all coins! How can I make sure I don’t miss the next one?
[c2_1716] (twitter, RVP) $RVP Can't remember where I saw this chart at but I picked up some just before close last week. Great R:R here for potential reversal breakout [Nebraskangooner tweeted about RVP]
[c2_1717] (twitter, AAPL) If you have been struggling over the last few weeks Do yourself a favor and start only trading: Leading stocks in leading sectors The opportunity cost of trading strong names in a strong bull market is unparalleled So many of you will miss moves like $AAPL, $AMZN, $GOOGL and [Sean trades tweeted about AMZN, GOOGL, AAPL]
[c2_1718] (reddit, AMZN) Capex and TSM drop I dont understand what is happening, everyone is so afraid of the massive capex that theyre selling the companies that it goes to? I understand wanting to sell Google or Amazon, but why sell TSM or Broadcom?
[c2_1719] (reddit, ETH) For years, crypto built in the gap between innovation and interpretation.
[c2_1720] (twitter, AAPL) $AAPL Closed 10% of my short... can't justify closing anymore of it but wanted to find a place to shave some profit today. [Nebraskangooner tweeted about AAPL]
[c2_1721] (reddit, SOFI) When margin calls are a good thing (?) Surely there are some rare scenarios when getting a margin call is a blessing in disguise. I went all in on sofi when it was around $30, it dropped to $27 and fidelity sold most of my shares on me. I was pissed at the time, but now happy they did that given it’s been trading under $20 practically all year. Same thing happened with Microsoft when I bought above $500, got margin called around $460, and now I’m ready to buy more under $400.
[c2_1722] (reddit, AMZN) Teen burglars posing as Amazon couriers steal £3.1M in crypto
[c2_1723] (twitter, MC) $FXI Sentiment Alert: $FXI Increasing bullish pressure with an imbalance in net option delta of 518 K. [HackermanAce 🔁 MC_OptionTrades about FXI, FXI]
[c2_1724] (twitter, GOLD) Gold is currently 3% away from a new all time high which is a move that can literally happen on a single day. The next biggest catalyst for hard assets to start surging higher. #Gold [JJcycles tweeted about GOLD]
[c2_1725] (twitter, BAM) $BAM is deflationary. 🦌 > [@BambiCashCrypto]( > 7. There will be an NFT series - a collection of 1k per phase of his life for a total of 8k NFTs. Those NFTs will be "free mint" using $BAM tokens. The $BAM used to buy the NFTs will be burnt and/or used for something discussed later here in this thread. [MacroCRG 🔁 BambiCashCrypto about BAM]
[c2_1726] (reddit, NVDA) Who is leading or well-positioned in commercial robotics? I keep hearing that robotics is expected to be the next big wave of technology. Which companies or organizations are currently leading the way, or at least in a strong position, in commercial robotics? Also, are there any companies in this space that look especially interesting from an investment perspective right now?
[c2_1727] (twitter, SPY) $SPY seeing some similarities to April 22’ & Jan 22’ If true… sellers want to break $385 with volume From there I expect stops to get ran at $369 before a favorable rally can be in play [ATMSnipes 🔁 ATMSnipes about SPY]
[c2_1728] (reddit, UPS) Long UPS Bouncing off both horizontal and vertical support of 109.89. Went long 3/20 $115 calls.
[c2_1729] (news, AA) Real Corporate Profits Are Now The Lowest Since The Financial Crisis
[c2_1730] (twitter, RS) The largest drops in RS over the last two weeks have come from Financials, Italy, European Financials, EVs, Airlines, High Beta, and various Foreign countries $GREK $IYG $IAI $EWI $EUFN $DRIV $JETS $SPHB $EWU $EFA $VWO [the_chart_life tweeted about GREK, IYG, IAI, EWI, EUFN, DRIV, JETS, SPHB, EWU, EFA, VWO]
[c2_1731] (reddit, BTC) How are your shares going? I've lost 28% ($24,000) in the last month. * How is your portfolio going? * What sectors are you invested in?
[c2_1732] (reddit, GOLD) What are the economic ramifications of China’s diversifying energy sources and investing in alternatives to fossil fuels?
[c2_1733] (reddit, HOOD) What is your opinion on this plan that I have I'm 20 years old and pretty new to investing, my goal is to not be broke when I retire so I decide to invest, I just made a robinhood account and plan on open a roth ira to invest in it 60% on VOO 30% on QQQM 10% on VXUS I do want to invest in tech but not too much I know QQQM and VOO are overlapping but I don't know which other etf are good, but that is my plan for now Thanks you, and have a great Monday
[c2_1734] (reddit, GOOGL) GOOGLE $GOOGL REPORTED EARNINGS \- EPS of $2.82 beating expectations of $2.61 \- Revenue of $113.8B beating expectations of $104.8B Google said it expects to spend between $175B-$185B on CAPEX in 2026 above expectations of $120B
[c2_1735] (reddit, TSLA) TSLA 397.5C gained 6150% in the last 4 minutes. Option volume spiked at the exact bottom. Some dude turned $100K to $6.1M in 4 minutes. Makes you question your rules and discipline
[c2_1736] (reddit, AMD) Finally in the green after 8 years. Now to blow it all on options. Again.
[c2_1737] (twitter, COIN) $COIN - Making a nice move up already. [IncomeSharks tweeted about COIN]
[c2_1738] (twitter, TSLA) If you are bearish $TSLA red box zone from 880-909 is where you put your money where your mouth is and buy puts I would, but premiums are just too ridiculous for me lol [StonkTrump tweeted about TSLA]
[c2_1739] (twitter, NVDA) $AMD When it ended up not confirming the reversal it was seemingly a buy... even $NVDA all over today, tough to call short term with earnings, better to be further out with your calls [Don't follow Shardi B if you hate trading tweeted about AMD, NVDA]
[c2_1740] (reddit, META) META Structuring higher? Holding above stacked MAs. Looking for more upside. Pull back into MAs would be ideal. Stop 653 taker 691 Not financial advice
[c2_1741] (twitter, PYPL) $PYPL (weekly) below major supports. [kpak82 tweeted about PYPL]
[c2_1742] (reddit, BTC) MSTR Just curious, what happens if MSTR sell its bag? We saw a huge dip after someone offloaded 70,000 BTC
[c2_1743] (twitter, SPY) Blue pill looking kind of sexy right now [@ATMSnipes]( $SPY $ES_F $SPX Red Pill or blue pill today? (Yes I know the line is black) Either way you know I’ll be trading it. $431 and $427 are your key levels today. [ATMSnipes 🔁 ATMSnipes about SPY, ES_F, SPX]
[c2_1744] (twitter, SNOW) $SNOW Daily - breakout watch this week [ItsSqueezy tweeted about SNOW]
[c2_1745] (reddit, GOLD) Oil prices top $100 per barrel as big Middle East producers cut output amid Iran war
[c2_1746] (reddit, GOLD) OPEC+ approves further oil output increase as Hormuz exports start to recover
[c2_1747] (reddit, ORCL) Oracle Stock Soars 10% on Its Strongest Q3 in Years After 84% Cloud Surge
[c2_1748] (reddit, JPM) JPMorgan CEO Jamie Dimon says Trump tariffs will boost inflation, slow an already weakening U.S. economy
[c2_1749] (twitter, AMD) $PYPL - Downgraded at DZ Bank, PT $140, PT $190 from $310 at Berenberg, $185 from $235 at Citigroup $AMD - PT $140 from $125 at Citigroup $CAT - PT $225 from $230 at Citigroup $SPOT - PT $280 from $310 at Barclays [rachels_44 tweeted about PYPL, AMD, CAT, SPOT]
