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
----- ITEMS (50: c2_2200 .. c2_2249) -----
[c2_2200] (twitter, BACK) $BACK nice volume ah into new highs. Could be gapper like $PEGY was… > [@greatstockpicks]( > Watching $BACK. Low floater near lows [BullishTrade retweeted Zalmy about PEGY, BACK]
[c2_2201] (twitter, VS) $NFLX trying for green. Was strong yesterday. Could be a runner if yesterday's high breaks here [VS Options Trading tweeted about NFLX]
[c2_2202] (twitter, TSLA) A new strategy I have been trying for a few days 👇 $SPY doesn't have a lot of volatility, individual stocks can often move better, how to find the best ones? 1. Use MNF to gauge overall sentiment 2. Use top flow to find top stocks 3. Play them instead of $SPY e.g $TSLA today! [reciknows 🔁 Tradytics about TSLA, SPY]
[c2_2203] (reddit, PMAX) Small-Cap Low Float On My Radar **(NASDAQ: $PMAX)** LOW FLOAT AROUND 1M! Hammer on the 4 hour Gap around $0.58 Currently at $0.30 A Hong Kong Financial Communications Powerhouse Expanding into U.S. Solar, Offering Investors Exposure to Asia's $4.5T Capital Markets and New Energy Growth! Company info > [.]( Disclaimer > [.](
[c2_2204] (twitter, PLG) $plg ill be scaling into this one via commons. [HackermanAce tweeted about PLG]
[c2_2205] (reddit, DRS) 🟣 Reverse Repo 03/04 0.877B - BUY, HODL, DRS, Pure BOOK, SHOP, VOTE 🟣
[c2_2206] (twitter, NFLX) My trades today $NFLX 325p +32% $SPOT -1.24% Challenging slow sell off today! 😀 [optionsprochick tweeted about SPOT, NFLX]
[c2_2207] (reddit, GME) What's the going on with Newegg? Stock price just hockey sticked up on Friday with no major news. A few sites are claiming the rally is from the site having lower prices than Amazon during Prime Day. Think they're getting bought out? It would make sense for Gamestop to acquire them, and Gamestop is sitting on a lot of cash. Anyone else have other speculative theories?
[c2_2208] (reddit, AMD) Amd 250 call +1900% Made back about 1/4 of my losses this morning
[c2_2209] (reddit, VXX) $SPX $SPY $VIX $VXX $VXST $UVXY $TVIX Closing CBOE SPX and VIX Index summary for November 29th
[c2_2210] (reddit, TIGR) TIGR $7.00 Calls 2x TIGR closed today with a 6% push on high relative volume. This is going to be my first real options trading test, so it's super low stakes, but I’m betting tomorrow morning theres going to be some push to test the resistance. The RSI is at around 65, but I’m probably going to try for a quick exit tomorrow morning, or if not hope for a spike early Friday. Already up slightly in after-hours. Watching open to see if I can't catch a small profit. How regarded am I?
[c2_2211] (twitter, GBTC) This guy did nothing about the $gbtc trash but claims he cares about "investors" . He absolutely has no idea how to handle crypto. > [@GaryGensler]( > Until crypto platforms comply with time-tested securities laws, risks to investors will persist. It remains a priority of the SEC to use all of our available tools to bring the industry into compliance. [HackermanAce 🔁 GaryGensler about GBTC]
[c2_2212] (twitter, BABA) $BABA Eddie Wu has been appointed to lead Alibaba's primary e-commerce business, succeeding Trudy Dai. This leadership change comes amid concerns about growth in Alibaba's e-commerce segment. [Mr Chart Norris tweeted about BABA]
[c2_2213] (twitter, MU) So many amazing A+ setups out there Goal is to focus on a few of your favorite names That way you have conviction to hold for big moves My favorite names for the next few weeks: $MU $COIN $HOOD $APP $ENPH All leading names in strong emerging sectors This summer will be [Sean trades tweeted about HOOD, COIN, MU, APP, ENPH]
[c2_2214] (twitter, WW) $SFET $6 Target - Dawson James - 900% upside - WW here [Ultra_Calls tweeted about SFET]
[c2_2215] (twitter, SFM) $SFM Monthly chart is very bullish, 1.618 at 42.64 Clear this level and we start pushing. I am personally expecting new ATH's. [HackermanAce tweeted about SFM]
[c2_2216] (news, ABR) Moody's CMBS conduit delinquency tracker rises to 5.58% in April
[c2_2217] (reddit, AI) POET vs. LWLG – thoughts from this community? Hey everyone, quick question for those following the photonics/AI connectivity space. Between POET Technologies and Lightwave Logic (LWLG), which one are you currently more bullish on and why? I’m trying to get a better sense of how people compare their technology, management, and path to commercialization, especially since both seem promising, but still carry significant risk. Would really appreciate any insights or different perspectives.
[c2_2218] (reddit, RENT) Almost one year ago, I dumped all my INTC shares at $21.3 Not an actual loss, but definitely this sub worthy dumb decision and something that lives rent free in my head last few months.
[c2_2219] (twitter, AMZN) $AMZN weekly chart After losing major support and breaking down in April, Amazon dumped right to the level where the previous uptrend started near 102 Now consolidating at the top of the 2018-2020 range, $AMZN looks to be forming another range between ~100-130 [BTstocktrading tweeted about AMZN, AMZN]
[c2_2220] (twitter, FLR) $FLR (per request) Unfortunately we had a deviation above 99sma and now below. Showing strong daily bear trend right now [Nebraskangooner tweeted about FLR]
[c2_2221] (reddit, GOLD) Oil prices keep rising as Trump seeks coalition to reopen Strait of Hormuz | US-Israel war on Iran News
[c2_2222] (reddit, BTC) What do you prefer to do with your BTC? Since many people seem disappointed with the price lately, how are you managing your BTC? Holding, staking, or doing something else worth sharing? Who knows, we might be interested.
[c2_2223] (twitter, TSLA) I believe in $TSLA more than my bright future, but do we get the right shoulder? FYI Above 245 the right shoulder will get invalidated, but currently moving as per plan [Mr Chuck Norris tweeted about TSLA]
[c2_2224] (reddit, ETH) 26 optimism transactions fees on June -why? I have my ETH with blockchain don’t have a lot. Does anyone know why I have optimism base transaction almost every day? I have not done anything with my ETH. Small amounts. Biggest is 0,000006xxx ETH Started after I swapped LPT -> ETH
[c2_2225] (twitter, PFE) $PFE Stop loss break even from entry and see if this nuke candles today after further breakdown [Nebraskangooner tweeted about PFE]
[c2_2226] (twitter, COST) Ripppper 🚀🚀 Watch $WMT next [@rachels_44]( $COST backtest off trendline. Strong today📈 Needs last week's high [rachels_44 🔁 rachels_44 about WMT, COST]
[c2_2227] (twitter, CL) Oil ticking red 👀 [@StonkTrump]( $CL Working on a third weekly bear div off a double top just like it did in 2018? You better be watching oil right now if you are speculating in the market This thing tanks and things get real ugly [StonkTrump 🔁 StonkTrump about CL]
[c2_2228] (twitter, AMD) Watching $AMD $ROKU $CAT $XOM $TSLA today! [optionsprochick tweeted about AMD, ROKU, CAT, XOM, TSLA]
[c2_2229] (twitter, ATHS) The real question is do i fade this pump on $pfe or will we run it back to ATHS [@Beastlyorion]( Pfizer was forced by court order to release documents showing that their covid vaccine was in fact killing and crippling patients. They knew this at a very early stage. Yet anyone who spoke out about it publicly was cancelled and banned from social media. [HackermanAce 🔁 Beastlyorion about PFE]
[c2_2230] (twitter, PLTR) $PLTR got me again Downgraded today [Don't follow Shardi B if you hate Crypto tweeted about PLTR]
[c2_2231] (reddit, BTC) Just bought 2 more BTC I’ve been buying BTC since 2019, and about 1/3 of my total net worth is tied to it. I bought when it was 105k, then 115k, then 120k. On paper, I’ve lost more than 300k (from the ATH), but today I decided to buy 2 more coins. Look at the bigger picture folks, it’s going to come back and reach 200k in no time. 80k is the new 15k, save the tweet!
[c2_2232] (reddit, GOLD) Centera Gold had me hooked I bought 250 shares of CGAU for about 14.90 average per share. It was up to 21 before the war started so I wasn’t worried. But now it’s dipping to within a dollar of what I payed for it. I’m a new investor so I don’t know if I should hold or take the little profit that is left. Everyone says hold, and buy more. But this market turn happened right when I was thinking of trimming my position. It’s hard out here in these streets. I wish everyone good luck!
[c2_2233] (reddit, RENT) Are you all seeing a reduction in consumer discretionary shopping as well? This is not strictly investing per se but I run a vintage and collectibles item shop out here in Arizona. And I’m seeing a huge drop in sales this past month. It may just be anecdotal, but maybe there is more to it, are you guys seeing reduced consumer discretionary spending in your locations as well?
[c2_2234] (twitter, FF) #ff @davidbelle_ @VolaTim @ZFXtrading @TraderSkew @PriapusIQ [trader1sz tweeted about]
[c2_2235] (twitter, SPY) $SPY Is trying hard to defend the $457.14 today. If it can get above $461.69, this should revisit the $464.16 range. SPY was down 4.87% from the highs this as of this morning. [optionsprochick tweeted about SPY]
[c2_2236] (twitter, MBLY) $MBLY ☕️ > [@kholov23]( > $MBLY 1H Chart ☕️ > > Since it broke out from downtrend channel / bullish flag / descending channel , or call it whatever you want , it's time to think about its next move and how it should move > > In simple words , that's what I see [Mr Chart Norris quote tweeted Mr Chart Norris about MBLY]
[c2_2237] (twitter, GP) $GP Another random stock at the bottom of my watch list that's going crazy [Nebraskangooner tweeted about GP]
[c2_2238] (twitter, AMZN) $AMZN Single trader drops ~$2.8M into a handful of strikes in the FEB monthly. [ThetaWarrior tweeted about AMZN]
[c2_2239] (reddit, AMD) AMD options advice I bought a few option on a lark while waiting for coffee back in May, nothing massive but first foray into options. Looking for advice on best way to handle it. Both are pretty deep itm Both 140 calls 1 expires in Dec 19 up 956% 5 expire June 18th next year up 464% Should I sell the Dec and just hang onto the other 5 until next year sometime?
[c2_2240] (twitter, GOOG) $GOOG Nice R:R here after that rejection. Didn't go quite as high as I thought it would. [Nebraskangooner tweeted about GOOG]
[c2_2241] (reddit, BULL) Can yall tell me if I got everything right this my first day on swing trading
[c2_2242] (twitter, SCHW) Charles Schwab $SCHW (weekly) Currently below channel and horizontal support. [kpak82 tweeted about SCHW]
[c2_2243] (twitter, S) $S Strong reaction at 11ema today [Nebraskangooner tweeted about S]
[c2_2244] (reddit, AAPL) #219 - The Golden Age of Stock Picking, Volatility & Opportunity (ft. Scott Grossman of Turning Point Brands) • Cannabis Investing Network on Apple Podcast
[c2_2245] (twitter, AMD) Solid week for our watchlist from the #sevenminutesqueeze video. $TSLA 947.77 -> 1040.70 (power move) $AMD 118 -> 120.63 $BA rejected off mid channel as expected Check out the quick walkthrough if you missed it! [ItsSqueezy tweeted about TSLA, AMD, BA]
[c2_2246] (twitter, NFLX) $NFLX That stupid sell off candle this morning ruined my long set up so I never got entry. Hate when PA is choppy like this. Looks set to go higher [Nebraskangooner tweeted about NFLX]
[c2_2247] (twitter, AMD) Watching $FB $TSLA $AMD $UAL $NVDA Today! [optionsprochick tweeted about FB, TSLA, AMD, UAL, NVDA]
[c2_2248] (reddit, U) Top High Premium yield Tickers for Today.. Here are some High Premium Yield tickers I am tracking: High Premium Yield Calls: $IREN:12.5009% $U:12.5009% $SMCI:12.5009% High Premium Yield Cash Secured Puts: $WMG:64.8441% $MSTR:96.9971% $AU:244.922% [Source](
[c2_2249] (reddit, AMZN) What Are Your Moves Tomorrow, July 10, 2026 This post contains content not supported on old Reddit. [Click here to view the full post](
