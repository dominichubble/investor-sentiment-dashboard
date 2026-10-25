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
----- ITEMS (50: c2_0850 .. c2_0899) -----
[c2_0850] (twitter, QQQ) I’m opening 10 Spots at 50% off this weekend for my private community!🌐 Access to my private community includes: 🔸Morning Market Report with 🔑 $SPY & $QQQ levels 🔸 Weekend report W/ Trade ideas 🔸 Live Voice & Chat Analysis that is Actionable 🔸 Trade Setups W/ targets &… [Avery tweeted about QQQ, SPY]
[c2_0851] (news, AAP) Advance Auto Parts updates FY2018 outlook post Q2 earnings beat
[c2_0852] (twitter, SPCE) EVERYONE IS MAKING MONEY WITH ME! [@Amdtrades]( $SPCE I think it's worth a shot going long. [Amdtrades 🔁 Amdtrades about SPCE]
[c2_0853] (reddit, GOLD) Dow falls 1,000 points as oil resumes surge, hitting $80 a barrel amid Iran conflict
[c2_0854] (reddit, GOLD) What up now with the gold and silver ? The war is being continued and still the uncertainty persists with no one knowing what's the next step still how are we seeing a downward rally in the price of the real assets l, gold and silver that have been falling consistently from few days ? Any team on it ?
[c2_0855] (reddit, NVDA) Strategy for buying leap calls? Assuming you've done your DD and everything checks out, do you just buy the closest strike you can comfortably afford, check on it periodically and sell when you're content? I usually use TA to scalp ETF options but I want to dabble with leaps to save myself the daily stress. An example I'm considering is like am NVDA 220c 21Aug2026
[c2_0856] (reddit, GOOGL) Loss keeps getting worse invest in dogshit stocks = dogshit returns
[c2_0857] (twitter, NSSC) $NSSC Napco Security Technologies, Inc. engages in the development, manufacture, and distribution of security products Looks like consolidation of previous highs and then a breakout above, with a wick retest Again, garbage options, commons play 4/12 [StonkTrump tweeted about NSSC]
[c2_0858] (reddit, BULL) Shorting space stocks Pretty much what the title says. Considering long-dated put options on space companies like ASTS and RKLB. What am I missing. Why are they so great. Edit: more so ASTS, they seem to be solely invested in the satellite cellular thing and while it's a cool idea I don't see how it's the next big thing to invest in.
[c2_0859] (reddit, BNO) 14k YOLO BNO UCO UWM CORN long BNO UCO CORN short UWM The Russell 2000 is gonna get schwacked by rising interest rates coupled with rising input costs due to oil. Corn is gonna rocket due to the cost of fertilizer doubling. Wish me luck or wish me dead, your choice. no crying in the casino
[c2_0860] (reddit, BTC) Bitcoin Shorts Hit Most Extreme Since 2024, Last Time BTC Exploded 83%
[c2_0861] (reddit, ANET) MU Gain Porn - Enjoy Sold out of my SNDK the other day, sold ALAB and most of my ANET this morning and went big on MU before it started its big move for the day. I already had a sizeable MU leap position prior to earnings, got crushed down post earnings (too pussy to have shown the loss porn then) but bought even more - lets see what the next year brings. That Ain't working - That's the way you do it money for nothing and chicks for free.
[c2_0862] (twitter, TGT) $TGT Breaking higher from a bull pennant, pushing above the 200 SMA and next critical area of resistance. [ThetaWarrior tweeted about TGT]
[c2_0863] (reddit, BULL) Vanguard Wellesley Income Fund (VWIAX) bond portion Hello everyone, I am in retirement. I have kept Wellesley Income Fund (VWIAX) for a long time, however, it has generated a large portion of non-qualified dividends. Is VTEI a better replacement for the bond portion of VWIAX to reduce taxes? Thanks.
[c2_0864] (twitter, TSLA) $TSLA Daily close below Monday's strong candle would be a bad sign for lower levels [Nebraskangooner tweeted about TSLA]
[c2_0865] (twitter, NET) $NET Super clean bear flag breakdown. [Nebraskangooner tweeted about NET]
[c2_0866] (reddit, ETH) Eth How are you guys feeling about ETH. Taking profits? Do you think we’re going to have a correction soon & will dip? Alt coin season? I am unsure if I should take some profits off & Just keep my initial investment in. Any REALISTIC ATH? Im guessing around 5,800$ of all goes well…
[c2_0867] (twitter, ES) $ES clearing the range high vwaps squeeeeeeeze [StonkTrump tweeted about ES]
[c2_0868] (twitter, ETH) If you have any friends or family that are bear, please do a wellness check on them. That’s the least you can do. $SPY $SPX $QQQ $TSLA $BTC $ETH [JMA_Trades tweeted about ETH, SPX, TSLA, SPY, QQQ, BTC]
[c2_0869] (reddit, BTC) ProShares Expected to Launch SPCF ETF Targeting 2x Daily Returns of SpaceX on June 12
[c2_0870] (reddit, TY) 529 college fund question Hi I have a 12 yo kid whose 529 account just reached the goal for the total value that I figured the kid would need. In large part, this happened due to outsized stock market gains (since 2022 particularly). Since there are 6 years left before drawing from 529, how would one handle this investment wisely? Keep 529 in target funds? Move to short term bonds / cash? Keep on contributing? TY
[c2_0871] (reddit, ETH) Bitmine now has a floating loss of $1.9B on its ETH holdings
[c2_0872] (twitter, TSLA) HOW TO use @unusual_whales + Technical Bearish Pattern To day trade 50% Gains! $SPY ✅Sell off at Open ✅Bearish Rising Wedge pattern ✅Coming into resistance ✅Bearish Short Term Flow Pattern + Flow = 💰💵 Chart @TrendSpider $NVDA $AMD $ROKU $TSLA $AMZN $NFLX #investrLessons [unusual_whales tweeted about SPY]
[c2_0873] (twitter, ES) IF $ES follows this plan and gives trigger (arrow)...thats when i'll look to short btc and play ES to range lows. probly look for dxy bottom too [trader1sz tweeted about ES=F]
[c2_0874] (reddit, GLDM) Inflation Hedge How is this portfolio? I am retired with a pension but pension will not keep up with inflation. My monthly expenses is currently covered by said pension. U.S. Total Market VTI 45% International VXUS 20% TIPS SCHP 15% REITS VNQ 8% Gold GLDM 5% Emerging Markets VWO 3% Cash SGOV 4% I know emerging markets is represented in VXUS but thought a little more exposure will be better diversification.
[c2_0875] (twitter, UBER) $UBER is definitely going to be one of the companies/stocks where in 20 years I will either be saying "Yeah, of course it's worth $500 billion, they basically reinvented the idea of taxis" or "Yeah, of course it went bankrupt, they basically tried to reinvent the idea of taxis". [the_chart_life tweeted about UBER]
[c2_0876] (twitter, AAPL) $AAPL (weekly) Bearish engulfing with close below weekly 5/9ema and uptrend line for the first time since January. [kpak tweeted about AAPL]
[c2_0877] (twitter, ES) 3987-3916 remains the key pivot point 20ema is also there Bounce there worth longing, at least for a scalp Fail there and “bring out your dead” 🔔 > [@StonkTrump]( > $ES > > Just getting back into the swing of things after a few days off > > Thinking we get a deviation above the highs and 50ema that I will probably short > > Then I’ll either cover and flip long at a retest of the 3987-3916 region or just ride the shorts down if that doesn’t hold [StonkTrump 🔁 StonkTrump about ES]
[c2_0878] (reddit, AAPL) What are you planning to buy during the next recession? Nobody knows when the next recession will happen, but everyone knows one will happen - a recession every decade or so is practically a feature of our current economic system. Obviously we're all going to drop tech like hot coals, but what are you going to pivot to? I've read that discount retailers, pharma, and automotive repair are always good picks, as are the unkillable tech companies (Apple, Microsoft etc.) once they bounce back up.
[c2_0879] (reddit, AAPL) Did i find a money glitch What is this strategy bc it's obviously not real lol
[c2_0880] (reddit, BULL) Momentum Crash Hits YOLO Traders’ Returns by Most in Four Years
[c2_0881] (twitter, CMT) The Communication Services Sector ETF $XLC is now on the all-time high list. $GOOG $META $T $VZ $NFLX [Ian McMillan, CMT retweeted Sam Gatlin about VZ, XLC, GOOG, META, NFLX, T]
[c2_0882] (twitter, NVDA) $NVDA bouncing out of 212 support zone 👀 > [@BTstocktrading]( > $NVDA dump into support, that was quick 🗑️🔥 > > Ugly fade from the highs today, watching for a reaction here near 210-212. If the lows break, daily demand near 207 is the next level of support. [BTstocktrading 🔁 BTstocktrading about NVDA, NVDA]
[c2_0883] (reddit, INTC) Intel profit corn It's not all rainbows, I was questioning myself most days. Been holding since 05/09/2024. It's in a yolo account.
[c2_0884] (reddit, COIN) Coin Center warns stalled US crypto bill leaves industry exposed to future crackdowns
[c2_0885] (reddit, TSLA) Paper handed TSLA calls Exactly what it sounds like, I should've waited 2 days to sell but was afraid of theta. Filling out my Wendy's application now
[c2_0886] (twitter, TSLA) Tesla, $TSLA, earnings: - Revenue: $24.3 billion, est: $24.08 billion - EPS: $1.19, est: 1.12 [reciknows 🔁 unusual_whales about TSLA]
[c2_0887] (reddit, ETH) New ETF V500 Just dropped on Vanguard, any thoughts on this? Would it be wise to chuck a cheeky 10k at it? All my holdings are on VDAL at the moment.
[c2_0888] (twitter, TSLA) $TSLA (weekly) Keep it simple Above triangle = bullish Below = bad [kpak tweeted about TSLA]
[c2_0889] (twitter, IREN) $IREN up another 5% pre market Almost 150% gains on this At some point it'll make a correction but I don't want to give up my entry point so I'm gonna keep holding. [Nebraskangooner tweeted about IREN]
[c2_0890] (news, TSLA) A J.P. Morgan analyst sees 60% downside to Tesla stock—and he may be too optimistic
[c2_0891] (twitter, SPY) $SPY Man what an insane week of shared content here without any losses. Didnt think I could top the week before. If youre a new follower, make sure to scroll down and check out all the previous posts. I do not delete any of my losses or trades. > [@kpak82]( > $ES_F +180 points. New personal best on one ES trade. [kpak82 🔁 kpak82 about ES_F, SPY]
[c2_0892] (twitter, MARA) $MARA Breaking out nicely, added $10 calls yesterday [ShardiB2 tweeted about MARA]
[c2_0893] (twitter, ES) RT @MacroCRG: $ES - comparing current drive down to the drive down in April is interesting April drive down: -17.76% & 52 days Current… [MacroCRG 🔁 MacroCRG about ES]
[c2_0894] (twitter, UBER) $UBER Multiple rejections and OBV pushing down hard. If it loses this support then it has some room to fall [Nebraskangooner tweeted about UBER]
[c2_0895] (twitter, NVDA) NVDA beat, price flat > [@DeItaone]( > $NVDA > > *NVIDIA 2Q ADJ EPS 68C > > *NVIDIA 2Q REV. $30.0B, EST. $28.86B > > *NVIDIA 2Q DATA CENTER REVENUE $26.3B, EST. $25.08B [CRG quote tweeted *Walter Bloomberg about NVDA]
[c2_0896] (reddit, GME) Welcome to Charm School: Day 4 of buying near-term ITM Call Options to leverage positive Delta Decay for sustained price action. Did you know holding OTM Calls to expiration is a major source of the slow grind down of GME's price every week? Negative Charm from OTM Calls is deadly on the front week!
[c2_0897] (reddit, BTC) Alternatives to Bitcoin and Ethereum that actually have potential to reach their current levels? Do any of these options exist? I want to stop messing with alts because it’s a shitshow, but at the same time, the prices of BTC and ETH is too high to make any meaningful increases because I don’t have thousands and thousands to start with. Does SOL, AVAX, LINK, etc any of these have a chance to run high like BTC and ETH do? Open to suggestions.
[c2_0898] (reddit, TSLA) All in on AVGO 💹 AI capex is going to accelerate. Specially as the AVGO customers like GOOGL & META have already confirmed. And its trading cheaper than the sector’s growth-adjusted-multiples.
[c2_0899] (twitter, SPY) $SPY in the supply zone and retesting the prior breakdown area near 408-410, but still strong and making higher lows - needs to break down under 408 before considering shorts [BTstocktrading tweeted about SPY]
