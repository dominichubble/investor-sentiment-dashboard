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
----- ITEMS (50: c2_2250 .. c2_2299) -----
[c2_2250] (reddit, BULL) Why are markets rallying all of a sudden? Just saw a sudden rise in the market with no headlines whatsoever, are we winning now…? I thought we are going to have an oil crisis which would consequently hurt the economy in the near future? What changed?
[c2_2251] (news, AAP) Ford (F), Jaguar Announce Job Cuts in Europe as Demand Wanes
[c2_2252] (twitter, NVDA) $NVDA Is retesting the $280.03 supply. If it can get momentum above $280 I'm watching this to move higher towards $284.97, $289.44, $293.98 resistances next. [optionsprochick tweeted about NVDA]
[c2_2253] (reddit, MSCI) QQQ 100% hedge costs per year Hi, I am new here and would like to understand how much it would cost me to hedge my portfolio: mainly QQQ and MSCI World for one year. I hear everywhere that you cannot reasonably do it so my follow-up question would be what you can hedge with 1% of the portfolio value. Expectation is that QQQ is down in 365 days by 10%.
[c2_2254] (twitter, GFAI) They are gonna be all over $GFAI I bet > [@manpreetkailon]( > South Florida hospitals lost IT service in cyber attack > > [Ultra_Calls 🔁 manpreetkailon about GFAI]
[c2_2255] (reddit, CC) PMCC - spreadsheet Anyone have a great spreadsheet to track PMCC. Buying the LEAPS and selling CC against them.
[c2_2256] (twitter, UK) $UK love this one unfortunately can’t sell shares on e*trade without calling it in. Can buy them tho 🤔 Fix it @etrade > [@Bullish_Trades]( > $UK - One to accumulate $3.30 and under. Float not updated on any sites yet, but float is only 460k after that 1:12 split yesterday [BullishTrade quote tweeted BullishTrade about UK]
[c2_2257] (news, AA) Alcoa hits lowest in 16 months amid aluminum demand concerns
[c2_2258] (twitter, SPY) You can apply this tweet to tomorrow. Nothing new here it’s just Thursday😂 [@ATMSnipes]( $SPY $SPX $ES_F Bulls taking advantage of demand #2. There is a real high probability we make a bull flag here Line(s) in the sand $450-$452 Bulls want north of $452 🎯 $457 Bears want South of $447 🎯 $444 [ATMSnipes 🔁 ATMSnipes about SPY, SPX, ES_F]
[c2_2259] (twitter, AMC) > [@DeItaone]( > $AMC Paused due to volatility [StonkTrump 🔁 DeItaone about AMC]
[c2_2260] (twitter, CF) Oh man my eyes are getting big looking at some of these stock charts that haven't started breaking down much yet. Shorted $CF and it's down 8%+ in 2 days. I can (and have) derisk by taking some profit. Move the funds elsewhere into something that is where CF was two days ago. [Nebraskangooner tweeted about CF]
[c2_2261] (reddit, BTC) 55 and starting Have $20k to put into ETFs. Ideally I’d like to retire in 10 years. Any suggestions?
[c2_2262] (twitter, TSM) $TSM Again, we got the sign... $158 seems obtainable Crazy for a company that has inventory sold out for the next few years and beat earnings and raised guidance Slaughtered on great news [Don't follow Shardi B if you hate trading tweeted about TSM]
[c2_2263] (twitter, NVDA) $NVDA back down to daily demand ~207 last level of support [BTstocktrading tweeted about NVDA]
[c2_2264] (reddit, ETH) Crypto Macro Hold: BTC ~$67K, ETH ~$1.9–2K; Whales Accumulate as Retail Stays Strong
[c2_2265] (reddit, AAPL) The SOX has caught up to NVDA for the last 24 month time period. NVDA is the largest component of the SOX (10-13%) but has not participated much in the latest semi rally. Which makes the SOX move even more eye popping, since it has achieved this largely without NVDA contributing. If you include 1 more year to this chart, NVDA still has a sizable lead on the SOX.
[c2_2266] (reddit, GOOGL) GOOGL question? Y’all, Google is under investigation from DOJ or EU I think. Not sure if they are still ongoing or what not. If so, does anyone know when they conclude the investigation? I just want to know if they are still being investigated or they already got slapped with fine or penalty. OR, the expected outcome or estimated conclusion by when? I am interested in buying but I would like to know these timelines before getting my feet wet. Thanks!
[c2_2267] (twitter, RIVN) $RIVN EV manufacturer Rivian Automotive Inc has applied to the Environmental Protection Agency (EPA) for certification of its 2025 model year R1T and R1S. The application indicates notable changes, including a reduced battery size and the inclusion of a heat pump. [Mr Chart Norris tweeted about RIVN]
[c2_2268] (twitter, CR) RT @CR38TOR_: Paypal Technical Analysis Video - Might possibly be one of the plays of 2024! $pypl $paypal [Amdtrades retweeted about PAYPAL, PYPL]
[c2_2269] (twitter, JPM) $JPM (weekly) rejection at channel resistance followed by reversal back below prior swing high. [kpak82 tweeted about JPM]
[c2_2270] (reddit, SPY) Think I’m holding these SPY puts until TSLA announces their terrible earnings and drags SPY down with it.
[c2_2271] (twitter, COIN) $coin is a good buy but only if it moves %50 down from here. [HackermanAce tweeted about COIN]
[c2_2272] (reddit, MA) MetaMask and Mastercard Launch Self‑Custody Crypto Card as MA Stock Rises
[c2_2273] (reddit, AAPL) Thanks I’m using Webull. I had 363 Apple share and and as you can see the picture I used 300 share to sell covered call. My question is can I sell the rest of the 63 share which are not tied to any option? Will it cause any problem? Picture for reference.
[c2_2274] (twitter, ES) $SPY $ES_F failed breakout (multiple times), feels like we a bigger move lower is coming [btstocktrading tweeted about ES=F, SPY]
[c2_2275] (reddit, NVDA) Google unveils TPU 8t and 8i AI chips as Nvidia and AMD face rising competition from hyperscalers
[c2_2276] (twitter, F) Corrected typos for $F and $NIO setups in the newsletter. Make sure to look at the horizontal levels on all charts as that is accurate. [kpak82 tweeted about F, NIO]
[c2_2277] (twitter, AMD) $AMD Strong push and DTL breakout. 🚀 [@ThetaWarrior]( $AMD Breaking out on the daily timeframe. [ThetaWarrior 🔁 ThetaWarrior about AMD, AMD]
[c2_2278] (twitter, HD) I really want to short $HD and hold for a month or two (or more?) but I can not figure out where I should be looking to short at... Hmmm.. Guess I won't force it. [Nebraskangooner tweeted about HD]
[c2_2279] (twitter, PFE) 👀👀👀🤝 > [@TrendSpider]( > $PFE Price sitting on a trendline that has acted as support dating back to February 2021 👀 Big spot to hold here! 💉 > > [HackermanAce 🔁 TrendSpider about PFE]
[c2_2280] (reddit, META) Meta call thinking the dip is due to one time tax, is this good time to buy 3 mo call?
[c2_2281] (reddit, GME) Top Stocks on Reddit Last week - GME is #2 and the only one with a bullish sentiment!
[c2_2282] (news, ABG) Are You Looking for a Top Momentum Pick? Why Asbury Automotive Group (ABG) is a Great Choice
[c2_2283] (reddit, UPS) UPS to cut 20,000 jobs on lower Amazon shipments, profit beats estimates Guh
[c2_2284] (twitter, ES) This entire rally looks like a back-test of previous support. If true, then we are going head much lower the coming weeks. $SPX $ES [MFHoz tweeted about SPX, ES]
[c2_2285] (news, ABBV) What's Next For The Biotech Bull Market? Are you a Trader or an Investor?
[c2_2286] (reddit, AI) A word of caution on using AI I decided to sample Gemini for calculating complex option strategy net Greeks, the numbers I got were way off my manual math, so I asked it - EDIT: For the number of down-votes and insultive comments, I'm guessing I hit a nerve with a few who work in the industry. This verifies AI is in a bubble. The words above are from Gemini, not me, but I'm getting down-votes.
[c2_2287] (reddit, ETH) ETH / BTC just fell below .018 for the first time since January 2020. If ETH/BTC falls ~10% more it will hit an 8 year low last seen March 2017.
[c2_2288] (twitter, NFLX) Stopped out on $NFLX buys. Didn't get confirmation to add either. No breakout. [Nebraskangooner tweeted about NFLX]
[c2_2289] (reddit, COIN) What are the best crypto coins to invest in to see maximum profits throughout the years? I'm new to crypto, and I don't know what to invest in. Starting sometime this year, I'm going to invest 1,000-2,000 bucks each month in different crypto currencies. I just don't know what to invest in right now. \*\*I'm NOT accepting DMs. Scammers, be warned!\*\*
[c2_2290] (twitter, AAPL) $AAPL Solid daily close. Up from entry. I'll just tighten my stop loss and see if this can make a push to $184 area [Nebraskangooner tweeted about AAPL]
[c2_2291] (twitter, BANF) $BANF Working on a weekly evening star. I've been looking for entry on a bounce but haven't gotten a bounce [Nebraskangooner tweeted about BANF]
[c2_2292] (twitter, ES) $ES Daily supply zone Danger Will Robinson use caution [StonkTrump tweeted about ES=F]
[c2_2293] (twitter, MSFT) $MRNA $SNY $MSFT Sanofi SA unveiled its collaboration with OpenAI and Formation Bio on Tuesday, aiming to accelerate drug development projects. [Mr Chart Norris tweeted about MSFT, SNY, MRNA]
[c2_2294] (reddit, NVDA) Unrealized profit 8k to 1.8k I invested into NOK back in 2021 and then was negative like 4k on it for like 5 years but never sold. NOK blew up few months ago and I watched it climb to 8k unrealized profit, and then watched this thing drop all the way back down to 1.8-2k profit. Thoughts on how you would approach this, the NOK threads believe this will blow up so I’m reluctant to sell to miss a massive growth stock like NVDA or MU. But part of me is saying sell it and invest in VTI/VXUS.
[c2_2295] (twitter, SPY) Big spot... Approach could have been better, IMO. We'll see if it can push through. $ES_F $SPX $SPY [the_chart_life tweeted about SPX, SPY, ES=F]
[c2_2296] (twitter, AAPL) $AAPL Tight wedge forming just below previous 52-week lows - An important spot to say the least. 🍎 [reciknows 🔁 TrendSpider about AAPL]
[c2_2297] (twitter, DIA) Nice to see everyone getting on the $DIA & $IWM train Only took $QQQ selling off 7 points🤣 You're about a week late but hop on we still have plenty of room while people dream of $SPY $440/$450 next week [ATMSnipes tweeted about SPY, DIA, QQQ, IWM]
[c2_2298] (twitter, FSLR) $FSLR First Solar Q1 2024 GAAP EPS $2.20 Beats $1.97 Estimate Sales $794.000M Beat $702.175M Estimate [Mr Chart Norris tweeted about FSLR]
[c2_2299] (reddit, MSFT) Microsoft weighs legal action over $50B Amazon-OpenAI cloud deal, raising risk to Azure exclusivity in AI race
