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
----- ITEMS (50: c2_0300 .. c2_0349) -----
[c2_0300] (twitter, AMD) SEMIS NEW 52 WEEK LOW 🔬 $AMD $NVDA $TSM $MU $TXN $INTC $ASML $LRCX $AMAT $ADI $SMH > [@ThetaWarrior]( > $NVDA & $AMD 52 WEEK LOW [ThetaWarrior 🔁 ThetaWarrior about AMD, NVDA, TSM, MU, TXN, INTC, ASML, LRCX, AMAT, ADI, SMH, NVDA, AMD]
[c2_0301] (twitter, PLTR) -35% later... [@MFHoz]( $PLTR As rates continue to rise, this tech stock looks very vulnerable. IMO it will repeat what it did from Feb into May. The C wave points towards a sell-off to 12-13$ in the coming months. [MFHoz 🔁 MFHoz about PLTR]
[c2_0302] (twitter, TSLA) $TSLA what’s your problem with rising wedge ? [Mr Chart Norris tweeted about TSLA]
[c2_0303] (reddit, TSLA) Today same day expiry options After 10am Powell speech, I see the gains are SPX \~500%, TSLA \~500%, but NVDA \~80%. Can anyone explain why the difference?
[c2_0304] (twitter, XYZ) Reminder that if you’re seeing a Head & Shoulders on your $SPY timeline from traders I told you they would show you this before they told you > [@ATMSnipes]( > @AlexanderSzews I like $394 if the alt plays which is XYZ up from $383, sub-divided by ABC. > > But since most bears and even the level-headed traders I know want the gaps to get filled. > > Me think that won't happen as they trap shorts with a fake H&S and they run the highs for liquidity. > > again not… [ATMSnipes 🔁 ATMSnipes about SPY]
[c2_0305] (twitter, BABA) Ppaaaiiiinnn I am missing China! Look at $fxi reeee As far as $baba goes, if the pull backs stay above support ill grab. I dont wanna chase 12%+ pump . > [@SirMik3]( > @HackermanAce thoughts on BABA calls? [HackermanAce 🔁 SirMik3 about FXI, BABA]
[c2_0306] (reddit, BTC) Investment plan if J Powell Fired Let's say trump succeeds in firing Jerome Powell, interest rates are lowered due to pressure from Trump and we face likely hyperinflation and further devaluing of the US Dollar. What would be a wise investment decision? Would this be a point where you should pull out of the US Stock market and invest in Gold or land or Bitcoin?
[c2_0307] (twitter, NUE) $NUE Breakout continuation into new highs as steel names continue to show lots of strength. Second target at $150. [@Albert_trades]( $NUE Retested the breakout area, now giving a flag look. Should continue to see higher as long as it can hold 130's [Albert_trades 🔁 Albert_trades about NUE, NUE]
[c2_0308] (twitter, TEAM) $TEAM weekly close to trendline resistance 🧐 MACD bullish here [rachels_44 tweeted about TEAM]
[c2_0309] (reddit, ETH) ETH vs. BTC: Ethereum’s Superior Monetary Policy Fantastic overview when it comes to security of a blockchain.
[c2_0310] (twitter, TIVC) @TDAmeritrade posted an incorrect headline on $TIVC it released "Q2 sales $268k miss on $460k estimate thats bogus... sales were $528k ... +15% higher than estimate $268k revenue was Q2 2021... this quarter is +97% to $528k [Bullish_Trades 🔁 simbatrading about TIVC]
[c2_0311] (twitter, TSLA) $TSLA would bounce nice if it can reclaim this 30m breakdown [Nebraskangooner tweeted about TSLA]
[c2_0312] (twitter, GOOGL) $GOOGL Weekly case study on large rounded consolidation breakouts and the 1.618 Fibonacci Extension. 🎯 Cc: $AMZN [Theta Warrior tweeted about AMZN, GOOGL]
[c2_0313] (twitter, TT) Last night I had a dream that $TT stock pumped strong. So random. [Nebraskangooner tweeted about TT]
[c2_0314] (reddit, AP) It Mario Monday Ya'll #AMC After a Weekend full of angry bears, I'm glad I was able to decompress with a few cold ones and I will be continuing to hodl the line with ya'll and supporting our company with a little Mario action later in the week. Fired up for the movie and loving life. Bi and Hodl and enjoy life my fellow APEs
[c2_0315] (twitter, NVDA) There we go… > [@MFHoz]( > In case you wonder why $NVDA has been so strong despite the sell-off. > > It wants to reach the upper resistance line before falling back down to $110 per share. [HOZ quote tweeted about NVDA]
[c2_0316] (reddit, MRVL) Thanks for the shoutout daddy Jensen $MRVL Assuming everyone is holding thru s&p500 announcement Friday and then earnings next week?
[c2_0317] (twitter, TSM) $TSM Ripping.. [Don't follow Shardi B if you hate Crypto tweeted about TSM]
[c2_0318] (twitter, EMA) $AAPL Weekly falling wedge breakout! Apple gave all the clues last week for continuation in the indices, as it held up all week and continued to make new highs. Continuing to watch this name for clues on the indices and other leading names. [@Albert_trades]( $AAPL Weekly falling wedge setup off the 50 EMA. Apple should give clues if market is going to follow through. [Albert_trades 🔁 Albert_trades about AAPL, AAPL]
[c2_0319] (reddit, AI) Snap to cut 1,000 jobs after activist pressure, bets on AI efficiency
[c2_0320] (reddit, BMNR) BMNR SITUATION, IM HERE TO SHARE MY LOSS PORN! When will it recover? Time for a rotation market!!!
[c2_0321] (twitter, RBLX) $RBLX Symmetrical triangle attempting a breakout. [Theta Warrior tweeted about RBLX]
[c2_0322] (reddit, RENT) HPE, the printer company gains I park my car at my house at 3:57pm and open the app. Why not? Let's pull the lever at the slot today. Cha-chong 🤑
[c2_0323] (reddit, GOLD) Taking one for the team I’ve held this position for a while, one of the first ones I’d taken getting into investing. Turns out I sold it at a terrible time 😍
[c2_0324] (reddit, CI) CIBanco Forced Shut Down And Liquidation in Mexico — What You Need To Know.
[c2_0325] (twitter, PTON) RT @snorlax_uw: $PTON bearish beluga whale 33p 1/21/22, ~4,000 contracts @ 1.54 [Honeydew0101 tweeted about PTON]
[c2_0326] (twitter, RIVN) Watching $ROKU $DDOG $RIVN to name a few today! 😀 [optionsprochick tweeted about DDOG, RIVN, ROKU]
[c2_0327] (news, A) IBD Rating Upgrades: Agilent Technologies Shows Improved Relative Price Strength
[c2_0328] (twitter, SPY) Extremely bullish signal [Dyme retweeted The Prophet Reborn about SPY]
[c2_0329] (reddit, GOLD) Trump directed millions of dollars to support coal using emergency powers. Bullish flows picked up a couple of days prior in Warrior Met Coal (HCC) A couple of days piror to the US administration announcing a $700 million plan to upgrade coal plants, support new facilities and export infrastructure, bullish flow in the fossil fuel companies Warrior Met Coal (HCC) and Peabody Energy (BTU) picked up. Both of the stocks are up on the announcement.
[c2_0330] (twitter, IHS) every uranium stock on my watchlist has huge high timeframe IHS's or bull flags with the right volume profile. i think #uranium is about to go on that long awaited bull run. $uuuu $pdn $dyl $lot $pen [SmartContracter tweeted about UUUU, PDN, DYL, LOT, PEN]
[c2_0331] (news, A) Agilent Technologies (A) Q2 Earnings and Revenues Miss Estimates
[c2_0332] (reddit, AGX) Covered call options on argon, AGX. Straddle? Made some money on covered calls as it jumped 130 points this morning. How would you work a straddle on the stock, given the peaks and valleys that this stock does religiously?
[c2_0333] (reddit, AMD) AMD flags $800 million hit from new US curbs on chip exports to China
[c2_0334] (twitter, BB) Binance will list @bounce_bit in a new format of Launchpool - Megadrop In since day 0, we cook 🤝 $BB [Mac 🐺 tweeted about BB]
[c2_0335] (reddit, ETH) What do you guys think about my dividend etf portfolio. Any thoughts? Give a score to my portfolio out of 10
[c2_0336] (reddit, ETH) ETH price to $5K next? Ethereum rallied 120% the last time this happened
[c2_0337] (twitter, SPY) $SPY strong 4h demand. As long as it holds, long play is still valid [kpak82 tweeted about SPY]
[c2_0338] (reddit, SOFI) What's wrong with SOFI? Haven't read the muddy waters analysis on SOFI, anyone who read it might give a tldr?
[c2_0339] (twitter, COIN) $COIN - First real resistance we've seen. Stuck to my plan for a solid 100% play. Wouldn't be surprised to see it run more but usually when a chart looks really bullish I regret not selling. (Same reason I bought when it looked really bearish). [IncomeSharks tweeted about COIN]
[c2_0340] (twitter, QQQ) $QQQ potential pullback entry for longs if 375 holds today [BT tweeted about QQQ]
[c2_0341] (reddit, BTC) Does daily investment make sense? I’ve listened to some podcasts that daily investment is a “safer” option. If I, for example, invest $5 every day in S&P and Nasdaq each, is it a feasible “safer” option or will I just be locking my cash in an asset unworthy of waiting with low profit? I would say I’m pretty patient and I don’t have any large costs to cover, so I’m thinking small daily investment long-term.
[c2_0342] (twitter, ES) Short-term momentum getting a little diverge-y again here as we come into the October highs. Do we grind through it or roll over? $SPX $SPY $ES_F [Ian McMillan, CMT tweeted about SPX, SPY, ES=F]
[c2_0343] (reddit, BULL) Bull vs Bear Market How low or how high does the market have to be, and for how long, to officially say we're in bull or bear market?
[c2_0344] (news, ABR) Among Best Dividend Stocks, This Is Why Arbor Realty Trust Has A Whopping Yield
[c2_0345] (twitter, SPY) Half of Twitter: "I was the only one to call for a sell from 4100" Me after proposing the bounce... the sell... a week ago... > [@ChandlerTrading]( > $ES_F $SPX $SPY 6H - May12 > > One possibility to watch for... > > A corrective move to overhead supply (before reaching monthly demand) would set up another short opportunity [ChandlerTrading 🔁 ChandlerTrading about ES_F, SPX, SPY]
[c2_0346] (reddit, NVDA) Deepseek unveils updates to R1 model It’s going to be an interesting trading day tomorrow with a mix of good and bad news. The reciprocal tariff pause by the courts will certainly alleviate supply chain costs though there’s a high likelihood the pause will be short-lived as the ruling will certainly be appealed by the administration. Deepseek updates could lead to unease in stocks of US AI competitors though positive NVDA earnings have so far uplifted those.
[c2_0347] (reddit, BULL) Rip gays The gays can handle a lot but I guarantee they didn’t bring the right lube for this.
[c2_0348] (reddit, GOLD) Stocks slide, oil gains as Trump's Iran deadline approaches: Live updates
[c2_0349] (twitter, COST) Today's plays $AMD 122c +72% closed profit $COST 580c +52% closed profit $VLO 100c +40% closed profit $SPY 458p +26% Closed profit It's been a slow but profitable week so far, Take what the market gives you! 😀 [optionsprochick tweeted about AMD, COST, VLO, SPY]
