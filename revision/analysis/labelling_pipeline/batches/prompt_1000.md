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
----- ITEMS (50: c2_1000 .. c2_1049) -----
[c2_1000] (twitter, GEO) $GEO (Michael Burry last bag) The crash expert became pessimistic and sold his stocks in June (right before the bounce) As a #HTF thinker this could still age well, but his last bag $GEO actually looking good when you zoom out. A rally from here to resistance would be 550% [crypto_scar tweeted about GEO, GEO]
[c2_1001] (reddit, TSLA) Tomorrow’s play TQQQ 400 60 puts (4/11) and TSLA 280 puts (4/11) Still running hot for more than a month. Up over 400k in 5 weeks ( previous posts) This morning I made a gain post of 135k and intraday made another 30k Picked these up on the closing. As i have mentioned in the comments in previous posts : My only (broken clock) thesis is that Nasdaq will hit 16600 levels going back to August 2024. So all my entry points are local maxima. **Previous posts:** [ [ [
[c2_1002] (reddit, EOD) Profit is profit (QQQ Puts) Unfortunately not degen enough to hold until end of day (these were worth $75k EOD btw)
[c2_1003] (twitter, ENPH) $ENPH Daily Chart Breakdown / Bounce / Retest Scenario Rising wedge breakdown , psych level on watch / 100 / [Mr Chuck Norris tweeted about ENPH]
[c2_1004] (reddit, BLK) Buying ETH with 401k brokerage account Hi all, I’m planning to put all my 401k money into an ethereum ETF. I’ve seen a bunch of ETFs managed by blackrock, fidelity & grayscale. Can you please help me pick the best eth ETF that tracks ETH to buy? Thanks.
[c2_1005] (twitter, GME) We talked about $GME on live stream yesterday and how it was no good for longs after the 1H structure break. GME selling millions of shares didn't help either > [@wallstreetbets]( > The loss porn in our X community is incredible, God bless the apes 🦍 > > join : [Nebraskangooner quote tweeted wallstreetbets about GME]
[c2_1006] (reddit, BULL) Everything nonstop dumping Can’t even buy dips anymore in this scam market.
[c2_1007] (twitter, NVDA) $NVDA $15 move after this tweet, big question will be if we can confirm this week... this was a nice trade thus far, I remain long with a stops in profit > [@ShardiB2]( > $NVDA > > Printing a rare weekly reversal candle not seen since the pandemic [ShardiB2 🔁 ShardiB2 about NVDA, NVDA]
[c2_1008] (twitter, ES) 4688 is key pivot for me...if that goes...should drop to 4640s $ES [trader1sz tweeted about ES]
[c2_1009] (reddit, ETH) Need help deciphering if this is a scam. So, I made a post earlier today about getting emails where an item on opensea sold and they are trying to deliver just over 1ETH to me, but I don't believe this is real. My gut is telling me this is a scam since the email is not an open sea email and the site it brings me to is etherscan. Screenshots are above and any input would be nice. Earlier post
[c2_1010] (reddit, AIXI) $AIXI still a great opportunity . So many dark pooled shares. could squeeze to 40 or $.60 and with a $AAPL settlement News multi dollars. Imo Due for a super squeeze .. keep on watch
[c2_1011] (twitter, NXPI) Oh semiconductor $nxpi beat ERs > [@DeItaone]( > $NXPI Earnings [HackermanAce 🔁 DeItaone about NXPI, NXPI]
[c2_1012] (reddit, BTC) Jefferies‘ Wood Drops Bitcoin on Threat from quantum computing [ Institutional money reacting on quantum threat - too early, overreacting or right timing. Should we Take quantum threat on crypto more serious, what do you think?
[c2_1013] (reddit, RENT) Don’t know what to do could use some advice I’m 26, I have about 24,000 in the s&p 14,000 in company stock (UPS) and 35,000$ in cash I want to invest the 35 but don’t know the right time I keep waiting and feel like I’m just missing out. Was gonna use the money for down payment on house but I’m starting to think that renting would be smarter as my portfolio grows. any advice ?
[c2_1014] (reddit, COST) Costco sells 'record-breaking' amounts of gas as fuel costs soar
[c2_1015] (reddit, BTC) Don't do CFD Please for the love of god, stay far away from it, or if you use it, use tight stop loss, don't be stupid like me. Also, don't go to sleep (like I did) with risky open positions. I could have avoided this. Just woke up to my positions being closed. This day completely crashed me..I can't tell anyone about this, taking it to the grave. Another advice, DON'T GAMBLE ON PENNY STOCKS ON CFD.
[c2_1016] (twitter, AMD) $AMD is consolidating on this $176.84 core support. This is a pretty big range for support. Keep an eye on it if it can get momo> $183 👀 Resistances: $185.33, $190.34, $194.99 [Pro Trader - Kay tweeted about AMD]
[c2_1017] (twitter, SPY) $SPY if we struggle with supply at 417 like we did at 458, an abrupt move down looks likely (like last time). No predicting, but preparation is key. Shoutout to @ThetaWarrior pivot as well [ItsSqueezy tweeted about SPY]
[c2_1018] (twitter, ATAI) Picture perfect Top Goon X buy approaching alert playing out on the $ATAI chart. For those of you who know the TG X rules this is beautiful [Nebraskangooner 🔁 TopGoonX about ATAI]
[c2_1019] (reddit, NVDA) $124,000 to $500 - Liberation Day - 99.5% LOSS! I started this year with $124,000 and the lowest point was after liberation day. NVDA calls cooked me good and I hit 99.5% LOSS with a portfolio value of $526. I had to deposit $10,000 to pick myself up from there. I've taken out $50,000 since then. Let's see how the 2nd half of the year goes.
[c2_1020] (twitter, SSNC) Successful launch for $SSNC today, taking my initials and staking them for the $PTON airdrop since GCV has invested in both > [@SatoshiSync]( > Did you know that you can stake your $SSNC tokens to gain stable rewards? > > > > There are two possibilities for you to choose from: 3 months long with 15% APR and 6 months with 20% APR > > Importantly, there will be a snapshot taken, on the 8th of April. > > Stakers… [Zoran Kole quote tweeted SatoshiSync about SSNC, PTON]
[c2_1021] (twitter, ES) $ES_F (1H) [@kpak82]( $ES_F (1H) [kpak82 🔁 kpak82 about ES_F, ES_F]
[c2_1022] (reddit, TJX) TJX – why street so bullish Why is the street so bullish on TJX – valuation’s so high already, plus it has a fair amount of debt, can’t justify another 20% upside this upcoming year like it’s done in the past. Am I missing something that the street’s seeing?
[c2_1023] (reddit, NFLX) Glorious lol Follow up. Actually had 7 spot contracts overnight. Ending on a high note. Letting 1 run for a bit longer but will book soon. Going to take it all out and put it towards debt. So much love to all the support and hate I got on yesterday’s post. ❤️ and no I am not some nepo rich kid that’s getting an allowance. I have a full time job with two side hustles and in a ton of debt now.
[c2_1024] (twitter, TW) $SPY First trade of the week from our algo. Following the entry: Profit available: +5.52pts Heat taken: -1.51pts @TWEliteAlgo Strikes again. 🤖 [ThetaWarrior tweeted about SPY]
[c2_1025] (reddit, WY) Senator Lummis pushes $300 de minimis crypto tax exemption to make Bitcoin a spending currency Sen. Cynthia Lummis (R-WY) is pushing a $300 de minimis crypto tax exemption — capped at $5,000 annually — to let Americans use Bitcoin and other digital assets for everyday purchases without triggering capital gains taxes.
[c2_1026] (reddit, GOOGL) Bitcoin Crosses Google to Become Fifth-Largest Asset as Fed Rate Cut Bets Rise
[c2_1027] (twitter, SPY) $SPY $SPX short play + 50points at the lows today before basing and going on a ridiculous rally. Always lock in profits and raise profit stops. [kpak82 tweeted about SPX, SPY]
[c2_1028] (twitter, ES) In context: #BTC -4.5% $ES -6% $NQ -11% [MacroCRG tweeted about ES, NQ]
[c2_1029] (reddit, TSLA) Sold my 295 TSLA Call NOV 21 25 today. Been holding since April 21
[c2_1030] (twitter, AAPL) $AAPL Gotta hold this candle base otherwise will sell off quickly [Nebraskangooner tweeted about AAPL]
[c2_1031] (reddit, MSTR) Keeping it safe? I bought 10% of 1 btc before the last halving thinking to make some quick cash. Started reading more about btc. Based on companies, governments and states holding. I’ve decided to hold half indefinitely, the other half I may sell. I’m not sure if it’s safe enough sitting in the exchange platform or should I move it somewhere else like a wallet?
[c2_1032] (twitter, SB) 🧵 For about a month, we have been quietly supporting $SB price and managing buybacks via DCA. During that time, we accumulated over 2500 $SB with the recent larger buyback for 100 $SOL #Solbank #Solana #DeFi [ANBESSA retweeted Solbank (🏦, 🏦) about SB, SOL, SOLBANK, DEFIPERP, SOLANA]
[c2_1033] (reddit, GME) What percentage of GME's profit came from operations of the core business? I'm trying to figure out how much of GME profit actually came from core busineas activities (selling games, collectibles, etc) as opposed to, say, interest payments on investments, and other unrelated items.
[c2_1034] (reddit, NVDA) spy analysis 3/24 **spy 3/24 gex read 🔴** spot $653.87, already below the $655 king node and $659 gex flip. dealers are in negative gamma — they amplify moves, not stabilize them. dex shows dealers actively selling, vol rising. $665 call wall ($35.86m) + $670 king node above = hard ceiling on any bounce. every rip is a fade. targets: $650 → $640. only flip bullish on reclaim of $659. *signal: strong buy puts. entry near $665, target $640, stop $672.50*
[c2_1035] (twitter, HUT) $HUT Benchmark analyst Mark Palmer initiates coverage on Hut 8 with a Buy rating and announces a price target of $12 [Mr Chart Norris tweeted about HUT]
[c2_1036] (twitter, AMC) Sure would be fun to see the hedge funds squirm if $GME and $AMC did another leg up [Nebraskangooner tweeted about GME, AMC]
[c2_1037] (twitter, DEFI) $DEFI pay attention on this chart in next days .... Understand? [eliz883 tweeted about DEFI]
[c2_1038] (reddit, SPY) SPY and NVDA, AAPL Monday movement How High do you think these three are going Monday? I bought some cheap SPY 581 calls and I’m trying to figure a good baseline percentage to sell at. I always get burned because I get greedy with my calls and puts.
[c2_1039] (twitter, EMA) $EURUSD looking solid for a bottom Bullish market structure intact and bouncing hard from 200EMA [MacroCRG tweeted about EURUSD]
[c2_1040] (reddit, BTC) BTC at extreme fear 15 Last recent time this year where fear was 15, when BTC was at ~83k, after which it rallied to ATH. This time any same? 👀
[c2_1041] (twitter, LPL) $LPL Watching for pending breakout. [Nebraskangooner tweeted about LPL]
[c2_1042] (reddit, HESM) ARCC bull and bears I have started looking into high yield stocks after seeing some success with my purchases of HESM last year and I’d love to know your opinions on the stock. The good and the bad. Thank you in advance!
[c2_1043] (twitter, SPY) This week's plays in the OPCC Group. In effort of full transparency, all plays, wins & losses are always shown. Play of the week goes to $SPY +130% closed profit📉 You can check out more in the blog below. 😀 [optionsprochick tweeted about SPY]
[c2_1044] (reddit, BULL) GLD and SLV Put calendars Both metals have taken a big hit, and IV on the April 24 puts is elevated sharply while May 1 IV is roughly equal in value and stable at 426 and 67 strikes respectively Pretty easy money on the short duration calendar here with a hedged delta
[c2_1045] (twitter, GOOGL) I don't push my luck & succumb to greed. I know when to walk away when there isn't an edge to my system. 3 plays today & I'm ending it on a high note. I encourage the same always. The market will be here tomorrow. $GOOGL +244% $AMD +116% $SPOT +129% Closed profit. [optionsprochick tweeted about GOOGL, AMD, SPOT]
[c2_1046] (twitter, LRCX) $LRCX Big break higher from it's base. Opened position upon the initial retest, locked in gains yesterday at the 1.618 Fib extension. $10.5 -> $45.64 [+334.67%] 💰 [Theta Warrior tweeted about LRCX]
[c2_1047] (reddit, META) Survived the GOOGL stalemate Although i lost all the gains I made since April, I got the 2500 I started with back by not selling. Also avoided the classic warsh rugpull. Im happy. I learned to use less money to win more than I was in the beginning, so if I can make 20k in 2 months as a greenhorn, then I can definitely do it again now that ive been witnessing warshs casino. Im happy
[c2_1048] (twitter, HOOD) $HOOD Flying today! Up over 50% from entry now [Nebraskangooner tweeted about HOOD]
[c2_1049] (reddit, AMZN) 90k basis - Income focused account going off the rails Saved 100k for the next "event" and ended up sending $10k to my mom and rolling with 90k to start. Iran started sending us to the 200ma on the daily. Event has arrived. Began loading @ SPY 650. Rode it to 630 and started using dividends to grab options to gen more $ for more shares. Really just wanted some weekly cash... but I can't complain. Cheers to another week degens 🥂 Pre-emptive: NAV EROSION! I know, i know.
