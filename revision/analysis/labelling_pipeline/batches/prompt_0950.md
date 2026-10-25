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
----- ITEMS (50: c2_0950 .. c2_0999) -----
[c2_0950] (reddit, TM) Car loan in 550 credit score Looking to buy a Toyota Corolla 2022 for 25-35k but am curious on everyone’s opinion. My credit score is lower since when I was abit younger I applied for loans which affected my credit score. I currently have a personal loan with 5k left and never missed a payment. Make around 50k a year as I’m part time and studying. I don’t want to apply for anything and not get approved whilst affecting my credit score.
[c2_0951] (twitter, GE) $GE (daily) Bearish false breakout reversal [kpak tweeted about GE]
[c2_0952] (twitter, BA) Watching $COIN $TSLA $BA to name a few today! 👀 [optionsprochick tweeted about COIN, BA, TSLA]
[c2_0953] (news, GS) A Confused Goldman No Longer Expects S&P To Drop To 2,400 But Sees No Market Upside Either
[c2_0954] (news, ABR) Dividend Increases: May 6-10, 2019 (Part 3: Remaining Sectors)
[c2_0955] (reddit, BTC) Do you guys believe BTC is only a store of value? Hmm, how people see Bitcoin evolving from here. Is it really just digital gold, mainly for holding and preserving value? Or do you think it still has room to grow as something more, like a medium of exchange or a base layer for broader financial infrastructure? With all the innovation happening in the space, I wonder if we’re limiting BTC by labeling it strictly as a store of value. What’s your take?
[c2_0956] (reddit, MSFT) Bought 9k in MSFT/NVDA weekly puts into the close - figuring an AI crash will happen after people spend all weekend ripping GPT5 a new ahole
[c2_0957] (twitter, TSLA) $TSLA Wedbush analyst Daniel Ives has maintained an "Outperform" rating on Tesla and raised the price target from $310 to $350. [Mr Chart Norris tweeted about TSLA]
[c2_0958] (reddit, RDDT) RDDT retard RDDT 250 Call Oct’ 26. Loss 90% of 27k so far. Believe it or not, I was up 90% early July. And I was like “chill bro, diamonds hands to the moon, trust RDDT management, generational wealth opportunity” - probably after a few beers 🍺 - never drink and trade anymore. Then, [u/spez](u/spez) talking about “choppy traffic”. The rest is history. I’m a true retard. Fuck [u/spez](u/spez)
[c2_0959] (twitter, RSI) $COIN 213 almost > [@kholov23]( > $COIN Daily Chart > > Breakdown / Bounce / Retest Scenario > > Ascending broadening wedge BREAKDOWN, continuation to the downside > > A close above 210 with RSI & MACD EXTENDED to the downside ( short term reversal looks possible ) [Mr Chart Norris quote tweeted Mr Chart Norris about COIN]
[c2_0960] (twitter, SNAP) This months stock calls $TSLA 60% TP $TWLO 60% TP $SNAP 20% TP, re-entry 20% TP $VFF 30% TP (25%) $MVIS $17% TP Most entries during loudest FUD (recession calls) Out of all positions. Patient now till opportunities show up again. Screening soon. #Timing and #Patience [Anbessa100 tweeted about MVIS, SNAP, TSLA, PATIENCE, TIMING, TWLO, VFF]
[c2_0961] (twitter, COST) $COST 🟦🟦 Costco Wholesale Q3 2024 GAAP EPS $3.78 Beats $3.70 Estimate Net Sales $58.52B Beat $58.07B Estimate [Mr Chart Norris tweeted about COST]
[c2_0962] (reddit, BABA) Panic mode activated ? G7 countries are considering releasing 400 million barrels of crude oil reserves. So the problem is big enough that they’re reaching for emergency reserves ?
[c2_0963] (twitter, MSFT) $MSFT Rebroke the flag today on decent volume. If it can get momo above $314.48, it looks like it can move higher towards $320.70, $328.12 resistance next. Alot of support to watch at 311/310 area still. [optionsprochick tweeted about MSFT]
[c2_0964] (twitter, COIN) $COIN - Profits secured, played the risk and got some nice gains on the stock and calls. I'm closing ALL my positions though cause I can see it pulling back more. SEC seems to be going after them more. Could have been a relief rally followed by a larger dump. [IncomeSharks tweeted about COIN]
[c2_0965] (twitter, META) META'S TWITTER RIVAL THREADS OVERTAKES CHATGPT AS FASTEST-GROWING PLATFORM $META [*Walter Bloomberg tweeted about META]
[c2_0966] (twitter, BTC) Day 182 of sideways chop. Mt. Gox offloaded nearly 100,000 #Bitcoin while the ETFs accumulated nearly 300,000 coins -- and many people got shaken out of their holdings. Been quite the ride. Expecting new highs in Q4. [Jelle tweeted about BTC]
[c2_0967] (twitter, UBER) $uber losing daily clouds flow on the 7/26 65p today [StonkTrump tweeted about UBER]
[c2_0968] (reddit, SPY) VIX has dropped. Time to buy puts if you're bearish Edit: I'm talking about puts on SPY / QQQ or other stocks, NOT on VIX The market is up today, but check the VIX. Also, check the low volume on SPY. It is struggling to go up. If you are bearish about all this long term, it may be time to buy some puts. Personally, I feel this is a bull trap, and I bought SPY 500P JUN 20th expiry. Do your own research, and play accordingly. Good luck!
[c2_0969] (reddit, RENT) ABN or PAYG? My new role gives me the opportunity to choose between ABN and PAYG. I am not currently in Australia, but I will be relocating to Sydney soon as I have accepted this offer as a software engineer. What are the opinions and complexities around registering a company and getting invoices through ABN ( a PTE ltd) I have around 2K to shift outside Australia every month for maintaining investments. 650-700 Rent per week. Want to bring all devices for work.
[c2_0970] (reddit, BRK-B) Berkshire’s operating earnings jump 34%, Buffett buys back no stock and raises cash hoard to $381 billion
[c2_0971] (reddit, PLTR) 85% CAGR on my Covid Refunds My biggest investment was 1800 shares on PLTR at 12.67. Sold half at 116. Other half was 67. No regrets Now, my biggest holding is just AMD because everyone here says its advanced money destroyer. I am also starting covered calls and cash secured poots to sell to gamblers My goal is to franchise a Wendys so I can help out mi amigos
[c2_0972] (reddit, UBS) UBS gets final approval for full-banking US charter [ Might be related to [past UBS moves]( signaling a potential move to the US, mainly thanks to its convenient capital requirements.
[c2_0973] (twitter, SPY) Patiently waiting to screenshot the bear posts $SPY [JMA aka The Milkman🥛 tweeted about SPY]
[c2_0974] (twitter, AIXI) $AIXI they’ve been dropping PR’s constantly over the past few weeks. This news looks attractive this morning: Xiao-I Corporation Launches Its New Product "Hearview"- "AI Empowered Glasses for Deaf People" [BullishTrade tweeted about AIXI]
[c2_0975] (twitter, CF) $CF Short is up 12% from entry and approaching first main target!! Beauty of a trade here. Biggest win all month so far :) [Nebraskangooner tweeted about CF]
[c2_0976] (reddit, META) META: WhatsApp introducing Message Translations on Android and iPhone "With more than 3 billion users in over 180 countries, we’re always working to keep our users closely connected, no matter where they are in the world. But we understand that sometimes language can be a barrier to getting things done or expressing how you truly feel. That's why we’re excited to bring message translations to WhatsApp, so you can more easily communicate across languages."
[c2_0977] (twitter, CRM) $SNAP - MS maintains overweight, PT $55 to $24 $TSLA - GS maintains buy, PT $1200 to $1000 $QS - GS maintains neutral, PT $17 to $12 $CRM - PT $208 to $218 at Barclays, $225 to $235 at Wells Fargo $XOM - PT $98 to $111 at Barclays $NKE - PT $165 to $150 at Wells Fargo [rachels_44 tweeted about SNAP, TSLA, QS, CRM, XOM, NKE]
[c2_0978] (reddit, RH) Is Robinhood now paying interest on CSP collateral? Saw this post and was wondering if it was legit or not. If it is, definitely gonna be getting RH Gold to take advantage.
[c2_0979] (reddit, ETH) Crypto market cap manipulation ? Seen it drop for a few seconds today was this market makers or what
[c2_0980] (reddit, COIN) Looking for metal commemorative BONK coin... To celebrate the upcoming burn, does anyone know where to buy a metal BONK coin as a memento? Or how to make one?
[c2_0981] (reddit, MSFT) MSFT earnings MSFT earnings less than a month out. Prediction that it will run up to 462 just before earnings. Microsoft will post record earnings for another quarter. It will proceed to crash back down to 350 or lower. Source: my ass. I’m holding 160 shares at 460 so it’s definitely gonna fuck me.
[c2_0982] (twitter, NVDA) Still down for $nvda, no signs of up yet. > [@HackermanAce]( > $nvda down only since open with no signs of strength, > > Structure flip and i grab calls, if not keep sending it down. [HackermanAce 🔁 HackermanAce about NVDA, NVDA]
[c2_0983] (twitter, TK) Joined for .20+ > [@TK_Tradess]( > Adding some $BBI down here, has the potential to be the next runner like we got for $COMS. High risk high reward [Ultra_Calls 🔁 TK_Tradess about BBI, COMS]
[c2_0984] (twitter, CF) $CF My favorite looking chart right now. Looks to be breaking down on high sell volume. Great risk:reward as invalidation is less than 4% above [Amdtrades 🔁 Nebraskangooner about CF]
[c2_0985] (twitter, AAPL) $AAPl I really don't think I can make these stock calls any easier [Shardi B Trading #teamshardi tweeted about AAPL]
[c2_0986] (twitter, ROKU) $ROKU what a run through 59 break! > [@rachels_44]( > $ROKU nice open. Just tested the daily 50sma, has been beaten down. Through 59 can go [rachels_44 🔁 rachels_44 about ROKU]
[c2_0987] (reddit, NVDA) What does it mean when an entire option chain has lower Asks than Bids? I’m looking at Nvidia’s options, calls and puts, and seeing this for every one that I check. It was happening before market close, too. For instance, the May 2^(nd) $70 Put has a Bid of $1.11 and an Ask of $.91. Has anyone seen something like this happen in the past? Edit: Turns out it's not every single contract: [
[c2_0988] (twitter, AIR) $CHWY 16 - > 29 / I got a lot of negative comments regarding Chewy , but yeah it is what it is > [@kholov23]( > $CHWY Monthly Chart > > Tightening falling wedge pattern, where price is approaching the lower boundary of the wedge > > And yeah this is the spot where you open your bags but not in the AIR > > Now prove me WRONG [Mr Chart Norris quote tweeted Mr Chart Norris about CHWY]
[c2_0989] (reddit, NFLX) NFLX Option help I am new to Options, If NFLX goes to $1257 what will be most profitable Option? ITM, OTM, expiration? I trade shares and that is easy for me. Few times I tried Options and the time killed me. Thank you!
[c2_0990] (twitter, SNAP) I took profit on half of my open stock positions this morning and fully closed $TSLA, $WFC, $BAC and $SNAP [Nebraskangooner tweeted about WFC, BAC, SNAP, TSLA]
[c2_0991] (twitter, MA) $XLE, is the #1 performer over the past year. This pattern is near completion, suggesting a big move is near! Bulls do not want it to break support & 200-MA line!!! #IBDpartner @MarketSmith @IBDinvestors [the_chart_life 🔁 KimbleCharting about IBDPARTNER, XLE]
[c2_0992] (reddit, COIN) Is my unfunded satoshi ravenbit coin worth anything near the $10k guy is selling his for? Mine is numbered 0091. Dude has one on ebay for $10k. Is mine worth anything? I still have stickers in tact. I never funded it.
[c2_0993] (reddit, ETH) what your take on high yield ETFS ? Last year was the best year for high-yield ETFs like Ymax and BITO, but now I have trimmed them, and so far this is my projected income... im using stackzai . com tracking
[c2_0994] (reddit, TSLA) Daily Discussion Thread for May 19, 2025 This post contains content not supported on old Reddit. [Click here to view the full post](
[c2_0995] (reddit, BTC) General Mills Hold or Fold? The App "Dividend Fox" says Holding is better. What do you think?
[c2_0996] (news, AMZN) Verint, Alcoa, Amazon, Walmart and Target as Zacks Bull and Bear of the Day
[c2_0997] (twitter, ES) $ES_F $SPX $SPY 3H -- March21 The bearish rejection today looked good until that final ten minute 30pt squeeze. Probably a tad higher before this leg finishes. Bulls still have a ton of work to do if they're going to get through this zone Also - $RTY_F giving hints 👀 [@rbc_trading]( $ES_F $SPX $SPY 3H -- March18 Bulls got it into the zone, no sign of reversal yet. Let's see what next week has in store. Have a great weekend 🍻 [rbc_trading 🔁 rbc_trading about ES_F, SPX, SPY, RTY_F, ES_F, SPX, SPY]
[c2_0998] (reddit, BTC) How do I accept BTC payments via my website, Offerup, & facebook? Maybe this is a stupid question but I have to ask. I want to know from people real world experience. I would like to allow BTC payments from my customers. This would include payments direct from my website store and via sites like Offerup and Facebook Marketplace. Do I just provide my wallet address? Thanks in advance.
[c2_0999] (twitter, LMT) $LMT Get above and breakout is confirmed [Amdtrades tweeted about LMT]
