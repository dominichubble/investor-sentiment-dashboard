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
----- ITEMS (50: c2_0700 .. c2_0749) -----
[c2_0700] (reddit, ETH) Ether more ‘like a memecoin,’ says trading firm as ETH drops 45% YTD
[c2_0701] (twitter, QQQ) $QQQ #QQQ Two untested levels of caution overhead along with the 200 EMA. [_bread911 tweeted about QQQ]
[c2_0702] (reddit, PLTR) Do I go all in on April 3rd after Liberation Day? Dear reddit, I know im gonna beat myself up if i dont take advantage of the stock markets current volatility, ive conducted some Technical analysis([Xynth]( on stocks i think would be best to go all in on: **TSLA:** [ **PLTR:** [ **MARA:** [** **Technical Indicators (RSI, ATR%, Signal Counts):** [** What do you guys think? Whats the move here?
[c2_0703] (twitter, MS) $NFLX Morgan Stanley analyst Benjamin Swinburne reaffirms an Overweight rating on Netflix and increases the price target from $700 to $780. [Mr Chart Norris tweeted about NFLX]
[c2_0704] (reddit, MCD) McDonald's beats Q1 earnings estimates despite challenging environment with value meals and new menu items
[c2_0705] (news, ABG) Should Value Investors Buy Asbury Automotive Group (ABG) Stock?
[c2_0706] (twitter, ROKU) $ROKU small trendline break with MACD cross on daily. Above 164 can try for a run📈 [rachels_44 tweeted about ROKU]
[c2_0707] (reddit, GS) Predicated 2026 S&P500 ROI Goldman Sachs strategists predict a 12% ROI for the index by 2026. Since we are currently at a 7% rate, the market must maintain its current pace to exceed the 12% ceiling. GS analysts have also factored in a slowdown in their 2026 projections. Additionally, the first half of the year is often front-loaded, followed by a "sideways" period or a cooling-off during the summer. What's your prediction for your portfolio?
[c2_0708] (twitter, SPY) Finally closed my $SPXL $SPY position after a 2 month hold from early November +66%. Amazing way to end the year. Hope everyone is enjoying this Santa rally 🎅 [JMA aka The Milkman🥛 tweeted about SPY, SPXL]
[c2_0709] (reddit, AMD) Is AMD still a good buy at this moment? Hey everyone, I’ve been adding to my tech positions and AMD is one of the names on my radar. With the recent run-up, I’m not sure if it’s still a good idea to add at the current price. How are you all viewing AMD’s valuation and long-term growth outlook right now? Is it still worth averaging in, or better to wait for a pullback?
[c2_0710] (reddit, NVDA) How should I structure my investing during this time? During this period of volatility I want to really maximize my future growth potential. I had around 100k (VOO , VTSAX, and NVDA) in the market prior to the market free fall. I’ve been putting in 1k every few days into VOO while it crashes. Would you guys recommend to just throw a giant lump sum in or keep the ladder investments going? I have another few decades before I exit the market so I’m not super risk averse.
[c2_0711] (twitter, MA) $Oil $WTIC - Update Took the Red-Route, and has slipped its Daily 42 & 150/MA's. Failure to recover quickly will question the 'Truncated-(E)' as Spprt There is a 'Broadening' Plot formed (green). Which I would class as bullish, yet need a signal... [Ace retweeted Sahara about OIL, WTIC]
[c2_0712] (reddit, INR) Guidance on portfolio structure Age:37. Goal: Retirement at age of 60 (preferably) Investment duration: 20+ years Risk tolerance: moderate ICICI prudential nifty 50 index : INR 8000 ICICI prudential nifty next 50 index : INR 4000 Parag flexi cap: INR 9000 ICICI prudential balanced advantage: INR 11000 Nippon India small cap: inr 4000 ICICI prudential corporate bond: INR 4000 Kindly advice on portfolio structure. Also, if there is lumpsump amount - where should I invest?
[c2_0713] (twitter, QQQ) $SPY up 7 straight days $QQQ up 8 straight days [kpak tweeted about SPY, QQQ]
[c2_0714] (twitter, IBM) $ibm lets see if it swings off the monthly mid bbands pops back up, if not , looking to do puts on any bearish rejection at the dashed blue line. [HackermanAce tweeted about IBM]
[c2_0715] (news, A) IBD Stock Of The Day Keysight Technologies Hits Buy Zone As Wall Street Cheers 5G-Driven Growth
[c2_0716] (twitter, QQQ) $QQQ flexing a daily bull flag break [BigCheds tweeted about QQQ]
[c2_0717] (twitter, NVDA) This week's plays in the OPCC Group. In effort of full transparency, all plays, wins & losses are always shown. Play of the week goes to $NVDA +65% closed profit but a 750% Move! WOW! 📉 You can check out more in the blog below. 😀 [optionsprochick tweeted about NVDA]
[c2_0718] (reddit, ASX) Just want to know If I've made a big mistake? I've allocated 100% of the ASX : IVV to a custom portfolio. Would it have been better to buy the actual holding or is it fine as it is? The monthly fee for the portfolio is $4. I don't necessarily want to sell and trigger a tax event just to buy the same holdings again. Does this matter long term or should I keep investing into it as it is. Thanks guys
[c2_0719] (reddit, AAPL) This stock portfolio tracks top companies on the Apple App Store — and it’s winning
[c2_0720] (twitter, ES) $ES Bulls got horny at range highs again didn't they? [StonkTrump tweeted about ES]
[c2_0721] (twitter, AAPL) $AAPL up a bit on earnings [Don't follow Shardi B if you hate Money tweeted about AAPL]
[c2_0722] (twitter, NVDA) $NVDA ripper day. +9 off that 288 break > [@rachels_44]( > $NVDA 280 support zone holding. Earnings next week. Above 288 strong into number [rachels_44 🔁 rachels_44 about NVDA]
[c2_0723] (reddit, SSNLF) 'Give me my money back': South Korean traders' leveraged SK Hynix, Samsung bets unravel after selloff No crying in the casino.
[c2_0724] (twitter, GME) $GME retest processing…… > [@kholov23]( > $GME Daily Chart > > Breakout / Retest processing...... [Mr Chart Norris quote tweeted Mr Chart Norris about GME]
[c2_0725] (reddit, QQQ) Bought 0dte QQQ $721c and 10 minutes later the bottom fell out So I bought 0dte puts and ended up in the green. Gave back all my profits for the year on Friday. Doubled my account between Monday and Tuesday. I'd rather be lucky than smart.
[c2_0726] (twitter, NFLX) $NFLX Weekly coming up to an important inflection point. [Theta Warrior tweeted about NFLX]
[c2_0727] (reddit, BTC) BTC dipping hard? Perfect timing to bump my DCA to $75 daily Back in September I got absolutely wrecked with some ridiculous margin trades, lost a massive chunk of change No point crying over spilled milk though right Strategy now is simple - just keep stacking sats through regular buys and forget about it Bitcoin crashing? mate I'm loving every minute of it, more coins for less cash so I'm cranking up my daily buys These prices are a gift if you ask me
[c2_0728] (twitter, SPY) $SPY puts .82 +100% make sure to lock in profits. > [@kpak82]( > Bought $SPY Aug23 495 puts @.39 > Full risk no stop. > Ie. If normal stop is -30% of 1k position, can use $300 position. [kpak quote tweeted kpak about SPY]
[c2_0729] (twitter, ES) $ES Bearish retest of the structure change level we mentioned before the runup? [@StonkTrump]( $ES Very significant level here This level represents, what, in my estimation, is the last lower high of the downtrend before we entered the range Breaking this level could signal a structure change [StonkTrump 🔁 StonkTrump about ES, ES]
[c2_0730] (twitter, NVDA) $NVDA (daily) failed breakout reversal [kpak82 tweeted about NVDA]
[c2_0731] (twitter, GOOGL) Bullish [@DeItaone]( *GOOGLE CONSIDERING LETTING USERS STORE CRYPTO IN DIGITAL CARDS $GOOGL [CryptoGodJohn 🔁 DeItaone about GOOGL]
[c2_0732] (twitter, SPY) $SPY When breakout retest? [Theta Warrior tweeted about SPY]
[c2_0733] (reddit, CD) Not good with saving So i just found out about Marcus banking and decided to close my eyes and put 2500 in a CD. i don't know how to save money at all if longer than a year. i would LOVE pro tips from everyone that has more than 15k in their savings. I'm a psychotherapist and I being home around $6800-$7200 a month, and I pay $3100 in total bills a month. i really do appreciate it and so does my family
[c2_0734] (reddit, AAPL) Full Port ROTH IRA w/ $AAPL With AI’s sharp mind and Siri’s wise voice, Innovation sparks, making markets rejoice. HEY SIRI, CAN U PLS BRING APPLE TO ALL TIME HIGH.
[c2_0735] (reddit, GOOGL) Alphabet capex 100% up yoy to 44B revenue is up 24% yoy. Numbers probably good for semi space, not sure about google [ Deck - [
[c2_0736] (twitter, NVDA) $NVDA Weekly trendline breakout after finding support at the 50 SMA and current demand zone. [ThetaWarrior tweeted about NVDA]
[c2_0737] (twitter, FF) #FF my bro @trader1sz Hard pressed to find anyone that has traded the bitcoin range better than he has over the last 9 months both ways. He’s printed some ridiculous figures. [Trader_XO tweeted about FF]
[c2_0738] (reddit, TSLA) $DELL wins 9.7B government contract I saw other guy posted MSFT won but news says DELL
[c2_0739] (twitter, AMZN) $AMZN Added a few % back to my short at $2838 yesterday.. Weekly turning red. OBV falling off a cliff, Top Goon X still showing strong downwards momentum. Holding my shorts with a tightened stop loss below $3,000 now (Avg. Entry was $3286) [Nebraskangooner tweeted about AMZN]
[c2_0740] (reddit, NVDA) Do you use stop losses on all of your portfolio? I see such varied opinion in Reddit regarding stop losses/trailing stop losses. My question is: do you set trailing stop losses for all of your stocks/etfs and if not, why not? If you do, what % do you choose? For example what would your % stop losses be for 3 very different stocks/etfs such as VTI, NVDA or a penny stock like RVPH?
[c2_0741] (reddit, AMZN) Shout out to AWS for making it impossible for me to get into 670 calls this morning. Lost out on a 200% move body text
[c2_0742] (twitter, SILO) $SILO - Made a strong move to $2.57 at open from premarket news. Now holding $2 support, keep on watch for a move back to those levels > [@Bullish_Trades]( > $SILO Broke above $2.40 here at open > > Silo Pharma Submits Pre-Investigational New Drug Application to FDA for SPC-15 as a Treatment for PTSD and Anxiety [BullishTrade quote tweeted BullishTrade about SILO]
[c2_0743] (twitter, RUM) $RUM Daily Candle Chart Mini bullish flag within the right shoulder of IH&S 6 processing....... [Mr Chart Norris tweeted about RUM]
[c2_0744] (twitter, ES) Oh gee double bottom lost on $es Kek [@StonkTrump]( $es looking like a fake double bottom bull trap If I’m wrong I’ll cope, but not buying that lol [StonkTrump 🔁 StonkTrump about ES, ES]
[c2_0745] (twitter, XPEV) $XPEV reported a loss of $0.36, consensus was ($0.46), Earnings Whisper was ($0.48) via @eWhispers #whisperbeat [EPSGUID tweeted about XPEV, WHISPERBEAT]
[c2_0746] (reddit, MCO) MCO vs SPGI ? was looking major date points of these 2 similar companies. SPGI is larger, diversified and I feel more stable (higher market cap 130b ish vs 80B mco; recenue 15B vs 8B avd pe ratio 29 vs 31. Seems MCO is strangly overvalued if u do even DCF . ANy thougts? which one would u buy today ?
[c2_0747] (twitter, OPEN) No changes, everything is going according to plan. Chop today/tomorrow before another potential leg up across corn and equities. [@CryptoCX1]( $BTC $HTR $WOO $GLD $COUP $CLOU $Z $OPEN All I’m in recently. Got a chart to shill me? Drop it below, I won’t even look at tickers. Give me a why via chart. [CryptoCX1 🔁 CryptoCX1 about BTC, HTR, WOO, GLD, COUP, CLOU, Z, OPEN]
[c2_0748] (reddit, BULL) Good time to buy ETFs? I see everything is going down so would it not be a good time to get any? Or would it be good time since I’m buying them lower ? My money’s been sitting in a checking doing nothing so trying to make a little off it in the long run.
[c2_0749] (twitter, SPY) $SPY Rejection from highs. Hanging under this TL now [HackermanAce 🔁 CheddarFlow about SPY]
