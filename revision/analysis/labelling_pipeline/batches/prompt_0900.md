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
----- ITEMS (50: c2_0900 .. c2_0949) -----
[c2_0900] (reddit, BTC) Daughter has $6K in bank savings account. Just money saved from working, etc. Going off to college in the fall (non of this money will go towards college expenses). But, I want to help her put the $ to work other than the terrible bank interest rate she's getting. Would you put in a high yield savings account (what are they? 4% of so?)? I also have a Fidelity account for her I could move the money into and put in a couple of ETFs or MFs? Thoughts?
[c2_0901] (reddit, TSLA) Tesla faces NHTSA investigation into Model 3 emergency door releases The NHTSA confirmed on Wednesday that it will conduct an investigation into complaints about the mechanical door releases of 2022 Tesla （TSLA） Model 3 vehicles. The investigation will cover 179,071 vehicles. Investigate defects in the mechanical door release mechanism. Tesla (TSLA) shares fell 1.0% in early trading on Wednesday.
[c2_0902] (twitter, BLNK) $BLNK Weekly Chart Descending triangle pattern , weekly close above 3.10 things will change [Mr Chart Norris tweeted about BLNK]
[c2_0903] (twitter, ES) This was the fastest and biggest bear market rally since January 2022. Maximum overbought The market is ready for a big rug pull. $ES $SPX [MFHoz tweeted about ES, SPX]
[c2_0904] (twitter, GOOGL) $GOOGL Possible outcome if the bottom support of the bear flag doesn't hold. [MFHoz tweeted about GOOGL]
[c2_0905] (twitter, AAPL) $AAPL Morning star on daily [Nebraskangooner tweeted about AAPL]
[c2_0906] (twitter, BA) $BA 3 day run to the 200ema. I'm watching how this acts at the $161.42 resistance. If it can consolidate & or move higher from here it can move towards the $163, $165.01, $169.26 next. Id like to see a flag form possibly above the $155. 20/50 emas are starting to cross up also [optionsprochick tweeted about BA]
[c2_0907] (twitter, QQQ) $QQQ Nasdaq and a lot of the individual names all did tests this morning... watched them bounce on the line.. Jobs tomorrow, which have been strong, could cause for some turbulence.. [Don't follow Shardi B if you hate Money tweeted about QQQ]
[c2_0908] (reddit, AVAX) Anyone else feel like the market is just shaking out weak hands and leverage? I get the bear argument. 4 year cycle this... inflation that... ect ect Alts have come to price levels that just flat out don't make sense. And im talking GOOD projects... not junk meme coins. like VET, XLM, LINK, AVAX have all dropped to near pre-bull run prices. Are you buying? waiting? DCAing? Something on my shoulder is telling me I'm going to regret not taking advantage of these prices.
[c2_0909] (reddit, BULL) Gold crash from 4550 to 4200 this week My view says gold will crash from 4550 to 4200 this week because my setup indicates
[c2_0910] (twitter, FF) #FF my friends @FakePearBear @BrownMoneyDoll1 @RyanRozbiani @meetsung @SlayBaeTrades @TraderTurbs @FreeTheSauce827 @Titan_Traders_ @TradrNate @Stockrateez94 @T_Tradezz @adcallday @turtlemankamXT @paperhanding_GC @karmaoptions @BestCoastTrader @PlandemicTrader [Honeydew0101 tweeted about]
[c2_0911] (reddit, JPM) Should I Move my JPM holdings into KBWB? So back when I started my investment journey early 2022/Late 2021 I bought some JPM shares as a long-term term hold around 130-140$ per share, I was hoping to just always hold JPM. Now it's up to almost $330 a share I'm up well above 60%. However my biggest concern is its currently about 13.25% of my portfolio. Should I just keep holding or move it into KBWB or a ETF with a primary holding of JPM to negate some risk?
[c2_0912] (twitter, QQQ) $ES_F $NQ_F $SPY $QQQ I was clear about one thing headed into this week: Do not go long or get overly bullish just yet. Many think we're about to get a historic relief rally but bears are still showing plenty of strength and defending and pushing at key levels. [kpak82 tweeted about ES_F, NQ_F, SPY, QQQ]
[c2_0913] (reddit, MU) Yeah I’m cooked😭🙏 Haven’t sold but expecting all of these to expire worthless other then coin🤞 turns out I’m only the disabled part of Stephen hawking, MU ER gonna pop off new ath, tlt was supposed to be my hedge but since they’re buying short dated T-bills long end yields will rise to most likely 5% before EOY not worried QE should be announced in 2026 and they’ll be buying up the long end, I expect to be down 20 grand next week & up 50 by 02/15/26
[c2_0914] (reddit, PLTR) Thoughts on CVI as a stock in the long term? CVI is an energy stock that has been around for a while, that has risen before but always tends to settle down. A stock that could be the next nvidia or palantir but falls short and never gains any longterm momentum. The dividends is also falling extremely short. It has a strong sell but u never know. Wanted to know thoughts since I feel like it can go either way, or if it might take too long to pan out.
[c2_0915] (reddit, NVDA) This is fake news, what you're seeing on the screen is known as a group hallucination, BUY NOW!
[c2_0916] (reddit, BULL) Auto Loan is Upside Down and I’m Screwed Long story short I owe $32k and my car is worth $17k on KBB. Low mileage just dumb decisions. Question is, if I will be unable to drive it for a year (potentially more) should I sell it, take the lost OR keep paying it off for the year I cannot drive it and keep it till the wheels fall off? I have the funds to pay it off but that’s a lot … I’m 28 and have stable income if that helps….
[c2_0917] (twitter, SPY) $SPY +130% now. I took puts expiring tomorrow by mistake. Meant to buy 2 weeks out 😅. Make sure to close out and roll out a few weeks if u plan to swing > [@kpak82]( > $SPY +32% winner [kpak82 🔁 kpak82 about SPY, SPY]
[c2_0918] (reddit, MSFT) capital gains on selling a home so i read that if u sell a home, the amount u make off that home has to be put into another property within a year or ill get hit w capital gains which is like 35% so i bought my house in 2017 for 179k i refied to get my husband off in 2022 which brought the loan to 210k so to date i owe 190 on the home. if i sell im thinking i will make 50-80k i live in md. i dont think ill get hit with the tax because its under 250k. but if i do, im not selling
[c2_0919] (twitter, SPY) This is a key level for $SPY here Trying to hold onto the 449.46 dark pool level Bulls want to see lunchtime pump here to reclaim some losses [StonkTrump tweeted about SPY]
[c2_0920] (reddit, QQQ) VOO + QQQ + VB + VXUS I’m 29 and started investing about a month ago. This is my current ETF portfolio: VOO 50% QQQ 30% VB 10% VXUS 10% Do you think this is a solid approach, or am I missing exposure to other growth sectors?
[c2_0921] (twitter, ES) Does not appear that $ES closed below the yearly vwap (40 mins to go). Cautiously bullish but if we close below it next week, abandon ship until a new trend surfaces. [CryptoParadyme tweeted about ES]
[c2_0922] (reddit, MU) Having to sell $10k to pay bills. Which one would you pick? Currently have to dip into my investments to pay off some medical bills. Have 300k between a few stocks but not sure which one will be the better ones to sell as to not lose too much upside. SPY, QQQI, MU, AMZN, RKLB, GOOG. I wouldn’t mind if I was to be buying back soon but probably won’t be able to contribute to my account for a while.
[c2_0923] (twitter, FANG) I just grabbed some $FANG calls slightly OTM for May. - Oil is bullish af - Diamondback is a solid operator w/ great acreage + balance sheet - Currently testing 2018 ATH - Structure looks similar to Chevron (CVX) prior to its breakout into price discovery a few weeks ago [CryptoKaleo tweeted about FANG]
[c2_0924] (twitter, ITP) $ITP strong earnings out on this penny. $160.88 million in revenue with an increase of 59% year-over-year and approximately $11.02 million in gross profit with an increase of 93% year-over-year. [Bullish_Trades tweeted about ITP]
[c2_0925] (reddit, TSLA) Which of the Magnificent 7 do you think is most likely to fail and the ones you plan to hold for at least the next 10 years? Based on just Business model alone and moat, which of the Magnificent 7 tech giants, Apple, Microsoft, Amazon, Alphabet, Meta, Nvidia, and Tesla do you think is most likely to lose its dominance or even disappear entirely over the next 20 years? What trends, risks, or shifts in technology, will drive your decisions?
[c2_0926] (twitter, SPY) Nice comeback today fellas! Closed all $SPY calls. Will wait to see what happens tomorrow. [JMA_Trades tweeted about SPY]
[c2_0927] (reddit, AMZN) Fun Fact and Hopium We are basically retesting the previous ATH given inflation from 2021 to 2025.
[c2_0928] (twitter, SPY) $SPY $SPX Darkpool index did not improve despite todays rally. Retail going all in again. [MFHoz tweeted about SPY, SPX]
[c2_0929] (reddit, BULL) Other than VTI/VOO, what Vanguard ETF would make a good core U.S. portfolio fund? It can cover all caps or some caps or one cap. Growth, value, or blend. I'm looking for a long term alternative hold to diversify a bit away from the VTI that I hold in another account.
[c2_0930] (reddit, ETH) Starting to invest as a 16 year old So, my dad recently helped me open an investment account, and I’d love to hear your advice and experiences. What helped you the most when you started investing, or what do you wish you’d done differently? I have about $550 saved up that I’m planning to invest. Thanks for your replies!
[c2_0931] (news, ABR) Arbor Realty Trust Schedules First Quarter —…–8 Earnings Conference Call
[c2_0932] (reddit, IBEX) IBEX IBEX 35 (Spain) Current Level: About 11,786 points Next Few Weeks Outlook: The IBEX 35 is facing headwinds from domestic political instability and concerns over bank exposure to real estate However, if tourism continues to recover, this could provide a boost. The outlook is mixed, with recovery potential if political tensions ease.
[c2_0933] (twitter, AMD) $NVDA This chart is so insane, RSI is crazy BUT I rolled out my calls yesterday so we riding this baby... There is no question this is just the start IMO, generative AI will infect everything, therefore everything gonna need a chip IMO Long $NVDA and $AMD Buy dips IMO [ShardiB2 tweeted about NVDA, AMD]
[c2_0934] (twitter, C) $C setup from the newsletter. Shared entry. Good win. [kpak82 tweeted about C]
[c2_0935] (twitter, NVDA) $NVDA Adding and losing $300B in market cap on a daily basis for reference 😂 [Daan Crypto Trades tweeted about NVDA]
[c2_0936] (reddit, NVDA) NVDA 2600% ($56k) gain (shares only) I've been holding this small $2000 position for years now. But the age of custom chips is here. Time to close it out
[c2_0937] (reddit, ETH) Where is all this money coming from pumping the stock market? Ok, might sound like a bit of a silly question. Although I’m genuinely curious as from where all this money is flowing into the stock market? Is there some tracker or overview that traces the flow of cash from bonds, crypto, creation of debt, savings accounts, etc into the stock market. At the time of COVID it kinda made some ‘sense’ as the printer sang his brrt brrt harmony, but now wazzup!?
[c2_0938] (reddit, AI) Avoid Oanda, difficult withdrawing funds Getting a withdrawal from Oanda is like pulling teeth. They constantly request a recent bank statement even when you honor this request. It seems like everything at Oanda is Ai generated. i don't think they care. I would never recommend this company
[c2_0939] (twitter, GME) $GME > [@kholov23]( > $GME Monthly Chart Update 🔨 > > MACD cross to the upside / hidden bullish divergence was on point 🪄 > > S / R flip above 27 [Mr Chart Norris quote tweeted Mr Chart Norris about GME]
[c2_0940] (reddit, AMZN) RealT I'm curious what the crypto community thinks about realT and how bad their making people's lives. These guys have been, and the token holders don't seem to give a crap about the people living in these homes. What are some ways tokenized assets in a world market can ensure they aren't negatively effecting communities? How can we ensure some type of responsibility or restitution for communities when crypto jerks like this will probably always exist?
[c2_0941] (reddit, MU) Please help me understand...MU csp The day of MU earnings I sold 4/17 $450 puts for $39. Over the next five days, MU traded down to $320 or so...and the put to $150. I thought for sure I would get early assignment. Never did. Now MU back to $420 and the put is $34. I certainly would have closed or exercised if I was the put buyer. What am I missing? Please explain the different reasonings for not closing/exercising.
[c2_0942] (twitter, COIN) Gonna grab some $coin call lottos today for the summer. [HackermanAce tweeted about COIN]
[c2_0943] (reddit, ETH) Ethereum Wipes Out 7 Years of Gains in Crash to $1,400, Where Next for ETH?
[c2_0944] (reddit, AMC) $AMC weekly Gamma Exposure (GEX) ☢️🧲🔋 \> Data changes day to day and intraday so please only use the latest data 🥺 \## Disclaimer None of this is financial advice. I believe the majority of price action is the result of managing the multidimensional risk picture. GEX is only a part of the volatility environment risk, one risk of many in the risk picture.
[c2_0945] (twitter, NVDA) $NVDA 130.8 is the gap from 6/20, strong if through. Fib support 128.8 [VS Options Trading tweeted about NVDA]
[c2_0946] (reddit, BTC) How is HDFC Silver ETF for one time investment? I have been thinking of investigating in silver but wasn’t sure where. And this seems like the best option so far. My plan is to invest one time and not do SIP for 3 years or more years. Is it okay or should I rethink my plan? Note: I am not very literate in this area 🥲
[c2_0947] (reddit, GOOGL) FREE OPTION BOOK For those of you who are interested in learning more about options, a free copy of "Options as a Strategic Investment" by Lawrence G. McMillan is available here [
[c2_0948] (twitter, ES) $NQ Making it back to the top of the range while $ES is testing 4000 flat. Looks like the market is really setting up for some volatility next week. Whichever way the market goes will be aggressive. [Amdtrades tweeted about NQ, ES]
[c2_0949] (twitter, FF) I don't do #FF because I think it's a bit useless... but a few names that have hit me in the last few months @CryptoGodJohn @Trader_XO @cryptocevo @trader1sz @CryptoRezeh @astekz @IncomeSharks @CryptoTony__ @jtradestar [CryptoGodJohn 🔁 eliz883 about FF]
