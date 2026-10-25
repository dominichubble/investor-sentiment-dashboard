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
----- ITEMS (50: c2_0450 .. c2_0499) -----
[c2_0450] (reddit, JPM) 'The demographic dividend of the last 40 years is ending': JPMorgan says the world is running out of the two things that kept interest rates down
[c2_0451] (reddit, GOOGL) Google position 2122 stocks now In my last post I added my purchase of 1000 stocks.. I added more this week, now more than 2/3rd done will buy another 900 if there is pullback. Not averaging up anymore. Hoping these bets buys me financial freedom some day. Happy 4th my regards🇺🇸🫡
[c2_0452] (twitter, MARA) $MARA 3Monthly Chart ☕️ Cup and handle formation within the right shoulder of inverted head and shoulders , where price is forming the handle Worth noting price is holding right above the All time high Anchored VWAP [Mr Chart Norris tweeted about MARA]
[c2_0453] (twitter, DKNG) Weekly & PA setup. $DKNG Lets see if we can flip n get into the clouds. [HackermanAce tweeted about DKNG]
[c2_0454] (twitter, DUOL) Just closed the remaining longs I had ( $DUOL and $PL - which was my lottery earnings bag) Did a lot of LTF trading today and just going to stick to that strategy now. I make way more money on LTF's then trying to hold shit, especially in this choppy market. [Nebraskangooner tweeted about PL, DUOL]
[c2_0455] (twitter, GME) $GME flow: - Put/call ratio: 0.651, call premium: 84 million - Avg 3d call volume: 231K, Fri: 199K - Avg 3d put volume: 96K, Fri: 129K - 5K 🐳: 57% 🐂 - 15K 🐳: 56% 🐂 - 30K 🐳: 57% 🐂 - 🐂 targeting 2022-03-18 expiry, $340 strike [unusual_whales tweeted about GME]
[c2_0456] (reddit, HBM) $TRT -Anyone looking into this? Earnings on Monday pre-market. Anyone looking into this company? Could be interesting for asymmetric upside for exposure to burn-in systems + testing for SiC/GaN qualification. $TRT trades at just a \~1.2x EV/Sales vs. $AEHR at \~60x All while we know that the testing tailwind is real: \- $TER AI mix rising to 70%+ of Q1 2026 revenue \- $AEHR $ 92M H2 bookings \- Advantest operating profit +135% on HBM/HPC \- ChipMOS Q1 2026 revenue +25% on AI memory
[c2_0457] (twitter, FLEX) Don't miss our CEO @therealleslamb0 Twitter Space Q/A with @LBank_Exchange ahead of $FLEX listing 👀 Set a reminder and tune in tomorrow @ 1PM UTC / 9AM EST [zhusu 🔁 OPNX_Official about FLEX]
[c2_0458] (twitter, META) $FB Shorted Meta with a tight stop. I am looking for $165.71 and $150.25 if this is going to drop from here. [Amdtrades tweeted about FB]
[c2_0459] (twitter, BAC) $BAC Not sure this is going to work out but it's a nice R:R and Top Goon X has a sell alert on LTF so I'll have a go. Could just straight moon here but tight stop loss. [Nebraskangooner tweeted about BAC]
[c2_0460] (reddit, BULL) High Yield OAS as a stock market bull/bear indicator Has anyone used High Yield Option adjusted spread as a bull/bear indicator? Its the spread of non-investment grade bonds over treasuries. The idea is the tighter the spread gets, the weaker the credit standards are. The data is available St. Louis Federal Reserve site. [
[c2_0461] (twitter, QQQ) $SPY $QQQ $ES_F $NQ_F 4H RSI hit extreme oversold levels so a reset with some chop and consolidation would be healthy. Doesnt take much to reset 4H RSI. [kpak tweeted about ES=F, QQQ, NQ=F, SPY]
[c2_0462] (twitter, ES) $ES_F $SPX $SPY 6H -- May20 Here we are... Round trip to supply and back through the lows Probably has another 100pts or so, but for the most part my job here is done Have a good weekend ✌️ > [@ChandlerTrading]( > $ES_F $SPX $SPY 6H - May12 > > One possibility to watch for... > > A corrective move to overhead supply (before reaching monthly demand) would set up another short opportunity [ChandlerTrading 🔁 ChandlerTrading about ES_F, SPX, SPY, ES_F, SPX, SPY]
[c2_0463] (reddit, BULL) locked in ndx profits weekend mode activated Riding this market rally I locked in over $62000 in realized profits I am currently sitting comfortably at home thoroughly enjoying the weekend thanks to these earnings The entire process was clean, seamless and completely stress free achieved through nothing more than patience and precise timing
[c2_0464] (reddit, NVDA) Sold NVDA for some TSLA gains. Had a bunch of nvda calls from the other day that I was green on, but wanted to jump on this Tesla hype. Now I have more money for nvda 🤑 Probably wait for money before buying back in nvda to see how this china talk goes.
[c2_0465] (twitter, AMD) $AMD 1H Pretty clear rising wedge here into the gap fill resistance at $76 Breakdown and $72 can Trade Should that level breakdown there is a nice gap fill at $67.93 > [@JohnHumeRoss1]( > @ATMSnipes AMD, please. [ATMSnipes 🔁 JohnHumeRoss1 about AMD]
[c2_0466] (twitter, RS) The largest jumps in RS over the last two weeks have come from Fracking, Utilities, Real Estate, Medical Devices, Low Vol, Infrastructure, and Cybersecurity $FRAK $IDU $XLRE $IHI $SPLV $IFRA $CIBR [the_chart_life tweeted about FRAK, IDU, XLRE, IHI, SPLV, IFRA, CIBR]
[c2_0467] (reddit, ASX) ASX 200 LIVE: ASX turns down, oil spikes after US Donald Trump’s speech; KMD tumbles 55pc
[c2_0468] (twitter, CMT) $IWM closes within 15 bps of Wednesdays intraday high. [Ian McMillan, CMT tweeted about IWM]
[c2_0469] (twitter, SPY) yes > [@MichaelNaussCMT]( > In my opinion, the April highs are super important to watch here on a closing basis. $SPY [Ian McMillan, CMT quote tweeted Michael Nauss, CMT, CAIA about SPY]
[c2_0470] (reddit, ES) Hard morning session - ES Daily review 3/31 ES Daily review I always see a big bull bar and think a bull trend is about to start — but it never does. The market keeps going in the opposite direction of what I expect. How should I change this mindset? My biggest loss today: I entered long at the big bull bar 9:50 and got fully stopped out at 10:50 after 6 good bear bar, -$600.
[c2_0471] (reddit, LTC) Is LTC the new meta? Can people invest in it now? Well, I saw Lite coins drop its value recently. I'm only 17, but I do wanna invest in stocks and crypto, before it's too late. I think lite coin will go up soon, but I'm not sure if it's worth investing or not since I don't know much about finance, crypto etc.. Well, pls help me out in this. If no, if it's not worth investing, tell me where else I could invest my money for now?
[c2_0472] (twitter, NVDA) $NVDA 💨 > [@kholov23]( > $NVDA high and higher [Mr Chart Norris quote tweeted Mr Chart Norris about NVDA]
[c2_0473] (twitter, CRWD) $CRWD Monthly tight consolidation just above the falling wedge. [Theta Warrior tweeted about CRWD]
[c2_0474] (twitter, SPY) Up the calls with them sitting at 3.65 n now ITM. Love to see it ! Gonna let it run just a little n close out by 11. Will re-enter near end. $spy Been trying to tweet out right when I buy my contracts so you all can try to follow. Just remember to not chase if u missed. > [@HackermanAce]( > Gonna close before 12 and likely by 1030. fyi. [HackermanAce 🔁 HackermanAce about SPY]
[c2_0475] (reddit, BTC) Leveraged ETF on world stock diversification? Hey guys, I'm looking for long term holds but with a bit more risk tolerance and was wondering if there are leveraged ETFS on world diversification stocks like VT / XEQT. Similar to TQQQ / SPMO, are there similar ETFS that leverage VT or XEQT? I couldn't seem to find one.
[c2_0476] (reddit, GLD) Begginer here first buy: should i buy UCTIS ETFs or US? Eu based Hello, can someone please explain kindly what to do, I have 5k EUR to invest, i'm EU based. First time investing. I decided to go for VT and GLD but I am not sure whether I should then use european market tracker like VWCE and SGLN? Sorry if my question is a bit stupid I am still learning!! Thanks guys.
[c2_0477] (twitter, AFRM) $AFRM Rejection at the upper boundary of bullish flag ( 30 - > 39 ) > [@kholov23]( > $AFRM Weekly Chart Update > > Price is breaking out from mini falling wedge pattern within the bullish flag on weekly candle chart > > Hope it makes sense [Mr Chart Norris quote tweeted Mr Chart Norris about AFRM]
[c2_0478] (reddit, AAPL) To sell or to not sell At what point do you sell a losing position? At what percentage do you set a stop loss and dump the position all together? I’ll give a personal example, I’m holding aapl 4/25 $165p that are down 59% (about $4k). I have not sold because I still think we could see another leg down, and there’s still a few days left till expiration. I’m just curious what everybody else does.
[c2_0479] (twitter, QQQ) RT @TrendSpider: Remember these TW Pivot buy signals from the October lows? $SPY $QQQ Thanks @ThetaWarrior 👏 [Theta Warrior retweeted TrendSpider about SPY, QQQ]
[c2_0480] (reddit, BULL) Capital gains tax advise Have a commercial investment property with some family members, set up as a partnership. In the process of selling it and looking like I will have about $70k that will be taxable, after the long term assets capital gain discount. Want to talk to someone to find out the options I have to reduce tax liability. ie put it in super, pay off chunk of investment property etc etc What type of accountant would be best to get advice from, happy to pay for their time./advise.
[c2_0481] (reddit, BTC) At some point, the weak hands won’t have any BTC left to sell
[c2_0482] (twitter, AMZN) $QQQ Weekly. #QQQ relative strength compared to #SPY and still holding the breakout. Potential bull flag and the 200sma around ~285 can offer support $NDX $NQ_F $XLK #Nasdaq $AAPL $MSFT $AMZN $SPY #StockMarket $SVIB $SBNY [reciknows tweeted about XLK, NQ=F, QQQ, AMZN, NASDAQ, STOCKMARKET, MSFT, NDX, SBNY, SVIB, SPY, AAPL]
[c2_0483] (twitter, CAT) $CAT Almost halfway to target. [Nebraskangooner tweeted about CAT]
[c2_0484] (reddit, TD) TCSH Im looking to park a nice chunk of money on the sidelines and since im with TD, i figure TCSH is a good option. I notice every month it declines after the payout the slowly goes back up again. Would it be wise to wait next payout before buying it? If anyone has more information on how TCSH works it would greatly be appreciated. Thanks!
[c2_0485] (twitter, DIA) Sellers seem to be getting exhausted Punted some small size $DIA calls since I had this daily demand box marked But overall not looking to play too much on opex [StonkTrump tweeted about DIA]
[c2_0486] (twitter, ES) $ES still expecting range low to be tested aslong as we remain range bound [trader1sz tweeted about ES]
[c2_0487] (reddit, TACO) Lot of new blood! Not gonna lie, the posting about Feb/Mar 2026! It’s as if you weren’t here for the DeepSeek implosion, trumps tariffs and TACO, 2022 inflation, the omnicron variant… there’s been so much turbulence in the last 5 years, I can’t believe you people are seasoned veterans. And if this is new to you, you’re probably not making money, right?
[c2_0488] (twitter, GME) Darn just missed the $GME impending breakout I mentioned Friday. [Nebraskangooner tweeted about GME]
[c2_0489] (twitter, SPOT) $SPOT Too bad I closed for tiny loss lol. I don't like when I chase a play and there's no momentum so I decided to get out instead of risking any more. Onto the next one. > [@Nebraskangooner]( > $SPOT > > Good R:R for a little short scalp here [Nebraskangooner 🔁 Nebraskangooner about SPOT, SPOT]
[c2_0490] (twitter, GOOGL) $GOOGL starting to break flag on daily here [rachels_44 tweeted about GOOGL]
[c2_0491] (reddit, BULL) Looking for feedback on a trading-themed apparel idea for traders Hi traders, I’m thinking about starting a small apparel brand for traders (Forex, crypto, day trading). The idea is simple T-shirts with trading quotes or jokes like “I am not virgin trading fuck me every day ”, “Buy the Dip”, etc. Just wanted to ask: • Would you actually wear something like this? • What kind of trading quotes/designs would you like? Appreciate any honest feedback
[c2_0492] (twitter, SPY) The comments😂😂 > [@ATMSnipes]( > $SPY - this continues to be the reference point for bullish upside > > Still aging like wine🫡 > > Looking back on it for the 6th time it shows how fintwit can get you caught up in the noise > > (as I got a little too bearish near the lows if you followed me at the time) [Avery quote tweeted Avery about SPY]
[c2_0493] (reddit, BTC) Bitcoin Whales Are Finally Accumulating Again... The good and surprising thing is that whales are accumulating Bitcoin. Wallets which were been not used from years is now active and accumulating crypto. Whale wallets holding 10000 BTC have flipped into accumulation. What they want? Why they sell instead they are accumulating? This shift is bigger then it looks. More buying coming.. source [
[c2_0494] (twitter, DKNG) $DKNG indeed > [@kholov23]( > $DKNG short term BOUNCE ? [Mr Chart Norris quote tweeted Mr Chart Norris about DKNG]
[c2_0495] (twitter, APPS) RT @kholov23: $APPS Monthly Chart Is it the time for reversal? 1. Noting Monthly Bullish Divergence 2. Monthly Candle Close - Doji Morni… [Mr Chuck Norris retweeted Mr Chuck Norris about APPS]
[c2_0496] (reddit, CMA) Unite plc in oversold territory? The REIT is now yielding close to 7%. A stronghold over accommodation in red brick university. The M&A, pending CMA investigation and spending of equity capital are key risk yet that was priced in at 900p. If rates fall this helps but in the meantime I’m picking up the carry. Feels like a golden opportunity for an entry point.
[c2_0497] (reddit, BTC) How are KOSPI gains sustainable? Kospi is up 170% this year. 170 FUCKING PERCENT. This seems like a ridiculous amount for a country’s ETF. Imagine if SPY had a 170 percent year, it would be absolutely absurd. I feel like this is bound to have a big pullback sometime soon. I’m curious to see how this could possibly continue pumping and how this 170% pump has affected the country of Korea.
[c2_0498] (twitter, RBLX) $RBLX weekly Here's one for my stock and options guys Clear accumulation Could either slowly buy dated calls in this range to build a position for an eventual breakout, or to play it safe, wait to buy range lows or buy a weekly close above the top wap First target would be [StonkTrump tweeted about RBLX]
[c2_0499] (twitter, PYPL) A couple Mobile Payments names putting in bases and pushing to new short-term highs... $SQ $PYPL [the_chart_life tweeted about SQ, PYPL]
