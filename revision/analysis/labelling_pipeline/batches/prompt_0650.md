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
----- ITEMS (50: c2_0650 .. c2_0699) -----
[c2_0650] (twitter, DDOG) Closed $DDOG > [@JMA_Trades]( > $DDOG at Daily demand > Bought 2/10 $80 Calls [JMA_Trades 🔁 JMA_Trades about DDOG]
[c2_0651] (reddit, OKLO) OKLO +18% VST +17% after Meta signs 6+ GW nuclear power deals for AI data centers
[c2_0652] (twitter, QQQ) $QQQ Daily. #QQQ possibly looking at re-test of the breakdown from the bear flag with more downside continuation incoming. Bulls need to hold > 259 or 2022 lows will be tested and likely broken $NDX $NQ_F $XLK #SPY #Nasdaq $AAPL $MSFT $SPY [reciknows tweeted about MSFT, NDX, QQQ, SPY, AAPL, NASDAQ, XLK, NQ=F]
[c2_0653] (reddit, NWGL) $NWGL still one to watch .. Bottomed and being accumulated . Watch for a spike soon IMO . Volume Profile , demand zone at $1.30 ish ... usually chart tells a story ... could see News and breakout. ..
[c2_0654] (reddit, VIST) VIST - hold or sell after spike this week? I have roughly 1000 shares of $VIST @$53 average. Single largest holding. Thesis is: 1. I think Milei is good for Argentinian economy 2. Lowest lifting cost in world 3. Well run 4. Good margins, growth rate, PEG, etc. With price per barrel increasing from $70 to $110 potentially, there might be a massive spike in share price. Anyone have thoughts on VIST, oil stocks, and whether I should take my profits and run or continue holding?
[c2_0655] (twitter, ES) $ES 1H -- (Jan28) Interesting development overnight -- this bounce opens the door to a more complex w4. Just a short term alt, still plenty of options. Stay nimble! [@rbc_trading]( $ES 2H -- (Jan27) So far so good. Still looking for ~4150 W5 can either be an impulse or an ending diagonal - The exact target will depend on which structure develops [rbc_trading 🔁 rbc_trading about ES, ES]
[c2_0656] (reddit, MSTR) HILARIOUS😂🤣 Just yesterday ppl were locking in 9% losses lmfao. Bitcoin volatility teaches us discipline and patience! Chart go up? Happy. Chart go down? EXCITED FOR CHEAPER COINS!!
[c2_0657] (twitter, AMZN) $AMZN so it was the bottom ? 🤔 > [@kholov23]( > $AMZN 175 - 176 would be a good spot lower boundary of bullish flag 💎 [Mr Chart Norris quote tweeted Mr Chart Norris about AMZN]
[c2_0658] (twitter, QQQ) $QQQ Monthly another successful "BRB" 📈 [Theta Warrior tweeted about QQQ]
[c2_0659] (twitter, SHOP) $SHOP Quick easy profit. Just closed out this scalp. > [@Nebraskangooner]( > $SHOP looking rough on low time frames > > probably sees a little dip here (if not a larger dip) [Nebraskangooner 🔁 Nebraskangooner about SHOP, SHOP]
[c2_0660] (reddit, SI) Continue SIP in this volatile market? Need one advice from reddit! In this volatile market, do you suggest to keep the SIPs running in mutual funds or should I stop them and hold on to cash?
[c2_0661] (twitter, CRSP) $CRSP Low time frame scalp with good R:R [Nebraskangooner tweeted about CRSP]
[c2_0662] (twitter, HOG) $HOG BULL SWEEPER DETECTED [HackermanAce 🔁 trndtrader about HOG]
[c2_0663] (reddit, AMC) Luv AMC never leavin. I love movies and I hate corruption and there’s a lot of negative noise right now almost like it was pumped in here and manufactured lol we get it never leaving
[c2_0664] (reddit, GOLD) War/energy crisis Which if any Alternative sources of energy will see a bump from this war/energy crisis? I understand it's not possible for them to scale quickly but will stuff like retail solar panels jump, cheaper EVs?
[c2_0665] (twitter, EXC) $EXC 🟧🟦 Exelon Q1 2024 Adj EPS $0.68 Misses $0.70 Estimate Sales $6.043B Beat $5.568B Estimate [Mr Chart Norris tweeted about EXC]
[c2_0666] (twitter, PLTR) Few $pltr > [@HackermanAce]( > $pltr ohhhh ? > > I expect most weeklies to do this and follow thru with a breakout [HackermanAce 🔁 HackermanAce about PLTR]
[c2_0667] (twitter, WDAY) It’s cyber week! Keeping this up until Friday 📈 this week we did $SHOP $TSLA $WDAY > [@rachels_44]( > 🚨Black Friday Sale!🚨 > > Finding relative strength💪 Loving the cheap plays lately🥳 > > ✅Real-time Alerts > ✅Trade ideas > ✅Charts & Chatroom > > [rachels_44 🔁 rachels_44 about WDAY, TSLA, SHOP]
[c2_0668] (reddit, ETH) Where do I start? So I didn't know about this subreddit but apparently the algorithm thought it was interesting for me... I have been lurking for a few days now and seen a bit of posts. I mainly invest in ETF but this is totally different and sparkled my curiosity. Before I deep dive, is there something I need to know or watch? Tkz
[c2_0669] (twitter, ES) $ES There it is Range tag and 441.84 dark pool support Bought some April spy calls worth a shot here accept back into range and ill call it invalid and cut calls [@StonkTrump]( $ES The spooz looks good, but getting overextended and it's probably time to punish fake bulls I'm still looking to long a confirmed range flip with a retest of range highs at the 4408-4425 region This would correspond with the heavy dark pool accumulation on $spy at 441.84 [StonkTrump 🔁 StonkTrump about ES, ES, SPY]
[c2_0670] (reddit, MSFT) spy analysis 3/16/26 Spy Monday 3/16 breakdown 👇 Spot $662.29 — already below the flip level at $666. Lower king at $660 is 5× stronger than upper. $185M put wall at $660 is the first magnet. Break $660 → $650 air pocket. Negative gamma means every move down gets amplified. Bearish until we reclaim $666.
[c2_0671] (twitter, TSM) DISCLAIMER: HIGH RISK #LottoFriday Keep bets small. Lock in gains. $CHWY 43C >42.17 | 40P &lt;40.62 $NVDA 255C >250.52 | 235P &lt;239.24 $SNOW 285C >283.12 | 275P &lt;278.52 $TSM 130C >129.59 | 127P &lt;128.2 Good luck! 🎲 [ThetaWarrior tweeted about CHWY, NVDA, SNOW, TSM]
[c2_0672] (reddit, ETH) Should I sell all my profits I’ve currently gained 22, 000 dollars in profit with only xrp and eth in my portfolio my current portfolio is at 30k total. Been pretty tempted to sell everything but I feel like we have one more big push? Thoughts on what to do? Sell everything or take partial profits if so how much
[c2_0673] (twitter, MVIS) $MVIS up 8% today 💯✅ Target hit 20% in two days. Next breakout would be HTF long signal. [Anbessa100 🔁 Anbessa100 about MVIS]
[c2_0674] (twitter, PLTR) $PLTR Daily Candle Chart They say when there is a BREAKOUT , don't panic , don't stress , don't FOMO, because there certainly will be a RETEST, and I reply ok Chupapi Munianio [Mr Chart Norris retweeted Mr Chart Norris about PLTR]
[c2_0675] (reddit, SPY) IC setups Current premium environment check — what are you seeing? IV rank on SPY and QQQ is elevated right now. Premium environment looks rich for IC setups around 36 DTE with POP above 80%. Anyone else positioning this week or waiting for more clarity on direction?
[c2_0676] (reddit, AMZN) Tapestry Call Options Aug15 strike $105 Hey guys, Pretty new to options trading and been wondering what to do with this one. I have a Tapestry call option for $105 expiring on Aug 15th. I am up 178% already. Stock keeps on going up now at $109 and I expect it would go even higher after earnings call next week. I don't plan on exercising the option but try to maximize profit. Should I sell now or wait a little longer ?
[c2_0677] (reddit, SPY) Trump teases a "very very big announcement" for Thurs/Fri/Mon A "very, very, very big announcement, like, as big as it gets". Says they'll announce Thurs/Fri/Mon probably. Hush hush on details but says it will be "very **positive**... one of the most important announcements made in many years." If we assume a 5% gain in SPY by Tuesday, there's an 18:1 R:R call out there: Any ideas? What it could be? Source: [
[c2_0678] (twitter, MARA) $MARA 1W This is one that will have to prove itself... But could get some relief off weekly demand. Will obviously depend on crypto mkts > [@STW_2020]( > @ChandlerTrading $MARA [ChandlerTrading 🔁 STW_2020 about MARA, MARA]
[c2_0679] (twitter, WMT) Blockchain is the key behind the potential explosive growth of the World Mobile Network 📳 Customers will use the network like any other, to make calls, connect to the internet, and more. Under the hood, now that's where things get really clever. $WMT [TheEuroSniper 🔁 wmtoken about WMT]
[c2_0680] (twitter, NVDA) $NVDA Up or Down - > it's a win ( Long-term 🧠 ) [Mr Chart Norris tweeted about NVDA]
[c2_0681] (reddit, TSLA) Does Fed Decision Matter? It's the Fed day, but I feel what Powell says will have little impact on markets. Microsoft, Meta, and Tesla report after market close. Their results will decide whether SP500 soars above 7000 or not. Markets will ignore interesr rates and focus on the key topic - AI. What do you think?
[c2_0682] (twitter, AMZN) DISCLAIMER: HIGH RISK #LottoFriday $AMZN 105C >104.58 | 100P &lt;101.26 $CHWY 29C >28.73 | 27P &lt;27.8 $DDOG 85C >84 | 80P &lt;81.12 $GME 130C >128 | 120P &lt;120.58 Above Green = Bullish (long call / short put) Below Red = Bearish (long put / short call) Good luck! 🎲 [ThetaWarrior tweeted about AMZN, CHWY, DDOG, GME]
[c2_0683] (reddit, UK) Yen Tsunami Coming The finance minister of Japan has issued a statement that they would shore up he yen since it's hit a 40 year low. That most likely means they're going to sell US treasuries which is a position of 1.21 trillion dollars. If they sell that what is your next move? I will be buying UVXY calls
[c2_0684] (reddit, RA) RealEstate cold call Hey Crew, Just cleared my mortgage today.. officially debt free I get a cold call from a RA that Ive never met.., Am I being paranoid or is something hinkey going on?
[c2_0685] (reddit, BULL) On SpaceX's controlled lock-up periods and how it will affect price? Thoughts on SpaceX's controlled lock-up periods and how it will affect price? With most stock, when the lock-up period ends, the stock price goes down. However, with their tiered lock-up release, wondering how others feel this will impact the selling price? Do you think it will go down significantly?
[c2_0686] (twitter, EMA) $SHOP Trendline breakdown and retest. Clinging to EMA support but has some room to fall if that doesn't hold. For me, I'll just be looking for LTF short scalp opportunities. missed my swing entry but should have some good LTF chances [Nebraskangooner tweeted about SHOP]
[c2_0687] (twitter, MBLY) $MBLY 🟦🟦 Mobileye Global Q2 2024 Adj. EPS $0.09 Beats $0.07 Estimate Sales $439.000M Beat $424.802M Estimate [Mr Chart Norris tweeted about MBLY]
[c2_0688] (reddit, ETH) Automatic DCA? Hi guys, new to Bitcoin here. I started in December a 25€ weekly DCA. I use Kraken Pro for the small fees but I don’t see a way to automatically buy bitcoin every week. Is there an exchange that allows automatic DCA while having low fees ? Or do I have to keep doing it manually?
[c2_0689] (twitter, RSI) $NDX back to a weekly RSI > 70 Sounds spooky. But what does history tell us? A year later... 87% chance of a gain, average return of 20%: [Ian McMillan, CMT tweeted about NDX]
[c2_0690] (reddit, SBI) Am I being greedy? Buying these since the fall : 1. Niftybees 2. SBI next nifty etf 3. goldbees Today I bought with all my capital, since it was falling crazy, My goal is for longterm 7-9 years. Am I being greeedy or following a safe path? Please suggest! Thanks!.
[c2_0691] (reddit, BTC) Cex or dex Hi, I have some money and I want to buy some btc. I don't like binance so I was thinking to go at a broker (or whaever they're called) and have him send my money to a bitcoin wallet like bluewallet or similar. Do you guys think dex is a good option? I will be buying and forgetting so not much swapping going on.
[c2_0692] (reddit, RBA) Are people massively underestimating what's coming? Rising oil feeds into fuel, transport, food and production costs - or in other words inflation. The RBA's role is to keep inflation within the 2-3% band and its only tool is to increase rates. Ultimately, nobody knows how long this conflict will last. But one thing we know about conflicts is that they are not easily and readily resolved. It may well take another 5-10 years for all we know.
[c2_0693] (twitter, RACE) $RACE weekly breakdown below uptrend line and box supports. [kpak tweeted about RACE]
[c2_0694] (twitter, PG) $PG Daily Chart Price moving within bull flag pattern, where it's holding below the upper boundary / resistance / of the channel [Mr Chart Norris tweeted about PG]
[c2_0695] (twitter, SPY) $SPY Daily Chart Good news inverted cup and handle got INVALIDATED ☕️ ❌ Symmetrical triangle breakout ✅ Finally closed above all time highs AVWAP ( March 28 ) MACD cross to the upside on daily chart What else ? [Mr Chart Norris tweeted about SPY]
[c2_0696] (twitter, TSLA) $TSLA 2022-01-07 C $1210 Underlying: $1146.59, % Diff: 5.53% 🤡🚨🐂🛍️ Bid-Ask: $8.3 - $8.45 Interest: 2,016 Volume: 9,795 IV: 69.25% Daily $ Volume: $8,203,313 Sector: Consumer Cyclical Strategy: short_vertical_spread Details + emoji legend: [unusual_whales tweeted about TSLA]
[c2_0697] (twitter, META) $FAST is Now Live and Trading! - 100% of tokens unlocked - Zero VC's - Podcast Summary Product is Already Live - Summaries shared by Michael Saylor - Reflection Token: Automatically get paid to hold - Founder @JeffKirdeikis has built @TeamFinance_ ($500M TVL) & @TheCryptoApp… > [@PodFast_]( > This is very meta. > > A summary of PodFast by @JeffKirdeikis [TheEuroSniper 🔁 PodFast_ about FAST]
[c2_0698] (twitter, SPY) $SPY and $QQQ at all time highs You know what that means right? BLUE SKY [Don't follow Shardi B if you hate Crypto tweeted about QQQ, SPY]
[c2_0699] (reddit, AAPL) Apple. This morning, 227.5c 0dte hit a low of $.02/contract. They are currently at $3.00 and climbing. $500 could’ve turned into $80,000, in <4hrs. An Apple call spread would’ve been like hitting the lottery this week.
