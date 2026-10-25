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
----- ITEMS (50: c2_0500 .. c2_0549) -----
[c2_0500] (twitter, RIOT) Miners, like $COIN starting to get some life $RIOT [Shardi B Trading #teamshardi tweeted about RIOT, COIN]
[c2_0501] (twitter, MARA) $MARA approaching level of interest - Prior highs confluence with DMA 200 [Cheds (Trading Quotes) tweeted about MARA]
[c2_0502] (twitter, ES) $ES - aslong as we hold above the level broken i remain bullish btc. Clean break below it then its over and short fuck out of btc [trader1sz tweeted about ES]
[c2_0503] (twitter, BAM) Why we only back solid projects with BIG teams! $BAM will take off 🚀 🔜 enough! LFG 👊 @BambiCashCrypto > [@jimmykwilson82]( > While most memecoins are still bleeding out, $BAM has spent the last 2 weeks creating an accumulation base, preparing for the next cycle higher. > > Instead of punting random hype driven plays w/ high chances of getting rugged, I invest with conviction in well connected teams w/… [MacroCRG 🔁 jimmykwilson82 about BAM]
[c2_0504] (twitter, AMD) $AMD downtrend break on daily 📈 Needs to hold above Friday's high for strength. 50sma is at 89 [rachels_44 tweeted about AMD]
[c2_0505] (twitter, FF) It's #FollowFriday, here are some amazing accounts you should be following 👇 @CryptoNewton @CryptoBoss1984 @VegetaCrypto1 @CryptoTony__ @DonCryptoDraper @devchart @CryptoGodfatha @j0hnnyw00 @SimonHayesBTC @DMTLAND_ @ShardiB2 @DaanCrypto #FF [ShardiB2 🔁 Altcoin1hunter about FOLLOWFRIDAY, FF]
[c2_0506] (twitter, DKNG) $DKNG Daily Candle Chart 🔨 Breakout / Retest ✅ Simple S / R Flip [Mr Chart Norris tweeted about DKNG]
[c2_0507] (reddit, GOOG) GOOG LEAP Mafia where you at? Post your screenshots below👇 Ok but FR fellow regards, do we hedge OTM leaps during earnings? Sell weekly calls and puts to go short vol? Buy puts to protect downside? Or just let it ride? I’m definitely not selling until I get long term capital gains, New Mexico tax law is weird.
[c2_0508] (twitter, GOLD) 2 more RTs and i post > [@HackermanAce]( > Thread of $gold related Stonk plays EZ WINS. [Ace quote tweeted Ace about GOLD]
[c2_0509] (twitter, ES) $ES Trap the "fade every rally" crowd and then send ✅ Can't celebrate too early as we have fed talks today, but I suspect we Gucci as news events usually magically follow the momentum of current price action Test of prior resistance to flip support and den up? > [@StonkTrump]( > $ES 1hr > > Range between avaps for now > > I personally think they are gonna trap the "fade every rally" crowd one more time and then send it, but lets see what happens [StonkTrump 🔁 StonkTrump about ES, ES]
[c2_0510] (twitter, SHOP) $SHOP May also be a good R:R short opportunity here on breakdown/retest. Odd that it's lagging a bit though, usually it more leads. I'm not going to enter any new trades today. Just taking profit on open ones. [Nebraskangooner tweeted about SHOP]
[c2_0511] (twitter, EG) All you had to do was hold $hood. This is my best spot play for 2024 > [@HackermanAce]( > @EG_Stocks $hood is prolly the easiest play in the stock market imo. > > We all know its going to break out of the range, head to 15 first and then just start ripping face. > > Tempting to buy leaps but worried about decay since its range stuck. [Ace quote tweeted Ace about HOOD]
[c2_0512] (reddit, BTC) I lost all my money I cannot sleep or eat , this is not for the weak. Don’t use all your capital. Goodnight world
[c2_0513] (reddit, WEN) Wendy’s, $WEN, Beats on Earnings this Morning! 20% short interest. 3 days to cover. Stock already jumped up 4.20% this morning. Shares available to borrow briefly hit 0 yesterday. Higher than average FINRA volume yesterday too. The positive earnings growth his morning are the perfect catalyst. Get your shares now!
[c2_0514] (twitter, NVDA) $nvda Which way are you betting for earnings? Im sitting out but if after ERs this holds the clouds then i dont wanna jump to early puts. Will wait for a cloud entry before shorting. [Ace tweeted about NVDA]
[c2_0515] (twitter, CNQ) Buying some dated out $cnq calls. I have commons but adding options. [HackermanAce tweeted about CNQ]
[c2_0516] (twitter, COIN) Coinbase reports earnings in 6 days. I'm looking forward to seeing how it goes and what is said going into the next quarter. $COIN [Amdtrades tweeted about COIN]
[c2_0517] (reddit, GOOGL) Sell 2000 of Google or 2000 of bitcoin? I kinda need some extra cash. What would you recommend me to sell
[c2_0518] (reddit, NVDA) Oh yeah, I’m getting the fuck out… 👋 Retard this, retard that… I’m too lazy to post all my trades but I can confirm they’re all retard lucky trades. I think there’s a bull trap coming so for that reason I AM OUT. Good luck to you regards! ❤️
[c2_0519] (reddit, MSTR) Thoughts on MSTR vs Bitcoin? What do you think is better to hold objectively for the purpose of long term wealth gains
[c2_0520] (reddit, TSM) Tim Apple Warned by CIA That China Could Move on Taiwan by 2027 (AAPL +2.25%, TSM +4.25%) [ bombshell report released today that the feds warned tech companies about china invading Taiwan, yet the two companies most likely to be hit the hardest are up today. How does that make sense
[c2_0521] (twitter, LEN) Consumer discretionary names looking pretty solid across the board... > [@adaptiv]( > 11/13/2022 > > Relative Strength Rankings - $RCD > > $BWA $SBUX $LEN $HD $DHI $YUM $WYNN $TPR $TJX $RL $F [the_chart_life 🔁 adaptiv about SBUX, RCD, TJX, RL, DHI, YUM, LEN, WYNN, BWA, TPR, F, HD]
[c2_0522] (twitter, SPY) $ES_F rejected key horizontal level plus daily 9ema and 20sma overnight. $SPY did the same thing at the highs today. Back into chop range now as we wait for Powell tomorrow. [kpak82 tweeted about ES_F, SPY]
[c2_0523] (twitter, ENPH) $enph crushes earnings today after hours ☀️ 🔋 * Revenues rose 67.7% YOY to $530.2M vs $505.44M Consensus Estimate * Also raises Q3 Guidance *$enph stonk +47% YTD Enphase trading up 7%+ in After Hours Trading! [hyperbitcoins tweeted about ENPH]
[c2_0524] (twitter, SPY) $SPY Sweeper locks in solid gains from a new opening position made this morning (exit highlighted in blue). [ThetaWarrior tweeted about SPY]
[c2_0525] (twitter, MSFT) We may have some bearish engulfing candles today but the charts may be influenced by more than just daily signals as everyone gets positioned into the major earnings & FOMC finale next week. Tues 7/26 AMC: $MSFT $GOOGL $V Wed 7/27 AMC: $Meta **FOMC** Thurs7/28 AMC: $AAPL $AMZN [kpak82 tweeted about MSFT, GOOGL, V, META, AAPL, AMZN]
[c2_0526] (twitter, MSFT) $MSFT just can get away from $280 in either direction [the_chart_life tweeted about MSFT]
[c2_0527] (reddit, MU) Micron $MU YOLO at yesterday's close ($1,090 strike call, 6/26 exp) yesterday was hell all the way until earnings. today's open should put me near 100% unrealized gains gg and gl to all
[c2_0528] (reddit, UK) Trump’s H200 chip shift exposes the real battleground in the AI race with China Trump's move on H200 shows blanket chip controls aren't enough anymore. China wins by building massive, reliable systems fast. Allowing regulated H200 sales gives us oversight, billions in revenue for US innovation, and stops them from fully breaking free.
[c2_0529] (twitter, CL) The relative weakness in Crude oil (versus US stocks) is also showing up in countries with high economic exposure to the Crude oil industry... $KSA $QAT $CL_F [the_chart_life tweeted about QAT, KSA, CL=F]
[c2_0530] (reddit, PT) Best yield trading platform 2026 and why? My vote goes to Pendle for offering the best fixed rates on stablecoins and RWA through their PT products. I'm curious what you guys enjoy using? Can be on any chain and ecosystem
[c2_0531] (reddit, MSFT) Bill Ackman says it’s one of the best times in a long time to buy quality stocks Someone in this sub said here, you know we hit bottom when Bill Ackman comes and says sell it all and creates fear (did in 2020). He goes ahead and scoops up stocks cheap right after. Now he said to go ahead and buy the quality stocks and brushes off fear - with the opposite rule I’d assume these guys are going to unload a bunch of ‘quality’ stocks to retail and creating the story for it.
[c2_0532] (reddit, AAPL) Best trade of my life Bought Apple Calls July 31st when it was 203$ a share on the hopes on them being too big and smart to get hit by tariffs as well as their AI program getting better
[c2_0533] (twitter, ARM) $ARM miss but not trading down much [Don't follow Shardi B if you hate Money tweeted about ARM]
[c2_0534] (twitter, COST) Dude it feels like $COST is just the inflation hedge [ThetaTendies 🔁 ThetaTendies about COST]
[c2_0535] (twitter, AMZN) $AMZN Whale scalps quick $420K? [Theta Warrior tweeted about AMZN]
[c2_0536] (reddit, ETH) Wallet drain As title says, Two days ago I was bridging my eth from eclipse chain to base chain using relay link and when I bridged the eth to my metamask wallet, the eth vanished. I checked through blockscan and it showed an internal transaction being made after two seconds of bridging. Is there any way the transaction being made after reversed or are my funds lost forever?
[c2_0537] (reddit, IREN) Top High Premium yield Tickers for Today.. Here are some High Premium Yield tickers I am tracking: High Premium Yield Calls: $HL:94.043% $NBIS:93.8721% $CRWV:113.062% High Premium Yield Cash Secured Puts: $IREN:130.469% $MSTR:101.514% $AU:263.965% [Source](
[c2_0538] (twitter, GOOGL) $GOOGL 2779 fib resistance $TSLA all about yesterday's high, close [rachels_44 tweeted about GOOGL, TSLA]
[c2_0539] (reddit, RGTI) How to deal with missing out (RGTI) I made over 100% on RGTI but I could’ve made so much more if I held for 10 more minutes. 🤡 Overtaken by regret, should I fomo back? 🫣
[c2_0540] (twitter, ROKU) Well... that was a beautiful mistake. I was going to ladder into a $ROKU position. I entered, looked away and looked back... I was down $4,000 in 3 minutes! I accidentally added an extra zero onto my order!! Whoooops Doubled down, got a rejection and got the hell out for profit [Nebraskangooner tweeted about ROKU]
[c2_0541] (reddit, COIN) What's your best recommended coins to invest in as the market recovers. Curious what people's thoughts are on some underrated coins that may have fallen pretty far from their highs. Also, what makes your underrated coin different from all the others. What's the value proposition in the next 3 months to a year?
[c2_0542] (twitter, THC) Cannabis under high magnification with an increased depth of field. Tens of thousands of images are composited to make a video in hyperfocus. #science #nature #photography #plants #wow #amazing #thc #cbd #cannabis #marijuana #indica #sativa" [hyperbitcoins 🔁 CannabisHubHQ about]
[c2_0543] (twitter, NVDA) ”MAY BE A BIT OVERHYPED” Lol > [@Investingcom]( > ⚠️ JUST IN: > > *BILLIONAIRE INVESTOR STANLEY DRUNKENMILLER CUT HIS NVIDIA STAKE IN LATE MARCH, SAYS AI MAY BE A BIT OVERHYPED SHORT TERM > > $NVDA [HZ quote tweeted Investing.com about NVDA]
[c2_0544] (reddit, AMZN) $74k AMZN Gain Today- Happy Halloween Kiddos! Sold half of my January $250 calls this morning. Have another 50 contracts riding. Next stop $300!
[c2_0545] (reddit, GOOGL) Debit Spread I have a question and will appreciate if someone can answer. I am new to debit spread and trying to learn paper trading in thinkorswim. If you see the image, cost or trade price is .60 for each trade and option spread is 1 so my maximum profit can not be more than .40 per trade. I do not understand how can my P/L be 520...
[c2_0546] (reddit, XRP) I’m thinking of selling my 66% XRP Bag. If It reaches 2,3€ per XRP, I might sell. Why? I have lost conviction. Last year I was full XRP soldier, but after these months I only trust Bitcoin (and some Meme to play). No utility, winning vs the SEC did nothing, RIPPLE holding the mayority… What you guys think? Has anyone did something similar? I would keep 33% incase it goes very high in 5-10 years.
[c2_0547] (reddit, CBOE) CBOE plans to launch 24-hour trading for U.S. equities to meet rising global demand for access to American markets. 👍 The exchange operator says it will seek approval from the SEC to move forward with all-hours trading, five days a week.
[c2_0548] (reddit, SI) How's your XIRR right now? My XIRR is currently at the lowest depths of the ocean lol. I’ve been doing SIPs for about 1.5 years but most of the amount was invested in the last 6 months. What's your XIRR and how long have you been invested?
[c2_0549] (twitter, VS) $TSLA broke monthly downtrend. Like it above Friday's high (252) for possible run into earnings report on 7/23. I have major support zone now at 240. > [@RajAlaparthi1]( > @rachels_44 $TSLA [VS Options Trading quote tweeted rk about TSLA]
