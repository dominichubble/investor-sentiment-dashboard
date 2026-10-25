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
----- ITEMS (50: c2_2150 .. c2_2199) -----
[c2_2150] (twitter, TIVC) $TIVC bios are hot. Won’t be selling a share under $5. Full DD thread coming this week 💎 [Bullish_Trades tweeted about TIVC]
[c2_2151] (twitter, GLD) $GLD Gold still working on the monthly cup & handle breakout ☕️ timing is starting to feel right [BTstocktrading 🔁 TrendSpider about GLD]
[c2_2152] (twitter, TSLA) $tsla The higher strikes for today. [HackermanAce tweeted about TSLA]
[c2_2153] (reddit, QQQ) Diary of a regard Lost about $100k in 2025, revenge traded into major green this year, mostly with SPY/QQQ 0DTEs. Need a hypnotist or some shit to break the cycle. Thankfully I’ve parked about $395k of it into safe stocks but I know I’ll need to sell some for Uncle Sam so 🥭 can continue to squander it.
[c2_2154] (twitter, PLUG) $PLUG Daily Chart We have 3 scenarios : 1. Falling wedge breakout, price is approaching the previous broken out trend line 2. Broadening bottoms , price is approaching the lower boundary of the wedge 3. Zooming out - Inverse head and shoulders pattern / Price is forming the… [Mr Chart Norris tweeted about PLUG]
[c2_2155] (reddit, HOOD) Robinhood vs Fidelity I’m currently 17, almost 18 in a few months, and I have a custodial Fidelity account. I was thinking about switching my individual account to Robinhood for IPOs, options, individual stocks, etc. However, if I did, I would keep Fidelity for my Roth IRA just for long-term security. Thoughts on all this? Thank you.
[c2_2156] (reddit, BULL) Advice for splitting money between short term options and long term holdings Been trading options for 1.5 years now. And made some significant money. I do want to set aside some money to play it safe and just invest that for the long term while also trading options too. Right now I have it as 80% is shares I own and is meant for long term. 20% is options trading. How does everyone else go about this?
[c2_2157] (twitter, TIVC) $TIVC Juicy earnings came out early today. Still waiting for the news that will move this to all time highs. It’s coming [Bullish_Trades tweeted about TIVC]
[c2_2158] (twitter, EBAY) Of course watching $SPY but also watching $EBAY $HOOD $TSLA today. If we do have a red day, I'll be looking for big names coming into demand levels. [JMA_Trades tweeted about EBAY, TSLA, HOOD, SPY]
[c2_2159] (reddit, CRSR) The OMER play! Ladies and Gents, I present to you the OMER play..... For those who missed the CRSR boat. A heavily beaten down stock with a 800m market cap and a new FDA approved drug, the only kind in the market. Soon to be approved in the European market. Already has a deal in place with Novo Nordisk valued at 2.1 billion. With a low trading volume, a price target of $33 and a short float of 23%..... The only way is up! To the MOON!
[c2_2160] (news, AB) Stocks With Rising Relative Price Strength: AllianceBernstein
[c2_2161] (reddit, TSLA) You all told me to sell and guess what I did? You regards have taught me a lot but one thing I refuse to do is follow the advice that people give here
[c2_2162] (reddit, ETH) 📢 Join All Core Devs - Testing 36 | ACDT 36! 🗓 Monday, May 12, 2025 🕓 14:00 UTC 📺 Live on EthCatHerders X/Twitter😺 Stay informed on Ethereum's latest updates including upcoming Fusaka upgrade, and more. Don't miss it! ✨ [
[c2_2163] (twitter, SPY) Trap the bulls one more time for me send it to 444 440 into the weekend and everyone gets a kiss on the cheek > [@ATMSnipes]( > I'm going to make it simple > > We're likely heading to $440, $435 then $430 for W4 on $SPY > > If we fail to make new highs into $470 (I think we do) it's chalked.. see you at $410 - $415 > > For now enjoy the opportunity to short & if anyone is screaming the world is ending; ignore it [Avery quote tweeted about SPY]
[c2_2164] (twitter, PT) $AMZN PT raised to $235 at Jeffrries Now Bezos just needs to GTFO the way 😤 [Don't follow Shardi B if you hate trading retweeted Paper Gains about AMZN]
[c2_2165] (twitter, AAPL) The $NASDAQ on an impressive rally led by $AAPL which is almost at All time highs again. Currently at the golden pocket fibonacci retracement so an interesting area to watch. To trade Indices, FX & Commodities using Crypto, consider trying out: [DaanCrypto tweeted about AAPL, NASDAQ]
[c2_2166] (twitter, PFE) $PFE Just finished building my short position. I want to see it reject today otherwise I may have to consider closing. Currently a underwater about 1% on my short. [Nebraskangooner tweeted about PFE]
[c2_2167] (twitter, CMT) $BTI also sporting a 9%+ yield, although I don't know how safe that is. you'd have to ask the dividend peeps. [Ian McMillan, CMT tweeted about BTI]
[c2_2168] (reddit, TSLA) +130% on Astrotech 🚀 Jeez, my friend told me about it pre-market and I thought heck why not and bought in at $7, sold at $15.9 (2nd pic) for gains of 43.93k SGD (abt 34k USD) Rest of the positions still holding, if it continues to rise, whatevs, I don't think I can be upset Note: The ticker is banned due to too low cap, this post will prob get taken down 🫡 but I just had to share
[c2_2169] (reddit, GOOGL) According to this pattern, Bitcoin will reach $10B dollars next year.
[c2_2170] (twitter, AAPL) $AAPL Will have a final rally up to $210 before beginning a downtrend that takes it down to $53 and less the coming years. This downward trend mirrors the company's apparent loss of innovative edge, as seen in the repetitive design of its last four iPhones. [HZ tweeted about AAPL]
[c2_2171] (reddit, MCD) McDonald's McRib Has Returned - Historically, Every McRib Comeback Since 2020 Has Marked A Bitcoin Rally Source: [
[c2_2172] (twitter, SPY) $SPY Rising wedge into resistance. 🔪 [ThetaWarrior tweeted about SPY]
[c2_2173] (twitter, AAPL) Let's roll...Bye Felicia $AAPL [ShardiB2 tweeted about AAPL]
[c2_2174] (twitter, ALT) Should be some nice bounces on $ALTS after we wick 600B [@TheEuroSniper]( If we get a $ALT bounce. Red box is where cut excitement to zero. [TheEuroSniper 🔁 TheEuroSniper about ALTS, ALT]
[c2_2175] (reddit, QQQ) 🚨Oil Just Broke The Market And The Fed Cannot Save You 🚨 - YouTube The Fed cannot fix this. Powell told Harvard he is looking through the oil shock. Meanwhile [$USO]( is ripping, [$QQQ]( is bleeding, and [$MU]( just lost 30% in 8 days. Stagflation is the word nobody wants to say out loud. Game plan and key levels in this video for swing trading including how I'm hedgin in this market.
[c2_2176] (reddit, BTC) IBIT Option Gains/YOLO I didn’t want to make two posts so this is a gains from today post and the continuing YOLO. My first ever 6 figures day of realized + unrealized as well
[c2_2177] (twitter, FF) #FF @jimcramer Nailed this one ledge. Well played. [TraderMotif tweeted about FF]
[c2_2178] (twitter, DDOG) $ABNB - Downgraded to Neutral at BTIG $TWLO - PT $270 from $420 at JPM $REGN - PT $728 from $754 at Truist Securities $DDOG - PT $190 from $236 at Needham $DASH - PT $150 from $205 at Mizuho $FB - Downgraded at KGI Securities, PT $270 [rachels_44 tweeted about ABNB, TWLO, REGN, DDOG, DASH, FB]
[c2_2179] (reddit, ADBE) On the AWS ddos attack The Amazon Web Services have been ddos attacked (thats the rumor). companies that use AWS (netflix, twitch, adobe, mcdonalds, capitol one, pfizer, bmw, coca cola, shell, all use it just to name a few) will have limited use. Is this an opportunity to bet against amazon? (im new and dumb so idk how this works)
[c2_2180] (twitter, BB) $TLT - Hit the lower BB again yesterday. If the April low is broken the next target would be a gap fill at 85.12. TLT tried bottoming a few times in 2024 but the Fed has repeatedly postponed its first rate cut outlook. Until the 1st cut is a fact lower lows are still possible. [ReciKnows retweeted CyclesFan about TLT]
[c2_2181] (twitter, UNG) The largest jumps in RS over the last two weeks have come from Retail, Pharma, Nat Gas, Healthcare, Aerospace/Defense, Biotech, and a large grouping of Fixed Income areas $RTH $PPH $XLV $UNG $PPA $BBH $SHY $MUB $JNK $CMBS [Ian McMillan, CMT tweeted about UNG, SHY, MUB, CMBS, XLV, PPH, BBH, RTH, PPA, JNK]
[c2_2182] (twitter, BTO) $SPY Whale Profits: 6/15 BTO 362P @ $6.4 x 34.1K contracts 6/16 STC 362P @ $9.03 = +$8.9M 6/16 BTO 354P @ $6.55 x 28.4K contracts 6/16 STC 354P @ $8.5 = +$5.5M One Day Gain: +$14,400,000 💰 [ThetaWarrior tweeted about SPY]
[c2_2183] (reddit, BULL) Cramer said that the stock market was resilient on Thursday...and then Friday happened
[c2_2184] (twitter, FGI) $FGI Alerted $4.35 currently trading $6.25 🔥 hit a Premarket high of $7.50+ [Ultra_Calls tweeted about FGI]
[c2_2185] (news, GEO) Top Dividend Challenger Net Gains Are By GEO, DDR, And Andeavor In December
[c2_2186] (reddit, GLD) Someone tell me I got lucky Bought GLD call on Thursday @ $307 for 4/21. Avg cost was $1.71 and just sold at $8.45. Only $675 in profit but I’m a newb and told myself I wouldn’t put more than $300 into this account to test the waters. Why did that work and did I get lucky?
[c2_2187] (reddit, MSFT) Thoughts on Accenture’s recent performance? Accenture had a very good run in the 4 weeks leading up to the last week of Jan. Then in the last 5 days dropped from 288 to 258, which is pretty nasty. Today only, in the wake of a meltdown for MSFT, it is dropping 4%. What are everyone’s thoughts about this behavior? I am surprised by this drastic shift in momentum
[c2_2188] (twitter, RIVN) Would really love if $RIVN did something like this because I might have been an idiot last week and aped some more calls [K A L E O tweeted about RIVN]
[c2_2189] (reddit, AI) Is EWY still a good investment? Since the war South Korea markets have been getting cooked and I want to hear others opinions if it was overvalued due to AI hype in the first place. Is it worth cutting my losses after putting money in recently and moving it into VT instead? I don’t mind risk but don’t want to have money in there doing nothing or going negative when I could put it somewhere else.
[c2_2190] (twitter, OXY) $OXY This looks like a bullish reaccumulation setting up. I'm going to pick a little up [Nebraskangooner tweeted about OXY]
[c2_2191] (twitter, DE) $DE Nice sweeper this morning too [StonkTrump tweeted about DE]
[c2_2192] (news, ABEV) 27 'Safer' Dividend Consumer Defensive WallStar Stocks Declared To August 2019
[c2_2193] (twitter, SMFL) $SMFL Highs, let’s break $3 > [@Bullish_Trades]( > $SMFL probably my favorite from the list with acquisition catalyst expected any day now [BullishTrade quote tweeted BullishTrade about SMFL]
[c2_2194] (reddit, IBKR) [Investing in ETFs] hi all!! i’m 18 and i want to invest in ETFs. I can only invest a few 100 dollars a month and i’m looking for a cheap broker. For some reason i can’t make an acc in Tiger brokers. IBKR fees is alr quite high. Apparently TDA sg can only make margin acc so idw to use that. Is there any other broker that offers low commission fee? Thankss!!!
[c2_2195] (twitter, GOOGL) My trades today $GOOGL 105c +15%, open Not a lot of great risk to reward plays that met my style. Had to sit & exercise patience today! 😀 [optionsprochick tweeted about GOOGL]
[c2_2196] (twitter, AAPL) $AAPL major support to note at 159 (200sma) [rachels_44 tweeted about AAPL]
[c2_2197] (twitter, MA) Lets all be honest here and call a spade a spade. $TWTR has been ran by failures for far too long. I've worked in tech my entire career, this is what we call a zombie company. Rarely above the 200MA and shit growth performance. Investors should be lucky to cash out there. [@elonmusk]( Taking Twitter private at $54.20 should be up to shareholders, not the board [CryptoCX1 🔁 elonmusk about TWTR]
[c2_2198] (twitter, BTC) #Bitcoin continues its seasonal chop season, which tends to run from March until early October. RSI still hovering around the weekly level - as is customary as well. Couple more weeks to go until Q4 👇🏼👀 [Jelle tweeted about BTC]
[c2_2199] (twitter, NKE) $NKE has added a letter to their ticker this morning $NUKE [Nebraskangooner tweeted about NKE, NUKE]
