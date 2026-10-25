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
----- ITEMS (50: c2_0400 .. c2_0449) -----
[c2_0400] (twitter, TLRY) I take it $TLRY earnings went poorly? Not a mega nuke, just round tripped today's gains. [Nebraskangooner tweeted about TLRY]
[c2_0401] (twitter, FUTU) are these the actions of a tyrannical man-baby? yes. no doubt. but at the end of the day - and this is the important part - all is fair in love and markets > [@BongCapital]( > So basically, Bill Hwang got squeezed for 400% on a short position $FUTU, and responded by… manipulating and squeezing other shorts. > > This is the worst supervillain origin story ever. [the_chart_life 🔁 BongCapital about FUTU]
[c2_0402] (reddit, CRM) Software and Cyber Security stocks are likely going higher: Jensen Huang says the market got them wrong Once the next round of earnings comes out for companies like NOW, PANW, CRM, and CRWD they are going right back up to their previous highs. Companies simply cannot use ai to replace these software/SAAS products yet, and it may be years before ai is good enough for that to happen.
[c2_0403] (reddit, CLF) CLF Yolo update.. record day and havent sold one position. All in CLF
[c2_0404] (twitter, ROKU) $ROKU Weekly Chart The chart was ready forgot to post it several days before / however it moved as i've expected Breakout / Retest Scenario [Mr Chart Norris tweeted about ROKU]
[c2_0405] (reddit, BTC) With Bitcoins drop below 73k, Bitcoin has erased all gains since the 2024 Trump election win.
[c2_0406] (twitter, AAPL) $AAPL first profit miss since 2016 and reversal candles everywhere [MacroCRG tweeted about AAPL]
[c2_0407] (reddit, BULL) Best 3 ETF portfolio I’m a 22 year old from Canada that wants to get into investing. I’ve been doing quite a bit of research on what to invest in at my age with an investing range of 30-40 years. My goal is to choose 2-3 ETFs that are great for long term growth. My core ETF would be xeqt (50-60%) for lots of exposure VFV (15-20%) for tech and I’m still searching for a third with more risk but more chance of more profit. I’m leaving towards SMH or SOXQ. Let me know what I should do
[c2_0408] (twitter, SMA) $COIN Rejected at yearly resistance, firm loss of the 200SMA, a double top in some cultures. Send it to hades. > [@News_Of_Alpha]( > FORTUNE: Coinbase CFO says Kamala Harris is using firm to accept crypto donations > > [Crypto Mikey quote tweeted Tree News about COIN]
[c2_0409] (twitter, NVDA) One of the greatest trader of all time is shorting $NVDA [Mr Chart Norris tweeted about NVDA]
[c2_0410] (reddit, BYD) Sell Tsla BYD earnings are eating them for lunch and dinner. Buy TSLQ Sell Tsla BYD earnings are eating them for lunch and dinner. Buy TSLQ
[c2_0411] (twitter, COIN) $COIN Coinbase set to gap down and test near all-time lows [BigCheds tweeted about COIN]
[c2_0412] (reddit, MA) Who’s ready for some saucy price action 🦍🟢 This is a textbook bullish pennant. And it will line up perfectly with the third golden cross (50MA/200MA crossover) since April 2024. Each of the previous 2 precipitated huge upswings. Let’s gooo boys Hedgies are Fuk, they’re out of Luck, it sucks to Suck, harder than Zuck, okay I’ve reached character requirements
[c2_0413] (reddit, HOOD) 5 Years Ago Today, Cascading Clearing Firm Default Was Occurring, January 28, 2021. Robinhood was 2nd place among the defaulting firms.
[c2_0414] (twitter, FA) #ff my bro @Trader_XO shares solid FA and TA in his free youtube streams. half of ct now copy his shit and regurgitate it [trader1sz tweeted about FF]
[c2_0415] (reddit, ETH) Explain like I’m 5 Early 2025 I went into a rabbit hole of ETFs etc. eventually, I forgot or lost interest. Today I visited the apps and I am looking into continuing investing. Can explain to me which ones I should or shouldn’t continue?
[c2_0416] (twitter, WSBC) Is it safe to get back into $wsbc? i had cut on the recent pump. [HackermanAce tweeted about WSBC]
[c2_0417] (reddit, GOLD) RWA Tokenization is gaining serious momentum. Real estate, private credit, treasury assets—even infrastructure—are slowly moving on-chain. **The promise:** • Fractional ownership • Global liquidity • Faster settlement • Transparent asset management But here’s the big question **Which asset class do you think will dominate RWA tokenization first?** A) Real Estate B) Private Credit C) Treasury/Bonds D) Commodities Curious to hear your thoughts.
[c2_0418] (reddit, AI) First good gain Glad I held when the wife told me to sell months ago.
[c2_0419] (reddit, TSLA) META gap up on Cloud business to sell excess AI Compute Meta is reportedly developing a cloud business to sell access to excess AI compute, per Bloomberg. The internal initiative is called Meta Compute. The plans being considered: AI model access hosted on Meta infrastructure, similar to AWS Bedrock Raw AI compute capacity, closer to CoreWeave Developer access to Meta’s data centers, chips, and models Meta stock up in the pre market while neo cloud providers are gapping down
[c2_0420] (twitter, SPY) $TSLQ Daily (inverse $TSLA). #TSLQ printed a hammer candlestick marking a potential reversal pattern. Break above 45 confirms the move Note bullish divergence on MACD #TSLA $QQQ $SPY $IWM $ARKK $XLK $Nasdaq [reciknows tweeted about XLK, ARKK, NASDAQ, IWM, TSLA, SPY, TSLQ, QQQ]
[c2_0421] (twitter, RKLB) $RKLB > [@kholov23]( > $RKLB Weekly Chart Update 0.2 > > Inverse head and shoulders breakout / retest [Mr Chart Norris quote tweeted Mr Chart Norris about RKLB]
[c2_0422] (reddit, MU) Some Memory Quickies Realized gains on overnight MU calls, couple weeks on the SNDK shares. Still holding DRAM shares and calls, and MU stock.
[c2_0423] (twitter, META) Added 500+ $META calls while it was below support today... truly stomach turning watching every tick.. my biggest options positions ever Playing the bounce here with my finger on the trigger if I need to unload.. in profits now of course [Don't follow Shardi B If You Hate Money tweeted about META]
[c2_0424] (twitter, COIN) $COIN- This is why you trade the risk and charts and not the news. All week it was FUD and negative attacks on Coinbase. Rating agencies moved it to a sell, people saying they were going bankrupt, etc. Meanwhile it's up 15% today. [IncomeSharks tweeted about COIN]
[c2_0425] (reddit, NVDA) Nvidia Calls +1100% Gain Bought these as a levered directional play for the earnings run up, closed all today. May buy a few options to actually play earnings too. Probably my best trades return % wise ever
[c2_0426] (twitter, TIVC) $PGY from $2.40 to $20+ in a week. Truly believe $TIVC has all the ingredients for a move like that too, will continue to add under $2 🤙🏼 [Bullish_Trades tweeted about PGY, TIVC]
[c2_0427] (twitter, SPY) 🧐 [@BCalusinski]( $SPY $1 BILLION singular Dark Pool print at open Someone cashing out after this big rally @CheddarFlow [CryptoParadyme 🔁 BCalusinski about SPY]
[c2_0428] (reddit, AMZN) Amd buying/selling or holding Hi ladies and gentlemen what you all doing with ur AMD position buying, selling or holding. I m up over hundred percent jst lil confused about selling Amd but again longer horizon Amd should do well or not which is why I am at stage of trimming half 25 percent to put cash into something else. For example meli, unh even amazon I believe there are still undervalued names out there. Thanks 🙏
[c2_0429] (twitter, DOCS) $DOCS Coiling up nicely here. Its made another inside day, looking for any momentum above $34 Resistance: $35.75, $38.34, $40.31 [optionsprochick tweeted about DOCS]
[c2_0430] (reddit, DAO) so ACI is leaving Aave DAO — anyone else worried about governance? just saw the news, ACI (one of the biggest delegates) is winding down over the next 4 months. marc zeller said something about governance standards and voting issues. tbh this feels like a bigger deal than people are treating it? they've been pretty central to how Aave runs. curious what happens to proposal quality now
[c2_0431] (twitter, ETH) Fundamentally I'm bearish on $sol when it comes to memecoins I prefer ones off $eth. [Amdtrades tweeted about SOL, ETH]
[c2_0432] (reddit, BULL) I know just lines but MACD is about to cross the 0 line MACD is about to cross once it crosses that zero line in the trend line looks very bullish Dorito on the MACD could be it to break that indicates bears are getting tired.... gme gme gme gme gme gme gme gme gme gme gme gme gme gme. gme gme gme gme gme gme gme gme gme gme gme gme gme gme gme gme gme gme gme
[c2_0433] (reddit, CFR) Polypropylene prices rise globally following outbreak of Middle East war The Middle East accounts for close to 25% of global PP and polyethylene exports, according to S&P Global Energy data. Polypropylene is a type of plastic used to make everything from food containers and water bottles to household appliances and automotive parts. Asian prices have seen the most significant increase. Platts assessed the PP injection CFR Far East Asia spot price at $1,180/mt, up $330/mt since March 2.
[c2_0434] (twitter, QQQ) $SPY $QQQ Long time followers know I have a solid record sniffing out major reversals. You can go through my twitter feed. I dont erase anything. Anyway, Its really really stinky right now. My nose is bleeding. Thats my technical analysis. [kpak82 tweeted about QQQ, SPY]
[c2_0435] (reddit, BABA) China’s Treasury gambit: A decade in the making "The real question isn’t whether the US dollar will lose its crown overnight. It won’t. The real question is whether China’s moves will convince everyone else to jump ship at the same time. Right now, other countries are hesitant. But the second they see that it is safe to ditch the dollar, it becomes a self-fulfilling prophecy. Once the first few dominoes fall, the rest will follow."
[c2_0436] (reddit, ETH) Meanwhile I’m just trying to retire peacefully someday… * Tech stocks crashing * Inflation still high * The Fed is raising rates * Commodities going crazy * Real estate overpriced * Crypto collapsing again * Economic slowdown fears And I’m just trying to build a stable portfolio.
[c2_0437] (twitter, QQQ) $TLT Daily. #TLT failed to reclaim 50sma, may find support at the rising support from the ascending triangle (ideally holds > 98) $SPY $QQQ $IWM $DIA $XLF $GLD $VIX $KRE $DXY $TBT [reciknows tweeted about GLD, KRE, DXY, XLF, DIA, VIX, SPY, TBT, TLT, IWM, QQQ]
[c2_0438] (reddit, INTC) Intel stock price just passed dotcom bubble level EPS 5 years ago 5$ SHARE PRICE 60$ TODAY 2026 EPS WILL BE 1$ AND SHARE PRICE IS NOW AT 77.
[c2_0439] (twitter, DIA) gotta be on a full stomach at this point. 3 months at the same spot. either this thing has been carb loading for another leg, or it's gonna puke it up over the coming weeks. > [@the_chart_life]( > bulls really trying hard to eat through this supply at 34000 on the Dow. $DJI $DIA [the_chart_life 🔁 the_chart_life about DJI, DIA]
[c2_0440] (twitter, GOLD) Have not seen this perspective on gold, not even once. $GOLD [JJcycles tweeted about GOLD]
[c2_0441] (reddit, META) Implementing Transfer Hooks on Solana (Anchor 0.31) A technical breakdown of integrating Transfer Hooks using the updated Anchor framework. This covers the architecture for the `Execute` instruction, managing `ExtraAccountMetaList`, and standard validation patterns for Token Extensions. Read the full breakdown here: [
[c2_0442] (twitter, PHG) Europe has been strong. Medical Devices have been strong. A little of both worlds... $PHG [the_chart_life tweeted about PHG]
[c2_0443] (twitter, DBD) $DBD Still going strong. Currently up 12% on position. [Nebraskangooner tweeted about DBD]
[c2_0444] (reddit, DRS) Did we DRS all the warrants yet? I know I was able to DRS a bunch of warrants a few months ago, have we not DRSd them all yet? There’s so few, and they’re so cheap. It would be a piece of cake. I saw 66 million shares drs’d, and 6000 warrants exercised. But one sizable investor could buy and DRS every warrant issued. Or one overly enthusiastic sub. Just saying
[c2_0445] (twitter, BTC) Markets ready to rip your face off. #BTC #ALTS #SPX #Metals [JJcycles tweeted about]
[c2_0446] (reddit, JPM) Clarity Act Passage Could Trigger Crypto Rally Says JPMorgan Its clear as day that they are suppressing the markets and trying to use a potential market rally as a hostage in order to get passage of the Clarity Act.
[c2_0447] (news, A) Agilent: 2 Decades Of Portfolio Moves Left Investors With A Good Business
[c2_0448] (reddit, META) Goodbye to AI - Meta CEO Mark Zuckerberg joins Sam Altman and acknowledges that artificial intelligence could be on a bubble
[c2_0449] (twitter, WMT) $WMT Wally World is still stuck in this rising channel. W/ ER approaching next week a rally into $130 supply would set this up for a nice dump 🎯$117, $112 [ATMSnipes tweeted about WMT]
