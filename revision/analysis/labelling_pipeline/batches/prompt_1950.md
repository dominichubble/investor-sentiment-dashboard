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
----- ITEMS (50: c2_1950 .. c2_1999) -----
[c2_1950] (reddit, AMZN) Corning shares jump 4% after striking deal to power Amazon AI data centers in U.S.
[c2_1951] (reddit, MU) Micron Up 60% in 4 Weeks. Writing Puts. After Micron surged 60%+ in 4 weeks, cash-secured puts are generating rich premium right now. This week I break down how to use high IV to get paid to wait for a pullback on $MU.
[c2_1952] (twitter, SPY) $SPY dark pools Biggest accumulation of the day at 441.84 ... over 5mil shares at 2.4Bil in premium Would love to see a retest of that level for a nice intraday dip buy [StonkTrump tweeted about SPY]
[c2_1953] (twitter, PLG) Looks like $plg did the flip today, will look to scale in with platinum taking off. [HackermanAce tweeted about PLG]
[c2_1954] (reddit, AMD) Finally gain porn Got spooked by the drop at open and sold way too , but I’ll take the gains
[c2_1955] (reddit, COIN) 32 Hi guys. Just got my 32nd coin. What do you think about validator? Is this good idea to become a validator? Thank you guys 🤓
[c2_1956] (twitter, ENPH) $ENPH 1D Reversal structure on the 1D. Looking for a rejection from the range. Plus, everyone seems bullish on this one🤔 [rbc_trading 🔁 rbc_trading about ENPH]
[c2_1957] (reddit, TWLO) It's not too late to join Hedgies already know from insider information that Figma will crush earnings expectations, that's why it's rising into earnings. This will blow past $35 after earnings. TEAM and TWLO already showed what happens when a SaSS companies beat earning this quarter. You can still get in.
[c2_1958] (reddit, TER) New Diversified ETFs dropped by Xtrackers * XEQ2 - LU3116008346 - Xtrackers Diversified Portfolio 20% Equity UCITS ETF 1C * XEQ4 - LU3116008429 - Xtrackers Diversified Portfolio 40% Equity UCITS ETF 1C * XEQ6 - LU3116008692 - Xtrackers Diversified Portfolio 60% Equity UCITS ETF 1C * XEQ8 - LU3116008775 - Xtrackers Diversified Portfolio 80% Equity UCITS ETF 1C TER 0.24% + Transaction costs 0.11-0.16%
[c2_1959] (twitter, FF) #FF My compadre @FoftyOerney Sharp with his macro game and gets it. Solid timeline. [Trader_XO tweeted about FF]
[c2_1960] (twitter, ES) $SPY Daily. #SPY testing 50sma and testing the breakout from the downtrend line from ATH's. To keep bullish upside in play, needs to hold > 390 $SPX $ES_F #SPX #ES_F $XLK $IWM $QQQ $AAPL $TSLA [reciknows 🔁 reciknows about SPY, TSLA, IWM, XLK, AAPL, ES=F, QQQ, SPX]
[c2_1961] (reddit, BULL) What Are Your Moves Tomorrow, June 04, 2026 This post contains content not supported on old Reddit. [Click here to view the full post](
[c2_1962] (reddit, MSTR) For the coming week - sell put 1. MSTR 200-230 2. BE 3. IREN 4. RGTI Take a good 30% discount for a week, max 2 weeks.
[c2_1963] (twitter, CHWY) $CHWY Weekly Chart 🐈 Falling wedge is breaking out on weekly candle chart MEOW [Mr Chart Norris tweeted about CHWY]
[c2_1964] (reddit, AMD) Dove into OVV today Feeling very bullish on it, I don’t touch options just longer term holds. Plan is to ride this and maybe pluck my initial if it hits 100%.
[c2_1965] (news, WBD) Paramount beats earnings expectations for Q1 with WBD deal on the horizon
[c2_1966] (reddit, UWMC) Any TWO$ Bagholders? Remember to Vote against the merger with UWMC$ Any TWO$ Bagholders? TWO is worth a lot more than 2.3 shares of UWMC... especially after a JOKE of a CEO they have over there. TWO book value is $11 while UWMC book is like 50 cents. UWMC also has a lot of nontrading shares that keep converting to trading shares and that is why the CEO keeps selling because he knows UWMC is WORTHLESS. So remember to VOTE against the Merger.
[c2_1967] (reddit, ASTS) Which companies am i missing from my "Space" themed investments? My current "Space" themed investments include Rocket Lab (RKLB), AST Space Mobile (ASTS) and Intuitive Machines (LUNR). And of course soon to be SpaceX due to its soon listing in many ETFs. Though other than RKLB, ASTS and LUNR. What am i really forgetting?
[c2_1968] (reddit, AI) Fuck Semiconductors Qqq and SOXX have now gone in reverse. Qqq was up today almost one percent while soxx was down 1% Semis have no more room to grow. People's risk exposure is rotating out of semis The gamma squeeze is done. Fuck your calls on semis. Im not selling this shit until it goes to 1mln. Get fucked
[c2_1969] (twitter, AMZN) $AMZN (weekly) hanging man with bearish confirmation [kpak82 tweeted about AMZN]
[c2_1970] (reddit, TSLA) AMD; SPY; HUT8 AMD: $2160->$5495 SPY: $1300-> $7800 (this one stings a little because I sold at open and looked back later and saw where I could’ve been, BUT profit is profit.) HUT8: $2960->$32440 Pocket change to some, but my single best swing trading day ever.
[c2_1971] (twitter, QQQ) $QQQ Struggling with resistance here.. [Don't follow Shardi B if you hate trading tweeted about QQQ]
[c2_1972] (reddit, NFLX) Two minutes of silence for Netflix covered calls this week As I see, there is a very high open interest at $84. Stock squeezed to $94 in the after hours. Let’s see if this goes down tomorrow.
[c2_1973] (news, AA) The Good Business Portfolio: Q3 2018 Earnings And Performance Review
[c2_1974] (reddit, BTC) OMG, What’s going on? the market started going up again. shorts are getting liquidated
[c2_1975] (reddit, MSTR) Bernstein expects Bitcoin to Rise, Strategy (MSTR) Buys More Bitcoin
[c2_1976] (reddit, BULL) So help me god NASA YOLO Crashed it and loaded up again just going with the full port now. Boom or bust no dd. Dd is Elon saving my port on June 12th gl regards.
[c2_1977] (reddit, TSLA) Gay Bear Summer Mostly traded call credit spreads on SPX and one big put purchase on TSLA (would not recommend.)
[c2_1978] (twitter, LH) > [@IncomeSharks]( > #SPX 6 weeks later. > > Still no HH, and no downtrend break > > Historically high inflation, and still tightening which we have yet to see all downstream effects of. Early stages are the job cuts we are starting to see > > Imo another LH soon @Pentosh1 I'll be taking some profits too. It's been a great rally since October. Some of these stock plays been moving like altcoins. [IncomeSharks tweeted about SPX]
[c2_1979] (twitter, LYFT) $LYFT Setting alerts in case loss of support. Head and shoulders potentially setting up here. [Nebraskangooner tweeted about LYFT]
[c2_1980] (twitter, SPY) $SPX $SPY I said $507-$509.50 was the sweet spot for bears.. quite sweet it was. Demand at $497.50 - $500 is no longer fresh and can easily be broken. Price structure is suggesting no X wave and we likely have a B wave high in. continuation down can happen into $493.39 then [Avery tweeted about SPX, SPY]
[c2_1981] (reddit, GOLD) After Venezuela blow, Iran supply risks test China's oil strategy
[c2_1982] (twitter, BEN) I think this is just starting and has more room to pump #ben $ben meme High risk meme [Prof.Noan.Ai 𓃵 tweeted about BEN]
[c2_1983] (twitter, TIVC) $AERC Wow printing $14’s now. Up over 800% in a week, those beat down IPO’s are the ones to load up on $TIVC $BEAT [Bullish_Trades tweeted about AERC, TIVC, BEAT]
[c2_1984] (reddit, STWD) Anyone have any feedback Anyone looking at Starwood Property Trust (STWD)? Just announced $0.48 cent quarterly dividend payable to owners as of March 31
[c2_1985] (reddit, CRCL) Biggest win for me I am gonna put all the proceeds to buy more CRCL!
[c2_1986] (reddit, DTE) Defend vertical credit spread opened a 0DTE 6480/6490 call credit spread. today's spike blew through my long leg and closed for a loss. looked to roll up and out but vol was much higher today so it negated any theta gains for later this week. how would you have defended in this instance?
[c2_1987] (reddit, BABA) Whats your opinon of the wisdomtree europe equity income etf? I like the high distribution of the ETF. It has performed very well lately. Probably not the safest ETF either?
[c2_1988] (twitter, SNAP) $SNAP (per request) Looks awful here but is at a key trendline. I'd prefer to wait for a weekly close either way here. Back above $10.70 would be potential for more upside. [Nebraskangooner tweeted about SNAP]
[c2_1989] (twitter, AAPL) $AAPL Trying to hold a LTF double bottom support here with Top Goon X showing bull div [Nebraskangooner tweeted about AAPL]
[c2_1990] (reddit, COIN) ETC? What do you guys think about Ethereum classic? It had a big jump today but I know it could just be a small jump in the grand scheme of things and go back down. Alot of people have written it off as a dead coin/project. What do you guys think of the future of ETC?
[c2_1991] (reddit, GD) Volume coming into aerospace and defense Flow of volume to $ITA which is already a top 10 theme. Something to look out for ($LMR $RTX $GD). On the other hand rare earths and lithium ($REMX $LIT) are losing traction.
[c2_1992] (twitter, F) What do you think $f will do on earnings ? [Ace tweeted about F]
[c2_1993] (twitter, ARM) $ARM Daily Chart Breakout / Retest Scenario After 2 weeks of consolidation under 65, finally price made an impulsive move to the upside ( 65 zone breakout ) aiming for a test of previous highs / 69 / Range : 65 - 69 [Mr Chuck Norris tweeted about ARM]
[c2_1994] (reddit, BULL) gold to 3500 as the title states, does anyone else have a feeling we are gonna keep crashing after some false hope... we shall see how 2026 shapes out, it'll be interesting. Good luck in Q2 yall!
[c2_1995] (twitter, CMT) Oct 2022 had a 100% chance of a recession We are instead looking at real GDP >2% for seven consecutive quarters for the f… [Cheds (Trading Quotes) retweeted Ryan Detrick, CMT about QQQ, DIA, SPY, SPX]
[c2_1996] (reddit, NVDA) If I waited I could’ve made 30k on these contracts Gain is gain but I regret selling for this low
[c2_1997] (twitter, NKE) $NKE Rejected at the resistance I wanted to see flipped. Out for small loss on this one. [Nebraskangooner tweeted about NKE]
[c2_1998] (reddit, NVDA) YOLO nvda calls Could be the best decision of my life or the worst lol
[c2_1999] (twitter, AAPL) $AAPL 2H Seemingly corrective into supply. Hidden bearish divergence. Indices at a key spot as well. Make or break here, bulls need to see this slice through supply. Bears want a rejection to send it to 129 [ChandlerTrading tweeted about AAPL]
