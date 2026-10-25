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
----- ITEMS (50: c2_0250 .. c2_0299) -----
[c2_0250] (twitter, AMZN) $AMZN Monthly all time closing highs. [Theta Warrior tweeted about AMZN]
[c2_0251] (reddit, TD) META triggered a TD9 buy signal With consideration to the current sentiment and geopolitical issues, I do not think that it's a good time to buy right here, despite that TD9 buy signals seems very effective on Meta. Another interesting thing to notice is that the buying point is higher and higher. Any takes?
[c2_0252] (twitter, HOOD) Impressive month for #teamshardi followers! > [@GrimTheWizard]( > @ShardiB2 thank u shardi. sold my $HOOD calls for now, going to touch grass now 🫡🥂 [Don't follow Shardi B if you hate trading 🔁 GRIM about HOOD, TEAMSHARDI]
[c2_0253] (reddit, NVR) Vistry Group PLC (LON:VTY) One of the biggest home builders in the UK. There are a ton of amazing analyses on youtube about how it could be the next NVR. Paying off their debt and repurchasing shares with a deep moat in the market and upcoming / ongoing catalysts with the UK facing a housing crisis. There has been a major drop since the CEO left. Wondering if anyone else is buying the dip at this price range. Thanks!
[c2_0254] (twitter, ALTS) $ALTS - Get back above this support and I'll start posting charts again. This was hopefully the fakeout/capitulation we needed before we can resume back up. [IncomeSharks tweeted about ALTS]
[c2_0255] (reddit, ES) 1 min supply and demand strategy + liquidity ES/NQ This is my 1 min supply and demand + liquidity strategy for ES. I also trade NQ. I would love to hear from other supply and demand traders for tips and just how to become a better and more discipline trader. People who already getting payouts from prop firms please let me know what made you stop bleeding as a newbie ? What trading rules has helped you elevate and become a better trader? How do you stick to your rules everyday ?
[c2_0256] (twitter, KITT) $KITT Joining back in with you guys @MoonMarket_ @greatstockpicks @stockballa > [@Bullish_Trades]( > $KITT time for another run back to .20’s imo. Nonstop chatter about @TheRoaringKitty, this penny could catch momentum with the ticker symbol [BullishTrade quote tweeted BullishTrade about KITT]
[c2_0257] (reddit, SPY) SPY Options Move to $5 Increment Strikes? Apologies if this is a dumb question, but I am only seeing SPY strikes in $5 increments now on two separate trading platforms when they have previously been in $1 increments. I’ve searched high and low for an answer but cannot find any news indicating a change. Does anyone happen to know why the change?
[c2_0258] (twitter, ES) Really ugly way to start the week for stocks. No sign of a relief premarket. $ES [Amdtrades tweeted about ES]
[c2_0259] (reddit, AMD) AMD CC got smoked Yes ofc AMD finally decides to blast off after I sell a covered call. Cheers
[c2_0260] (news, ABR) John And Jane's October Dividend Increases And Income Tracker - Taxable Account
[c2_0261] (reddit, BABA) Accumulating dividends etf mechanics. Im pretty new to etf investing and have now held some SWLD for around 6 months on T212 (UK). My understanding is that dividends on this etf are auto invested back into my portfolio. I cannot see any info on how much/when dividends are paid on any website for this etf, nor have I received any on the app (to my knowledge). I would love to be able to see the benefit of the dividends. Please enlighten me as to how I find out this information.
[c2_0262] (reddit, TSLA) 3 Reasons why TSLA will dump to below 400$ before end of the Year Yeah ya'll just clicked because you saw that fancy chart huh? But in truth we all know TSLA is immune to everything. Buuuuut Reason one: TSLA will dump after 6 of Nov. because Musks 1T Package will be accepted and its already priced in. Reason two: I got puts, and I am always winning. Reason three: Never trust your gut, and my gut says TSLA will moon. Thank you.
[c2_0263] (reddit, AMZN) Snowflake soars 30% on earnings beat and plan to spend $6 billion on Amazon cloud
[c2_0264] (reddit, NVDA) ber & fomc bet losses whoops does not pay to be ber *positions: in general, I bet that fomc would cut >25 bps, and had puts on nvda and orcl
[c2_0265] (twitter, SPY) Took the direct route on CPI... New highs made, structure appears nearly complete... Now watching for a reversal structure 🙂 > [@ChandlerTrading]( > $ES_F $SPX $SPY 8H -- Dec12 > > •No changes. Still looking for the same outcome > > • I suspect the Dec07 low will get swept before the rally, but it's a coin flip with CPI tomorrow > > • I'm risk free at the moment. Just gonna see where CPI takes us and go from there [ChandlerTrading 🔁 ChandlerTrading about SPY, SPX, ES=F]
[c2_0266] (reddit, GME) It's the Way Too Early GME Q1 2026 Earnings Estimate! Lot's of great guidance in the FY 2025 Report -- notably the lack of anticipated significant store closures!! Looking for Revenue of $662M with Net Income of $75M. $0.17 EPS. Revisions and Balance Sheet projections closer to quarter end!
[c2_0267] (twitter, ASTS) Sold the last a little early this morning at 1🫠 but room is on fire! $ASTS squeeze mode [rachels_44 tweeted about ASTS]
[c2_0268] (twitter, MAPS) $MAPS - Closed Profit: 12k [@0xd0n]( $MAPS, +10k [0xd0n 🔁 0xd0n about MAPS, MAPS]
[c2_0269] (news, AB) Mining Automation Market Perspective By Global Size, Share, Competitive Analysis & Regional ...
[c2_0270] (reddit, NVDA) NVDA $135 Put June 13th I’ve been looking at NVDA and since it touched it 500 day MA it’s been bouncing up. However, now it seems to be approaching its $135.58 resistance bar. Also, last 4 earning, the stock declined after. Buying a June 13th PUT is $735. IV is 51.48%. Good or bad idea? I would like to reasoning behind answers please. Thanks in advance
[c2_0271] (news, CSCO) 29 Firms Going Ex-Dividend Next Week, Including Comcast, Cisco And JP Morgan
[c2_0272] (twitter, ES) $ES_F ALT Keep the possible flat in mind here... [rbc_trading tweeted about ES_F]
[c2_0273] (reddit, BULL) advice on deep itm call I have a UNH call expiring in 01/27 with strike of $320 that i bought for $23.00. Is it best to keep it with so much time left, roll the call up to take some profit or just exercise the call and sell covered calls to make up the loss of exercising?
[c2_0274] (twitter, SPY) $SPY Weekly Could be working on an inverse head and shoulders if buyers start to take some control-off the lows in Q1. Retest of the trendline from the Jan 22 highs most likely in play. [Albert_trades tweeted about SPY]
[c2_0275] (reddit, GLD) Why hasn't gold gone parabolic? GLD had a rush earlier this year but hasn't rallied since. Inflation concerns, poor economy, fed yields increasing, everything points towards gold being the defacto investment. Why isn't GLD pumping? Unrelated, I bought a shitload of GDXJ in January.
[c2_0276] (reddit, NVDA) Google could make 100 Billion on the SpaceX IPO Google owns around 5% of Spacex and if the IPO goes for 2 Trillion then that means 100 Billion in profit for them. [
[c2_0277] (reddit, NVDA) Tool to rank stock for option selling Hey r/options, I’ve been building a tool that ranks S&P 500 stocks using fundamental data and option metrics, and I’d love your feedback. [
[c2_0278] (twitter, MSFT) $MSFT Down big but guess what? Was ALSO overbought, needed this, and hit the trendline support... this is one I would easily shift more money into Jan calls if we get a big down day and more opportunity [ShardiB2 tweeted about MSFT]
[c2_0279] (reddit, GOLD) Lost $90k because i went to make breakfast Checked back and already down, so i might as well hold, kept holding and lost it all.
[c2_0280] (twitter, QQQ) $SPY $QQQ One of my rules on swing trades is to lock in most of my profits if theres a short term move of over 5%. Rules are made from hard lessons learned. Currently we rallied close to 6% in only 2 days. Chances of consolidation or pullback is high. [kpak82 tweeted about SPY, QQQ]
[c2_0281] (twitter, ES) $ES 5min Low time frame moving average throwbacks are a clear sign of trend (in either direction) Don’t fight the trend anon [StonkTrump tweeted about ES=F]
[c2_0282] (twitter, NFLX) $NFLX - Earnings come out July 19th. Usually best to see how it goes before adding to a position. Would have to be pretty bad earnings to drop it much more. But will update it once it's released. [IncomeSharks tweeted about NFLX]
[c2_0283] (twitter, MP) $MP Weekly Candle Chart 🐉 🔨 Double bottoms formation in conjunction of bullish divergence on weekly chart = 🪄 [Mr Chart Norris tweeted about MP]
[c2_0284] (reddit, ETH) Warning: Scam landing in your chats. It's not Wallstreetbets, it's Wailstreetbets with funny capitalization. Don't click anything and report
[c2_0285] (reddit, BOOM) Guys… The Dow! The Dow is over 50,000 Was looking for just a day trade of maybe 20%. Opened my brokerage at the end of the work day and BOOM! Don’t think this news will follow through into the weekend, so I’ll take my gains. The Dow is over 50,000!
[c2_0286] (reddit, WEN) In on WEN!!! NEW CEO, NEW CFO, LETS FUCKING GO! Position posted - 3300 shares at basis of $8.58 per share.
[c2_0287] (twitter, AAPL) $AAPL gap filled & stalling Needs to get above if this tech rally is to have serious legs [MacroCRG tweeted about AAPL]
[c2_0288] (reddit, GOOGL) BlockDAG Scam or Legit? What Our Investigation Found Behind the Massive Presale
[c2_0289] (reddit, BULL) Next bull run expectations? Very early estimates. Imo 250,000 would be a low estimate and nearer to 500,000 being a higher estimate. definitely consider the impact companies will have (you know the one I’m talking about).
[c2_0290] (twitter, QQQ) $QQQ Weekly The 200 SMA ~$284 is in the bullish .50-.618 fib zone near the 200 day SMA. That's about as low as this one can continue to pull back before we're alerted to something much more bearish as a possibility. For now, I include only the most bearish outcome, if needed. [reciknows 🔁 PikesPeakTrades about QQQ]
[c2_0291] (reddit, BTC) Bitcoin Price just fell below 80k for the first time since the April 2025 Tariff Scare. (Low of $78,159 so far)
[c2_0292] (twitter, BLNK) $BLNK Was planning on buying some of this earlier close to $10 and got distracted... bullish engulfing candle in the works. Key support. [Nebraskangooner tweeted about BLNK]
[c2_0293] (reddit, ETH) Is $ETH realistically on track to reach $7,500 in this cycle? With ETH trading around current levels, some analysts are calling for $7,500 by the end of the cycle based on institutional demand, staking dynamics, and Ethereum’s role in DeFi and stablecoins. Do you think this is a realistic target, or is it too optimistic given current macro conditions and regulatory uncertainty? What do you think would be the main catalysts needed for ETH to reach that level?
[c2_0294] (twitter, TEAM) $TEAM with a double beat…yet down -11% after hours. It works both ways, folks. I don’t say it to be mean, just to tell the truth. > [@VenturaCaptl]( > $TEAM Atlassian Corporation Q2 Adj. EPS $0.45 Beats $0.31 Estimate, Sales $872.70M Beat $851.32M Estimate [the_chart_life 🔁 VenturaCaptl about TEAM]
[c2_0295] (twitter, PFE) $PFE - Shorting this stock. Lower highs and lower lows forming. Looks like a breakdown and re-test. Nice spot by @Nebraskangooner [IncomeSharks tweeted about PFE]
[c2_0296] (reddit, AMZN) 70K loss POV when you have thousands of stocks to pick from and you pick the worst 3
[c2_0297] (twitter, SPY) $SPY Im still in this. Its down -50%. Obviously market has been A-hole and chopping for 2 weeks. One thing we have no control over is chop and how fast the market decides to move its butt at a reversal area. Like being stuck in traffic, only person you hurt by stressing is > [@kpak82]( > Bought $SPY sept20 $500 puts @ .88 > Full risk. Could go to zero. Follow at your own risk. [kpak quote tweeted kpak about SPY]
[c2_0298] (reddit, MSTR) Ambitious developers are biggest risk to Bitcoin, says Michael Saylor
[c2_0299] (reddit, AMZN) Meta is testing an AI bedtime story app for people with no imagination
