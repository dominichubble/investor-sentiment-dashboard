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
----- ITEMS (50: c2_0800 .. c2_0849) -----
[c2_0800] (twitter, TSLA) Today's plays $FSLR 80c 103% closed quick profit $NVDA 290c +20%, then -45% $TSLA 1200c +32% closed profit $FDX 230c +4% closed profit Shake out city today. Smaller gains. Reboot tomorrow! 😀 [optionsprochick tweeted about FSLR, NVDA, TSLA, FDX]
[c2_0801] (twitter, HIMS) $HIMS Daily Candle Chart 🔨 🐉 Breakdown / Bounce / Retest Scenario Risk is there - so does the Reward I like it for a short term Reverse [Mr Chart Norris tweeted about HIMS]
[c2_0802] (reddit, GOLD) At what point do u transition? At what point do u transition from Funded account to a live account? rules are annoying and also Is nasdaq better than Gold for beginners?
[c2_0803] (reddit, TFSA) XBAL for FHSA? Hi there I’m looking to buy a house in the next 2-3 years. 75% of I’m savings that I don’t plan to ever touch are in TFSA-XEQT. I understand that XEQT is not good for something I will need in 2-3 years, so I have looked into XBAL given its low-medium risk rating. What do you think? Currently I’m just holding on to my FHSA in CASH because I’m not sure where I want to put it yet.
[c2_0804] (twitter, NVDA) $NVDA held 230 support zone on the daily📈Above 241 can try for 250, 256 Back under 220, weak [rachels_44 tweeted about NVDA]
[c2_0805] (reddit, SPXC) $25k on the real SpaceX fat-finger play: SPXC Dyslexia is real and the company even does space stuff. HVAC: High Vector Altitude Capsule (I think). In for 100 shares.
[c2_0806] (twitter, ES) $ES Key area here. A breakout of the trendline would draw in breakout traders. I've shared this type of pattern all year long and it's worked a treat. False breakout of trendline, back within and the market sells off. [AltcoinPepe tweeted about ES=F]
[c2_0807] (reddit, V) Visa/Mastercard down around 5% since Trump suggested capping interest rates. Which makes no sense, since they don't even charge interest, they just process transactions. Am I missing something here or there's an opportunity for making a quick buck here? I don't think that congress will allow that idea to go anywhere anyways, either.
[c2_0808] (twitter, GOLD) When buying physical gold you pay a premium of at least 4% (I'm not even including shipping and most places charge more). When you sell it, it's another 4%. Meaning $GOLD has to pump 8% for you to break even on your buy. This is why people hold #GOLD for so long, they have to. [IncomeSharks tweeted about GOLD]
[c2_0809] (reddit, TSLA) U.S. launches Tesla probe following more crashes involving full self-driving feature
[c2_0810] (reddit, EMA) Turning $40,000 to $400,000 So i turned $40,000 into $400,000 in 3 months trading options. Strategy consists of gap fills on the daily timeframe on large/mid cap stocks such as $CVS, $DIS, $NKE. Utilizing flow and the 8EMA as extra conviction. I know that’s pretty good but is it time to start selling options with larger capital? I’ve heard 1% portfolio gain per month is realistic, but I haven’t sold options extensively before. Thoughts?
[c2_0811] (reddit, ETH) Value accrual thesis for ETH ETH beginner. Can someone point me to the source where Vitalik or the Foundation itself make a commitment to ETH performance as a financial asset? Is there any such commitment? Or are the commitments made for other metrics such as security and other things, and any financial value growth only a side effect? Is it the community that must solve the L2 value drain issue?
[c2_0812] (news, ABG) Asbury Automotive Group Schedules Release of First Quarter —…—… Financial Results
[c2_0813] (reddit, AMZN) Amzn calls Posting this as sort of an update. Decided to not only hold but buy more 🤡 32 days till expiration but I’ll hold through the rest of the month.
[c2_0814] (reddit, TSLA) RH account deficit TSLA IC call excercise $380 and call assignment $375 Hello guys I did an TSLA Iron Condor and i got a buy call excercise $380 and a Sell call assignment $375 The RH app said that i have an account deficit What is gonna happen? Or what is happening? Please help! Thank you!
[c2_0815] (twitter, TSLA) PlainSite accuses $TSLA of fraud, stating they lied to New York and investors. [unusual_whales tweeted about TSLA]
[c2_0816] (reddit, GOLD) Oil price vs stock prices Thursday oil was up 5% and XOM dropped by 3%, divulging from earlier pattern of moving with the price of oil. Strangley USO another us based oil stock went up by 10% that day. Can anyone give some insight and what to expect next week with further infrastructure damage and general Iran escalation so far this weekend.
[c2_0817] (reddit, AMC) Something is knocking AMC down right before close Todays close was $0.9653 Someone is batting AMC down at close trying to keep AMC under a $1 average for 30 days so that delisting or a reverse split becomes a possibility. I guess there isnt much we can do, unless Execs show confidence in AMC by buying with their own money. Today's last minute drop was pretty decent. If it continues like this, it might be a scalping opportunity.
[c2_0818] (twitter, MET) #earnings after the close today $FB $QCOM $SPOT $TMUS $SU $ALGN $LSPD $KLIC $MCK $QRVO $CCS $SITM $ALGT $AFL $EGHT $MET $MXL $ALL $SKY $OMF $MUSA $AVNW $ARWR $NTGR $CTSH $MAA $HOLX $CTVA $ADTN $FBHS $DXC $LNC $OHI $AVB $YELL $ELF $RRR $UGI $ESS $BDN [eWhispers tweeted about FB, QCOM, SPOT, TMUS, SU, ALGN, LSPD, KLIC, MCK, QRVO, CCS, SITM, ALGT, AFL, EGHT, MET, MXL, ALL, SKY, OMF, MUSA, AVNW, ARWR, NTGR, CTSH, MAA, HOLX, CTVA, ADTN, FBHS, DXC, LNC, OHI, AVB, YELL, ELF, RRR, UGI, ESS, BDN]
[c2_0819] (twitter, BTC) $BTC - Retest of the former breakout in progress, then continuation to the upside should be expected. [BATMAN ⚡ tweeted about BTC]
[c2_0820] (twitter, ES) I got caught slackin today. I was short on #ES_F today and made a very nice profit. Got faked out around 2:30 thinking the selling finished (closed position and opened a long). Silly mistake to fight the very obvious trend. Had to take the L. We’re all human. [ItsSqueezy tweeted about]
[c2_0821] (reddit, BTC) First time buyer I have always wanted to buy BTC after a big crash and hold out until I almost double my money. This week was the perfect opportunity. I bought $5k worth of BTC when it was at $63,300 and then another $5k worth when it went back up to $64,500, fearing the $62,000 was the lowest it would get this cycle. So wish me luck! If anybody has any advice for noobies, I am all ears! But I’m feeling pretty good about this so far!
[c2_0822] (reddit, DASH) Privacy Coins Rally: Zcash (ZEC) and Dash (DASH) Lead Sector Surge to ~$26.6 B Market Cap
[c2_0823] (twitter, FF) honestly I see this #FF as a bit ridiculous on CT lol [eliz883 tweeted about FF]
[c2_0824] (twitter, SMA) $BTC logical spot to bounce here with throwback to 4H SMA 200 and prior range peak Note acceleration into lower BB dip and volume on this candle [Cheds (Trading Quotes) tweeted about BTC]
[c2_0825] (twitter, NIO) Went up to .97 after my entry. You guys couldve taken profit or raised stop. I fell asleep and my stop triggered at .63 > [@kpak82]( > $NIO jul1 23p .83 [kpak82 🔁 kpak82 about NIO]
[c2_0826] (reddit, CE) Much needed guidance For all those struggling but trying to understand what's wrong with their process and trying to be better, listening to Mark Douglas becomes very important. "Kaunsa CE, PE buy karu" people will be utterly bored. Enjoy.
[c2_0827] (twitter, GS) $TWTR - Downgraded to Sector Weight at Keybanc $DAL - PT $50 from $45 at Jefferies $PYPL - PT $155 from $190 at Credit Suisse $TSLA - PT $1,260 from $1,350 at Piper Sandler $PTON - PT $28 from $40 at Telsey Advisory $GS - PT $486 from $488 at BMO Capital [rachels_44 tweeted about TWTR, DAL, PYPL, TSLA, PTON, GS]
[c2_0828] (reddit, RH) The changing of Pattern Day Trading on Thursday This was my day today, however had it been Thursday or go forward, it could of looked different. I honestly think the removal of the Pattern Day Trade warning esp on RH is going to unlock upside for folks who were “holding onto hope” SPY $760 calls for reference
[c2_0829] (twitter, MA) Here's a stock I like, $MA / Mastercard. Basically, refuses to go down. Big moat around the business model, adding this to my potential buy list. [CryptoJelleNL tweeted about MA]
[c2_0830] (news, AMZN) Amazon-made Rekognition software now able to detect fear The facial recognition tool has faced much backlash. With the inclusion of "Fear," there are now eight emotions integrated into the software. Read more here. Read more... More about Tech, Amazon, Mashable Video, Facial Recognition, and Aclu
[c2_0831] (reddit, ETH) Why is everyone bearish about September? Title. I keep hearing that September is going to be a red month, why? Because historically it was.. but we have an essentially guaranteed rate cut, OG Bitcoin whales and institutions buying crazy amounts of ETH, and so on. Doesn't this prediction make no sense?
[c2_0832] (reddit, GOOGL) got charged by google st. louis hey guys so i just got charged 0 dollars and it said “purchase verification of (currency) 0 at Google St. Louis USAApproved” is this something i should be worried about? should i go change my card number? i have a monthly subscription for google storage
[c2_0833] (reddit, AMZN) Amazon snaps 9-day losing streak during which it lost more than $450 billion in value
[c2_0834] (reddit, AI) 19.1K > 65.8K > 11K on semiconductors SNDK, MU, STX took me to the top and it all came crashing down. I don’t really care if I lose my last 11k I’m just tired.
[c2_0835] (twitter, NVDA) $NVDA at a critical level [Don't follow Shardi B if you hate Crypto tweeted about NVDA]
[c2_0836] (reddit, BTC) Will BTC surge again before EOY? We've seen BTC recover a little bit, but what do we think will happen in the next few weeks? Any possibility of an end-of-year surge?
[c2_0837] (twitter, AMZN) $AMZN (Perfect 🤝) [@0xd0n]( $AMZN - If you receive such a gift, buy Green - buy zone [0xd0n 🔁 0xd0n about AMZN, AMZN]
[c2_0838] (twitter, UPST) $UPST Strong day so far. [Nebraskangooner tweeted about UPST]
[c2_0839] (reddit, BTC) Crypto sentiment I’ve been here since 2017—through the taxi-driver euphoria and the -75% crashes. But this cycle feels different. There’s a strange tension beneath the institutional growth; it’s the calm before something massive. I had to step away from the charts last night just to breathe. Then I came back and stacked more $BTC. I'm playing the long game. How are you holding up? Anxious or Zen?
[c2_0840] (twitter, TW) $SPY TW Pivot sell signal alerted at HOD today on both 5min and 15min charts. Great example of the power of multiple timeframes lining up. Indicator available only* via @TrendSpider [ThetaWarrior tweeted about SPY]
[c2_0841] (reddit, BULL) Micron shares are rising again despite weak overall market. Why memory chip rally seems unstoppable
[c2_0842] (reddit, AMD) Is anyone else noticing how every dip gets bought instantly… except the ones you’re in? It’s crazy the whole market looks like it’s running on autopilot. Micro-pullback → instant reversal → new high. But your stock? Full dead silence until it drops another 3–5%. Is this just bad luck, or are we all living inside the same simulation?
[c2_0843] (reddit, COIN) Why coin not notes As Bitnote or MoreNote or Bills as BitBills or MoreBills🤌🏻
[c2_0844] (twitter, KSCP) $KSCP What up? $8.2 breakout next baby [Ultra_Calls tweeted about KSCP]
[c2_0845] (reddit, BULL) Need review on US market portfolio I have been researching US ETFs offered through INDmoney and figuring out how various themes could potentially complement one another over the long haul. Allocation currently looking into: VOO – 55% (broad US markets) SMH – 25% (semiconductor/ AI infrastructure) ARKX – 12% (innovation/space) UFO – 8% (commercial space)
[c2_0846] (reddit, TSLA) Near miss with $tsla covered calls. Sold to open 2 of these TSLA25091200C360000 for $14.00 each in May with a 370 breakeven. Had to fly to China on 12th, so closed them at a day early at $1.00. Then crazy Elon started buying few hours later...(screen shot of my buy to close history on September 12th.) No idea how high they went. 😬 Careful with the meme stock CCs.
[c2_0847] (twitter, QQQ) $QQQ #QQQ $303 line in the sand, an important high in 2020 lost, now trying to be regained Fold here, we revisit $285 Hold here can see $317 / $335 [_bread911 tweeted about QQQ]
[c2_0848] (twitter, NVDA) $NVDA the big bull winner today [Nebraskangooner tweeted about NVDA]
[c2_0849] (twitter, SPY) $SPY Is testing the hourly trend line. Still a gap below that could be filled 👀 $381.51, $378.73 [optionsprochick tweeted about SPY]
