"""Lightweight ticker detection from financial text.

Extracts stock ticker mentions without heavy NLP dependencies (no spaCy).
Uses the project's stock_database.json for validation and name resolution.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Reddit/social text often uses $nvda / $GME — match case-insensitively, normalize to upper.
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_BARE_TICKER_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,5})(?![a-z])")

# Common uppercase words / abbreviations that coincide with real tickers.
# These are almost never stock references in running text.
_BARE_TICKER_BLACKLIST: Set[str] = {
    # Single-letter are already excluded by the 2-char minimum in the regex.
    # 2-letter
    "AI", "AM", "AN", "AS", "AT", "BE", "BY", "DD", "DO", "GO", "HE",
    "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO",
    "TO", "UP", "US", "WE",
    # 3-letter
    "ALL", "ARE", "BIG", "CAN", "CAR", "DAY", "EAR", "ERA", "FAR", "FEW",
    "FOR", "FUN", "GET", "GOT", "GUN", "HAS", "HER", "HIS", "HOW", "ITS",
    "JOB", "LET", "LOW", "MAN", "MAY", "MEN", "MOM", "NET", "NEW", "NOT",
    "NOW", "ODD", "OLD", "ONE", "OUR", "OUT", "OWN", "PAY", "PER", "PRO",
    "PUT", "RAN", "RAW", "RED", "RUN", "SAT", "SAW", "SAY", "SET", "SHE",
    "SIT", "SIX", "THE", "TEN", "TOO", "TOP", "TRY", "TWO", "USE", "VIA",
    "WAR", "WAS", "WAY", "WHO", "WHY", "WIN", "WON", "YET", "YOU",
    # 4-letter common English / Reddit / finance jargon
    "ALSO", "BACK", "BEAR", "BEEN", "BEST", "BODY", "BOLD", "BOTH", "BULL",
    "BURN", "CALL", "CAME", "CASH", "COME", "CORE", "COST", "CULT", "DARE",
    "DEAL", "DEBT", "DEEP", "DIPS", "DOES", "DONE", "DOWN", "DROP", "DUMP",
    "EACH", "EARN", "EASE", "EDIT", "ELSE", "EVEN", "EVER", "FACE", "FACT",
    "FAIL", "FAIR", "FALL", "FAST", "FEAR", "FEEL", "FILL", "FIND", "FINE",
    "FIRE", "FIVE", "FLAT", "FLEW", "FLIP", "FLOW", "FOLD", "FORM", "FOUR",
    "FREE", "FROM", "FUEL", "FULL", "FUND", "GAIN", "GAME", "GAVE", "GIVE",
    "GLAD", "GOES", "GOLD", "GONE", "GOOD", "GRAB", "GREW", "GRIP", "GROW",
    "GURU", "HACK", "HALF", "HAND", "HANG", "HARD", "HATE", "HAVE", "HEAD",
    "HEAR", "HEAT", "HELP", "HERE", "HIGH", "HINT", "HITS", "HOLD", "HOLE",
    "HOME", "HOPE", "HUGE", "HUNG", "HURT", "IDEA", "INFO", "INTO", "ITEM",
    "JUMP", "JUST", "KEEN", "KEEP", "KEPT", "KICK", "KILL", "KIND", "KNEW",
    "KNOW", "LACK", "LAID", "LAND", "LAST", "LATE", "LEAD", "LEAN", "LEFT",
    "LEND", "LESS", "LIFE", "LIFT", "LIKE", "LINE", "LINK", "LIST", "LIVE",
    "LOCK", "LONE", "LONG", "LOOK", "LOOP", "LOSE", "LOSS", "LOST", "LOTS",
    "LOVE", "LUCK", "MADE", "MAIN", "MAKE", "MANY", "MARK", "MASS", "MATH",
    "MEAN", "MERE", "MILD", "MIND", "MINE", "MISS", "MODE", "MOOD", "MOON",
    "MORE", "MOST", "MOVE", "MUCH", "MUST", "NAME", "NEAR", "NEAT", "NEED",
    "NEXT", "NICE", "NINE", "NONE", "NOTE", "ODDS", "OKAY", "ONCE", "ONLY",
    "ONTO", "OPEN", "OVER", "PACE", "PACK", "PAGE", "PAID", "PAIR", "PALE",
    "PART", "PASS", "PAST", "PATH", "PEAK", "PICK", "PILE", "PLAN", "PLAY",
    "PLOT", "PLUS", "POLL", "POOL", "POOR", "POST", "POUR", "PREV", "PULL",
    "PUMP", "PURE", "PUSH", "QUIT", "RACE", "RAIN", "RANK", "RARE", "RATE",
    "READ", "REAL", "RELY", "RENT", "REST", "RICH", "RIDE", "RING", "RISE",
    "RISK", "ROAD", "ROCK", "RODE", "ROLE", "ROLL", "ROOF", "ROOM", "ROOT",
    "ROPE", "ROSE", "RUIN", "RULE", "RUSH", "SAFE", "SAID", "SAKE", "SALE",
    "SAME", "SAND", "SANG", "SAVE", "SEED", "SEEK", "SEEM", "SEEN", "SELF",
    "SELL", "SEND", "SENT", "SHIP", "SHOP", "SHOT", "SHOW", "SHUT", "SICK",
    "SIDE", "SIGN", "SITE", "SIZE", "SKIN", "SLIP", "SLOW", "SNAP", "SOFT",
    "SOIL", "SOLD", "SOLE", "SOME", "SONG", "SOON", "SORT", "SOUL", "SPOT",
    "STAR", "STAY", "STEM", "STEP", "STOP", "SUCH", "SUIT", "SURE", "SWAP",
    "TAIL", "TAKE", "TALE", "TALK", "TALL", "TANK", "TAPE", "TEAM", "TECH",
    "TELL", "TEND", "TERM", "TEST", "TEXT", "THAN", "THAT", "THEM", "THEN",
    "THEY", "THIN", "THIS", "THUS", "TIED", "TIER", "TILL", "TIME", "TINY",
    "TIPS", "TIRE", "TOAD", "TOLD", "TOLL", "TONE", "TOOK", "TOOL", "TOPS",
    "TORE", "TORN", "TOUR", "TRAP", "TREE", "TRIM", "TRIP", "TRUE", "TUBE",
    "TUNE", "TURN", "TWIN", "TYPE", "UGLY", "UNIT", "UPON", "URGE", "USED",
    "USER", "VARY", "VAST", "VERY", "VIEW", "VOID", "VOTE", "WAGE", "WAIT",
    "WAKE", "WALK", "WALL", "WANT", "WARM", "WARN", "WASH", "WAVE", "WEAK",
    "WEAR", "WEEK", "WELL", "WENT", "WERE", "WEST", "WHAT", "WHEN", "WHOM",
    "WIDE", "WIFE", "WILD", "WILL", "WIND", "WINE", "WING", "WIRE", "WISE",
    "WISH", "WITH", "WOKE", "WOOD", "WORD", "WORE", "WORK", "WORM", "WORN",
    "WRAP", "YARD", "YEAR", "YOUR", "ZERO", "ZONE",
    # 5-letter
    "ABOUT", "ABOVE", "AFTER", "AGAIN", "BEING", "BELOW", "BEGAN", "COULD",
    "EVERY", "THEIR", "THERE", "THESE", "THINK", "THOSE", "TODAY", "UNDER",
    "UNTIL", "WATCH", "WEIRD", "WHERE", "WHICH", "WHILE", "WORLD", "WOULD",
    "BONDS", "CHEAP", "CRASH", "HEDGE", "LUNAR", "MONEY", "NEVER", "NOTED",
    "OTHER", "PRESS", "PRICE", "QUITE", "RALLY", "SINCE", "STAKE", "STOCK",
    "STORE", "SHORT", "TRADE", "TREND", "TRUST", "VALUE", "WORTH",
    # Common abbreviations / acronyms
    "CEO", "CFO", "COO", "CTO", "GDP", "IPO", "USA", "USD", "SEC", "FBI",
    "CIA", "FDA", "EPA", "FED", "ETF", "EPS", "IMO", "PSA", "TIL", "FYI",
    "FAQ", "BTW", "LOL", "ATH", "OTC", "ITM", "OTM", "NAV", "APR", "APY",
    # Reddit / social-media jargon
    "LMAO", "YOLO", "FOMO", "HODL", "MOASS",
    # Ambiguous 2-3 letter tickers that commonly appear as abbreviations
    "ET", "III", "PM", "RE", "TV", "UK", "EU", "DC", "LA", "NY",
    "GDP", "DOJ", "DOD", "IRS", "ICE", "DHS", "NSA", "CDC",
    "GP", "IP", "IT", "HR", "PR", "PC", "AI", "AR", "VR",
}

# Curated mapping of well-known company names / brands to their tickers.
# Only distinctive names that are unlikely to be common English words.
# This avoids the massive false-positive problem from auto-extracting
# the first word of every company name in the 10k+ stock database.
_WELL_KNOWN_NAMES: Dict[str, str] = {
    # Mega-cap / household names
    "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA", "microsoft": "MSFT",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "netflix": "NFLX", "disney": "DIS", "boeing": "BA", "intel": "INTC",
    "qualcomm": "QCOM", "broadcom": "AVGO", "oracle": "ORCL", "adobe": "ADBE",
    "salesforce": "CRM", "cisco": "CSCO", "ibm": "IBM", "samsung": "SSNLF",
    "toyota": "TM", "sony": "SONY", "walmart": "WMT", "costco": "COST",
    "starbucks": "SBUX", "mcdonald": "MCD", "mcdonalds": "MCD",
    "coca-cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "nike": "NKE", "visa": "V", "mastercard": "MA", "paypal": "PYPL",
    "jpmorgan": "JPM", "goldman": "GS", "blackrock": "BLK",
    "berkshire": "BRK-B", "morgan stanley": "MS",
    # Popular tech / fintech / retail investor favourites
    "coinbase": "COIN", "robinhood": "HOOD", "palantir": "PLTR",
    "gamestop": "GME", "shopify": "SHOP", "spotify": "SPOT",
    "airbnb": "ABNB", "uber": "UBER", "lyft": "LYFT",
    "snowflake": "SNOW", "crowdstrike": "CRWD", "datadog": "DDOG",
    "rivian": "RIVN", "lucid": "LCID", "nio": "NIO",
    "sofi": "SOFI", "affirm": "AFRM", "roblox": "RBLX",
    "microstrategy": "MSTR", "supermicro": "SMCI",
    # Major ETFs often discussed as tickers
    "spdr": "SPY",
}


class TickerDetector:
    """Fast ticker detection using stock_database.json — no spaCy needed."""

    _instance: Optional["TickerDetector"] = None

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = (
                Path(__file__).resolve().parents[3]
                / "data"
                / "stocks"
                / "stock_database.json"
            )

        self._valid_tickers: Set[str] = set()
        self._name_to_ticker: Dict[str, str] = {}

        if db_path.exists():
            self._load_database(db_path)
        else:
            logger.warning("Stock database not found at %s — ticker detection disabled", db_path)

    def _load_database(self, db_path: Path) -> None:
        with db_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for ticker in data.get("stocks", {}):
            self._valid_tickers.add(ticker.upper())

        # Use the curated well-known name list rather than auto-extracting
        # first-words from 10k+ company names (too many false positives).
        self._name_to_ticker = dict(_WELL_KNOWN_NAMES)

        logger.info(
            "TickerDetector loaded %d tickers, %d name→ticker mappings",
            len(self._valid_tickers),
            len(self._name_to_ticker),
        )

    @classmethod
    def get_instance(cls) -> "TickerDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_valid_ticker(self, symbol: str) -> bool:
        """True if *symbol* is in the loaded stock universe (uppercase)."""
        if not symbol:
            return False
        return symbol.strip().lstrip("$").upper() in self._valid_tickers

    def detect(self, text: str) -> List[Tuple[str, str]]:
        """Return ``[(ticker, mentioned_as), ...]`` found in *text*.

        ``mentioned_as`` is the surface form (e.g. ``"$AAPL"``, ``"TSLA"``,
        ``"Tesla"``).  Each ticker appears at most once.
        """
        if not text:
            return []

        found: Dict[str, str] = {}

        # 1. Cashtags  –  $AAPL, $aapl  (highest confidence)
        for m in _CASHTAG_RE.finditer(text):
            sym = m.group(1).upper()
            if sym in self._valid_tickers:
                found.setdefault(sym, f"${sym}")

        # 2. Bare uppercase tickers  –  AAPL, TSLA
        for m in _BARE_TICKER_RE.finditer(text):
            sym = m.group(1)
            if sym in _BARE_TICKER_BLACKLIST:
                continue
            if sym in self._valid_tickers:
                found.setdefault(sym, sym)

        # 3. Known company names  –  "Tesla", "Apple", "Nvidia", etc.
        text_lower = text.lower()
        for name, ticker in self._name_to_ticker.items():
            if ticker in found:
                continue
            if name not in text_lower:
                continue
            pattern = rf"\b{re.escape(name)}\b"
            m = re.search(pattern, text_lower)
            if m:
                original = text[m.start() : m.end()]
                found.setdefault(ticker, original)

        return list(found.items())
