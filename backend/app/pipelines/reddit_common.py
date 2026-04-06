"""Shared Reddit keyword batches and ticker hints for search-based ingestion."""

from __future__ import annotations

from typing import Dict, List, Set

# Smaller OR-queries return different result sets than one giant OR (see reddit_bulk_ingest).
DEFAULT_KEYWORD_GROUPS: List[List[str]] = [
    ["stock", "stocks", "market", "trading", "investor"],
    ["earnings", "eps", "revenue", "guidance", "quarter"],
    ["fed", "rates", "inflation", "recession", "tariff"],
    ["nvda", "tsla", "aapl", "msft", "amzn", "meta", "googl", "amd"],
    ["bull", "bear", "rally", "crash", "dip", "short", "long"],
    ["crypto", "bitcoin", "btc", "eth", "etf"],
    ["bankruptcy", "debt", "lawsuit", "sec", "merger"],
    ["401k", "ira", "roth", "retirement", "pension"],
    ["puts", "calls", "covered", "wheel", "theta", "gamma"],
    ["oil", "energy", "opec", "gold", "commodity", "copper"],
    ["china", "baba", "europe", "uk", "japan", "emerging"],
    ["reit", "mortgage", "housing", "rent", "landlord"],
    ["ipo", "listing", "offering", "secondary", "split"],
    ["ai", "semiconductor", "software", "cloud", "datacenter"],
]

# Optional extra terms per ticker (symbol-only search misses many posts).
TICKER_SEARCH_ALIASES: Dict[str, List[str]] = {
    "GOOGL": ["google", "alphabet"],
    "GOOG": ["google", "alphabet"],
    "META": ["facebook", "fb"],
    "AMZN": ["amazon"],
    "BRK.B": ["berkshire", "buffett"],
    "BRK.A": ["berkshire", "buffett"],
}


def hint_tickers_from_keyword_group(group: List[str]) -> List[str]:
    """Extract uppercase symbols from a search keyword group (for import hint_tickers)."""
    ordered: List[str] = []
    seen_syms: Set[str] = set()
    for raw in group:
        sym = raw.strip().lstrip("$").upper()
        if not sym or len(sym) > 5:
            continue
        if not sym.replace(".", "").isalnum():
            continue
        if sym in seen_syms:
            continue
        seen_syms.add(sym)
        ordered.append(sym)
    return ordered


def keyword_groups_for_tickers(tickers: List[str]) -> List[List[str]]:
    """One search batch per ticker: plain symbol + $SYMBOL + optional aliases."""
    groups: List[List[str]] = []
    for raw in tickers:
        sym = raw.strip().lstrip("$").upper()
        if not sym or not sym.replace(".", "").isalnum():
            continue
        terms = [sym, f"${sym}"]
        for alias in TICKER_SEARCH_ALIASES.get(sym, []):
            if alias not in terms:
                terms.append(alias)
        groups.append(terms)
    return groups
