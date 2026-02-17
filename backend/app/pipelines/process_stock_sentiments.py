"""
Stock-sentiment processing pipeline.

Reads all processed news and Reddit data, extracts stock tickers from the text,
calculates sentiment (FinBERT or keyword fallback), and populates
data/stock_sentiments/stock_sentiments.json for the correlation analysis engine.

Usage:
    cd backend
    python -m app.pipelines.process_stock_sentiments              # keyword mode (fast)
    python -m app.pipelines.process_stock_sentiments --use-finbert # FinBERT mode (accurate)
    python -m app.pipelines.process_stock_sentiments --fast        # explicit keyword mode
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Ticker extraction ----

# Regex for $AAPL or (AAPL) style tickers
_DOLLAR_TICKER = re.compile(r"\$([A-Z]{1,5})\b")
_PAREN_TICKER = re.compile(r"\(([A-Z]{1,5})\)")

# Major company name -> ticker mapping (covers top ~100 stocks commonly mentioned
# in financial Reddit / news). This avoids needing spaCy NER.
COMPANY_TICKERS: Dict[str, str] = {
    "apple": "AAPL", "iphone": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "aws": "AMZN",
    "meta": "META", "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "palantir": "PLTR",
    "coinbase": "COIN",
    "gamestop": "GME",
    "disney": "DIS",
    "paypal": "PYPL",
    "uber": "UBER",
    "airbnb": "ABNB",
    "snapchat": "SNAP", "snap inc": "SNAP",
    "twitter": "TWTR", "x corp": "TWTR",
    "spotify": "SPOT",
    "shopify": "SHOP",
    "zoom": "ZM",
    "roku": "ROKU",
    "roblox": "RBLX",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "ibm": "IBM",
    "adobe": "ADBE",
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "citigroup": "C", "citibank": "C",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "visa": "V",
    "mastercard": "MA",
    "boeing": "BA",
    "lockheed": "LMT", "lockheed martin": "LMT",
    "raytheon": "RTX",
    "caterpillar": "CAT",
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "eli lilly": "LLY",
    "united health": "UNH", "unitedhealth": "UNH",
    "coca-cola": "KO", "coca cola": "KO",
    "pepsi": "PEP", "pepsico": "PEP",
    "walmart": "WMT",
    "costco": "COST",
    "home depot": "HD",
    "target": "TGT",
    "nike": "NKE",
    "starbucks": "SBUX",
    "mcdonald": "MCD", "mcdonalds": "MCD",
    "exxon": "XOM", "exxonmobil": "XOM",
    "chevron": "CVX",
    "conocophillips": "COP",
    "ford": "F",
    "general motors": "GM",
    "rivian": "RIVN",
    "lucid": "LCID", "lucid motors": "LCID",
    "nio": "NIO",
    "alibaba": "BABA",
    "baidu": "BIDU",
    "tencent": "TCEHY",
    "samsung": "SSNLF",
    "toyota": "TM",
    "sony": "SONY",
    "broadcom": "AVGO",
    "qualcomm": "QCOM",
    "micron": "MU",
    "arm holdings": "ARM", "arm": "ARM",
    "snowflake": "SNOW",
    "crowdstrike": "CRWD",
    "datadog": "DDOG",
    "cloudflare": "NET",
    "sofi": "SOFI",
    "robinhood": "HOOD",
    "block inc": "SQ", "square": "SQ",
    "affirm": "AFRM",
    "draftkings": "DKNG",
    "dick's sporting": "DKS", "dicks sporting": "DKS",
    "chewy": "CHWY",
    "etsy": "ETSY",
    "doordash": "DASH",
    "lyft": "LYFT",
    "twilio": "TWLO",
    "unity": "U",
    "pinterest": "PINS",
    "reddit": "RDDT",
}

# Common non-stock tickers to skip (common English words that look like tickers)
_FALSE_TICKERS: Set[str] = {
    "A", "I", "IT", "ON", "OR", "AM", "AT", "BE", "DO", "GO", "IF",
    "IN", "IS", "MY", "NO", "OF", "OK", "SO", "TO", "UP", "US", "WE",
    "ALL", "ARE", "CAN", "FOR", "HAS", "HER", "HIS", "HOW", "NEW",
    "NOW", "OLD", "OUR", "OUT", "OWN", "PUT", "RUN", "SAY", "THE",
    "TOO", "TOP", "TRY", "TWO", "USE", "WAR", "WAY", "WHO", "WHY",
    "WIN", "WON", "YET", "CEO", "CFO", "COO", "CTO", "SEC", "IPO",
    "ETF", "GDP", "CPI", "FBI", "CIA", "USA", "EUR", "GBP", "USD",
    "API", "AI", "ML", "EV", "PE", "PS", "DD", "FYI", "FYP", "IMO",
    "EDIT", "TLDR", "LMAO", "YOLO", "HODL", "FOMO", "LETS", "JUST",
    "BUY", "SELL", "HOLD", "LONG", "SHORT", "CALL", "PUT",
    "NYSE", "NASDAQ", "SP", "DOW", "FTSE", "ASX",
    "ANY", "BIG", "DAY", "DID", "END", "FAR", "GOT", "HAD", "LET",
    "MAY", "MAN", "MEN", "NOT", "ONE", "SET", "WAS", "AGO", "BAD",
    "LOW", "PAY", "SEE", "ACE", "ADD", "AGE", "AIM", "AID", "BIT",
}

# Tickers that match common English words and need financial context to be valid
_CONTEXT_REQUIRED_TICKERS: Set[str] = {
    "ARM", "MU", "NET", "SNOW", "U", "F", "V", "C", "T",
}

# Company name mappings that need financial context (they're common English words)
_CONTEXT_REQUIRED_NAMES: Set[str] = {
    "target", "ford", "arm", "unity", "visa",
}

# Financial context keywords that suggest the surrounding text is about stocks
_FINANCIAL_CONTEXT_WORDS: Set[str] = {
    "stock", "stocks", "share", "shares", "price", "market", "trading",
    "invest", "investor", "investing", "investment", "portfolio",
    "earnings", "revenue", "profit", "dividend", "ipo", "eps",
    "bullish", "bearish", "ticker", "nasdaq", "nyse", "sp500",
    "s&p", "dow", "etf", "fund", "hedge", "analyst", "valuation",
    "pe ratio", "market cap", "short", "long", "options", "calls",
    "puts", "squeeze", "rally", "crash", "surge", "plunge",
    "quarterly", "fiscal", "guidance", "forecast", "estimate",
    "upgrade", "downgrade", "buy", "sell", "hold", "overweight",
    "underweight", "outperform", "underperform", "$",
}


def _has_financial_context(text: str) -> bool:
    """Check if the text contains financial context keywords."""
    text_lower = text.lower()
    words = set(re.findall(r"\b[a-z&$]+\b", text_lower))
    matches = words & _FINANCIAL_CONTEXT_WORDS
    return len(matches) >= 1 or "$" in text


def extract_tickers_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract stock tickers from text using regex + company name dictionary
    with context-aware filtering for ambiguous tickers.

    Returns list of (ticker, mentioned_as) tuples.
    """
    found: Dict[str, str] = {}
    text_lower = text.lower()
    has_context = _has_financial_context(text)

    # 1. $TICKER patterns (highest confidence - explicit stock mention)
    for match in _DOLLAR_TICKER.finditer(text):
        ticker = match.group(1).upper()
        if ticker not in _FALSE_TICKERS and len(ticker) >= 1:
            found[ticker] = f"${ticker}"

    # 2. (TICKER) patterns - common in Reddit posts like "(DKS) DICK'S Sporting"
    for match in _PAREN_TICKER.finditer(text):
        ticker = match.group(1).upper()
        if ticker not in _FALSE_TICKERS and len(ticker) >= 2:
            if ticker in _CONTEXT_REQUIRED_TICKERS and not has_context:
                continue
            found[ticker] = f"({ticker})"

    # 3. Company name matching
    for name, ticker in COMPANY_TICKERS.items():
        if ticker in found:
            continue

        # Skip context-required names unless financial context is present
        if name in _CONTEXT_REQUIRED_NAMES and not has_context:
            continue

        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, text_lower):
            found[ticker] = name.title()

    # 4. Filter out "reddit" / RDDT from Reddit-sourced data (self-referential)
    if "RDDT" in found and found["RDDT"] in ("Reddit", "reddit"):
        reddit_keywords = {"subreddit", "r/", "redditor", "upvote", "downvote"}
        if any(kw in text_lower for kw in reddit_keywords):
            del found["RDDT"]

    return [(ticker, mentioned_as) for ticker, mentioned_as in found.items()]


# ---- Sentiment scoring ----

# Financial sentiment lexicon - curated for financial text
_POSITIVE_WORDS = {
    "surge", "surges", "surging", "surged", "soar", "soars", "soaring", "soared",
    "rally", "rallies", "rallied", "rallying", "gain", "gains", "gained",
    "rise", "rises", "rising", "rose", "jump", "jumps", "jumped", "jumping",
    "climb", "climbs", "climbed", "climbing", "boom", "booming", "boomed",
    "bullish", "bull", "upbeat", "optimistic", "optimism", "outperform",
    "beat", "beats", "beating", "exceeded", "exceeds", "exceeding",
    "profit", "profits", "profitable", "profitability",
    "revenue", "growth", "growing", "grew", "expand", "expanding",
    "strong", "stronger", "strongest", "strength",
    "upgrade", "upgraded", "upgrades", "buy", "overweight",
    "record", "high", "highs", "breakout", "breakthrough",
    "positive", "upside", "momentum", "acceleration",
    "recover", "recovers", "recovered", "recovery",
    "dividend", "buyback", "buybacks", "repurchase",
    "innovation", "innovative", "opportunity", "opportunities",
    "success", "successful", "win", "wins", "winning",
    "demand", "robust", "solid", "impressive", "impressive",
    "outpace", "outpaced", "exceeds", "fantastic", "excellent",
}

_NEGATIVE_WORDS = {
    "crash", "crashes", "crashed", "crashing",
    "plunge", "plunges", "plunged", "plunging",
    "drop", "drops", "dropped", "dropping",
    "fall", "falls", "fell", "falling",
    "decline", "declines", "declined", "declining",
    "loss", "losses", "lost", "losing",
    "miss", "misses", "missed", "missing",
    "bearish", "bear", "pessimistic", "pessimism",
    "sell", "selling", "selloff", "sell-off",
    "downgrade", "downgraded", "downgrades", "underweight",
    "cut", "cuts", "slashed", "reduce", "reduced",
    "weak", "weaker", "weakest", "weakness",
    "risk", "risks", "risky", "volatile", "volatility",
    "debt", "deficit", "bankruptcy", "bankrupt",
    "layoff", "layoffs", "fired", "restructuring",
    "lawsuit", "sued", "investigation", "probe", "scandal",
    "warning", "warns", "warned", "caution", "cautious",
    "fear", "fears", "panic", "concern", "concerns", "worried",
    "inflation", "recession", "downturn", "slowdown",
    "overvalued", "bubble", "fraud", "default",
    "underperform", "disappoint", "disappointing", "disappointed",
    "struggle", "struggles", "struggling",
    "tumble", "tumbles", "tumbled", "tank", "tanks", "tanked",
    "worst", "terrible", "awful", "disaster",
}


def calculate_financial_sentiment(text: str) -> Tuple[str, float]:
    """
    Calculate sentiment using a financial keyword lexicon (fast mode).
    Returns (label, confidence_score).
    """
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))

    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return "neutral", 0.5

    net_score = (pos_count - neg_count) / total

    if net_score > 0.15:
        confidence = min(0.95, 0.6 + (pos_count / max(total, 1)) * 0.3)
        return "positive", round(confidence, 4)
    elif net_score < -0.15:
        confidence = min(0.95, 0.6 + (neg_count / max(total, 1)) * 0.3)
        return "negative", round(confidence, 4)
    else:
        return "neutral", round(0.5 + abs(net_score) * 0.2, 4)


def calculate_finbert_sentiments(texts: List[str], batch_size: int = 32) -> List[Tuple[str, float]]:
    """
    Calculate sentiment using the FinBERT model in batches.
    Returns list of (label, confidence_score) tuples.
    """
    from app.models.finbert_model import get_model

    logger.info("Loading FinBERT model...")
    model = get_model()
    logger.info(f"FinBERT model loaded on {model.device}")

    results: List[Tuple[str, float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch_num = i // batch_size + 1

        if batch_num % 10 == 1:
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({i}/{len(texts)} texts)")

        try:
            predictions = model.predict(batch)
            if isinstance(predictions, dict):
                predictions = [predictions]

            for pred in predictions:
                results.append((pred["label"], round(pred["score"], 4)))
        except Exception as e:
            logger.warning(f"  FinBERT batch error at {i}: {e}")
            for text in batch:
                results.append(calculate_financial_sentiment(text))

    return results


# ---- Data loading ----


def load_reddit_file(filepath: Path) -> List[Dict]:
    """Load a processed Reddit data file and extract records."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            return []

        results = []
        for item in records:
            text = ""
            title = item.get("title", "")
            selftext = item.get("selftext", "")

            if title:
                text = title
            if selftext and len(selftext) > 10:
                text = f"{title}. {selftext}" if title else selftext

            if not text or len(text) < 15:
                continue

            # Parse timestamp
            created_utc = item.get("created_utc")
            if created_utc:
                try:
                    timestamp = datetime.fromtimestamp(created_utc).isoformat()
                except (ValueError, OSError, OverflowError):
                    timestamp = None
            else:
                timestamp = None

            # Try to get date from filename if no timestamp
            if not timestamp:
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filepath.name)
                if date_match:
                    timestamp = f"{date_match.group(1)}T12:00:00"

            if not timestamp:
                continue

            results.append({
                "text": text[:2000],
                "source": "reddit",
                "timestamp": timestamp,
                "source_id": item.get("id", ""),
                "subreddit": item.get("subreddit", ""),
            })

        return results

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return []


def load_news_file(filepath: Path) -> List[Dict]:
    """Load a processed news data file and extract records."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            return []

        results = []
        for item in records:
            # Prefer clean fields if available
            title = item.get("clean_title", item.get("title", ""))
            description = item.get("clean_description", item.get("description", ""))
            content = item.get("clean_content", item.get("content", ""))

            text = title
            if description and len(description) > 20:
                text = f"{title}. {description}"

            if not text or len(text) < 15:
                continue

            # Parse timestamp
            published_at = item.get("published_at", "")
            if published_at:
                try:
                    # Handle various ISO formats
                    timestamp = published_at.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filepath.name)
                    timestamp = f"{date_match.group(1)}T12:00:00" if date_match else None
            else:
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filepath.name)
                timestamp = f"{date_match.group(1)}T12:00:00" if date_match else None

            if not timestamp:
                continue

            results.append({
                "text": text[:2000],
                "source": "news",
                "timestamp": timestamp,
                "source_id": item.get("source_id", ""),
                "source_name": item.get("source_name", ""),
            })

        return results

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return []


def load_prediction_file(filepath: Path) -> List[Dict]:
    """Load a prediction JSON/CSV file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            text = item.get("text", "")
            if not text or len(text) < 15:
                continue

            results.append({
                "text": text[:2000],
                "source": item.get("source", "reddit"),
                "timestamp": item.get("timestamp", ""),
                "source_id": item.get("reddit_id", ""),
                "existing_label": item.get("label"),
                "existing_confidence": item.get("confidence"),
            })

        return results

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return []


# ---- Main processing ----


def process_all_data(data_dir: Path, use_finbert: bool = False) -> Dict:
    """
    Process all available data files and generate stock-sentiment records.

    Args:
        data_dir: Path to the data directory.
        use_finbert: If True, use FinBERT model for sentiment; otherwise use keyword lexicon.

    Returns statistics about processing.
    """
    sentiment_mode = "FinBERT" if use_finbert else "keyword"
    logger.info(f"Sentiment mode: {sentiment_mode}")
    all_records: List[Dict] = []

    # 1. Load processed Reddit data
    reddit_dir = data_dir / "processed" / "reddit"
    if reddit_dir.exists():
        reddit_files = sorted(reddit_dir.glob("reddit_finance_*.json"))
        # Skip meta files
        reddit_files = [f for f in reddit_files if "_meta" not in f.name]
        logger.info(f"Found {len(reddit_files)} Reddit data files")

        for filepath in reddit_files:
            records = load_reddit_file(filepath)
            all_records.extend(records)
            if records:
                logger.info(f"  Loaded {len(records)} records from {filepath.name}")

    # 2. Load processed news data
    news_dir = data_dir / "processed" / "news"
    if news_dir.exists():
        news_files = sorted(news_dir.glob("news_finance_*.json"))
        news_files = [f for f in news_files if "_meta" not in f.name]
        logger.info(f"Found {len(news_files)} news data files")

        for filepath in news_files:
            records = load_news_file(filepath)
            all_records.extend(records)
            if records:
                logger.info(f"  Loaded {len(records)} records from {filepath.name}")

    # 3. Load prediction data
    predictions_dir = data_dir / "predictions"
    if predictions_dir.exists():
        pred_files = sorted(predictions_dir.glob("*.json"))
        logger.info(f"Found {len(pred_files)} prediction files")

        for filepath in pred_files:
            records = load_prediction_file(filepath)
            all_records.extend(records)
            if records:
                logger.info(f"  Loaded {len(records)} records from {filepath.name}")

    logger.info(f"\nTotal records loaded: {len(all_records)}")

    # 4. Extract tickers per record (pre-pass)
    logger.info("Extracting tickers from records...")
    records_with_tickers: List[Tuple[Dict, List[Tuple[str, str]]]] = []
    skipped = 0

    for record in all_records:
        text = record["text"]
        tickers = extract_tickers_from_text(text)
        if not tickers:
            skipped += 1
            continue
        records_with_tickers.append((record, tickers))

    logger.info(f"Records with ticker mentions: {len(records_with_tickers)}")
    logger.info(f"Records without ticker mentions: {skipped}")

    # 5. Calculate sentiment for all records
    if use_finbert:
        logger.info("Running FinBERT batch sentiment analysis...")
        texts_to_score: List[str] = []
        indices_needing_scoring: List[int] = []

        for idx, (record, _tickers) in enumerate(records_with_tickers):
            existing_label = record.get("existing_label")
            existing_confidence = record.get("existing_confidence")
            if existing_label and existing_confidence:
                continue
            texts_to_score.append(record["text"][:512])
            indices_needing_scoring.append(idx)

        logger.info(f"Texts to score with FinBERT: {len(texts_to_score)} "
                     f"(skipping {len(records_with_tickers) - len(texts_to_score)} with existing labels)")

        start_time = time.time()
        finbert_results = calculate_finbert_sentiments(texts_to_score, batch_size=32)
        elapsed = time.time() - start_time
        logger.info(f"FinBERT completed in {elapsed:.1f}s ({len(texts_to_score) / max(elapsed, 0.1):.0f} texts/sec)")

        finbert_lookup: Dict[int, Tuple[str, float]] = {}
        for i, idx in enumerate(indices_needing_scoring):
            finbert_lookup[idx] = finbert_results[i]

    # 6. Build stock-sentiment records with Pydantic validation
    from app.schemas.sentiment import StockSentimentRecord

    stock_sentiments: List[Dict] = []
    ticker_stats: Dict[str, int] = {}
    processed = 0
    validation_errors = 0

    for idx, (record, tickers) in enumerate(records_with_tickers):
        text = record["text"]

        existing_label = record.get("existing_label")
        existing_confidence = record.get("existing_confidence")

        if existing_label and existing_confidence:
            label = existing_label
            confidence = float(existing_confidence)
        elif use_finbert:
            label, confidence = finbert_lookup.get(idx, calculate_financial_sentiment(text))
        else:
            label, confidence = calculate_financial_sentiment(text)

        for ticker, mentioned_as in tickers:
            record_id = f"{ticker}_{hash(text) % 10**10}_{record['timestamp'][:10]}"

            record_data = {
                "id": record_id,
                "ticker": ticker,
                "mentioned_as": mentioned_as,
                "sentiment_label": label,
                "sentiment_score": confidence,
                "context": text[:300],
                "source": record["source"],
                "source_id": record.get("source_id", ""),
                "full_text": None,
                "position": None,
                "timestamp": record["timestamp"],
                "sentiment_mode": "finbert" if use_finbert else "keyword",
            }

            try:
                validated = StockSentimentRecord(**record_data)
                stock_sentiments.append(validated.model_dump())
            except ValidationError:
                validation_errors += 1
                stock_sentiments.append(record_data)

            ticker_stats[ticker] = ticker_stats.get(ticker, 0) + 1

        processed += 1

    if validation_errors > 0:
        logger.warning(f"Validation errors (records kept as-is): {validation_errors}")

    logger.info(f"\nProcessed: {processed} records with stock mentions")
    logger.info(f"Skipped: {skipped} records with no stock mentions")
    logger.info(f"Generated: {len(stock_sentiments)} stock-sentiment records")
    logger.info(f"Unique tickers: {len(ticker_stats)}")

    # Show top tickers
    top_tickers = sorted(ticker_stats.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info(f"\nTop 20 tickers by mentions:")
    for ticker, count in top_tickers:
        logger.info(f"  {ticker}: {count} mentions")

    # 5. Save to stock_sentiments.json
    output_dir = data_dir / "stock_sentiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "stock_sentiments.json"

    output_data = {
        "sentiments": stock_sentiments,
        "metadata": {
            "total_sentiments": len(stock_sentiments),
            "unique_tickers": len(ticker_stats),
            "last_updated": datetime.now().isoformat(),
            "sentiment_mode": "finbert" if use_finbert else "keyword",
            "processing_stats": {
                "total_records_loaded": len(all_records),
                "records_with_stocks": processed,
                "records_without_stocks": skipped,
                "ticker_distribution": dict(top_tickers),
            },
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved to {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        "total_records": len(all_records),
        "stock_sentiments": len(stock_sentiments),
        "unique_tickers": len(ticker_stats),
        "top_tickers": dict(top_tickers),
    }


def main():
    """Run the stock sentiment processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process stock sentiment from collected data."
    )
    parser.add_argument(
        "--use-finbert",
        action="store_true",
        default=False,
        help="Use FinBERT model for sentiment analysis (accurate, requires GPU/CPU time).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Use keyword-based sentiment (fast, no model loading). This is the default.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for FinBERT inference (default: 32).",
    )
    args = parser.parse_args()

    use_finbert = args.use_finbert and not args.fast

    logger.info("=" * 60)
    logger.info("Stock Sentiment Processing Pipeline")
    logger.info(f"  Mode: {'FinBERT' if use_finbert else 'Keyword (fast)'}")
    logger.info("=" * 60)

    backend_dir = Path(__file__).parent.parent.parent
    data_dir = backend_dir.parent / "data"

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info(f"Data directory: {data_dir}")
    logger.info("")

    stats = process_all_data(data_dir, use_finbert=use_finbert)

    logger.info("\n" + "=" * 60)
    logger.info("Processing Complete!")
    logger.info(f"  Sentiment mode: {'FinBERT' if use_finbert else 'Keyword'}")
    logger.info(f"  Total input records: {stats['total_records']}")
    logger.info(f"  Stock-sentiment pairs: {stats['stock_sentiments']}")
    logger.info(f"  Unique tickers: {stats['unique_tickers']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
