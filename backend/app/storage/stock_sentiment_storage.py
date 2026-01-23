"""
Storage module for stock-sentiment pairs.

Manages saving and retrieving stock sentiment analysis results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class StockSentimentStorage:
    """Stores and retrieves stock sentiment data."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize stock sentiment storage.

        Args:
            storage_dir: Directory to store stock sentiment data.
                        Defaults to data/stock_sentiments/
        """
        if storage_dir is None:
            # Default to data/stock_sentiments/ in project root
            backend_dir = Path(__file__).parent.parent.parent
            storage_dir = backend_dir.parent / "data" / "stock_sentiments"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.main_file = self.storage_dir / "stock_sentiments.json"
        self._data: Dict = {"sentiments": [], "metadata": {}}
        self._loaded = False

    def load(self) -> None:
        """Load existing stock sentiment data."""
        if self._loaded:
            return

        if self.main_file.exists():
            with open(self.main_file, "r", encoding="utf-8") as f:
                self._data = json.load(f)
                logger.info(
                    f"Loaded {len(self._data.get('sentiments', []))} stock sentiments"
                )
        else:
            self._data = {"sentiments": [], "metadata": {}}
            logger.info("No existing stock sentiment data found")

        self._loaded = True

    def save(self) -> None:
        """Save stock sentiment data to file."""
        self._data["metadata"]["total_sentiments"] = len(
            self._data.get("sentiments", [])
        )
        self._data["metadata"]["last_updated"] = datetime.now().isoformat()

        with open(self.main_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

        logger.info(f"Saved stock sentiment data to {self.main_file}")

    def save_stock_sentiment(
        self,
        ticker: str,
        sentiment: Dict,
        context: str,
        mentioned_as: str,
        source: Optional[str] = None,
        source_id: Optional[str] = None,
        full_text: Optional[str] = None,
        position: Optional[Dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Save a single stock sentiment record.

        Args:
            ticker: Stock ticker symbol
            sentiment: Sentiment dict with 'label' and 'score'
            context: Context text around mention
            mentioned_as: How stock was mentioned (e.g., "Apple" or "$AAPL")
            source: Source type (reddit/twitter/news)
            source_id: Source post/tweet/article ID
            full_text: Full original text
            position: Position dict with 'start' and 'end'
            timestamp: Timestamp (defaults to now)

        Returns:
            Record ID
        """
        if not self._loaded:
            self.load()

        record_id = f"{ticker}_{int(datetime.now().timestamp() * 1000)}"

        record = {
            "id": record_id,
            "ticker": ticker,
            "mentioned_as": mentioned_as,
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "context": context,
            "source": source,
            "source_id": source_id,
            "full_text": full_text,
            "position": position,
            "timestamp": (
                timestamp.isoformat() if timestamp else datetime.now().isoformat()
            ),
        }

        self._data["sentiments"].append(record)
        self.save()

        return record_id

    def save_analysis_result(self, result: Dict, source: Optional[str] = None) -> List[str]:
        """
        Save full stock sentiment analysis result.

        Args:
            result: Result from StockSentimentAnalyzer.analyze()
            source: Source type (reddit/twitter/news)

        Returns:
            List of record IDs
        """
        record_ids = []

        for stock in result.get("stocks", []):
            record_id = self.save_stock_sentiment(
                ticker=stock["ticker"],
                sentiment=stock["sentiment"],
                context=stock.get("context", ""),
                mentioned_as=stock["mentioned_as"],
                source=source,
                full_text=result["text"],
                position=stock.get("position"),
            )
            record_ids.append(record_id)

        return record_ids

    def get_stock_sentiment(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get sentiment records for a specific stock.

        Args:
            ticker: Stock ticker symbol
            start_date: Filter by start date
            end_date: Filter by end date
            source: Filter by source type

        Returns:
            List of sentiment records
        """
        if not self._loaded:
            self.load()

        results = []

        for record in self._data.get("sentiments", []):
            # Filter by ticker
            if record["ticker"] != ticker:
                continue

            # Filter by date range
            if start_date or end_date:
                record_date = datetime.fromisoformat(record["timestamp"])
                if start_date and record_date < start_date:
                    continue
                if end_date and record_date > end_date:
                    continue

            # Filter by source
            if source and record.get("source") != source:
                continue

            results.append(record)

        return results

    def get_trending_stocks(
        self, min_mentions: int = 5, hours: int = 24
    ) -> List[Dict]:
        """
        Get stocks with most mentions in recent period.

        Args:
            min_mentions: Minimum number of mentions
            hours: Time period in hours

        Returns:
            List of dicts with ticker and mention count
        """
        if not self._loaded:
            self.load()

        # Calculate cutoff time
        cutoff = datetime.now().timestamp() - (hours * 3600)

        # Count mentions per ticker
        ticker_counts: Dict[str, int] = {}

        for record in self._data.get("sentiments", []):
            record_time = datetime.fromisoformat(
                record["timestamp"]
            ).timestamp()

            if record_time >= cutoff:
                ticker = record["ticker"]
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        # Filter and sort
        trending = [
            {"ticker": ticker, "mentions": count}
            for ticker, count in ticker_counts.items()
            if count >= min_mentions
        ]

        trending.sort(key=lambda x: x["mentions"], reverse=True)

        return trending

    def aggregate_sentiment(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Aggregate sentiment for a ticker over time period.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for aggregation
            end_date: End date for aggregation

        Returns:
            {
                'ticker': str,
                'total_mentions': int,
                'average_score': float,
                'sentiment_distribution': {
                    'positive': int,
                    'negative': int,
                    'neutral': int
                }
            }
        """
        records = self.get_stock_sentiment(ticker, start_date, end_date)

        if not records:
            return {
                "ticker": ticker,
                "total_mentions": 0,
                "average_score": 0.0,
                "sentiment_distribution": {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                },
            }

        # Calculate statistics
        total = len(records)
        scores = [r["sentiment_score"] for r in records]
        avg_score = sum(scores) / total if total > 0 else 0.0

        distribution = {"positive": 0, "negative": 0, "neutral": 0}
        for record in records:
            label = record["sentiment_label"]
            distribution[label] = distribution.get(label, 0) + 1

        return {
            "ticker": ticker,
            "total_mentions": total,
            "average_score": round(avg_score, 4),
            "sentiment_distribution": distribution,
        }

    def get_statistics(self) -> Dict:
        """
        Get overall statistics.

        Returns:
            Statistics dict
        """
        if not self._loaded:
            self.load()

        sentiments = self._data.get("sentiments", [])
        unique_tickers = set(s["ticker"] for s in sentiments)

        return {
            "total_records": len(sentiments),
            "unique_tickers": len(unique_tickers),
            "last_updated": self._data.get("metadata", {}).get(
                "last_updated", None
            ),
        }
