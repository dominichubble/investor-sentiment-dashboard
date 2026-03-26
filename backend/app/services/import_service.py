"""Import service for ingesting external records into SQLite."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.analysis.financial_sentiment_enrichment import (
    build_rationale,
    collect_unique_snippets,
    enrich_aspects_with_scores,
    extract_aspect_snippets,
    normalized_label_entropy,
)
from app.storage.record_ids import make_record_id
from app.storage.sqlite_storage import SQLiteSentimentStorage
from app.utils.ticker_detection import TickerDetector

logger = logging.getLogger(__name__)

# Serialized into source_meta_json for traceability (Reddit permalink, scores, etc.).
_SOURCE_META_KEYS = frozenset(
    {
        "permalink",
        "score",
        "num_comments",
        "upvote_ratio",
        "url",
        "link_flair_text",
        "is_self",
        "domain",
        "author",
        "subreddit",
    }
)


class ImportService:
    """Import records from local datasets and persist FinBERT predictions."""

    TEXT_KEYS = (
        "text",
        "clean_content",
        "content",
        "selftext",
        "clean_description",
        "description",
        "clean_title",
        "title",
        "raw_text",
    )
    SOURCE_KEYS = ("source", "source_name", "subreddit")
    SOURCE_ID_KEYS = ("source_id", "id")
    TIMESTAMP_KEYS = ("timestamp", "created_at", "published_at", "publishedAt")

    def __init__(
        self,
        storage: SQLiteSentimentStorage | None = None,
        analyzer: Any | None = None,
    ) -> None:
        self.storage = storage or SQLiteSentimentStorage()
        self.analyzer = analyzer

    def import_from_data_dirs(
        self,
        data_root: Path | None = None,
        include_raw: bool = True,
        include_processed: bool = True,
    ) -> dict[str, int]:
        """Import and classify records from local raw/processed data folders."""
        if data_root is None:
            data_root = Path(__file__).resolve().parents[3] / "data"

        source_dirs: list[Path] = []
        if include_raw:
            source_dirs.append(data_root / "raw")
        if include_processed:
            source_dirs.append(data_root / "processed")

        payloads: list[dict[str, Any]] = []
        for directory in source_dirs:
            payloads.extend(self._load_payloads_from_dir(directory))

        inserted = self._classify_and_store(payloads)
        return {"records_loaded": len(payloads), "records_inserted": inserted}

    def import_from_records(self, records: list[dict[str, Any]]) -> dict[str, int]:
        """
        Import records already fetched from external APIs.

        This path is used to keep ingestion and storage separated while avoiding
        mandatory local file persistence.
        """
        inserted = self._classify_and_store(records)
        return {"records_loaded": len(records), "records_inserted": inserted}

    def _load_payloads_from_dir(self, directory: Path) -> list[dict[str, Any]]:
        if not directory.exists():
            return []

        payloads: list[dict[str, Any]] = []
        for json_path in directory.rglob("*.json"):
            if json_path.name.endswith("_meta.json"):
                continue
            payloads.extend(self._load_json_records(json_path))

        for csv_path in directory.rglob("*.csv"):
            payloads.extend(self._load_csv_records(csv_path))

        return payloads

    def _load_json_records(self, file_path: Path) -> list[dict[str, Any]]:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable JSON file %s: %s", file_path, exc)
            return []

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = []
            for key in ("items", "records", "data", "posts", "articles", "tweets"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
        else:
            rows = []

        return [r for r in rows if isinstance(r, dict)]

    def _load_csv_records(self, file_path: Path) -> list[dict[str, Any]]:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                return [row for row in csv.DictReader(handle)]
        except OSError as exc:
            logger.warning("Skipping unreadable CSV file %s: %s", file_path, exc)
            return []

    def _classify_and_store(self, records: list[dict[str, Any]]) -> int:
        prepared_rows = [self._normalize_record(row) for row in records]
        prepared_rows = [row for row in prepared_rows if row is not None]
        if not prepared_rows:
            return 0

        analyzer = self.analyzer
        if analyzer is None:
            from app.models.sentiment_inference import analyze_batch as analyzer

        texts = [row["text"] for row in prepared_rows]
        batch_out = analyzer(texts, batch_size=32, return_all_scores=True)
        sentiments = batch_out[0] if isinstance(batch_out, tuple) else batch_out

        detector = TickerDetector.get_instance()

        work_items: list[dict[str, Any]] = []
        skipped = 0
        for row, sentiment in zip(prepared_rows, sentiments):
            if not sentiment:
                continue
            text = row["text"]
            label = str(sentiment.get("label", "neutral"))
            score = float(sentiment.get("score", 0.5))
            raw_scores = sentiment.get("scores")
            if not isinstance(raw_scores, dict) or not raw_scores:
                rest = (1.0 - score) / 2.0 if score <= 1.0 else 0.0
                scores = {
                    "positive": rest,
                    "negative": rest,
                    "neutral": rest,
                }
                if label in scores:
                    scores[label] = score
            else:
                scores = {
                    "positive": float(raw_scores.get("positive", 0.0)),
                    "negative": float(raw_scores.get("negative", 0.0)),
                    "neutral": float(raw_scores.get("neutral", 0.0)),
                }

            entropy = normalized_label_entropy(scores)
            rationale = build_rationale(label, score, scores, entropy)
            aspects = extract_aspect_snippets(text)

            tickers = detector.detect(text)
            if not tickers:
                skipped += 1
                continue

            work_items.append(
                {
                    "row": row,
                    "label": label,
                    "score": score,
                    "scores": scores,
                    "entropy": entropy,
                    "rationale": rationale,
                    "aspects": aspects,
                    "tickers": tickers,
                }
            )

        aspect_lists = [w["aspects"] for w in work_items]
        unique_snips = collect_unique_snippets(aspect_lists)
        snip_sentiment: dict[str, dict[str, Any]] = {}
        if unique_snips:
            snip_batch = analyzer(
                unique_snips, batch_size=32, return_all_scores=True
            )
            snip_results = snip_batch[0] if isinstance(snip_batch, tuple) else snip_batch
            for snip, res in zip(unique_snips, snip_results):
                if res:
                    snip_sentiment[snip] = res

        db_rows: list[dict[str, Any]] = []
        for item in work_items:
            row = item["row"]
            timestamp = row["timestamp"]
            source = row["source"]
            source_id = row["source_id"]
            data_source = row.get("data_source")
            source_meta_json = row.get("source_meta_json")
            text = row["text"]
            aspects_json = (
                enrich_aspects_with_scores(item["aspects"], snip_sentiment)
                if item["aspects"]
                else None
            )

            for ticker, mentioned_as in item["tickers"]:
                db_rows.append(
                    {
                        "id": make_record_id(
                            "stock", source, source_id, ticker, mentioned_as
                        ),
                        "text": text,
                        "ticker": ticker,
                        "mentioned_as": mentioned_as,
                        "sentiment_label": item["label"],
                        "sentiment_score": item["score"],
                        "score_positive": item["scores"]["positive"],
                        "score_negative": item["scores"]["negative"],
                        "score_neutral": item["scores"]["neutral"],
                        "sentiment_uncertainty": item["entropy"],
                        "rationale": item["rationale"],
                        "aspects_json": aspects_json,
                        "source": source,
                        "data_source": data_source,
                        "source_id": source_id,
                        "source_meta_json": source_meta_json,
                        "timestamp": timestamp,
                    }
                )

        total = self.storage.save_records_batch(db_rows)
        if db_rows:
            logger.info(
                "Ticker detection: %d stock rows from %d documents (%d skipped, no ticker found)",
                len(db_rows),
                len(prepared_rows),
                skipped,
            )
        return total

    def _normalize_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
        text = self._extract_text(row)
        if not text:
            return None

        timestamp = self._extract_timestamp(row)
        source = self._extract_source(row)
        source_id = self._extract_source_id(row)
        data_source = self._extract_data_source(row)
        ticker = row.get("ticker")
        if isinstance(ticker, str):
            ticker = ticker.upper()
        else:
            ticker = None

        return {
            "text": text,
            "source": source,
            "source_id": source_id,
            "data_source": data_source,
            "source_meta_json": self._serialize_source_meta(row),
            "timestamp": timestamp,
            "ticker": ticker,
        }

    @staticmethod
    def _serialize_source_meta(row: dict[str, Any]) -> str | None:
        payload: dict[str, Any] = {}
        for k in _SOURCE_META_KEYS:
            if k not in row:
                continue
            v = row[k]
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                payload[k] = v
            else:
                payload[k] = str(v)
        if not payload:
            return None
        return json.dumps(payload, ensure_ascii=False)

    def _extract_text(self, row: dict[str, Any]) -> str:
        # Reddit entries are best represented as title + body.
        title = str(row.get("title", "")).strip()
        selftext = str(row.get("selftext", "")).strip()
        if title and selftext:
            return f"{title}\n\n{selftext}"
        if title:
            return title

        for key in self.TEXT_KEYS:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_source(self, row: dict[str, Any]) -> str:
        for key in self.SOURCE_KEYS:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return "unknown"

    def _extract_data_source(self, row: dict[str, Any]) -> str | None:
        """Platform channel: reddit | news | twitter (not outlet/subreddit name)."""
        raw = row.get("data_source")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
        if row.get("subreddit") is not None:
            return "reddit"
        if row.get("author_id") is not None:
            return "twitter"
        if row.get("source_name") is not None or row.get("clean_title") is not None:
            return "news"
        return None

    def _extract_source_id(self, row: dict[str, Any]) -> str:
        for key in self.SOURCE_ID_KEYS:
            value = row.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                return value_str
        return ""

    def _extract_timestamp(self, row: dict[str, Any]) -> str:
        # Support Unix timestamp fields from Reddit-like payloads.
        created_utc = row.get("created_utc")
        if isinstance(created_utc, (int, float)):
            return (
                datetime.fromtimestamp(created_utc, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        for key in self.TIMESTAMP_KEYS:
            value = row.get(key)
            if not value:
                continue

            if isinstance(value, (int, float)):
                dt = datetime.fromtimestamp(value, tz=timezone.utc)
                return dt.isoformat().replace("+00:00", "Z")

            if isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt.isoformat().replace("+00:00", "Z")
                except ValueError:
                    continue

        return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
