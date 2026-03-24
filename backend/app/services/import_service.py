"""Import service for ingesting external records into SQLite."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.storage.record_ids import make_record_id
from app.storage.sqlite_storage import SQLiteSentimentStorage

logger = logging.getLogger(__name__)


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
            # Lazy import to avoid loading heavy ML dependencies unless this job runs.
            from app.models.sentiment_inference import analyze_batch as analyzer

        texts = [row["text"] for row in prepared_rows]
        batch_out = analyzer(texts, batch_size=32, return_all_scores=False)
        # analyze_batch returns (results, failures); support plain list for tests/mocks.
        sentiments = batch_out[0] if isinstance(batch_out, tuple) else batch_out

        db_rows: list[dict[str, Any]] = []
        for row, sentiment in zip(prepared_rows, sentiments):
            if not sentiment:
                continue
            timestamp = row["timestamp"]
            source = row["source"]
            source_id = row["source_id"]
            text = row["text"]
            db_rows.append(
                {
                    "id": make_record_id(
                        "doc", source, source_id, timestamp, text[:120]
                    ),
                    "text": text,
                    "ticker": row["ticker"],
                    "mentioned_as": "",
                    "sentiment_label": sentiment["label"],
                    "sentiment_score": float(sentiment["score"]),
                    "source": source,
                    "source_id": source_id,
                    "timestamp": timestamp,
                }
            )

        return self.storage.save_records_batch(db_rows)

    def _normalize_record(self, row: dict[str, Any]) -> dict[str, Any] | None:
        text = self._extract_text(row)
        if not text:
            return None

        timestamp = self._extract_timestamp(row)
        source = self._extract_source(row)
        source_id = self._extract_source_id(row)
        ticker = row.get("ticker")
        if isinstance(ticker, str):
            ticker = ticker.upper()
        else:
            ticker = None

        return {
            "text": text,
            "source": source,
            "source_id": source_id,
            "timestamp": timestamp,
            "ticker": ticker,
        }

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
