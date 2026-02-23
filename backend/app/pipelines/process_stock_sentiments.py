"""DB-native stock enrichment pipeline.

This job reads existing document-level sentiment rows from SQLite, runs stock
entity analysis, and writes stock-level sentiment rows back into the same DB.
It no longer depends on local `data/raw/*` files.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict

from app.stocks import StockSentimentAnalyzer
from app.storage.database import SentimentRecordRow, get_session
from app.storage.record_ids import make_record_id
from app.storage.sqlite_storage import SQLiteSentimentStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MAX_TEXT_LEN = 2000
MAX_CONTEXT_LEN = 500


def _existing_stock_document_ids() -> set[str]:
    session = get_session()
    try:
        rows = (
            session.query(SentimentRecordRow.document_id)
            .filter(SentimentRecordRow.record_type == "stock")
            .filter(SentimentRecordRow.document_id.isnot(None))
            .distinct()
            .all()
        )
        return {r[0] for r in rows if r[0]}
    finally:
        session.close()


def process_documents(
    max_records: int | None = None,
    batch_size: int = 100,
    reprocess: bool = False,
) -> Dict[str, int]:
    """Generate stock-level rows from document rows already in SQLite."""
    storage = SQLiteSentimentStorage()
    storage.load()
    analyzer = StockSentimentAnalyzer()

    existing_stock_docs = _existing_stock_document_ids() if not reprocess else set()
    session = get_session()

    processed_docs = 0
    stock_rows_written = 0
    stock_batch: list[dict] = []
    ticker_stats: dict[str, int] = {}

    try:
        query = (
            session.query(SentimentRecordRow)
            .filter(SentimentRecordRow.record_type == "document")
            .order_by(SentimentRecordRow.timestamp.desc())
        )
        if max_records is not None:
            query = query.limit(max_records)

        for row in query.yield_per(batch_size):
            document_id = row.document_id or row.id
            if not reprocess and document_id in existing_stock_docs:
                continue

            text = (row.text or "").strip()[:MAX_TEXT_LEN]
            if len(text) < 10:
                continue

            try:
                analysis = analyzer.analyze(
                    text=text,
                    extract_context=True,
                    include_movements=True,
                )
            except OSError as exc:
                logger.error(
                    "spaCy model unavailable (%s). Run: python -m spacy download en_core_web_sm",
                    exc,
                )
                return {
                    "processed_documents": processed_docs,
                    "stock_rows_written": stock_rows_written,
                }
            except Exception as exc:
                logger.warning("Failed stock analysis for doc %s: %s", document_id, exc)
                continue

            for stock in analysis.get("stocks", []):
                ticker = stock.get("ticker")
                if not ticker:
                    continue
                position = stock.get("position") or {}
                context = (stock.get("context") or "")[:MAX_CONTEXT_LEN]

                stock_batch.append(
                    {
                        "id": make_record_id(
                            "stock",
                            document_id,
                            ticker,
                            stock.get("mentioned_as", ""),
                            str(position.get("start")),
                            str(position.get("end")),
                        ),
                        "record_type": "stock",
                        "document_id": document_id,
                        "text": text,
                        "ticker": ticker,
                        "mentioned_as": stock.get("mentioned_as", ""),
                        "sentiment_label": stock.get("sentiment", {}).get(
                            "label", row.sentiment_label
                        ),
                        "sentiment_score": float(
                            stock.get("sentiment", {}).get("score", row.sentiment_score)
                        ),
                        "context": context,
                        "source": row.source or "",
                        "source_id": row.source_id or "",
                        "position_start": position.get("start"),
                        "position_end": position.get("end"),
                        "timestamp": row.timestamp,
                        "sentiment_mode": "finbert",
                    }
                )
                ticker_stats[ticker] = ticker_stats.get(ticker, 0) + 1

            processed_docs += 1

            if len(stock_batch) >= 1000:
                stock_rows_written += storage.save_records_batch(stock_batch)
                stock_batch.clear()

        if stock_batch:
            stock_rows_written += storage.save_records_batch(stock_batch)

    finally:
        session.close()

    logger.info("Processed documents: %d", processed_docs)
    logger.info("Stock rows written: %d", stock_rows_written)
    logger.info("Unique tickers found: %d", len(ticker_stats))
    return {
        "processed_documents": processed_docs,
        "stock_rows_written": stock_rows_written,
        "unique_tickers": len(ticker_stats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enrich document records with stock-level sentiment rows.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of documents processed (debugging).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="DB streaming batch size (default: 100).",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-run stock extraction for documents that already have stock rows.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stock Sentiment Enrichment Pipeline (DB-first)")
    logger.info("=" * 60)
    process_documents(
        max_records=args.max_records,
        batch_size=args.batch_size,
        reprocess=args.reprocess,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
