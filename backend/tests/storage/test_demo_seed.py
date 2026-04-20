"""Local SQLite demo seed (no DATABASE_URL)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

# Import a service module first so `app.storage` finishes initialising before
# `app.storage.database` (same pattern as tests/services/test_statistics_service.py).
import app.services.statistics_service  # noqa: F401

from app.storage.database import Base, SentimentRecordRow
from app.storage.demo_seed import ensure_demo_dataset


def test_demo_seed_populates_sqlite(tmp_path: Path) -> None:
    db_file = tmp_path / "demo.sqlite"
    url = f"sqlite:///{db_file.as_posix()}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    ensure_demo_dataset(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        stocks = (
            session.query(func.count(SentimentRecordRow.id))
            .filter(SentimentRecordRow.ticker.isnot(None))
            .scalar()
            or 0
        )
        docs = (
            session.query(func.count(SentimentRecordRow.id))
            .filter(SentimentRecordRow.ticker.is_(None))
            .scalar()
            or 0
        )
        assert stocks >= 12
        assert docs >= 1
    finally:
        session.close()


def test_demo_seed_is_idempotent(tmp_path: Path) -> None:
    db_file = tmp_path / "demo2.sqlite"
    url = f"sqlite:///{db_file.as_posix()}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    ensure_demo_dataset(engine)
    ensure_demo_dataset(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        total = session.query(func.count(SentimentRecordRow.id)).scalar() or 0
        assert total > 0
    finally:
        session.close()
