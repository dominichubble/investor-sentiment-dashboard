"""Tests for oldest-first pruning helpers."""

from sqlalchemy.exc import OperationalError

from app.storage.sqlite_storage import _storage_pressure_error


def test_storage_pressure_detects_diskfull_name():
    class DiskFull(Exception):
        pass

    assert _storage_pressure_error(DiskFull("x"))


def test_storage_pressure_detects_message_tokens():
    assert _storage_pressure_error(RuntimeError("No space left on device"))
    assert _storage_pressure_error(RuntimeError("Error 53100"))
    assert _storage_pressure_error(RuntimeError("exceeded storage limit"))


def test_storage_pressure_wrapped_operational_error():
    inner = Exception("could not extend file: No space left on device")
    outer = OperationalError("stmt", {}, inner)
    assert _storage_pressure_error(outer)


def test_storage_pressure_false_for_unrelated():
    assert not _storage_pressure_error(RuntimeError("connection refused"))
    assert not _storage_pressure_error(ValueError("bad data"))
