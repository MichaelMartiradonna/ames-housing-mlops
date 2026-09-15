"""Failure paths that protect performance gates and release-data restoration."""

import hashlib
import io
import zipfile

import pytest

from src.data_storage import restore_remote
from src.evaluate import PerformanceGateError, enforce_performance


def make_bundle(name="files/md5/ab/example"):
    data = b"approved test object"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, data)
    content = stream.getvalue()
    return content, {
        "archive_sha256": hashlib.sha256(content).hexdigest(),
        "objects": {name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}},
    }


def test_failed_performance_is_rejected(config):
    with pytest.raises(PerformanceGateError, match="MAE"):
        enforce_performance({"mae": 35001, "rmse": 45000, "r2": 0.8}, config)
    with pytest.raises(PerformanceGateError, match="R²"):
        enforce_performance({"mae": 20000, "rmse": 30000, "r2": 0.74}, config)
    with pytest.raises(PerformanceGateError, match="finite"):
        enforce_performance({"mae": float("nan"), "rmse": 30000, "r2": 0.8}, config)


def test_verified_archive_restores_expected_object(tmp_path):
    content, manifest = make_bundle()
    restore_remote(content, manifest, tmp_path / "remote")
    assert (tmp_path / "remote/files/md5/ab/example").read_bytes() == b"approved test object"


def test_modified_archive_is_rejected_before_writing(tmp_path):
    content, manifest = make_bundle()
    destination = tmp_path / "remote"
    with pytest.raises(ValueError, match="checksum"):
        restore_remote(content + b"tampered", manifest, destination)
    assert not destination.exists()


def test_archive_cannot_escape_storage_directory(tmp_path):
    content, manifest = make_bundle("../outside.txt")
    with pytest.raises(ValueError, match="escapes"):
        restore_remote(content, manifest, tmp_path / "remote")
    assert not (tmp_path / "outside.txt").exists()
