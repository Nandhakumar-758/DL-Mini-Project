from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT / "data" / "sample"
SAMPLE_FILES = [SAMPLE_DIR / "english_sample.txt", SAMPLE_DIR / "tamil_sample.txt"]


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory) -> Path:
    """Train a tiny real LSTM on the sample corpus once per test session."""
    pytest.importorskip("tensorflow")
    from app.ml.training import TrainConfig, train

    out = tmp_path_factory.mktemp("artifacts")
    cfg = TrainConfig.development(SAMPLE_FILES, out)
    cfg.epochs = 3
    cfg.embedding_dim = cfg.lstm_units = cfg.dense_units = 16
    train(cfg, log=lambda _msg: None)
    return out


def make_client(artifact_dir: Path, db_dir: Path, **overrides) -> TestClient:
    settings = Settings(
        artifact_dir=artifact_dir, db_path=db_dir / "app.db", _env_file=None, **overrides
    )
    return TestClient(create_app(settings))


@pytest.fixture()
def client_ready(trained_dir, tmp_path):
    with make_client(trained_dir, tmp_path) as client:
        yield client


@pytest.fixture()
def client_missing(tmp_path):
    empty = tmp_path / "empty_artifacts"
    empty.mkdir()
    with make_client(empty, tmp_path) as client:
        yield client


@pytest.fixture()
def client_tokenizer_only(trained_dir, tmp_path):
    only = tmp_path / "tok_only"
    only.mkdir()
    shutil.copy(trained_dir / "tokenizer.json", only / "tokenizer.json")
    with make_client(only, tmp_path) as client:
        yield client
