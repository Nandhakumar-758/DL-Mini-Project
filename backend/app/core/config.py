"""Application settings.

Every value can be overridden with an environment variable prefixed with
``MOZHI_`` (for example ``MOZHI_ARTIFACT_DIR``) or through ``backend/.env``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOZHI_", env_file=".env", extra="ignore")

    # --- model artifacts (kept separate from application records) -----------
    artifact_dir: Path = PROJECT_DIR / "artifacts"
    model_filename: str = "next_word_lstm_model.keras"
    tokenizer_filename: str = "tokenizer.json"
    metadata_filename: str = "metadata.json"
    history_filename: str = "history.json"
    evaluation_filename: str = "evaluation.json"

    # --- application records (SQLite) ----------------------------------------
    db_path: Path = BACKEND_DIR / "var" / "mozhipredict.db"
    history_max_items: int = 200

    # --- web ------------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_request_bytes: int = 16 * 1024

    # When the LSTM is unavailable but a tokenizer exists, auto-complete may
    # fall back to vocabulary-frequency ranking. Fallback results are always
    # labelled as such and are never presented as LSTM predictions.
    allow_vocabulary_fallback: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
