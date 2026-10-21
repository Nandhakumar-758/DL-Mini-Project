"""Typed request / response models for the public API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 2000

Mode = Literal["next_word", "autocomplete"]
ScoreType = Literal["model_probability", "vocabulary_frequency"]
Source = Literal["lstm", "vocabulary_fallback"]


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        max_length=MAX_TEXT_CHARS,
        description="Text typed so far (Tamil, English or mixed). Never modified in the response.",
    )
    mode: Mode = Field("next_word", description="'next_word' or 'autocomplete'.")
    top_k: int = Field(5, ge=1, le=10, description="Number of suggestions (UI offers 3, 5, 10).")

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be empty.")
        return value


class Suggestion(BaseModel):
    word: str
    rank: int = Field(..., ge=1)
    score: float = Field(
        ...,
        description=(
            "For score_type 'model_probability': the LSTM softmax probability (0-1) of this word "
            "over the WHOLE vocabulary. Scores of the returned words do not sum to 1. "
            "For 'vocabulary_frequency': the word's relative frequency in the training corpus."
        ),
    )
    score_type: ScoreType
    source: Source


class ModelStatus(BaseModel):
    state: str
    ready: bool
    lstm_available: bool
    fallback_available: bool
    development_model: bool
    message: str


class Diagnostics(BaseModel):
    context_words: int = 0
    unknown_context_words: int = 0
    context_truncated: bool = False


class PredictResponse(BaseModel):
    input_text: str
    mode: Mode
    prefix: str | None = None
    suggestions: list[Suggestion]
    source: Source | None = Field(None, description="Where the suggestions came from, if any.")
    model_status: ModelStatus
    message: str | None = None
    diagnostics: Diagnostics = Diagnostics()


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: ModelStatus


class ErrorBody(BaseModel):
    code: str
    message: str
    errors: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    detail: ErrorBody


class HistoryItem(BaseModel):
    id: int
    created_at: str
    text: str
    mode: Mode
    top_k: int
    suggestions: list[str]


class HistoryResponse(BaseModel):
    enabled: bool
    items: list[HistoryItem]


class SettingsBody(BaseModel):
    history_enabled: bool


class DeleteHistoryResponse(BaseModel):
    deleted: int


class ModelInfoResponse(BaseModel):
    status: ModelStatus
    architecture: dict[str, Any] | None = None
    tokenizer: dict[str, Any] | None = None
    run: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    training: dict[str, Any] | None = None


class MetricsResponse(BaseModel):
    available: bool
    message: str
    development_model: bool = False
    history: dict[str, list[float]] | None = None
    evaluation: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    report_baseline: dict[str, Any]
