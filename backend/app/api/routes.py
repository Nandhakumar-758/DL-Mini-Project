"""HTTP endpoints. All paths are under /api."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from app.core.report_baseline import REPORT_BASELINE
from app.ml.engine import Engine, InferenceError, ModelUnavailable
from app.schemas.api import (
    DeleteHistoryResponse,
    ErrorResponse,
    HealthResponse,
    HistoryResponse,
    MetricsResponse,
    ModelInfoResponse,
    ModelStatus,
    PredictRequest,
    PredictResponse,
    SettingsBody,
    Suggestion,
)
from app.services.history import HistoryStore

log = logging.getLogger("mozhipredict.api")
router = APIRouter(prefix="/api")


def _engine(request: Request) -> Engine:
    return request.app.state.engine


def _history(request: Request) -> HistoryStore:
    return request.app.state.history


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _artifact(request: Request, filename: str) -> Any | None:
    return _read_json(Path(request.app.state.settings.artifact_dir) / filename)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Service liveness plus model readiness (the service is 'ok' even without a model)."""
    return HealthResponse(status="ok", model=ModelStatus(**_engine(request).status()))


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(request: Request) -> ModelInfoResponse:
    """Architecture and run metadata of the loaded model (no filesystem paths)."""
    engine = _engine(request)
    meta = engine.metadata or {}
    tok = engine.tokenizer
    return ModelInfoResponse(
        status=ModelStatus(**engine.status()),
        architecture=engine.architecture,
        tokenizer=(
            {
                "vocab_size": tok.vocab_size,
                "num_words": tok.num_words,
                "oov_token": tok.oov_token,
                "distinct_words_seen": len(tok.word_index) - (1 if tok.oov_token else 0),
                "has_frequency_table": tok.has_frequencies,
            }
            if tok
            else None
        ),
        run={k: meta.get(k) for k in ("run_id", "trained_at", "mode", "libraries")} if meta else None,
        config=meta.get("config"),
        dataset=meta.get("dataset"),
        training=meta.get("training"),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["prediction"],
)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Next-word prediction or prefix auto-complete.

    ``score`` is the LSTM softmax probability over the whole vocabulary
    (``score_type = model_probability``); it is not renormalised and not a
    confidence value. Fallback results carry ``score_type = vocabulary_frequency``.
    """
    engine = _engine(request)
    try:
        result = engine.predict(body.text, body.mode, body.top_k)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={"code": exc.code, "message": exc.message}) from exc
    except InferenceError as exc:
        raise HTTPException(
            status_code=500, detail={"code": "inference_failed", "message": str(exc)}
        ) from exc

    suggestions = [
        Suggestion(word=c.word, rank=i, score=round(c.score, 6), score_type=c.score_type, source=c.source)
        for i, c in enumerate(result.suggestions, start=1)
    ]
    store = _history(request)
    if suggestions and store.history_enabled():
        store.add(body.text, body.mode, body.top_k, [s.word for s in suggestions])

    return PredictResponse(
        input_text=body.text,
        mode=body.mode,
        prefix=result.prefix,
        suggestions=suggestions,
        source=suggestions[0].source if suggestions else None,
        model_status=ModelStatus(**engine.status()),
        message=result.message,
        diagnostics=result.diagnostics,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["model"])
def metrics(request: Request) -> MetricsResponse:
    """Measured training history and evaluation of the artifacts on disk.

    ``report_baseline`` is always included but is quoted from the report and
    labelled as such; it is never mixed into the measured fields.
    """
    engine = _engine(request)
    s = request.app.state.settings
    history = _artifact(request, s.history_filename)
    evaluation = _artifact(request, s.evaluation_filename)
    meta = engine.metadata or _artifact(request, s.metadata_filename) or {}
    available = bool(history or evaluation)
    return MetricsResponse(
        available=available,
        message=(
            "Measured metrics loaded from the training artifacts."
            if available
            else "Metrics not available. Run scripts/train_model.py (or scripts/evaluate_model.py "
            "for an existing model) to generate history.json and evaluation.json."
        ),
        development_model=engine.development_model,
        history=history if isinstance(history, dict) else None,
        evaluation=evaluation if isinstance(evaluation, dict) else None,
        training=meta.get("training"),
        dataset=meta.get("dataset"),
        report_baseline=REPORT_BASELINE,
    )


@router.get("/history", response_model=HistoryResponse, tags=["history"])
def get_history(request: Request) -> HistoryResponse:
    """Only entries the user opted in to saving. Empty unless history is enabled."""
    store = _history(request)
    return HistoryResponse(enabled=store.history_enabled(), items=store.list())


@router.delete("/history", response_model=DeleteHistoryResponse, tags=["history"])
def delete_history(request: Request) -> DeleteHistoryResponse:
    return DeleteHistoryResponse(deleted=_history(request).clear())


@router.put("/settings", response_model=SettingsBody, tags=["history"])
def update_settings(body: SettingsBody, request: Request, response: Response) -> SettingsBody:
    """Turn opt-in history on or off. Turning it off does not delete saved items."""
    _history(request).set_history_enabled(body.history_enabled)
    return body


@router.get("/settings", response_model=SettingsBody, tags=["history"])
def get_settings_endpoint(request: Request) -> SettingsBody:
    return SettingsBody(history_enabled=_history(request).history_enabled())
