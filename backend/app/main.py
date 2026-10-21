"""FastAPI application factory.

Run from the ``backend`` directory:  ``uvicorn app.main:app --reload``
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.ml.engine import Engine
from app.services.history import HistoryStore

log = logging.getLogger("mozhipredict")


def _error(status: int, code: str, message: str, errors: list[dict[str, str]] | None = None):
    body: dict = {"code": code, "message": message}
    if errors:
        body["errors"] = errors
    return JSONResponse(status_code=status, content={"detail": body})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine.load()  # never raises: failures become a status the UI can show
        log.info("Model state at startup: %s", app.state.engine.state)
        yield

    app = FastAPI(
        title="MozhiPredict AI API",
        version="1.0.0",
        description="Tamil/English next-word prediction and auto-complete with a stacked LSTM.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = Engine(settings)
    app.state.history = HistoryStore(settings.db_path, settings.history_max_items)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_request_bytes:
            return _error(413, "request_too_large", "The request is too large.")
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        errors = [
            {
                "field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
                "message": str(err.get("msg", "Invalid value")).removeprefix("Value error, "),
            }
            for err in exc.errors()
        ]
        first = errors[0]["message"] if errors else "Invalid request."
        return _error(422, "invalid_request", first, errors)

    @app.exception_handler(Exception)
    async def unexpected_handler(_: Request, exc: Exception):
        log.exception("Unhandled error")  # details stay in the server log only
        return _error(500, "internal_error", "Unexpected server error.")

    app.include_router(router)
    return app


app = create_app()
