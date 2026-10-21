"""Inference engine: loads the trained model + tokenizer and answers requests.

Design rules (from the project brief):

* The model and the tokenizer must come from the same training run. This is
  verified by tensor shapes and, when ``metadata.json`` exists, by a tokenizer
  fingerprint.
* LSTM scores are the model's softmax probabilities over the *whole*
  vocabulary. They are NOT re-normalised after filtering, so they do not sum
  to 100 % and must not be described as "confidence".
* Vocabulary-frequency suggestions (fallback) are always labelled as such.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.text import (
    clean_text,
    ends_with_whitespace,
    fold,
    split_trailing_token,
)
from app.ml.data import pad_context
from app.ml.tokenizer import TokenizerError, WordTokenizer

log = logging.getLogger("mozhipredict.engine")

SCORE_MODEL = "model_probability"
SCORE_FREQUENCY = "vocabulary_frequency"
SOURCE_LSTM = "lstm"
SOURCE_FALLBACK = "vocabulary_fallback"


class ModelUnavailable(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InferenceError(Exception):
    """The model was loaded but running it failed."""


@dataclass
class Candidate:
    word: str
    score: float
    score_type: str
    source: str


@dataclass
class PredictionResult:
    suggestions: list[Candidate]
    message: str | None = None
    prefix: str | None = None
    diagnostics: dict = field(default_factory=dict)


_STATE_MESSAGES = {
    "ready": "Trained LSTM model loaded and ready.",
    "missing_artifacts": (
        "No trained model found. Train one with scripts/train_model.py or copy your "
        "model and tokenizer into the artifacts directory (see docs/setup-guide.md)."
    ),
    "invalid_artifacts": "The model or tokenizer file is invalid or they do not belong together.",
    "tensorflow_unavailable": "TensorFlow is not installed, so the LSTM cannot run.",
    "load_failed": "The model file could not be loaded.",
}


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._reset()

    # ------------------------------------------------------------------ state
    def _reset(self) -> None:
        self.model = None
        self.tokenizer: WordTokenizer | None = None
        self.metadata: dict | None = None
        self.state = "missing_artifacts"
        self.sequence_length = 10
        self.architecture: dict | None = None

    def _fail(self, state: str, detail: str | None = None) -> None:
        self.model = None
        self.state = state
        if detail:
            log.warning("Model not loaded (%s): %s", state, detail)

    @property
    def lstm_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.state == "ready"

    @property
    def fallback_available(self) -> bool:
        return (
            self.settings.allow_vocabulary_fallback
            and self.tokenizer is not None
            and self.tokenizer.has_frequencies
        )

    @property
    def development_model(self) -> bool:
        return bool(self.metadata and self.metadata.get("mode") == "development_sample")

    def status(self) -> dict:
        message = _STATE_MESSAGES.get(self.state, "Unknown state.")
        if not self.lstm_ready and self.fallback_available:
            message += " Auto-complete can use a labelled vocabulary-frequency fallback."
        if self.lstm_ready and self.development_model:
            message = (
                "Development model loaded (trained on the tiny illustrative sample corpus). "
                "It is NOT the report's full model."
            )
        return {
            "state": self.state,
            "ready": self.lstm_ready,
            "lstm_available": self.lstm_ready,
            "fallback_available": (not self.lstm_ready) and self.fallback_available,
            "development_model": self.development_model,
            "message": message,
        }

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        s = self.settings
        self._reset()
        tokenizer_path = Path(s.artifact_dir) / s.tokenizer_filename
        model_path = Path(s.artifact_dir) / s.model_filename
        metadata_path = Path(s.artifact_dir) / s.metadata_filename

        if tokenizer_path.is_file():
            try:
                self.tokenizer = WordTokenizer.load(tokenizer_path)
            except TokenizerError as exc:
                self._fail("invalid_artifacts", f"tokenizer: {exc}")
                return
        if metadata_path.is_file():
            try:
                self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("metadata.json could not be parsed; ignoring it")

        if self.tokenizer is None or not model_path.is_file():
            self._fail("missing_artifacts", "tokenizer or model file absent")
            return

        try:
            from tensorflow import keras  # noqa: WPS433 (lazy import by design)
        except ImportError:
            self._fail("tensorflow_unavailable", "tensorflow import failed")
            return

        try:
            model = keras.models.load_model(model_path, compile=False)
        except Exception as exc:  # noqa: BLE001 - any loader failure => friendly state
            self._fail("load_failed", f"{type(exc).__name__}: {exc}")
            return

        # --- the model and the tokenizer must belong together -----------------
        try:
            out_dim = int(model.output_shape[-1])
            seq_len = int(model.input_shape[1])
        except Exception as exc:  # noqa: BLE001
            self._fail("invalid_artifacts", f"cannot read model shapes: {exc}")
            return
        if out_dim != self.tokenizer.vocab_size:
            self._fail(
                "invalid_artifacts",
                f"model output size {out_dim} != tokenizer vocab size {self.tokenizer.vocab_size}",
            )
            return
        expected = (self.metadata or {}).get("tokenizer_fingerprint")
        if expected and expected != self.tokenizer.fingerprint:
            self._fail("invalid_artifacts", "tokenizer fingerprint does not match metadata")
            return

        self.sequence_length = seq_len
        self.model = model
        self.architecture = self._describe_model(model)
        try:  # warm-up: surfaces inference problems at start-up, not on first request
            self._probabilities(pad_context([], seq_len))
        except Exception as exc:  # noqa: BLE001
            self._fail("load_failed", f"warm-up inference failed: {exc}")
            return
        self.state = "ready"

    @staticmethod
    def _describe_model(model) -> dict:
        layers = []
        for layer in model.layers:
            try:
                shape = [None if d is None else int(d) for d in layer.output.shape]
            except Exception:  # noqa: BLE001
                shape = None
            info = {"type": layer.__class__.__name__, "output_shape": shape, "parameters": int(layer.count_params())}
            for attr in ("units", "rate", "output_dim", "activation"):
                value = getattr(layer, attr, None)
                if value is not None:
                    info[attr] = value if isinstance(value, (int, float)) else getattr(value, "__name__", str(value))
            layers.append(info)
        return {
            "sequence_length": int(model.input_shape[1]),
            "vocab_size": int(model.output_shape[-1]),
            "total_parameters": int(model.count_params()),
            "layers": layers,
        }

    # -------------------------------------------------------------- inference
    def _probabilities(self, padded: np.ndarray) -> np.ndarray:
        assert self.model is not None
        try:
            with self._lock:
                out = self.model(padded, training=False)
            return np.asarray(out, dtype=np.float64)[0]
        except Exception as exc:  # noqa: BLE001
            log.exception("Inference failed")
            raise InferenceError("The model failed while computing predictions.") from exc

    def _top_words(self, probs: np.ndarray, k: int) -> list[Candidate]:
        tok = self.tokenizer
        assert tok is not None
        masked = probs.copy()
        masked[list(tok.special_indices)] = -np.inf  # never suggest padding / OOV
        masked[tok.vocab_size :] = -np.inf
        order = np.argsort(-masked)[:k]
        return [
            Candidate(tok.index_word[int(i)], float(probs[i]), SCORE_MODEL, SOURCE_LSTM)
            for i in order
            if np.isfinite(masked[i]) and int(i) in tok.index_word
        ]

    def _require_lstm(self) -> None:
        if not self.lstm_ready:
            raise ModelUnavailable(
                "model_unavailable",
                _STATE_MESSAGES.get(self.state, "The LSTM model is not available."),
            )

    def predict(self, text: str, mode: str, top_k: int) -> PredictionResult:
        if mode == "next_word":
            return self._next_word(text, top_k)
        if mode == "autocomplete":
            return self._autocomplete(text, top_k)
        raise ValueError(f"Unknown mode: {mode}")

    def _context_probs(self, cleaned_context: str) -> tuple[np.ndarray, dict]:
        tok = self.tokenizer
        assert tok is not None
        tokens = tok.texts_to_sequence(cleaned_context)
        unknown = sum(1 for t in tokens if t == tok.oov_index)
        diagnostics = {
            "context_words": len(tokens),
            "unknown_context_words": unknown,
            "context_truncated": len(tokens) > self.sequence_length,
        }
        padded = pad_context(tokens, self.sequence_length)
        return self._probabilities(padded), diagnostics

    def _next_word(self, text: str, k: int) -> PredictionResult:
        self._require_lstm()
        cleaned = clean_text(text)
        if not cleaned:
            return PredictionResult(
                [], "No usable words found. Type some Tamil or English words first.", None,
                {"context_words": 0, "unknown_context_words": 0, "context_truncated": False},
            )
        probs, diag = self._context_probs(cleaned)
        suggestions = self._top_words(probs, k)
        message = None
        if diag["context_words"] and diag["unknown_context_words"] == diag["context_words"]:
            message = (
                "None of the typed words are in the model vocabulary, so these suggestions "
                "come from unknown-word (<OOV>) context and are unreliable."
            )
        elif not suggestions:
            message = "The model returned no usable suggestions for this context."
        return PredictionResult(suggestions, message, None, diag)

    def _autocomplete(self, text: str, k: int) -> PredictionResult:
        tok = self.tokenizer
        empty_diag = {"context_words": 0, "unknown_context_words": 0, "context_truncated": False}
        if tok is None:
            raise ModelUnavailable("model_unavailable", _STATE_MESSAGES.get(self.state, ""))
        if ends_with_whitespace(text):
            return PredictionResult(
                [], "Type part of a word to auto-complete it, or switch to Next-Word mode.", "", empty_diag
            )

        context_raw, last_raw = split_trailing_token(text)
        prefix = clean_text(last_raw)
        if not prefix or prefix != fold(last_raw) or " " in prefix:
            return PredictionResult(
                [],
                "The word being typed contains digits or punctuation that the model ignores, "
                "so it cannot be auto-completed.",
                prefix or "",
                empty_diag,
            )

        matches = tok.prefix_matches(prefix)
        context = clean_text(context_raw)
        has_context = bool(context)

        if not matches:
            return PredictionResult(
                [], f"No vocabulary words start with “{prefix}”.", prefix, empty_diag
            )

        if self.lstm_ready and has_context:
            probs, diag = self._context_probs(context)
            ranked = sorted(
                matches,
                key=lambda m: (probs[m[1]], tok.word_counts.get(m[0], 0)),
                reverse=True,
            )[:k]
            suggestions = [Candidate(w, float(probs[i]), SCORE_MODEL, SOURCE_LSTM) for w, i in ranked]
            return PredictionResult(suggestions, None, prefix, diag)

        # ---- fallback: vocabulary frequency (never presented as an LSTM result)
        if not self.fallback_available:
            self._require_lstm()
            return PredictionResult(
                [],
                "Auto-complete needs at least one preceding word so the LSTM has context.",
                prefix,
                empty_diag,
            )
        total = float(sum(tok.word_counts.values())) or 1.0
        ranked = sorted(matches, key=lambda m: tok.word_counts.get(m[0], 0), reverse=True)[:k]
        suggestions = [
            Candidate(w, tok.word_counts.get(w, 0) / total, SCORE_FREQUENCY, SOURCE_FALLBACK)
            for w, _ in ranked
        ]
        reason = (
            "there is no preceding word for the LSTM to use"
            if self.lstm_ready
            else "the trained LSTM is not loaded"
        )
        return PredictionResult(
            suggestions,
            f"Vocabulary-frequency fallback (not an LSTM prediction): {reason}.",
            prefix,
            empty_diag,
        )
