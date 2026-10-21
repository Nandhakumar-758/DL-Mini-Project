"""Training and evaluation pipeline (report sections 3.1-3.8, 4.2-4.10, 4.14).

Used by ``scripts/train_model.py`` and ``scripts/evaluate_model.py``. All numbers
written to ``history.json`` / ``evaluation.json`` / ``metadata.json`` are
measured during the run; nothing is hard-coded.
"""
from __future__ import annotations

import json
import math
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.text import detect_script
from app.ml.data import (
    build_sequences,
    corpus_statistics,
    load_corpus,
    prepare_sentences,
    split_sentences,
)
from app.ml.model import ModelConfig, build_model
from app.ml.tokenizer import WordTokenizer

MODE_FULL = "full"
MODE_DEV = "development_sample"


@dataclass
class TrainConfig:
    corpus_paths: list[Path]
    output_dir: Path
    mode: str = MODE_FULL
    num_words: int = 10_000
    max_len: int = 11  # 10 context words + 1 target
    embedding_dim: int = 128
    lstm_units: int = 256
    dense_units: int = 256
    dropout: float = 0.3
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 0.001
    patience: int = 3
    val_split: float = 0.2
    seed: int = 42
    min_words: int = 3
    deduplicate: bool = True
    split_mode: str = "sentence"  # "sentence" (default) or "pairs" (report-compatible)
    model_filename: str = "next_word_lstm_model.keras"
    extra: dict = field(default_factory=dict)

    @classmethod
    def development(cls, corpus_paths: list[Path], output_dir: Path) -> "TrainConfig":
        """Tiny configuration for the illustrative sample corpus. NOT the report model."""
        return cls(
            corpus_paths=corpus_paths,
            output_dir=output_dir,
            mode=MODE_DEV,
            embedding_dim=32,
            lstm_units=64,
            dense_units=64,
            batch_size=32,
            epochs=40,
            patience=6,
        )


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def _topk_hits(probs: np.ndarray, targets: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    top1 = probs.argmax(axis=1) == targets
    k = min(k, probs.shape[1])
    top_idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    topk = (top_idx == targets[:, None]).any(axis=1)
    return top1, topk


def evaluate_model(
    model,
    tokenizer: WordTokenizer,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scripts: np.ndarray,
    batch_size: int = 512,
    split_strategy: str = "sentence",
) -> dict:
    """Validation loss/accuracy/top-5/perplexity plus known-word accuracy per script.

    Known-word accuracy excludes targets that are the OOV token, as in report 5.9.
    """
    if len(X_val) == 0:
        raise ValueError("The validation set is empty; cannot evaluate.")
    scores = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0, return_dict=True)
    loss = float(scores["loss"])

    top1_all: list[np.ndarray] = []
    top5_all: list[np.ndarray] = []
    for start in range(0, len(X_val), batch_size):  # batched to avoid a huge (N x vocab) matrix
        probs = np.asarray(model(X_val[start : start + batch_size], training=False))
        t1, t5 = _topk_hits(probs, y_val[start : start + batch_size])
        top1_all.append(t1)
        top5_all.append(t5)
    top1 = np.concatenate(top1_all)
    top5 = np.concatenate(top5_all)

    is_oov = (
        (y_val == tokenizer.oov_index) if tokenizer.oov_index is not None else np.zeros(len(y_val), bool)
    )

    def bucket(mask: np.ndarray) -> dict:
        total = int(mask.sum())
        known = mask & ~is_oov
        n_known = int(known.sum())
        return {
            "samples": total,
            "known_word_samples": n_known,
            "known_word_top1_accuracy": float(top1[known].mean()) if n_known else None,
            "known_word_top5_accuracy": float(top5[known].mean()) if n_known else None,
            "oov_target_share": float(is_oov[mask].mean()) if total else None,
        }

    known_word = {"all": bucket(np.ones(len(y_val), dtype=bool))}
    for name in ("english", "tamil", "mixed"):
        mask = scripts == name
        if mask.any():
            known_word[name] = bucket(mask)

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "validation",
        "split_strategy": split_strategy,
        "samples": int(len(y_val)),
        "loss": loss,
        "accuracy": float(scores.get("accuracy", float("nan"))),
        "top5_accuracy": float(scores.get("top5_accuracy", float("nan"))),
        "perplexity": float(math.exp(min(loss, 700))),  # exp(loss), as in report 3.8
        "oov_target_share": float(is_oov.mean()),
        "known_word": known_word,
    }


def scripts_for_pairs(sentences: list[str], owners: np.ndarray) -> np.ndarray:
    per_sentence = np.array([detect_script(s) for s in sentences])
    return per_sentence[owners] if len(owners) else np.array([], dtype=str)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def _library_versions() -> dict[str, str]:
    import keras
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
    }


def train(cfg: TrainConfig, log: Callable[[str], None] = print) -> dict:
    """Run the full pipeline and write all artifacts. Returns the metadata dict."""
    from tensorflow import keras

    started = time.time()
    keras.utils.set_random_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. load + clean -----------------------------------------------------------
    raw = load_corpus(cfg.corpus_paths)
    sentences, duplicates = prepare_sentences(raw, cfg.min_words, cfg.deduplicate)
    log(f"Loaded {len(raw)} lines -> {len(sentences)} usable sentences ({duplicates} duplicates removed)")
    if len(sentences) < 10:
        raise ValueError("Too few usable sentences to train (need at least 10).")

    # 2. split ------------------------------------------------------------------
    if cfg.split_mode == "sentence":
        train_s, val_s = split_sentences(sentences, cfg.val_split, cfg.seed)
        # Tokenizer is fitted on TRAINING sentences only so validation words
        # cannot leak into the vocabulary (the report fits on the whole corpus).
        tokenizer = WordTokenizer.fit(train_s, cfg.num_words)
        X_tr, y_tr, _ = build_sequences(train_s, tokenizer, cfg.max_len)
        X_val, y_val, owners = build_sequences(val_s, tokenizer, cfg.max_len)
        val_scripts = scripts_for_pairs(val_s, owners)
    elif cfg.split_mode == "pairs":
        from sklearn.model_selection import train_test_split

        tokenizer = WordTokenizer.fit(sentences, cfg.num_words)
        X, y, owners = build_sequences(sentences, tokenizer, cfg.max_len)
        all_scripts = scripts_for_pairs(sentences, owners)
        X_tr, X_val, y_tr, y_val, _, val_scripts = train_test_split(
            X, y, all_scripts, test_size=cfg.val_split, random_state=cfg.seed
        )
        train_s, val_s = sentences, sentences  # shared on purpose: this mode leaks
    else:
        raise ValueError("split_mode must be 'sentence' or 'pairs'.")

    log(f"Vocabulary size (model): {tokenizer.vocab_size}")
    log(f"Training pairs: {len(X_tr)} | Validation pairs: {len(X_val)}")
    if len(X_tr) == 0 or len(X_val) == 0:
        raise ValueError("Split produced an empty training or validation set; provide more text.")

    # 3. model ------------------------------------------------------------------
    mcfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        sequence_length=cfg.max_len - 1,
        embedding_dim=cfg.embedding_dim,
        lstm_units=cfg.lstm_units,
        dense_units=cfg.dense_units,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
    )
    model = build_model(mcfg)
    model.summary(print_fn=log)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience, restore_best_weights=True
    )
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[early_stop],
        verbose=2,
    )
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    epochs_run = len(hist["loss"])
    best_epoch = int(np.argmin(hist["val_loss"])) + 1

    # 4. evaluate with the restored best weights ---------------------------------
    evaluation = evaluate_model(
        model, tokenizer, X_val, y_val, val_scripts, split_strategy=cfg.split_mode
    )
    log(
        f"Validation: loss={evaluation['loss']:.4f} acc={evaluation['accuracy']:.4f} "
        f"top5={evaluation['top5_accuracy']:.4f} perplexity={evaluation['perplexity']:.2f}"
    )

    # 5. save ---------------------------------------------------------------------
    model.save(out / cfg.model_filename)
    tokenizer.save(out / "tokenizer.json")
    stats = corpus_statistics(sentences, tokenizer)
    metadata = {
        "format_version": 1,
        "run_id": uuid.uuid4().hex[:12],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": cfg.mode,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "config": {
            **{k: v for k, v in asdict(mcfg).items()},
            "num_words": cfg.num_words,
            "max_len": cfg.max_len,
            "batch_size": cfg.batch_size,
            "max_epochs": cfg.epochs,
            "patience": cfg.patience,
            "val_split": cfg.val_split,
            "seed": cfg.seed,
            "split_mode": cfg.split_mode,
            "deduplicate": cfg.deduplicate,
            "min_words": cfg.min_words,
        },
        "dataset": {
            "sources": [Path(p).name for p in cfg.corpus_paths],
            "raw_lines": len(raw),
            "duplicates_removed": duplicates,
            **stats,
            "train_sentences": len(train_s) if cfg.split_mode == "sentence" else None,
            "val_sentences": len(val_s) if cfg.split_mode == "sentence" else None,
            "vocab_size": tokenizer.vocab_size,
            "train_pairs": int(len(X_tr)),
            "val_pairs": int(len(X_val)),
            "split_strategy": cfg.split_mode,
        },
        "training": {
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "stopped_early": epochs_run < cfg.epochs,
            "restored_best_weights": True,
            "seconds": round(time.time() - started, 1),
        },
        "libraries": _library_versions(),
    }
    (out / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (out / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Artifacts written to {out}")
    return metadata
