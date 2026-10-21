"""Stacked-LSTM language model from report section 3.5 / 4.6-4.7.

TensorFlow is imported lazily so that the rest of the backend (and its tests)
can run on machines where TensorFlow is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int = 10  # context words; max_len - 1
    embedding_dim: int = 128
    lstm_units: int = 256
    dense_units: int = 256
    dropout: float = 0.3
    learning_rate: float = 0.001


def build_model(cfg: ModelConfig):
    """Embedding -> LSTM(return_sequences) -> Dropout -> LSTM -> Dropout -> Dense(relu) -> Dense(softmax)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(cfg.sequence_length,)),
            layers.Embedding(cfg.vocab_size, cfg.embedding_dim),
            layers.LSTM(cfg.lstm_units, return_sequences=True),
            layers.Dropout(cfg.dropout),
            layers.LSTM(cfg.lstm_units),
            layers.Dropout(cfg.dropout),
            layers.Dense(cfg.dense_units, activation="relu"),
            layers.Dense(cfg.vocab_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model
