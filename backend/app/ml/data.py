"""Corpus loading, cleaning, splitting and n-gram sequence generation.

Pure Python / NumPy: nothing here needs TensorFlow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.text import clean_text, detect_script
from app.ml.tokenizer import WordTokenizer


class CorpusError(RuntimeError):
    """Raised for unusable corpus input (missing file, bad encoding, empty)."""


def load_corpus(paths: Sequence[Path]) -> list[str]:
    """Read UTF-8 text files, one sentence per line."""
    lines: list[str] = []
    if not paths:
        raise CorpusError("No corpus files were given.")
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise CorpusError(f"Corpus file not found: {path.name}")
        try:
            with path.open(encoding="utf-8-sig") as handle:  # tolerate a BOM
                lines.extend(line.rstrip("\n") for line in handle)
        except UnicodeDecodeError as exc:
            raise CorpusError(f"{path.name} is not valid UTF-8.") from exc
    return lines


def prepare_sentences(
    raw_lines: Sequence[str], min_words: int = 3, deduplicate: bool = True
) -> tuple[list[str], int]:
    """Clean lines, drop short ones and (optionally) exact duplicates.

    Returns ``(sentences, duplicates_removed)``. Removing duplicates matters
    because an identical sentence in both train and validation is leakage.
    """
    cleaned = [clean_text(line) for line in raw_lines]
    cleaned = [s for s in cleaned if len(s.split()) >= min_words]
    if not deduplicate:
        return cleaned, 0
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in cleaned:
        if sentence not in seen:
            seen.add(sentence)
            unique.append(sentence)
    return unique, len(cleaned) - len(unique)


def split_sentences(
    sentences: Sequence[str], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split by *sentence* (not by n-gram pair) with a seeded shuffle.

    The report splits individual n-gram pairs, which puts prefixes of the same
    sentence in both sets (the model is validated on sentences it has seen).
    Splitting whole sentences removes that leakage.
    """
    if len(sentences) < 2:
        raise CorpusError("At least two sentences are needed for a train/validation split.")
    order = np.random.RandomState(seed).permutation(len(sentences))
    n_val = min(max(1, int(round(len(sentences) * val_fraction))), len(sentences) - 1)
    val_idx = set(order[:n_val].tolist())
    train = [s for i, s in enumerate(sentences) if i not in val_idx]
    val = [s for i, s in enumerate(sentences) if i in val_idx]
    return train, val


def build_sequences(
    sentences: Sequence[str], tokenizer: WordTokenizer, max_len: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n-gram sequences exactly as in report section 3.4 / 4.4.

    For every position ``i >= 1`` the window ``tokens[max(0, i-max_len+1) : i+1]``
    is taken; the last element is the target, the rest (left-padded with 0 to
    ``max_len - 1``) is the input.

    Returns ``(X, y, sentence_ids)`` where ``sentence_ids[k]`` is the index of
    the sentence that produced pair ``k``.
    """
    context_len = max_len - 1
    xs: list[list[int]] = []
    ys: list[int] = []
    owners: list[int] = []
    for sid, sentence in enumerate(sentences):
        tokens = tokenizer.texts_to_sequence(sentence)
        for i in range(1, len(tokens)):
            window = tokens[max(0, i - max_len + 1) : i + 1]
            context = window[:-1]
            xs.append([0] * (context_len - len(context)) + context)  # "pre" padding
            ys.append(window[-1])
            owners.append(sid)
    if not xs:
        return (
            np.zeros((0, context_len), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(owners, dtype=np.int32),
    )


def pad_context(tokens: Sequence[int], context_len: int) -> np.ndarray:
    """Keep the last ``context_len`` tokens and left-pad with zeros (1 x context_len)."""
    tail = list(tokens)[-context_len:]
    row = [0] * (context_len - len(tail)) + tail
    return np.asarray([row], dtype=np.int32)


def corpus_statistics(
    sentences: Sequence[str],
    tokenizer: WordTokenizer,
) -> dict[str, int]:
    scripts = [detect_script(s) for s in sentences]
    return {
        "sentences": len(sentences),
        "tamil_sentences": scripts.count("tamil"),
        "english_sentences": scripts.count("english"),
        "mixed_sentences": scripts.count("mixed"),
        "total_words": sum(len(s.split()) for s in sentences),
        "unique_words": len(tokenizer.word_index) - (1 if tokenizer.oov_token else 0),
    }
