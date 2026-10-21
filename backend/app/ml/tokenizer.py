"""Word-level tokenizer compatible with the report's Keras ``Tokenizer``.

The report uses ``Tokenizer(num_words=10000, oov_token="<OOV>", filters="")``
and stores it with ``pickle``. Unpickling files is unsafe and ties inference to
a specific Keras internals layout, so this project stores the tokenizer as
JSON and re-implements the (small) subset of behaviour that is needed:

* index 0 is reserved for padding,
* index 1 is the OOV token,
* real words get indices 2.. by descending frequency (ties: first seen first),
* only indices < ``num_words`` are emitted, everything else becomes OOV.

``tests/test_tokenizer.py`` checks equivalence with Keras' own implementation.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

FORMAT_VERSION = 1
PAD_INDEX = 0


class TokenizerError(ValueError):
    """Raised when a tokenizer file is missing fields or malformed."""


class WordTokenizer:
    def __init__(
        self,
        word_index: dict[str, int],
        num_words: int = 10_000,
        oov_token: str | None = "<OOV>",
        word_counts: dict[str, int] | None = None,
    ) -> None:
        self.word_index = dict(word_index)
        self.num_words = int(num_words)
        self.oov_token = oov_token
        self.word_counts = dict(word_counts or {})
        self.index_word = {i: w for w, i in self.word_index.items()}
        self.oov_index = self.word_index.get(oov_token) if oov_token else None
        self.vocab_size = min(self.num_words, len(self.word_index) + 1)
        self.special_indices = frozenset(
            i for i in (PAD_INDEX, self.oov_index) if i is not None
        )
        # Candidate words for auto-complete: real words inside the model vocabulary.
        self._candidates: list[tuple[str, int]] = sorted(
            (
                (w, i)
                for w, i in self.word_index.items()
                if i < self.vocab_size and i not in self.special_indices
            ),
            key=lambda item: item[1],
        )

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        sentences: Iterable[str],
        num_words: int = 10_000,
        oov_token: str | None = "<OOV>",
    ) -> "WordTokenizer":
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(sentence.split())
        # Counter preserves first-seen order and sorted() is stable, which
        # mirrors Keras' tie-breaking.
        ordered = [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
        if oov_token:
            ordered.insert(0, oov_token)
        word_index = {w: i for i, w in enumerate(ordered, start=1)}
        return cls(word_index, num_words, oov_token, dict(counts))

    # ------------------------------------------------------------ encoding
    def texts_to_sequence(self, text: str) -> list[int]:
        """Convert already-cleaned text into token ids (Keras semantics)."""
        ids: list[int] = []
        for word in text.split():
            idx = self.word_index.get(word)
            if idx is not None and idx < self.num_words:
                ids.append(idx)
            elif self.oov_index is not None:
                ids.append(self.oov_index)
        return ids

    # --------------------------------------------------------- vocabulary
    def prefix_matches(self, prefix: str) -> list[tuple[str, int]]:
        """Vocabulary words (within the model vocabulary) starting with ``prefix``.

        Matching is on Unicode code points, which is what a user typing Tamil
        expects: typing ``க`` must match ``கி...`` and ``கு...``.
        """
        return [(w, i) for w, i in self._candidates if w.startswith(prefix)]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {"w": sorted(self.word_index.items()), "n": self.num_words, "o": self.oov_token},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def has_frequencies(self) -> bool:
        return bool(self.word_counts)

    # ----------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "num_words": self.num_words,
            "oov_token": self.oov_token,
            "vocab_size": self.vocab_size,
            "fingerprint": self.fingerprint,
            "word_index": self.word_index,
            "word_counts": self.word_counts,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: object) -> "WordTokenizer":
        if not isinstance(data, dict):
            raise TokenizerError("Tokenizer file must contain a JSON object.")
        word_index = data.get("word_index")
        num_words = data.get("num_words")
        if not isinstance(word_index, dict) or not word_index:
            raise TokenizerError("Tokenizer is missing a non-empty 'word_index'.")
        if not isinstance(num_words, int) or num_words < 2:
            raise TokenizerError("Tokenizer has an invalid 'num_words'.")
        for word, idx in word_index.items():
            if not isinstance(word, str) or not isinstance(idx, int) or isinstance(idx, bool) or idx < 1:
                raise TokenizerError("Tokenizer 'word_index' must map words to positive integers.")
        if len(set(word_index.values())) != len(word_index):
            raise TokenizerError("Tokenizer 'word_index' contains duplicate indices.")
        oov_token = data.get("oov_token")
        if oov_token is not None and not isinstance(oov_token, str):
            raise TokenizerError("Tokenizer has an invalid 'oov_token'.")
        word_counts = data.get("word_counts") or {}
        if not isinstance(word_counts, dict):
            raise TokenizerError("Tokenizer has invalid 'word_counts'.")
        return cls(word_index, num_words, oov_token, word_counts)

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TokenizerError("Tokenizer file could not be read as UTF-8 JSON.") from exc
        return cls.from_dict(data)

    @classmethod
    def from_keras(cls, keras_tokenizer) -> "WordTokenizer":
        """Build from a fitted ``keras`` Tokenizer (used by the pickle converter)."""
        return cls(
            dict(keras_tokenizer.word_index),
            keras_tokenizer.num_words or len(keras_tokenizer.word_index) + 1,
            keras_tokenizer.oov_token,
            dict(keras_tokenizer.word_counts),
        )
