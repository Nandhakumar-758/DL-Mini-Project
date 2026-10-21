"""Text preprocessing shared by the training pipeline and the inference engine.

The same function MUST be used in both places, otherwise the tokenizer sees
different text at training and prediction time.

Behaviour follows the report (section 3.2 / 4.2) with two documented changes:

* Only ASCII capital letters are lower-cased. Tamil has no letter case, so
  nothing in the Tamil block is touched by case folding.
* Text is normalised to Unicode NFC (canonical equivalence only). This makes
  text typed with different but equivalent code-point sequences match the
  corpus. It never changes the meaning or script of Tamil text.
"""
from __future__ import annotations

import re
import string
import unicodedata

TAMIL_RANGE = "஀-௿"

_DISALLOWED = re.compile(rf"[^a-z{TAMIL_RANGE}\s]")
_WHITESPACE = re.compile(r"\s+")
_TAMIL_CHAR = re.compile(f"[{TAMIL_RANGE}]")
_LATIN_CHAR = re.compile("[a-z]")
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
_TRAILING_TOKEN = re.compile(r"^(?P<context>.*?)(?P<last>\S*)$", re.DOTALL)


def fold(text: str) -> str:
    """NFC-normalise and lower-case ASCII letters only (Tamil is untouched)."""
    return unicodedata.normalize("NFC", text).translate(_ASCII_LOWER)


def clean_text(text: str) -> str:
    """Keep English letters, Tamil characters and single spaces only."""
    text = _DISALLOWED.sub(" ", fold(text))
    return _WHITESPACE.sub(" ", text).strip()


def has_tamil(text: str) -> bool:
    return bool(_TAMIL_CHAR.search(text))


def detect_script(text: str) -> str:
    """Return ``tamil``, ``english``, ``mixed`` or ``none`` for a piece of text."""
    folded = fold(text)
    tamil = has_tamil(folded)
    latin = bool(_LATIN_CHAR.search(folded))
    if tamil and latin:
        return "mixed"
    if tamil:
        return "tamil"
    if latin:
        return "english"
    return "none"


def split_trailing_token(text: str) -> tuple[str, str]:
    """Split raw text into (everything before the last token, the last token).

    The last token is empty when the text ends in whitespace (or is empty).
    """
    match = _TRAILING_TOKEN.match(text)
    assert match is not None  # the pattern matches every string
    return match.group("context"), match.group("last")


def ends_with_whitespace(text: str) -> bool:
    return bool(text) and text[-1].isspace()
