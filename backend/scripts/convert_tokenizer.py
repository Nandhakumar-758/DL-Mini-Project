"""Convert the report's pickled Keras tokenizer (tokenizer.pkl) to tokenizer.json.

    python -m scripts.convert_tokenizer /path/to/tokenizer.pkl --out ../artifacts/tokenizer.json

SECURITY: unpickling executes code embedded in the file. Only convert a pickle
you created yourself (for example the one saved by your own Colab notebook).
The web application never reads pickle files.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pickle_file", type=Path)
    p.add_argument("--out", type=Path, default=PROJECT / "artifacts" / "tokenizer.json")
    args = p.parse_args(argv)

    from app.ml.tokenizer import WordTokenizer

    try:
        with args.pickle_file.open("rb") as handle:
            keras_tokenizer = pickle.load(handle)  # noqa: S301 - user-supplied, documented above
        tokenizer = WordTokenizer.from_keras(keras_tokenizer)
    except (OSError, pickle.UnpicklingError, AttributeError, EOFError, ImportError) as exc:
        print(f"error: could not convert tokenizer: {exc}", file=sys.stderr)
        return 1
    tokenizer.save(args.out)
    print(f"Wrote {args.out.name}: vocab_size={tokenizer.vocab_size}, words={len(tokenizer.word_index)}")
    print("Make sure the model file next to it was trained with THIS tokenizer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
