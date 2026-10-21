"""Evaluate a saved model + tokenizer on a held-out split and write evaluation.json.

    python -m scripts.evaluate_model --corpus ../data/english.txt ../data/tamil.txt

The validation set is rebuilt with the same seed and split mode that training
used. WARNING: if the model was trained elsewhere (for example in the report's
Colab notebook) on a different split, part of this "validation" data may have
been seen in training and the scores will be optimistic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", nargs="+", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, default=PROJECT / "artifacts")
    p.add_argument("--model-file", default="next_word_lstm_model.keras")
    p.add_argument("--tokenizer-file", default="tokenizer.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--split-mode", choices=["sentence", "pairs"], default="sentence")
    args = p.parse_args(argv)

    from tensorflow import keras

    from app.ml.data import (
        CorpusError,
        build_sequences,
        load_corpus,
        prepare_sentences,
        split_sentences,
    )
    from app.ml.tokenizer import TokenizerError, WordTokenizer
    from app.ml.training import evaluate_model, scripts_for_pairs

    try:
        tokenizer = WordTokenizer.load(args.artifact_dir / args.tokenizer_file)
        model = keras.models.load_model(args.artifact_dir / args.model_file, compile=False)
        model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")],
        )
        if int(model.output_shape[-1]) != tokenizer.vocab_size:
            print("error: model and tokenizer do not match (different vocabulary sizes).", file=sys.stderr)
            return 1
        max_len = int(model.input_shape[1]) + 1
        sentences, _ = prepare_sentences(load_corpus(args.corpus), args.min_words, not args.no_dedupe)
        if args.split_mode == "sentence":
            _, val_s = split_sentences(sentences, args.val_split, args.seed)
            X_val, y_val, owners = build_sequences(val_s, tokenizer, max_len)
            scripts = scripts_for_pairs(val_s, owners)
        else:
            from sklearn.model_selection import train_test_split

            X, y, owners = build_sequences(sentences, tokenizer, max_len)
            scripts_all = scripts_for_pairs(sentences, owners)
            _, X_val, _, y_val, _, scripts = train_test_split(
                X, y, scripts_all, test_size=args.val_split, random_state=args.seed
            )
        result = evaluate_model(model, tokenizer, X_val, y_val, scripts, split_strategy=args.split_mode)
    except (CorpusError, TokenizerError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = args.artifact_dir / "evaluation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"loss={result['loss']:.4f} accuracy={result['accuracy']:.4f} "
        f"top5={result['top5_accuracy']:.4f} perplexity={result['perplexity']:.2f}"
    )
    print(f"Wrote {out.name} ({result['samples']} validation pairs). Training history is not recomputed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
