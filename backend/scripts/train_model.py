"""Train the stacked-LSTM next-word model and write all artifacts.

Run from the ``backend`` directory:

    # real training on your own corpus (report configuration)
    python -m scripts.train_model --corpus ../data/english.txt ../data/tamil.txt

    # tiny development model on the illustrative sample corpus (NOT the report model)
    python -m scripts.train_model --dev

Artifacts written to --output-dir (default ../artifacts, or ../artifacts/dev_sample
with --dev): next_word_lstm_model.keras, tokenizer.json, metadata.json,
history.json, evaluation.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
SAMPLE_FILES = [PROJECT / "data" / "sample" / "english_sample.txt", PROJECT / "data" / "sample" / "tamil_sample.txt"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", nargs="+", type=Path, help="UTF-8 text files, one sentence per line.")
    p.add_argument("--output-dir", type=Path, help="Where to write artifacts.")
    p.add_argument("--dev", action="store_true", help="Tiny model on the illustrative sample corpus.")
    p.add_argument("--epochs", type=int, help="Maximum epochs (report: 20).")
    p.add_argument("--batch-size", type=int, help="Batch size (report: 128).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-words", type=int, default=3, help="Drop sentences shorter than this (default 3).")
    p.add_argument("--no-dedupe", action="store_true", help="Keep duplicate sentences (report behaviour).")
    p.add_argument(
        "--split-mode",
        choices=["sentence", "pairs"],
        default="sentence",
        help="'sentence' (default, no leakage) or 'pairs' (report-compatible random n-gram split; leaks).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from app.ml.training import TrainConfig, train  # imported late: TensorFlow is heavy

    if args.dev:
        corpus = args.corpus or SAMPLE_FILES
        out = args.output_dir or PROJECT / "artifacts" / "dev_sample"
        cfg = TrainConfig.development(list(corpus), out)
        print("DEVELOPMENT MODE: tiny model on illustrative sample data. This is NOT the report's model.")
    else:
        if not args.corpus:
            print("error: --corpus is required (or use --dev). See data/README.md.", file=sys.stderr)
            return 2
        out = args.output_dir or PROJECT / "artifacts"
        cfg = TrainConfig(corpus_paths=list(args.corpus), output_dir=out)

    cfg.seed = args.seed
    cfg.min_words = args.min_words
    cfg.deduplicate = not args.no_dedupe
    cfg.split_mode = args.split_mode
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    from app.ml.data import CorpusError

    try:
        train(cfg)
    except (CorpusError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
