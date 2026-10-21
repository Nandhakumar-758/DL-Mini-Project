# Artifacts

Trained model files live here and are **git-ignored** (they are large and run-specific).

A usable directory contains files that come from the **same training run**:

| File | Purpose |
|---|---|
| `next_word_lstm_model.keras` | Keras model |
| `tokenizer.json` | Word index, OOV token, word counts (JSON, never pickle) |
| `metadata.json` | Config, dataset statistics, tokenizer fingerprint, library versions |
| `history.json` | Per-epoch loss / accuracy / top-5 accuracy |
| `evaluation.json` | Validation loss, accuracy, top-5, perplexity, known-word accuracy per script |

On start-up the backend checks that the model's output size equals the tokenizer's vocabulary size
and (when `metadata.json` exists) that the tokenizer fingerprint matches, and refuses mismatched pairs.

Choose the directory with `MOZHI_ARTIFACT_DIR`. `dev_sample/` is a tiny development model trained
on the illustrative sample corpus. It is not the report's model and is labelled as such in the UI.

Generate real artifacts with `python -m scripts.train_model --corpus ...` (see `docs/setup-guide.md`).
