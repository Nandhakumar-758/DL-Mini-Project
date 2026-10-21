# Architecture

```
Browser (React)  ──/api──▶  FastAPI  ──▶  Engine ──▶ Keras LSTM + tokenizer.json   (artifacts/)
                               │
                               └──▶  SQLite (opt-in history, settings)              (backend/var/)
```

## Backend (`backend/app`)

| Module | Role |
|---|---|
| `main.py` | App factory, CORS, request-size guard, start-up model loading, generic error handler (no stack traces to clients) |
| `api/routes.py` | The `/api` endpoints |
| `schemas/api.py` | Pydantic request/response models and validation |
| `core/config.py` | Settings from `MOZHI_*` environment variables |
| `core/text.py` | Cleaning shared by training and inference (report §3.2) |
| `core/report_baseline.py` | Numbers quoted from the report, kept apart from measured metrics |
| `ml/tokenizer.py` | Word tokenizer with `num_words=10000`, `<OOV>`, JSON persistence, fingerprint |
| `ml/data.py` | Corpus loading, de-duplication, n-gram sequences, left padding, sentence split |
| `ml/model.py` | The stacked LSTM (report §3.5) |
| `ml/training.py` | Train, evaluate, write artifacts |
| `ml/engine.py` | Loads model + tokenizer, validates they match, next-word and auto-complete inference |
| `services/history.py` | SQLite store; saves nothing unless the user opted in |

### Inference flow (report §3.9, §4.11–4.12)

1. Validate request (length, mode, `top_k`).
2. Clean text with the same function used in training; tokenise; keep the last 10 tokens; left-pad with 0.
3. `model(x)` → softmax over the vocabulary.
4. **Next-word:** mask padding (index 0) and `<OOV>` (index 1) and anything ≥ vocabulary size; take top-k.
5. **Auto-complete:** split trailing partial word from the context; collect vocabulary words starting with the prefix; rank by LSTM probability for the context. Fallback ranking by corpus frequency is returned with a different `score_type`/`source`.

### Model–tokenizer consistency
At load the engine requires `model output size == tokenizer vocab size` and, when `metadata.json` exists, a matching tokenizer fingerprint. A mismatch yields `invalid_artifacts` rather than wrong predictions.

### Training details and deviations
See the README for the five documented deviations from the report. Tensor shapes were checked: input `(batch, 10)` → embedding `(10, 128)` → LSTM1 `(10, 256)` → LSTM2 `(256)` → Dense `(256)` → softmax `(vocab)`, with `vocab = min(10000, len(word_index) + 1)` (the report's formula) so the output layer always matches the tokenizer.

## Frontend (`frontend/src`)

- `pages/`: Prediction Studio, Model Performance, How It Works, Settings.
- `components/`: layout/navigation, model status badge, suggestion list, charts, alerts, privacy notice.
- `services/api.ts`: typed fetch client that turns HTTP/network failures into friendly errors; `hooks/` holds the debounce hook and model-status context.
- Live suggestions are debounced; Predict can always be pressed explicitly. Clicking a suggestion inserts it with correct spacing (replacing the partial word in auto-complete mode).

## Privacy design
No third-party calls. Text is held in memory for the request only. The `history` table is written only after the user enables history in Settings, and can be cleared from the UI or `DELETE /api/history`. Model files and application records live in separate locations.
