# MozhiPredict AI

Next-word prediction and auto-complete for Tamil and English text with a stacked LSTM.
This is the web application for the mini-project report *"Next-Word Prediction and
Auto-Complete Engine for Tamil/English Text Using LSTM"* (Deep Learning, Assignment 02).

- **Frontend:** React + Vite + TypeScript + Tailwind CSS + Recharts + Lucide
- **Backend:** FastAPI + Pydantic + SQLite (opt-in history only)
- **Model:** Keras/TensorFlow — Embedding(128) → LSTM(256, return_sequences) → Dropout(0.3) → LSTM(256) → Dropout(0.3) → Dense(256, ReLU) → Dense(vocab, softmax), as in report §3.5

## Honest status

| Item | State |
|---|---|
| Application (4 pages, API, history, tests) | Implemented; backend 64 tests and frontend 27 tests pass |
| Training pipeline following the report | Implemented, **not run on a real corpus here** — no Tamil/English corpus was supplied |
| Trained report model | **Not included.** Real LSTM predictions need artifacts you train (or copy from your Colab run) |
| `artifacts/dev_sample/` | A *tiny development model* trained on 153 illustrative sentences. It is labelled in the UI and is **not** the report's model; its suggestions are weak |
| Metrics shown | Read from `history.json` / `evaluation.json` on disk. The report's numbers appear only in a separate card labelled *report-reported baseline* |

Without any model the app still runs: the status badge says so, Next-Word returns a clear
"model unavailable" message, and Auto-Complete can use a **labelled** vocabulary-frequency
fallback (never presented as an LSTM result).

## Quick start

Requirements: Python 3.10+ (3.13 tested), Node 20+.

```bash
# 1. Backend
cd backend
python -m venv ../.venv && source ../.venv/bin/activate      # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt                              # or tensorflow-cpu for a smaller install
cp .env.example .env

# Try it with the tiny development model:
MOZHI_ARTIFACT_DIR=../artifacts/dev_sample uvicorn app.main:app --port 8000
# (Windows PowerShell: $env:MOZHI_ARTIFACT_DIR="../artifacts/dev_sample"; uvicorn app.main:app --port 8000)

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                                                  # http://localhost:5173 (proxies /api to :8000)
```

Check the backend: `curl localhost:8000/api/health`. Interactive API docs: <http://localhost:8000/docs>.

## Train the real model

Put one-sentence-per-line UTF-8 files in `data/` (see `data/README.md` for sourcing and
licensing), then:

```bash
cd backend
python -m scripts.train_model --corpus ../data/english.txt ../data/tamil.txt
MOZHI_ARTIFACT_DIR=../artifacts uvicorn app.main:app --port 8000
```

This uses the report's configuration (10,000-word vocabulary, 10-word context, 80:20 split,
batch 128, ≤20 epochs, Adam 0.001, early stopping with best-weight restore) and writes the model,
tokenizer, history, evaluation and metadata into `artifacts/`. Already trained in Colab?
See `docs/setup-guide.md` for converting your `tokenizer.pkl`.

## Deviations from the report (all deliberate and documented)

1. **Sentence-level train/validation split.** The report splits shuffled n-gram pairs, so
   prefixes of the same sentence land on both sides (leakage) and inflate validation scores.
   The default here splits by sentence. `--split-mode pairs` reproduces the report's method.
2. **Tokenizer fitted on training sentences only**, so validation words cannot enter the vocabulary.
3. **Tokenizer saved as JSON, not pickle.** The app never unpickles files; `scripts/convert_tokenizer.py`
   converts your own pickle once.
4. **Duplicate sentences removed** by default (`--no-dedupe` restores the report behaviour).
5. **"Probability", not "confidence".** Scores are raw softmax probabilities over the whole
   vocabulary, not renormalised after filtering, so they do not sum to 100 %.

The report's own results also contain an inconsistency worth knowing before you present it:
the training log (§5.5, 6 epochs, val accuracy ≈ 22 %) disagrees with the curves in Figs 5.2–5.4
(20 epochs, ≈ 57 % accuracy, loss ≈ 1.1–1.3). The app's baseline card uses the logged
epochs (§5.5) and evaluation table (§5.9), and about 20 % of validation targets are `<OOV>`, which inflates headline accuracy.

## Tests

```bash
cd backend && python -m pytest          # preprocessing, Tamil Unicode, tokenizer, padding, API, missing-model
cd frontend && npm test                 # loading, API errors, suggestion insertion, metrics states
cd frontend && npm run build            # typecheck + production build
```

## Privacy

Typed text is processed by your own backend only and is never sent to a third-party service.
It is **not stored** unless you switch on history in Settings (off by default); history can be
cleared at any time. Suggestions reflect the training corpus and may be biased, repetitive or
inappropriate.

## Layout

```
frontend/   React app (pages, components, hooks, services, types)
backend/    FastAPI app (api, core, ml, schemas, services), scripts/, tests/
data/       your corpus (git-ignored) + illustrative sample
artifacts/  trained model files (git-ignored)
docs/       architecture, setup guide, API reference
```

## Limitations

Word-level vocabulary of 10,000 words (rare and many inflected Tamil forms become `<OOV>`),
10-word context window, no sub-word modelling, no Romanised-Tamil support, auto-complete only
suggests words already in the vocabulary. See report §6–7 for future scope.
