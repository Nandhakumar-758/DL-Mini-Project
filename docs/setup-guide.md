# Setup guide

## 1. Backend

```bash
cd backend
python -m venv ../.venv
source ../.venv/bin/activate          # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt       # tensorflow-cpu is enough without a GPU
cp .env.example .env                  # optional; every variable has a default
uvicorn app.main:app --port 8000 --reload
```

Configuration (environment variables or `backend/.env`, prefix `MOZHI_`):

| Variable | Default | Meaning |
|---|---|---|
| `MOZHI_ARTIFACT_DIR` | `../artifacts` | Folder with model, tokenizer and JSON metadata |
| `MOZHI_MODEL_FILENAME` | `next_word_lstm_model.keras` | |
| `MOZHI_TOKENIZER_FILENAME` | `tokenizer.json` | |
| `MOZHI_DB_PATH` | `backend/var/mozhipredict.db` | SQLite file (history + settings only) |
| `MOZHI_CORS_ORIGINS` | Vite dev origins | JSON list of allowed origins |
| `MOZHI_ALLOW_VOCABULARY_FALLBACK` | `true` | Labelled non-LSTM auto-complete when no model is loaded |

If `artifacts/` has no model, the server still starts and reports `missing_artifacts`.

## 2. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, /api is proxied to http://localhost:8000
```

Different backend address: `VITE_BACKEND_URL=http://host:port npm run dev`.
Production bundle: `npm run build` (output in `frontend/dist`).

## 3. Get a model

### A. Development model (instant, weak)

```bash
cd backend
python -m scripts.train_model --dev          # → ../artifacts/dev_sample (153 sample sentences)
MOZHI_ARTIFACT_DIR=../artifacts/dev_sample uvicorn app.main:app --port 8000
```

This only proves the plumbing works. It is flagged as a development model everywhere.

### B. Train the report's model

1. Collect English and Tamil text (see `data/README.md`), one sentence per line, UTF-8:
   `data/english.txt`, `data/tamil.txt`.
2. Train (a GPU or Colab is recommended; the full model has ≈4.8 M parameters):
   ```bash
   cd backend
   python -m scripts.train_model --corpus ../data/english.txt ../data/tamil.txt
   ```
   Defaults match the report: vocabulary 10,000, context 10 words, embedding 128, LSTM 256+256,
   dropout 0.3, Dense 256 + softmax, Adam 0.001, sparse categorical cross-entropy, batch 128,
   ≤ 20 epochs, early stopping (patience 3, best weights restored), accuracy + top-5 accuracy.
3. Start the backend with `MOZHI_ARTIFACT_DIR=../artifacts`.

Useful flags: `--epochs`, `--batch-size`, `--seed`, `--min-words`, `--no-dedupe`,
`--split-mode pairs` (report-compatible but leaky; see README), `--output-dir`.

### C. You already trained in the report's Colab notebook

1. Download `next_word_lstm_model.keras` and `tokenizer.pkl` from Colab.
2. Convert the pickle once (only do this for a file you created yourself — unpickling runs code):
   ```bash
   cd backend
   python -m scripts.convert_tokenizer /path/to/tokenizer.pkl --out ../artifacts/tokenizer.json
   ```
3. Copy the `.keras` file into `artifacts/`.
4. Generate metrics against your corpus:
   ```bash
   python -m scripts.evaluate_model --corpus ../data/english.txt ../data/tamil.txt
   ```
   The report trained on a random pair split, so some "validation" sentences were seen in training and
   these scores will be optimistic. The model and tokenizer must come from the same notebook run; the
   backend rejects pairs whose vocabulary size differs.

## 4. Tests

```bash
cd backend && python -m pytest
cd frontend && npm test && npm run build
```

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Badge says "No trained model" | Nothing in `MOZHI_ARTIFACT_DIR`; use option A, B or C |
| "model or tokenizer file is invalid or do not belong together" | Output size ≠ tokenizer vocabulary, or fingerprint mismatch; retrain or re-convert from the same run |
| "TensorFlow is not installed" | `pip install tensorflow-cpu` |
| Tamil shows as boxes | Install a Tamil-capable system font (Noto Sans Tamil) |
| Browser says backend unreachable | Start uvicorn on port 8000 or set `VITE_BACKEND_URL` |
