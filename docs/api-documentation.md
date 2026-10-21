# API reference

Base path `/api`. Interactive docs at `/docs` (Swagger) when the backend runs.
Errors use `{"detail": {"code": "...", "message": "..."}}`; filesystem paths and stack traces are never returned.

## GET /api/health
`200` always while the service is up. `model.ready` says whether the LSTM is loaded.
```json
{"status":"ok","model":{"state":"ready","ready":true,"lstm_available":true,
 "fallback_available":false,"development_model":false,"message":"Trained LSTM model loaded and ready."}}
```
`model.state`: `ready`, `missing_artifacts`, `invalid_artifacts`, `tensorflow_unavailable`, `load_failed`.

## GET /api/model-info
Architecture (layers, shapes, parameters), tokenizer summary, and config/dataset/training metadata from `metadata.json`. Fields are `null` when unavailable.

## POST /api/predict
Request:
```json
{"text": "i want to ", "mode": "next_word", "top_k": 5}
```
- `text`: 1–2000 characters, Tamil/English/mixed, not blank. Returned unchanged as `input_text`.
- `mode`: `next_word` or `autocomplete`.
- `top_k`: 1–10 (the UI offers 3, 5, 10).

Response:
```json
{"input_text":"i want to ge","mode":"autocomplete","prefix":"ge",
 "suggestions":[{"word":"get","rank":1,"score":0.0019,"score_type":"model_probability","source":"lstm"}],
 "source":"lstm","model_status":{...},"message":null,
 "diagnostics":{"context_words":3,"unknown_context_words":0,"context_truncated":false}}
```

**Scores**
- `model_probability`: the LSTM's softmax probability (0–1) over the *whole* vocabulary. It is **not** renormalised after filtering (padding/OOV removed, prefix filter), so returned scores do not sum to 1, and it is not a calibrated confidence.
- `vocabulary_frequency`: relative word frequency in the training corpus. `source` is `vocabulary_fallback`. This is a dictionary lookup, not an LSTM prediction, and the UI labels it accordingly.

**Behaviour**
- `next_word`: needs the LSTM; padding and `<OOV>` are never suggested.
- `autocomplete`: the last token is the prefix, earlier words are the LSTM context; vocabulary words with that prefix are ranked by LSTM probability. With no preceding word, or no model, the labelled fallback is used. No matching word → empty `suggestions` and an explanatory `message` (nothing is invented).
- Tamil is never lowercased or normalised. A word containing digits or punctuation cannot be auto-completed (the model ignores them) and says so.

**Status codes:** `200` ok (possibly with empty suggestions and a `message`); `413` body larger than 16 KB; `422` invalid request (`code: invalid_request` with per-field `errors`); `503` model unavailable (`code: model_unavailable`); `500` inference failed (`code: inference_failed`).

## GET /api/metrics
```json
{"available":true,"message":"...","development_model":false,
 "history":{"loss":[...],"val_loss":[...],"accuracy":[...],"val_accuracy":[...],"top5_accuracy":[...],"val_top5_accuracy":[...]},
 "evaluation":{"loss":..,"accuracy":..,"top5_accuracy":..,"perplexity":..,"oov_target_share":..,"known_word":{...}},
 "training":{...},"dataset":{...},"report_baseline":{...}}
```
`history`, `evaluation`, `training`, `dataset` are measured values read from the artifacts on disk. When absent: `available=false` with instructions. `report_baseline` is always a separate, labelled copy of numbers quoted from the report; it is never mixed into the measured fields.

## GET /api/history · DELETE /api/history
History is **off by default**. `GET` returns `{"enabled":bool,"items":[...]}`; items exist only if the user enabled saving. `DELETE` clears them and returns `{"deleted": n}`.

## GET /api/settings · PUT /api/settings
`{"history_enabled": true|false}`. Disabling does not delete saved items; use `DELETE /api/history`.
