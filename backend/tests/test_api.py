"""API behaviour with a real (tiny) trained LSTM and with missing artifacts."""
from __future__ import annotations

import json
import shutil

from tests.conftest import make_client

TAMIL_PROMPT = "நான் "


# ----------------------------------------------------------------------- health
def test_health_ready(client_ready):
    r = client_ready.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["ready"] is True
    assert body["model"]["state"] == "ready"
    assert body["model"]["development_model"] is True  # trained from the sample corpus


def test_health_without_model_still_ok_but_not_ready(client_missing):
    r = client_missing.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["ready"] is False
    assert body["model"]["state"] == "missing_artifacts"
    assert "train" in body["model"]["message"].lower()


# ------------------------------------------------------------- next-word mode
def test_next_word_response_format(client_ready):
    r = client_ready.post("/api/predict", json={"text": "i want to ", "mode": "next_word", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["input_text"] == "i want to "
    assert body["mode"] == "next_word"
    assert body["model_status"]["ready"] is True
    assert body["source"] == "lstm"
    sugg = body["suggestions"]
    assert len(sugg) == 5
    assert [s["rank"] for s in sugg] == [1, 2, 3, 4, 5]
    for s in sugg:
        assert set(s) == {"word", "rank", "score", "score_type", "source"}
        assert s["score_type"] == "model_probability"
        assert s["source"] == "lstm"
        assert 0.0 <= s["score"] <= 1.0
        assert s["word"] not in ("<OOV>", "")
    scores = [s["score"] for s in sugg]
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) < 1.0  # whole-vocabulary probabilities are not renormalised
    assert len({s["word"] for s in sugg}) == 5


def test_top_k_is_respected(client_ready):
    for k in (3, 10):
        r = client_ready.post("/api/predict", json={"text": "the weather is", "mode": "next_word", "top_k": k})
        assert len(r.json()["suggestions"]) == k


def test_tamil_unicode_preserved_end_to_end(client_ready):
    text = "நான் பள்ளிக்கு "
    r = client_ready.post("/api/predict", json={"text": text, "mode": "next_word", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["input_text"] == text
    assert body["suggestions"], "expected suggestions for Tamil context"
    assert any("஀" <= ch <= "௿" for s in body["suggestions"] for ch in s["word"])


def test_mixed_script_input_is_accepted(client_ready):
    r = client_ready.post("/api/predict", json={"text": "இன்று நான் office ", "mode": "next_word", "top_k": 3})
    assert r.status_code == 200
    assert len(r.json()["suggestions"]) == 3


def test_unknown_words_are_reported_not_crashing(client_ready):
    r = client_ready.post("/api/predict", json={"text": "zzzzz qqqqq xxxxx ", "mode": "next_word", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["diagnostics"]["unknown_context_words"] == 3
    assert "vocabulary" in body["message"]


def test_long_input_is_truncated_to_model_context(client_ready):
    text = "i want to go home " * 50
    r = client_ready.post("/api/predict", json={"text": text, "mode": "next_word", "top_k": 3})
    assert r.status_code == 200
    assert r.json()["diagnostics"]["context_truncated"] is True


def test_text_with_only_symbols_returns_message_not_error(client_ready):
    r = client_ready.post("/api/predict", json={"text": "123 !!! ", "mode": "next_word", "top_k": 3})
    assert r.status_code == 200
    assert r.json()["suggestions"] == []
    assert "No usable words" in r.json()["message"]


def test_single_word_input_works(client_ready):
    r = client_ready.post("/api/predict", json={"text": "i ", "mode": "next_word", "top_k": 3})
    assert r.status_code == 200 and len(r.json()["suggestions"]) == 3


# --------------------------------------------------------------- auto-complete
def test_autocomplete_returns_only_prefix_matches_ranked_by_model(client_ready):
    r = client_ready.post("/api/predict", json={"text": "i want to g", "mode": "autocomplete", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["prefix"] == "g"
    assert body["suggestions"]
    assert all(s["word"].startswith("g") for s in body["suggestions"])
    assert all(s["source"] == "lstm" and s["score_type"] == "model_probability" for s in body["suggestions"])
    scores = [s["score"] for s in body["suggestions"]]
    assert scores == sorted(scores, reverse=True)
    assert "go" in [s["word"] for s in body["suggestions"]]


def test_autocomplete_tamil_prefix(client_ready):
    r = client_ready.post("/api/predict", json={"text": "நான் ப", "mode": "autocomplete", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["prefix"] == "ப"
    assert body["suggestions"]
    assert all(s["word"].startswith("ப") for s in body["suggestions"])


def test_autocomplete_without_match_returns_message_not_invention(client_ready):
    r = client_ready.post("/api/predict", json={"text": "i want to qzx", "mode": "autocomplete", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["suggestions"] == []
    assert "No vocabulary words start with" in body["message"]


def test_autocomplete_when_text_ends_with_space_asks_for_a_partial_word(client_ready):
    r = client_ready.post("/api/predict", json={"text": "i want to ", "mode": "autocomplete", "top_k": 5})
    assert r.status_code == 200
    assert r.json()["suggestions"] == []
    assert "partial word" in r.json()["message"].lower() or "part of a word" in r.json()["message"].lower()


def test_autocomplete_with_digits_in_last_word_is_explained(client_ready):
    r = client_ready.post("/api/predict", json={"text": "i want to g0", "mode": "autocomplete", "top_k": 5})
    assert r.status_code == 200
    assert r.json()["suggestions"] == []
    assert "digits or punctuation" in r.json()["message"]


def test_autocomplete_single_word_uses_labelled_frequency_fallback(client_ready):
    """No preceding context -> the LSTM has nothing to condition on."""
    r = client_ready.post("/api/predict", json={"text": "th", "mode": "autocomplete", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["suggestions"]
    assert all(s["source"] == "vocabulary_fallback" for s in body["suggestions"])
    assert all(s["score_type"] == "vocabulary_frequency" for s in body["suggestions"])
    assert "not an LSTM prediction" in body["message"]


# ------------------------------------------------------------ input validation
def test_empty_text_is_rejected_with_clear_message(client_ready):
    for text in ("", "   \n "):
        r = client_ready.post("/api/predict", json={"text": text, "mode": "next_word", "top_k": 5})
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["code"] == "invalid_request"
        assert "empty" in detail["message"].lower()


def test_malformed_requests_are_rejected(client_ready):
    bad_bodies = [
        {"mode": "next_word"},
        {"text": "hello ", "mode": "sing"},
        {"text": "hello ", "mode": "next_word", "top_k": 0},
        {"text": "hello ", "mode": "next_word", "top_k": 99},
        {"text": 42, "mode": "next_word"},
    ]
    for body in bad_bodies:
        r = client_ready.post("/api/predict", json=body)
        assert r.status_code == 422, body
        assert r.json()["detail"]["code"] == "invalid_request"
    r = client_ready.post("/api/predict", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_oversized_text_is_rejected(client_ready):
    r = client_ready.post("/api/predict", json={"text": "a " * 1500, "mode": "next_word"})
    assert r.status_code == 422  # > 2000 characters
    r = client_ready.post("/api/predict", json={"text": "a" * 40000, "mode": "next_word"})
    assert r.status_code == 413  # rejected before parsing


def test_validation_errors_do_not_echo_input(client_ready):
    secret = "my-private-text-" + "x" * 3000
    r = client_ready.post("/api/predict", json={"text": secret, "mode": "next_word"})
    assert "my-private-text" not in r.text


# ----------------------------------------------------------- missing artifacts
def test_predict_without_model_returns_503_with_helpful_message(client_missing):
    r = client_missing.post("/api/predict", json={"text": "i want to ", "mode": "next_word", "top_k": 3})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == "model_unavailable"
    assert "train" in detail["message"].lower()


def test_autocomplete_without_anything_returns_503(client_missing):
    r = client_missing.post("/api/predict", json={"text": "i want to g", "mode": "autocomplete", "top_k": 3})
    assert r.status_code == 503


def test_next_word_without_lstm_never_uses_fallback(client_tokenizer_only):
    r = client_tokenizer_only.post("/api/predict", json={"text": "i want to ", "mode": "next_word", "top_k": 3})
    assert r.status_code == 503


def test_autocomplete_fallback_is_labelled_when_only_tokenizer_exists(client_tokenizer_only):
    r = client_tokenizer_only.post("/api/predict", json={"text": "i want to g", "mode": "autocomplete", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["model_status"]["ready"] is False
    assert body["model_status"]["fallback_available"] is True
    assert body["suggestions"]
    for s in body["suggestions"]:
        assert s["source"] == "vocabulary_fallback"
        assert s["score_type"] == "vocabulary_frequency"
    assert "not an LSTM prediction" in body["message"]


def test_fallback_can_be_disabled(trained_dir, tmp_path):
    only = tmp_path / "tok"
    only.mkdir()
    shutil.copy(trained_dir / "tokenizer.json", only / "tokenizer.json")
    with make_client(only, tmp_path, allow_vocabulary_fallback=False) as client:
        r = client.post("/api/predict", json={"text": "i want to g", "mode": "autocomplete", "top_k": 3})
        assert r.status_code == 503


def test_invalid_tokenizer_file_is_reported_not_fatal(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "tokenizer.json").write_text("{ not json", encoding="utf-8")
    with make_client(bad, tmp_path) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["model"]["state"] == "invalid_artifacts"


def test_mismatched_model_and_tokenizer_are_rejected(trained_dir, tmp_path):
    mismatch = tmp_path / "mismatch"
    shutil.copytree(trained_dir, mismatch)
    tok = json.loads((mismatch / "tokenizer.json").read_text(encoding="utf-8"))
    tok["word_index"]["extra_word_from_other_run"] = max(tok["word_index"].values()) + 1
    (mismatch / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    with make_client(mismatch, tmp_path) as client:
        body = client.get("/api/health").json()
        assert body["model"]["ready"] is False
        assert body["model"]["state"] == "invalid_artifacts"


def test_corrupt_model_file_is_reported_not_fatal(trained_dir, tmp_path):
    broken = tmp_path / "broken"
    shutil.copytree(trained_dir, broken)
    (broken / "next_word_lstm_model.keras").write_bytes(b"this is not a keras model")
    with make_client(broken, tmp_path) as client:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["model"]["state"] == "load_failed"


def test_errors_do_not_leak_filesystem_paths(client_missing, tmp_path):
    for response in (
        client_missing.get("/api/health"),
        client_missing.get("/api/model-info"),
        client_missing.post("/api/predict", json={"text": "hello world ", "mode": "next_word"}),
    ):
        assert str(tmp_path) not in response.text
        assert "Traceback" not in response.text


# -------------------------------------------------------------- model-info etc
def test_model_info_reports_real_architecture(client_ready):
    body = client_ready.get("/api/model-info").json()
    arch = body["architecture"]
    assert arch["sequence_length"] == 10
    assert arch["vocab_size"] == body["tokenizer"]["vocab_size"]
    types = [layer["type"] for layer in arch["layers"]]
    assert types == ["Embedding", "LSTM", "Dropout", "LSTM", "Dropout", "Dense", "Dense"]
    assert body["config"]["batch_size"] == 32
    assert body["run"]["mode"] == "development_sample"
    assert body["dataset"]["tamil_sentences"] > 0


def test_model_info_without_model(client_missing):
    body = client_missing.get("/api/model-info").json()
    assert body["architecture"] is None and body["tokenizer"] is None
    assert body["status"]["ready"] is False


def test_metrics_come_from_artifacts_and_baseline_is_labelled(client_ready):
    body = client_ready.get("/api/metrics").json()
    assert body["available"] is True
    assert body["development_model"] is True
    assert set(body["history"]) >= {"loss", "val_loss", "accuracy", "val_accuracy", "top5_accuracy"}
    assert len(body["history"]["loss"]) >= 1
    ev = body["evaluation"]
    assert ev["perplexity"] > 1 and 0 <= ev["accuracy"] <= 1
    assert "all" in ev["known_word"]
    assert "Report-reported" in body["report_baseline"]["label"]
    # measured values are not the report's values
    assert ev["loss"] != body["report_baseline"]["evaluation"]["loss"]


def test_metrics_not_available_state(client_missing):
    body = client_missing.get("/api/metrics").json()
    assert body["available"] is False
    assert body["history"] is None and body["evaluation"] is None
    assert "Metrics not available" in body["message"]
    assert body["report_baseline"]["evaluation"]["perplexity"] == 492.78


# --------------------------------------------------------------------- history
def test_history_is_off_by_default_and_nothing_is_stored(client_ready):
    client_ready.post("/api/predict", json={"text": "i want to ", "mode": "next_word", "top_k": 3})
    body = client_ready.get("/api/history").json()
    assert body == {"enabled": False, "items": []}


def test_history_opt_in_store_and_clear(client_ready):
    assert client_ready.put("/api/settings", json={"history_enabled": True}).json() == {"history_enabled": True}
    client_ready.post("/api/predict", json={"text": TAMIL_PROMPT, "mode": "next_word", "top_k": 3})
    client_ready.post("/api/predict", json={"text": "i want to ", "mode": "next_word", "top_k": 3})
    body = client_ready.get("/api/history").json()
    assert body["enabled"] is True
    assert [i["text"] for i in body["items"]] == ["i want to ", TAMIL_PROMPT]  # newest first, Tamil intact
    assert len(body["items"][0]["suggestions"]) == 3

    deleted = client_ready.delete("/api/history").json()
    assert deleted == {"deleted": 2}
    assert client_ready.get("/api/history").json()["items"] == []

    client_ready.put("/api/settings", json={"history_enabled": False})
    client_ready.post("/api/predict", json={"text": "i want to ", "mode": "next_word", "top_k": 3})
    assert client_ready.get("/api/history").json()["items"] == []


def test_cors_allows_vite_dev_server(client_ready):
    r = client_ready.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = client_ready.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
