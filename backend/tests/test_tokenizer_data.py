import json

import numpy as np
import pytest

from app.ml.data import (
    CorpusError,
    build_sequences,
    load_corpus,
    pad_context,
    prepare_sentences,
    split_sentences,
)
from app.ml.tokenizer import TokenizerError, WordTokenizer

SENTENCES = ["i love deep learning", "i love music", "நான் பள்ளிக்கு செல்கிறேன்"]


def test_fit_reserves_padding_and_oov_indices():
    tok = WordTokenizer.fit(SENTENCES, num_words=100)
    assert tok.oov_index == 1
    assert 0 not in tok.index_word
    assert tok.word_index["i"] == 2  # most frequent real word (ties keep first-seen order)
    assert tok.vocab_size == len(tok.word_index) + 1


def test_unknown_words_become_oov_and_num_words_limits_vocabulary():
    tok = WordTokenizer.fit(SENTENCES, num_words=4)  # only indices 1..3 are allowed
    ids = tok.texts_to_sequence("i love deep learning")
    assert ids[0] == tok.word_index["i"]
    assert all(i < 4 for i in ids)
    assert ids[-1] == tok.oov_index
    assert tok.vocab_size == 4


def test_save_and_load_roundtrip_preserves_tamil(tmp_path):
    tok = WordTokenizer.fit(SENTENCES, num_words=100)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = WordTokenizer.load(path)
    assert loaded.word_index == tok.word_index
    assert "பள்ளிக்கு" in loaded.word_index
    assert loaded.fingerprint == tok.fingerprint
    assert loaded.texts_to_sequence("நான் பள்ளிக்கு") == tok.texts_to_sequence("நான் பள்ளிக்கு")


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", "{}", json.dumps({"word_index": {"a": 0}, "num_words": 10}),
     json.dumps({"word_index": {"a": 1, "b": 1}, "num_words": 10})],
)
def test_invalid_tokenizer_files_are_rejected(tmp_path, content):
    path = tmp_path / "t.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(TokenizerError):
        WordTokenizer.load(path)


def test_missing_tokenizer_file_is_rejected(tmp_path):
    with pytest.raises(TokenizerError):
        WordTokenizer.load(tmp_path / "nope.json")


def test_prefix_matches_excludes_special_tokens_and_supports_tamil():
    tok = WordTokenizer.fit(["good game gone", "good morning", "நிலா நிறம் நிலம்"], num_words=100)
    assert [w for w, _ in tok.prefix_matches("go")] == ["good", "gone"]
    assert {w for w, _ in tok.prefix_matches("நி")} == {"நிலா", "நிறம்", "நிலம்"}
    assert tok.prefix_matches("<") == []  # the OOV token is never a candidate
    assert tok.prefix_matches("zzz") == []


def test_equivalent_to_keras_tokenizer():
    pytest.importorskip("tensorflow")
    from tensorflow.keras.preprocessing.text import Tokenizer

    lines = SENTENCES + ["the cat sat on the mat", "the dog sat", "நான் நான் வீடு"]
    keras_tok = Tokenizer(num_words=6, oov_token="<OOV>", filters="")
    keras_tok.fit_on_texts(lines)
    ours = WordTokenizer.fit(lines, num_words=6)
    assert ours.word_index == dict(keras_tok.word_index)
    for line in lines + ["unseen words here the"]:
        assert ours.texts_to_sequence(line) == keras_tok.texts_to_sequences([line])[0]
    assert WordTokenizer.from_keras(keras_tok).word_index == ours.word_index


def test_sequences_match_report_example():
    tok = WordTokenizer.fit(["i love deep learning"], num_words=100)
    X, y, owners = build_sequences(["i love deep learning"], tok, max_len=11)
    assert X.shape == (3, 10)
    ids = tok.texts_to_sequence("i love deep learning")
    assert list(y) == ids[1:]  # targets: love, deep, learning
    assert list(X[0]) == [0] * 9 + [ids[0]]  # padded at the beginning ("pre")
    assert list(X[2]) == [0] * 7 + ids[:3]
    assert set(owners) == {0}


def test_long_sentences_use_a_sliding_window_of_ten_context_words():
    words = [f"w{i}" for i in range(15)]
    tok = WordTokenizer.fit([" ".join(words)], num_words=100)
    X, y, _ = build_sequences([" ".join(words)], tok, max_len=11)
    ids = tok.texts_to_sequence(" ".join(words))
    assert X.shape == (14, 10)
    assert list(X[-1]) == ids[4:14]
    assert y[-1] == ids[14]


def test_pad_context_keeps_last_tokens_and_pads_left():
    assert pad_context([5, 6], 4).tolist() == [[0, 0, 5, 6]]
    assert pad_context(list(range(1, 13)), 10).tolist() == [list(range(3, 13))]
    assert pad_context([], 3).tolist() == [[0, 0, 0]]
    assert pad_context([1], 3).dtype == np.int32


def test_prepare_sentences_cleans_filters_and_dedupes():
    raw = ["Hello World again!", "hello world again", "hi", "நான்  வீடு செல்கிறேன்"]
    sentences, dups = prepare_sentences(raw, min_words=3)
    assert sentences == ["hello world again", "நான் வீடு செல்கிறேன்"]
    assert dups == 1


def test_split_is_by_sentence_with_no_overlap_and_is_reproducible():
    sentences = [f"sentence number {c} here" for c in "abcdefghij"]
    train, val = split_sentences(sentences, 0.2, seed=1)
    assert len(val) == 2 and len(train) == 8
    assert not set(train) & set(val)
    assert split_sentences(sentences, 0.2, seed=1) == (train, val)


def test_split_needs_two_sentences():
    with pytest.raises(CorpusError):
        split_sentences(["only one sentence here"])


def test_load_corpus_errors(tmp_path):
    with pytest.raises(CorpusError):
        load_corpus([tmp_path / "missing.txt"])
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(CorpusError):
        load_corpus([bad])
    ok = tmp_path / "ok.txt"
    ok.write_text("நான் வீடு செல்கிறேன்\nhello world again\n", encoding="utf-8")
    assert load_corpus([ok]) == ["நான் வீடு செல்கிறேன்", "hello world again"]
