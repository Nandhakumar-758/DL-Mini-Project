from app.core.text import clean_text, detect_script, ends_with_whitespace, split_trailing_token

TAMIL = "நான் பள்ளிக்கு செல்கிறேன்"


def test_english_is_lowercased_and_punctuation_removed():
    assert clean_text("I Want TO, go! 123 home.") == "i want to go home"


def test_extra_whitespace_collapsed():
    assert clean_text("  hello \n\t  world  ") == "hello world"


def test_tamil_unicode_is_preserved_exactly():
    assert clean_text(TAMIL) == TAMIL
    assert clean_text("நான், பள்ளிக்கு!") == "நான் பள்ளிக்கு"


def test_tamil_combining_marks_not_stripped():
    word = "தமிழ்நாட்டின்"  # contains pulli and vowel signs
    assert clean_text(word) == word
    assert len(clean_text(word)) == len(word)


def test_mixed_script_keeps_both_scripts():
    assert clean_text("இன்று நான் Office செல்கிறேன்") == "இன்று நான் office செல்கிறேன்"


def test_clean_is_idempotent():
    once = clean_text("Hello, உலகம்!!")
    assert clean_text(once) == once


def test_only_symbols_gives_empty_string():
    assert clean_text("123 !!! ???") == ""


def test_detect_script():
    assert detect_script("hello") == "english"
    assert detect_script(TAMIL) == "tamil"
    assert detect_script("நான் office") == "mixed"
    assert detect_script("1234") == "none"


def test_split_trailing_token():
    assert split_trailing_token("i want to ge") == ("i want to ", "ge")
    assert split_trailing_token("i want to ") == ("i want to ", "")
    assert split_trailing_token("நான் நி") == ("நான் ", "நி")
    assert split_trailing_token("word") == ("", "word")
    assert split_trailing_token("") == ("", "")


def test_ends_with_whitespace():
    assert ends_with_whitespace("a ")
    assert not ends_with_whitespace("a")
    assert not ends_with_whitespace("")
