from text_classifier.text import clean_text


def test_clean_text_normalizes_email_url_and_space():
    value = clean_text("  Contact a@b.com\n at https://example.com  ")
    assert value == "Contact EMAIL at URL"


def test_clean_text_handles_none():
    assert clean_text(None) == ""
