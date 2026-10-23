import pandas as pd
import pytest

from finnews.preprocess import clean_text, preprocess

CONFIG = {
    "min_text_length": 3,
    "labels": {"negative": 0, "neutral": 1, "positive": 2},
}


def make_df(rows):
    return pd.DataFrame(rows, columns=["text", "sentiment"])


def test_clean_text_lowercases_and_strips():
    assert clean_text("  Profit ROSE  ") == "profit rose"


def test_clean_text_collapses_spaces_and_nbsp():
    assert clean_text("net sales   grew") == "net sales grew"


def test_clean_text_removes_urls():
    assert clean_text("see http://example.com/a now") == "see now"
    assert clean_text("see www.example.com now") == "see now"


def test_clean_text_unescapes_html():
    assert clean_text("A &amp; B") == "a & b"


def test_clean_text_keeps_digits_and_percent():
    assert clean_text("Sales rose 10 % to EUR 5.2 m") == "sales rose 10 % to eur 5.2 m"


@pytest.mark.parametrize("value", [None, float("nan"), 5])
def test_clean_text_non_string_gives_empty(value):
    assert clean_text(value) == ""


def test_clean_text_is_idempotent():
    once = clean_text("  Hello &amp;  WORLD http://x.io ")
    assert clean_text(once) == once


def test_preprocess_maps_labels_to_numbers():
    df = make_df(
        [("good news here", "positive"), ("bad news here", "negative"), ("plain news", "neutral")]
    )
    result = preprocess(df, CONFIG)
    assert dict(zip(result["text"], result["label"])) == {
        "good news here": 2,
        "bad news here": 0,
        "plain news": 1,
    }


def test_preprocess_drops_empty_short_and_unknown():
    df = make_df(
        [("", "neutral"), ("ab", "neutral"), ("fine text", "unknown"), ("fine text", "neutral")]
    )
    result = preprocess(df, CONFIG)
    assert list(result["text"]) == ["fine text"]
    assert list(result["label"]) == [1]


def test_preprocess_drops_duplicates():
    df = make_df([("same text", "neutral"), ("Same  Text", "neutral")])
    assert len(preprocess(df, CONFIG)) == 1


def test_preprocess_drops_conflicting_labels():
    df = make_df([("same text", "neutral"), ("same text", "positive"), ("other text", "neutral")])
    result = preprocess(df, CONFIG)
    assert list(result["text"]) == ["other text"]


def test_preprocess_does_not_change_input():
    df = make_df([("Good News", "positive")])
    preprocess(df, CONFIG)
    assert df["text"][0] == "Good News"
    assert "label" not in df.columns


def test_preprocess_is_deterministic():
    df = make_df([("Good News", "positive"), ("bad news", "negative")])
    first = preprocess(df, CONFIG)
    second = preprocess(df, CONFIG)
    pd.testing.assert_frame_equal(first, second)
