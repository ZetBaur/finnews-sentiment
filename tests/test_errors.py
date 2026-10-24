import pandas as pd

from finnews.errors import count_error_types, find_errors
from finnews.improve import build_variant

LABELS = {"negative": 0, "neutral": 1, "positive": 2}
CONFIG = {"max_iter": 1000, "random_state": 42}
PARAMS = {"model": "linearsvc", "C": 1, "ngram_range": [1, 1]}

TEXTS = pd.Series(
    [
        "profit rose strongly",
        "profit rose a lot",
        "loss grew badly",
        "loss grew a lot",
        "meeting is on monday",
        "meeting is on friday",
    ]
)
LABEL_VALUES = pd.Series([2, 2, 0, 0, 1, 1])


def fitted_model():
    return build_variant(PARAMS, CONFIG).fit(TEXTS, LABEL_VALUES)


def test_find_errors_empty_when_all_correct():
    errors = find_errors(fitted_model(), TEXTS, LABEL_VALUES, LABELS)
    assert errors.empty
    assert list(errors.columns) == ["text", "true", "pred", "margin", "n_words"]


def test_find_errors_finds_wrong_label():
    wrong = LABEL_VALUES.copy()
    wrong.iloc[0] = 0
    errors = find_errors(fitted_model(), TEXTS, wrong, LABELS)
    assert len(errors) == 1
    assert errors.iloc[0]["true"] == "negative"
    assert errors.iloc[0]["pred"] == "positive"
    assert errors.iloc[0]["margin"] > 0


def test_count_error_types_sorted_descending():
    errors = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "true": ["negative", "negative", "positive"],
            "pred": ["neutral", "neutral", "neutral"],
            "margin": [0.1, 0.2, 0.3],
            "n_words": [1, 1, 1],
        }
    )
    counts = count_error_types(errors)
    assert counts.iloc[0]["true"] == "negative"
    assert counts.iloc[0]["count"] == 2
