import pytest

from finnews.improve import build_variant
from finnews.predict import load_model, predict

CONFIG = {
    "max_iter": 1000,
    "random_state": 42,
    "min_text_length": 3,
    "labels": {"negative": 0, "neutral": 1, "positive": 2},
}
PARAMS = {"model": "linearsvc", "C": 1, "ngram_range": [1, 1]}
TEXTS = [
    "profit rose sharply",
    "profit increased strongly",
    "sales fell and loss widened",
    "loss grew and sales dropped",
    "the meeting is on monday",
    "the office is in helsinki",
]
LABELS = [2, 2, 0, 0, 1, 1]


@pytest.fixture
def model():
    return build_variant(PARAMS, CONFIG).fit(TEXTS, LABELS)


def test_predict_returns_text_and_label(model):
    result = predict(["profit rose sharply"], model=model, config=CONFIG)
    assert result == [{"text": "profit rose sharply", "label": "positive"}]


def test_predict_keeps_input_order(model):
    result = predict(["loss grew and sales dropped", "the office is in helsinki"], model=model, config=CONFIG)
    assert [r["label"] for r in result] == ["negative", "neutral"]


def test_predict_cleans_text_like_training(model):
    messy = predict(["  PROFIT   Rose &amp; sharply  "], model=model, config=CONFIG)
    clean = predict(["profit rose & sharply"], model=model, config=CONFIG)
    assert messy[0]["label"] == clean[0]["label"]


@pytest.mark.parametrize("bad", ["", "   ", "ab"])
def test_predict_rejects_empty_text(model, bad):
    with pytest.raises(ValueError):
        predict([bad], model=model, config=CONFIG)


def test_saved_model_gives_same_predictions(model, tmp_path):
    import joblib

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = load_model(path)
    assert list(loaded.predict(TEXTS)) == list(model.predict(TEXTS))
