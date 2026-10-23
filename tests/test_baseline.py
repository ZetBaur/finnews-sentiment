import json

import pandas as pd
import pytest

from finnews.baseline import build_model, evaluate, run_baseline, split_data

LABELS = {"negative": 0, "neutral": 1, "positive": 2}

CONFIG = {
    "labels": LABELS,
    "min_text_length": 3,
    "test_size": 0.2,
    "random_state": 42,
    "max_iter": 1000,
}


def make_df():
    rows = []
    for i in range(20):
        rows.append((f"profit rose strongly w{3 * i:02d}x", 2))
        rows.append((f"net loss widened sharply w{3 * i + 1:02d}x", 0))
        rows.append((f"company is located in city w{3 * i + 2:02d}x", 1))
    return pd.DataFrame(rows, columns=["text", "label"])


def test_split_sizes_and_no_overlap():
    df = make_df()
    x_train, x_test, _, _ = split_data(df, CONFIG)
    assert len(x_test) == 12
    assert len(x_train) == 48
    assert set(x_train).isdisjoint(set(x_test))


def test_split_is_stratified():
    df = make_df()
    _, _, _, y_test = split_data(df, CONFIG)
    assert y_test.value_counts().to_dict() == {0: 4, 1: 4, 2: 4}


def test_split_is_reproducible():
    df = make_df()
    first = split_data(df, CONFIG)[1]
    second = split_data(df, CONFIG)[1]
    assert list(first.index) == list(second.index)


def test_vectorizer_fitted_only_on_train():
    df = make_df()
    x_train, x_test, y_train, _ = split_data(df, CONFIG)
    model = build_model(CONFIG).fit(x_train, y_train)
    vocab = model.named_steps["tfidf"].vocabulary_
    test_only_words = {w for t in x_test for w in t.split()} - {
        w for t in x_train for w in t.split()
    }
    assert test_only_words
    assert test_only_words.isdisjoint(vocab)


def test_evaluate_returns_expected_keys():
    df = make_df()
    x_train, x_test, y_train, y_test = split_data(df, CONFIG)
    model = build_model(CONFIG).fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test, LABELS)
    assert set(metrics) >= {"accuracy", "macro_f1", "per_class", "confusion_matrix", "classes"}
    assert metrics["classes"] == ["negative", "neutral", "positive"]
    assert 0 <= metrics["accuracy"] <= 1
    assert len(metrics["confusion_matrix"]) == 3


def test_run_baseline_saves_files(tmp_path, monkeypatch):
    monkeypatch.setattr("finnews.baseline.load_data", lambda path: _raw_df())
    config = {
        **CONFIG,
        "data_path": "unused",
        "metrics_path": str(tmp_path / "metrics.json"),
        "model_path": str(tmp_path / "model.joblib"),
    }
    metrics = run_baseline(config)
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["accuracy"] == pytest.approx(metrics["accuracy"])
    assert (tmp_path / "model.joblib").exists()


def _raw_df():
    df = make_df()
    inverse = {v: k for k, v in LABELS.items()}
    return pd.DataFrame({"text": df["text"], "sentiment": df["label"].map(inverse)})
