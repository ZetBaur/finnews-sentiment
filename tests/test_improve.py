import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from finnews.improve import build_classifier, build_variant

CONFIG = {"max_iter": 1000, "random_state": 42}


def test_build_classifier_logreg():
    clf = build_classifier({"model": "logreg", "C": 10}, CONFIG)
    assert isinstance(clf, LogisticRegression)
    assert clf.C == 10


def test_build_classifier_linearsvc():
    clf = build_classifier({"model": "linearsvc", "C": 0.5}, CONFIG)
    assert isinstance(clf, LinearSVC)
    assert clf.C == 0.5


def test_build_classifier_unknown_model():
    with pytest.raises(ValueError):
        build_classifier({"model": "magic", "C": 1}, CONFIG)


def test_build_variant_uses_ngram_range():
    params = {"model": "logreg", "C": 1, "ngram_range": [1, 2]}
    pipeline = build_variant(params, CONFIG)
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)
