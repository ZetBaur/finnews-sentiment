import json
import logging
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from finnews.baseline import evaluate, split_data
from finnews.config import load_config
from finnews.load_data import load_data
from finnews.log import setup_logging
from finnews.preprocess import preprocess

log = logging.getLogger(__name__)


def build_classifier(params: dict, config: dict):
    if params["model"] == "logreg":
        return LogisticRegression(C=params["C"], max_iter=config["max_iter"])
    if params["model"] == "linearsvc":
        return LinearSVC(C=params["C"], random_state=config["random_state"])
    raise ValueError(f"unknown model: {params['model']}")


def build_variant(params: dict, config: dict) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=tuple(params["ngram_range"]))),
            ("clf", build_classifier(params, config)),
        ]
    )


def run_experiments(config: dict | None = None) -> dict:
    config = config or load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)
    folds = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"]
    )

    results = {}
    for name, params in config["experiments"].items():
        model = build_variant(params, config)
        cv_scores = cross_val_score(model, x_train, y_train, cv=folds, scoring="f1_macro")
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_test, y_test, config["labels"])
        results[name] = {
            "params": params,
            "cv_macro_f1_mean": float(cv_scores.mean()),
            "cv_macro_f1_std": float(cv_scores.std()),
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "n_features": len(model.named_steps["tfidf"].vocabulary_),
        }
        log.info(
            "%s: CV F1 %.3f ± %.3f | test accuracy %.3f, F1 %.3f | features %d",
            name,
            results[name]["cv_macro_f1_mean"],
            results[name]["cv_macro_f1_std"],
            results[name]["test_accuracy"],
            results[name]["test_macro_f1"],
            results[name]["n_features"],
        )

    path = Path(config["improve_metrics_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved %s", path)
    return results


if __name__ == "__main__":
    setup_logging()
    run_experiments()
