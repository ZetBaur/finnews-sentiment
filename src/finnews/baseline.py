import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from finnews.config import load_config
from finnews.load_data import load_data
from finnews.log import setup_logging
from finnews.preprocess import preprocess

log = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, config: dict):
    return train_test_split(
        df["text"],
        df["label"],
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=df["label"],
    )


def build_model(config: dict) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=config["max_iter"])),
        ]
    )


def evaluate(model: Pipeline, x_test: pd.Series, y_test: pd.Series, labels: dict) -> dict:
    pred = model.predict(x_test)
    names = sorted(labels, key=labels.get)
    report = classification_report(
        y_test, pred, labels=[labels[n] for n in names], target_names=names, output_dict=True
    )
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {n: report[n] for n in names},
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[labels[n] for n in names]).tolist(),
        "classes": names,
        "n_test": len(y_test),
    }


def run_baseline(config: dict | None = None) -> dict:
    config = config or load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)
    log.info("train %d, test %d", len(x_train), len(x_test))

    model = build_model(config)
    model.fit(x_train, y_train)

    metrics = evaluate(model, x_test, y_test, config["labels"])
    log.info("accuracy %.3f, macro F1 %.3f", metrics["accuracy"], metrics["macro_f1"])

    metrics_path = Path(config["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    model_path = Path(config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    log.info("saved %s and %s", metrics_path, model_path)
    return metrics


if __name__ == "__main__":
    setup_logging()
    run_baseline()
