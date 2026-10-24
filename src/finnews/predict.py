import logging
import sys
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from finnews.baseline import evaluate, split_data
from finnews.config import load_config
from finnews.improve import build_variant
from finnews.load_data import load_data
from finnews.log import setup_logging
from finnews.preprocess import clean_text, preprocess

log = logging.getLogger(__name__)


def train_and_save(config: dict | None = None) -> Path:
    config = config or load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)

    params = config["experiments"][config["best_experiment"]]
    model = build_variant(params, config).fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test, config["labels"])
    log.info("accuracy %.3f, macro F1 %.3f", metrics["accuracy"], metrics["macro_f1"])

    path = Path(config["final_model_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("saved %s", path)
    return path


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict(texts: list[str], model: Pipeline | None = None, config: dict | None = None) -> list[dict]:
    config = config or load_config()
    model = model or load_model(config["final_model_path"])
    names = {code: name for name, code in config["labels"].items()}

    cleaned = [clean_text(text) for text in texts]
    for original, text in zip(texts, cleaned):
        if len(text) < config["min_text_length"]:
            raise ValueError(f"text is empty or too short: {original!r}")

    codes = model.predict(cleaned)
    return [{"text": text, "label": names[int(code)]} for text, code in zip(texts, codes)]


if __name__ == "__main__":
    setup_logging()
    if len(sys.argv) < 2:
        train_and_save()
    else:
        for item in predict(sys.argv[1:]):
            print(f"{item['label']:<9} {item['text']}")
