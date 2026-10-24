import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.improve import build_variant
from finnews.load_data import load_data
from finnews.log import setup_logging
from finnews.preprocess import preprocess

log = logging.getLogger(__name__)


def find_errors(model: Pipeline, x_test: pd.Series, y_test: pd.Series, labels: dict) -> pd.DataFrame:
    names = {value: name for name, value in labels.items()}
    pred = model.predict(x_test)
    scores = model.decision_function(x_test)
    classes = list(model.classes_)

    rows = []
    for text, true, answer, row in zip(x_test, y_test, pred, scores):
        if true == answer:
            continue
        rows.append(
            {
                "text": text,
                "true": names[true],
                "pred": names[answer],
                "margin": float(row[classes.index(answer)] - row[classes.index(true)]),
                "n_words": len(text.split()),
            }
        )
    return pd.DataFrame(rows, columns=["text", "true", "pred", "margin", "n_words"])


def count_error_types(errors: pd.DataFrame) -> pd.DataFrame:
    counts = errors.groupby(["true", "pred"]).size().reset_index(name="count")
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def plot_confusion(model: Pipeline, x_test: pd.Series, y_test: pd.Series, labels: dict, path: Path) -> None:
    names = sorted(labels, key=labels.get)
    order = [labels[n] for n in names]
    matrix = confusion_matrix(y_test, model.predict(x_test), labels=order)
    share = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", color=color, fontsize=14)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("ответ модели")
    ax.set_ylabel("настоящий класс")
    ax.set_title("Матрица ошибок на test")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_error_analysis(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)

    params = config["experiments"][config["best_experiment"]]
    model = build_variant(params, config).fit(x_train, y_train)

    errors = find_errors(model, x_test, y_test, config["labels"])
    log.info("errors %d of %d (%.1f%%)", len(errors), len(y_test), 100 * len(errors) / len(y_test))

    errors_path = Path(config["errors_path"])
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(errors_path, index=False)
    plot_confusion(model, x_test, y_test, config["labels"], Path(config["confusion_path"]))
    log.info("saved %s and %s", errors_path, config["confusion_path"])

    for _, row in count_error_types(errors).iterrows():
        subset = errors[(errors["true"] == row["true"]) & (errors["pred"] == row["pred"])]
        log.info("%s -> %s: %d, средняя длина %.1f слов", row["true"], row["pred"], row["count"], subset["n_words"].mean())
        confident = subset.sort_values("margin", ascending=False).head(config["examples_per_type"])
        for _, example in confident.iterrows():
            log.info("  [%.2f] %s", example["margin"], example["text"])
    return errors


if __name__ == "__main__":
    setup_logging()
    run_error_analysis()
