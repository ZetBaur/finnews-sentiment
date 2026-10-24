from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.improve import build_variant
from finnews.load_data import load_data
from finnews.preprocess import preprocess

OUT_PATH = Path("reports/theory/errors.png")
NAMES = ["negative", "neutral", "positive"]


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)

    model = build_variant(config["experiments"]["svc_C1"], config).fit(x_train, y_train)
    pred = model.predict(x_test)

    matrix = confusion_matrix(y_test, pred)
    share = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            color = "white" if share[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{matrix[i, j]}\n{share[i, j]:.0%}",
                ha="center", va="center", color=color, fontsize=14,
            )
    ax.set_xticks(range(3), NAMES)
    ax.set_yticks(range(3), NAMES)
    ax.set_xlabel("ответ модели")
    ax.set_ylabel("настоящий класс")
    ax.set_title("Куда уходят предложения каждого класса\n(число и доля строки)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
