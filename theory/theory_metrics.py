from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from finnews.config import load_config
from finnews.load_data import load_data
from finnews.preprocess import preprocess

OUT_PATH = Path("reports/theory/metrics.png")
NAMES = ["negative", "neutral", "positive"]


def main() -> None:
    config = load_config()
    df = preprocess(load_data())
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter=1000).fit(vectorizer.fit_transform(x_train), y_train)
    pred = model.predict(vectorizer.transform(x_test))

    matrix = confusion_matrix(y_test, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [4, 5]})
    ax1.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax1.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=14)
    ax1.set_xticks(range(3), NAMES)
    ax1.set_yticks(range(3), NAMES)
    ax1.set_xlabel("ответ модели")
    ax1.set_ylabel("настоящий класс")
    ax1.set_title("Матрица ошибок")

    width = 0.25
    x = np.arange(3)
    for offset, values, label, color in [
        (-width, precision, "precision", "#2e86c1"),
        (0, recall, "recall", "#e67e22"),
        (width, f1, "F1", "#27ae60"),
    ]:
        bars = ax2.bar(x + offset, values, width, label=label, color=color)
        ax2.bar_label(bars, fmt="%.2f")
    ax2.set_xticks(x, NAMES)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.set_title("Метрики по классам")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")
    print(matrix)
    for name, p, r, f in zip(NAMES, precision, recall, f1):
        print(f"{name:9s} precision {p:.2f} recall {r:.2f} F1 {f:.2f}")


if __name__ == "__main__":
    main()
