from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.load_data import load_data
from finnews.preprocess import preprocess

C_VALUES = [0.01, 0.1, 1, 10, 100, 1000]
OUT_PATH = Path("reports/theory/c_param.png")


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)

    vectorizer = TfidfVectorizer()
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)

    train_scores = []
    test_scores = []
    for c in C_VALUES:
        clf = LogisticRegression(C=c, max_iter=config["max_iter"])
        clf.fit(train_matrix, y_train)
        train_scores.append(f1_score(y_train, clf.predict(train_matrix), average="macro"))
        test_scores.append(f1_score(y_test, clf.predict(test_matrix), average="macro"))
        print(f"C={c}: train {train_scores[-1]:.3f}, test {test_scores[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(C_VALUES, train_scores, marker="o", color="#c0392b", label="train")
    ax.plot(C_VALUES, test_scores, marker="o", color="#2e86c1", label="test")
    ax.set_xscale("log")
    ax.set_xlabel("C (меньше C: модель сдержаннее, больше C: подгоняется под train)")
    ax.set_ylabel("macro F1")
    ax.set_title("Чем больше C, тем сильнее разрыв между train и test")
    ax.legend()
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
