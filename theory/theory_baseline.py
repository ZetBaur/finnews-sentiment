from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from finnews.load_data import load_data
from finnews.preprocess import preprocess

OUT_PATH = Path("reports/theory/baseline.png")
SEED = 42


def main() -> None:
    df = preprocess(load_data())
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"]
    )

    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)

    vectorizer = TfidfVectorizer()
    model = LogisticRegression(max_iter=1000).fit(vectorizer.fit_transform(x_train), y_train)
    model_pred = model.predict(vectorizer.transform(x_test))

    names = ["всегда neutral", "TF-IDF + LogReg"]
    acc = [accuracy_score(y_test, dummy_pred), accuracy_score(y_test, model_pred)]
    f1 = [
        f1_score(y_test, dummy_pred, average="macro"),
        f1_score(y_test, model_pred, average="macro"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, values, title in zip(axes, [acc, f1], ["Точность (accuracy)", "F1 (macro)"]):
        bars = ax.bar(names, values, color=["#999999", "#2e86c1"])
        ax.bar_label(bars, fmt="%.2f")
        ax.set_ylim(0, 1)
        ax.set_title(title)
    fig.suptitle("Тест: глупое правило против baseline-модели")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")
    print(f"train {len(x_train)}, test {len(x_test)}")
    print(f"всегда neutral: accuracy {acc[0]:.3f}, F1 macro {f1[0]:.3f}")
    print(f"TF-IDF + LogReg: accuracy {acc[1]:.3f}, F1 macro {f1[1]:.3f}")


if __name__ == "__main__":
    main()
