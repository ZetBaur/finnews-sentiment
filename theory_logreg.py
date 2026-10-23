from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN = [
    ("profit rose sharply", 1),
    ("sales increased strongly", 1),
    ("net profit grew", 1),
    ("strong growth in sales", 1),
    ("profit fell sharply", 0),
    ("sales decreased", 0),
    ("net loss widened", 0),
    ("weak sales and loss", 0),
]
NEW = ["profit rose", "sales fell", "net profit grew sharply"]
OUT_PATH = Path("reports/theory/logreg.png")


def main() -> None:
    texts = [t for t, _ in TRAIN]
    labels = [y for _, y in TRAIN]

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(texts)
    model = LogisticRegression(C=10).fit(x_train, labels)

    words = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    order = weights.argsort()
    probs = model.predict_proba(vectorizer.transform(NEW))[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]})
    colors = ["#2e86c1" if w > 0 else "#c0392b" for w in weights[order]]
    ax1.barh(words[order], weights[order], color=colors)
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_title("Вес слова: синий — за positive, красный — против")

    bars = ax2.bar(range(len(NEW)), probs, color="#2e86c1")
    ax2.bar_label(bars, fmt="%.2f")
    ax2.set_xticks(range(len(NEW)), [f"«{t}»" for t in NEW], rotation=15)
    ax2.axhline(0.5, color="gray", linestyle="--")
    ax2.set_ylim(0, 1)
    ax2.set_title("Вероятность positive для новых предложений")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")
    for w, v in sorted(zip(words, weights), key=lambda p: p[1]):
        print(f"{w:10s} {v:+.2f}")
    for t, p in zip(NEW, probs):
        print(f"{t!r}: positive {p:.2f}")


if __name__ == "__main__":
    main()
