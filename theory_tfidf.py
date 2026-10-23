from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

SENTENCES = ["profit rose", "profit fell", "profit rose again"]
OUT_PATH = Path("reports/theory/tfidf.png")


def main() -> None:
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(SENTENCES).toarray()
    words = vectorizer.get_feature_names_out()

    fig, axes = plt.subplots(1, len(SENTENCES), figsize=(12, 3.8), sharey=True)
    for ax, sentence, row in zip(axes, SENTENCES, matrix):
        colors = ["#c0392b" if w == "profit" else "#2e86c1" for w in words]
        bars = ax.bar(words, row, color=colors)
        ax.bar_label(bars, fmt="%.2f")
        ax.set_title(f"«{sentence}»")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("вес слова (TF-IDF)")
    fig.suptitle("«profit» есть во всех предложениях, его вес ниже; редкие слова весят больше")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")
    for sentence, row in zip(SENTENCES, matrix):
        print(sentence, dict(zip(words, row.round(2))))


if __name__ == "__main__":
    main()
