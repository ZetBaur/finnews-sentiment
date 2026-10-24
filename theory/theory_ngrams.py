from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

SENTENCES = ["profit did rise", "profit did not rise"]
OUT_PATH = Path("reports/theory/ngrams.png")


def draw(ax, ngram_range, title):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(SENTENCES).toarray()
    features = vectorizer.get_feature_names_out()
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(features)), features, rotation=60, ha="right")
    ax.set_yticks(range(len(SENTENCES)), [f"«{s}»" for s in SENTENCES])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")
    ax.set_title(title)


def main() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [4, 9]}
    )
    draw(axes[0], (1, 1), "ngram_range=(1, 1): только слова")
    draw(axes[1], (1, 2), "ngram_range=(1, 2): слова и пары слов")
    fig.suptitle("Со словами по одному предложения различаются только словом «not»; пара «not rise» даёт отдельный признак")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
