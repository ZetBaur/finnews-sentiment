from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from finnews.config import load_config
from finnews.load_data import load_data
from finnews.preprocess import preprocess

OUT_PATH = Path("reports/theory/split.png")
NAMES = {0: "negative", 1: "neutral", 2: "positive"}
COLORS = {0: "#c0392b", 1: "#999999", 2: "#2e86c1"}


def shares(y):
    counts = y.value_counts(normalize=True).sort_index()
    return [counts.get(k, 0) * 100 for k in NAMES]


def main() -> None:
    config = load_config()
    df = preprocess(load_data())
    seed = config["random_state"]
    size = config["test_size"]

    rows = [("все данные", df["label"])]
    for stratify, title in [(None, "test без stratify"), (df["label"], "test со stratify")]:
        _, test = train_test_split(
            df["label"], test_size=size, random_state=seed, stratify=stratify
        )
        rows.append((title, test))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for i, (title, y) in enumerate(rows[:3]):
        left = 0
        for k, value in zip(NAMES, shares(y)):
            ax.barh(i, value, left=left, color=COLORS[k], label=NAMES[k] if i == 0 else None)
            ax.text(left + value / 2, i, f"{value:.1f}", ha="center", va="center", color="white")
            left += value
    ax.set_yticks(range(3), [r[0] for r in rows[:3]])
    ax.invert_yaxis()
    ax.set_xlabel("доля класса, %")
    ax.set_title("Доли классов")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.3))

    ax = axes[1]
    for stratify, label, color in [(None, "без stratify", "#e67e22"), (df["label"], "со stratify", "#2e86c1")]:
        negative_shares = []
        for s in range(1, 6):
            _, test = train_test_split(df["label"], test_size=size, random_state=s, stratify=stratify)
            negative_shares.append(shares(test)[0])
        ax.plot(range(1, 6), negative_shares, marker="o", label=label, color=color)
    ax.set_xlabel("номер случайного деления")
    ax.set_ylabel("доля negative в test, %")
    ax.set_title("Доля negative в test от деления к делению")
    ax.legend()

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")
    for title, y in rows[:3]:
        print(title, [round(v, 1) for v in shares(y)])


if __name__ == "__main__":
    main()
