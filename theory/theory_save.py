from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.improve import build_variant
from finnews.load_data import load_data
from finnews.preprocess import preprocess

OUT_PATH = Path("reports/theory/save.png")
WORDS = ["profit", "sales", "loss"]
NEW_TEXTS = ["profit rose sharply", "sales fell", "loss widened"]


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, _, y_train, _ = split_data(df, config)

    model = build_variant(config["experiments"]["svc_C1"], config).fit(x_train, y_train)
    saved_vocab = model.steps[0][1].vocabulary_
    fresh_vocab = TfidfVectorizer().fit(NEW_TEXTS).vocabulary_

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    left.bar(["сохранённый\nсловарь", "новый словарь\nиз 3 текстов"], [len(saved_vocab), len(fresh_vocab)],
             color=["#2b6cb0", "#c53030"])
    for i, value in enumerate([len(saved_vocab), len(fresh_vocab)]):
        left.text(i, value, str(value), ha="center", va="bottom", fontsize=13)
    left.set_ylabel("число столбцов (слов)")
    left.set_title("Модель ждёт столько столбцов,\nсколько было при обучении")

    saved_idx = [saved_vocab.get(w, 0) for w in WORDS]
    fresh_idx = [fresh_vocab.get(w, 0) for w in WORDS]
    x = range(len(WORDS))
    right.bar([i - 0.2 for i in x], saved_idx, width=0.4, label="сохранённый словарь", color="#2b6cb0")
    right.bar([i + 0.2 for i in x], fresh_idx, width=0.4, label="новый словарь", color="#c53030")
    right.set_xticks(list(x), WORDS)
    right.set_ylabel("номер столбца слова")
    right.set_title("Одно и то же слово — разные столбцы,\nесли словарь собрать заново")
    right.legend()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
