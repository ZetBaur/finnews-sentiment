import random

from sklearn.metrics import accuracy_score, f1_score

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.improve import build_variant
from finnews.load_data import load_data
from finnews.preprocess import preprocess


def shuffle_words(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)

    params = config["experiments"][config["best_experiment"]]
    model = build_variant(params, config).fit(x_train, y_train)

    rng = random.Random(config["random_state"])
    shuffled = x_test.map(lambda text: shuffle_words(text, rng))

    for name, texts in [("исходные", x_test), ("слова перемешаны", shuffled)]:
        pred = model.predict(texts)
        print(
            f"{name}: accuracy {accuracy_score(y_test, pred):.3f}, "
            f"macro F1 {f1_score(y_test, pred, average='macro'):.3f}"
        )
    print("пример:", x_test.iloc[0])
    print("после: ", shuffled.iloc[0])


if __name__ == "__main__":
    main()
