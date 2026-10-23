from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.load_data import load_data
from finnews.preprocess import preprocess


def score(vectorizer, x_train, x_test, y_train, y_test, max_iter):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(vectorizer.transform(x_train), y_train)
    pred = model.predict(vectorizer.transform(x_test))
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)
    max_iter = config["max_iter"]

    honest = TfidfVectorizer().fit(x_train)
    leaky = TfidfVectorizer().fit(df["text"])

    for name, vec in [("честно: TF-IDF на train", honest), ("с утечкой: TF-IDF на всех данных", leaky)]:
        acc, f1 = score(vec, x_train, x_test, y_train, y_test, max_iter)
        print(f"{name:36s} слов в словаре {len(vec.vocabulary_):6d}  accuracy {acc:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    main()
