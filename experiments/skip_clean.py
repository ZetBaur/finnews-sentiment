from sklearn.metrics import accuracy_score, f1_score

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.load_data import load_data
from finnews.predict import load_model
from finnews.preprocess import clean_text, preprocess

NOISE_PREFIX = "&amp; "
NOISE_SUFFIX = " http://example.com/page"


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)
    _, x_test, _, y_test = split_data(df, config)
    model = load_model(config["final_model_path"])

    noisy = x_test.map(lambda text: NOISE_PREFIX + text + NOISE_SUFFIX)

    variants = {
        "чистые тексты": x_test,
        "с мусором, через clean_text": noisy.map(clean_text),
        "с мусором, без clean_text": noisy,
    }
    for name, texts in variants.items():
        pred = model.predict(texts)
        print(
            f"{name:<30} accuracy {accuracy_score(y_test, pred):.3f}  "
            f"macro F1 {f1_score(y_test, pred, average='macro'):.3f}"
        )


if __name__ == "__main__":
    main()
