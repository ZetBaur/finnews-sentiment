from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from finnews.baseline import evaluate, split_data
from finnews.config import load_config
from finnews.load_data import load_data
from finnews.preprocess import preprocess

MIN_DF_VALUES = [1, 2, 3, 5]


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)
    x_train, x_test, y_train, y_test = split_data(df, config)
    folds = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=config["random_state"]
    )

    for min_df in MIN_DF_VALUES:
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df)),
                ("clf", LinearSVC(C=1, random_state=config["random_state"])),
            ]
        )
        cv_scores = cross_val_score(model, x_train, y_train, cv=folds, scoring="f1_macro")
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_test, y_test, config["labels"])
        n_features = len(model.named_steps["tfidf"].vocabulary_)
        print(
            f"min_df={min_df}: features {n_features}, CV F1 {cv_scores.mean():.3f} ± {cv_scores.std():.3f}, "
            f"test F1 {metrics['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
