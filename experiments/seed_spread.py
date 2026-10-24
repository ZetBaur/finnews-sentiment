import statistics

from sklearn.metrics import f1_score

from finnews.baseline import split_data
from finnews.config import load_config
from finnews.improve import build_variant
from finnews.load_data import load_data
from finnews.preprocess import preprocess

SEEDS = range(10)
VARIANTS = ["baseline", "svc_C1"]


def main() -> None:
    config = load_config()
    df = preprocess(load_data(config["data_path"]), config)

    scores = {name: [] for name in VARIANTS}
    for seed in SEEDS:
        x_train, x_test, y_train, y_test = split_data(df, {**config, "random_state": seed})
        for name in VARIANTS:
            model = build_variant(config["experiments"][name], config).fit(x_train, y_train)
            scores[name].append(f1_score(y_test, model.predict(x_test), average="macro"))

    for name, values in scores.items():
        print(
            f"{name}: min {min(values):.3f}, max {max(values):.3f}, "
            f"среднее {statistics.mean(values):.3f}, разброс (std) {statistics.stdev(values):.3f}"
        )
    diffs = [s - b for b, s in zip(scores["baseline"], scores["svc_C1"])]
    print("svc_C1 минус baseline по деления:", " ".join(f"{d:+.3f}" for d in diffs))
    print(f"svc_C1 лучше baseline в {sum(d > 0 for d in diffs)} из {len(diffs)} делений")


if __name__ == "__main__":
    main()
