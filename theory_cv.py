from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_FOLDS = 5
N_TRAIN = 3868
OUT_PATH = Path("reports/theory/cv.png")


def main() -> None:
    fold_size = N_TRAIN / N_FOLDS
    fig, ax = plt.subplots(figsize=(11, 4.2))

    for run in range(N_FOLDS):
        for part in range(N_FOLDS):
            is_check = part == run
            ax.barh(
                run,
                fold_size,
                left=part * fold_size,
                color="#e67e22" if is_check else "#2e86c1",
                edgecolor="white",
            )
            ax.text(
                part * fold_size + fold_size / 2,
                run,
                "проверка" if is_check else "обучение",
                ha="center",
                va="center",
                color="white",
            )

    ax.barh(N_FOLDS + 0.3, N_TRAIN * 0.25, left=N_TRAIN * 1.02, color="#7f8c8d")
    ax.text(N_TRAIN * 1.02 + N_TRAIN * 0.125, N_FOLDS + 0.3, "test", ha="center", va="center", color="white")
    ax.text(N_TRAIN * 1.02 + N_TRAIN * 0.125, N_FOLDS - 0.6, "не трогаем", ha="center", va="center", fontsize=9)

    ax.set_yticks(range(N_FOLDS), [f"запуск {i + 1}" for i in range(N_FOLDS)])
    ax.invert_yaxis()
    ax.set_xlabel("строки train (3868), 5 частей по ~774")
    ax.set_title("5 запусков: каждая часть train по разу становится проверочной; test ждёт до конца")
    ax.set_xlim(0, N_TRAIN * 1.3)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
