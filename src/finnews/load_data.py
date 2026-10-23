import csv

import pandas as pd

DATA_PATH = "data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt"


def load_data(path=DATA_PATH):
    df = pd.read_csv(
        path,
        sep="@",
        header=None,
        names=["text", "sentiment"],
        encoding="latin-1",
        quoting=csv.QUOTE_NONE,
    )
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.head())
