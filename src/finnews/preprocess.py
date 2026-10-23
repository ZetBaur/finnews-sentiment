import html
import logging
import re
import unicodedata

from finnews.config import load_config
from finnews.load_data import load_data
from finnews.log import setup_logging

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACES_RE = re.compile(r"\s+")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = SPACES_RE.sub(" ", text)
    return text.strip()


def preprocess(df, config=None):
    config = config or load_config()
    labels = config["labels"]
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    before = len(df)
    df = df[df["text"].str.len() >= config["min_text_length"]]
    log.info("Отброшено коротких или пустых текстов: %d", before - len(df))

    before = len(df)
    df = df[df["sentiment"].isin(labels)]
    log.info("Отброшено строк с неизвестным классом: %d", before - len(df))

    df["label"] = df["sentiment"].map(labels)

    labels_per_text = df.groupby("text")["label"].transform("nunique")
    before = len(df)
    df = df[labels_per_text == 1]
    log.info("Отброшено строк с конфликтом классов: %d", before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    log.info("Отброшено дубликатов: %d", before - len(df))

    return df.reset_index(drop=True)


if __name__ == "__main__":
    setup_logging()
    raw = load_data()
    clean = preprocess(raw)
    log.info("Было %d строк, стало %d", len(raw), len(clean))
    for text in raw["text"].head(5):
        print("до :", text)
        print("после:", clean_text(text))
