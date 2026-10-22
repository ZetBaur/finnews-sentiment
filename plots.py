import os

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

os.makedirs("reports", exist_ok=True)

df = load_data()
df["text_length"] = df["text"].str.len()

sns.countplot(data=df, x="sentiment", order=["negative", "neutral", "positive"])
plt.title("Распределение классов")
plt.savefig("reports/sentiment_distribution.png")
plt.close()

df["text_length"].hist(bins=50)
plt.title("Длина текстов")
plt.xlabel("Символов в тексте")
plt.ylabel("Количество текстов")
plt.savefig("reports/text_length_distribution.png")
plt.close()

print("Графики сохранены в reports/")
