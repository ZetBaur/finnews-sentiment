from finnews.load_data import load_data

df = load_data()

print("Размер:", df.shape)
print()
df.info()
print()
print("Пропуски:")
print(df.isnull().sum())
print()
print("Классы (штук):")
print(df["sentiment"].value_counts())
print()
print("Классы (доля):")
print(df["sentiment"].value_counts(normalize=True).round(3))
print()

df["text_length"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()
print("Длины текстов:")
print(df[["text_length", "word_count"]].describe())
print()

print("Дубликаты текстов:", df["text"].duplicated().sum())
print()

for sentiment in df["sentiment"].unique():
    print(f"=== {sentiment} ===")
    for text in df[df["sentiment"] == sentiment]["text"].head(3):
        print("-", text)
    print()
