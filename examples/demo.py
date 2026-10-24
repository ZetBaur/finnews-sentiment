from finnews.predict import predict

PHRASES = [
    "Operating profit rose to EUR 20 million from EUR 15 million a year earlier.",
    "The company reported a net loss for the quarter.",
    "The meeting will be held on Monday in Helsinki.",
    "Sales decreased by 15% compared to the previous year.",
    "The board proposes a dividend of EUR 0.50 per share.",
    "The company won a large order from a Chinese customer.",
]


def main() -> None:
    for item in predict(PHRASES):
        print(f"{item['label']:<9} {item['text']}")


if __name__ == "__main__":
    main()
