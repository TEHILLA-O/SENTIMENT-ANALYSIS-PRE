"""Stage 14 - run the saved model on arbitrary text.

The vectorizer is baked into the model, so raw strings go straight in: no manual
cleaning or tokenising at inference time.

    python src/predict.py
    python src/predict.py --model bilstm "Best film I have seen all year"
    python src/predict.py --file my_reviews.txt
"""

import argparse
from pathlib import Path

import tensorflow as tf

from config import MODEL_DIR
# clean_text must be imported (and therefore registered) before load_model.
from preprocess import clean_text  # noqa: F401

DEMO_REVIEWS = [
    "This movie was absolutely brilliant. I loved every minute of it.",
    "This was boring, badly written and a complete waste of time.",
    "The film had some good moments but overall it was average.",
    "The acting was fine, but the plot made no sense whatsoever.",
    "What a masterpiece. I especially enjoyed wasting two hours of my life.",
    "Not bad at all - I expected far worse from the trailer.",
]


def describe(score: float) -> str:
    """Turn a probability into a readable verdict, keeping the strength visible."""
    if score >= 0.85:
        return "Positive (strong)"
    if score >= 0.5:
        return "Positive"
    if score >= 0.15:
        return "Negative"
    return "Negative (strong)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", nargs="*", help="one or more review texts")
    parser.add_argument("--model", choices=["pooling", "bilstm"], default="pooling")
    parser.add_argument("--file", type=Path, help="text file with one review per line")
    args = parser.parse_args()

    reviews = list(args.reviews)
    if args.file:
        reviews += [line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not reviews:
        reviews = DEMO_REVIEWS
        print("(no input given - using the built-in demo reviews)\n")

    model_path = MODEL_DIR / f"sentiment_model_{args.model}.keras"
    if not model_path.exists():
        raise SystemExit(f"No trained model at {model_path} - run src/train.py first")

    model = tf.keras.models.load_model(model_path)
    # tf.constant keeps these as string tensors; a plain numpy array of strings
    # becomes a fixed-width unicode dtype that Keras refuses.
    scores = model.predict(tf.constant(reviews), verbose=0).ravel()

    for review, score in zip(reviews, scores):
        score = float(score)
        bar = "#" * round(score * 30)
        print(f'"{review}"')
        print(f"  Sentiment : {describe(score)}")
        print(f"  Score     : {score:.4f}  |{bar:<30}|")
        print()


if __name__ == "__main__":
    main()
