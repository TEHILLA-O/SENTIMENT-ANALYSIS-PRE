"""Stages 3, 5 and 6 - text cleaning, dataset loading and text -> numbers.

Run it directly to see a before/after cleaning demo plus the learned vocabulary:

    python src/preprocess.py
"""

import re
import string

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

from config import (
    BATCH_SIZE,
    MAX_FEATURES,
    SEED,
    SEQUENCE_LENGTH,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_SPLIT,
)

PUNCTUATION_PATTERN = f"[{re.escape(string.punctuation)}]"

# The reviews are scraped HTML and contain broken byte sequences (stray \x96 and
# friends). Restricting the vocabulary to ASCII letters and digits drops that
# noise, and keeps the saved vocabulary file readable on any platform - Keras
# writes it with the system's default text encoding, which is cp1252 on Windows.
NON_ASCII_PATTERN = "[^a-z0-9 ]"


# Registering makes the function findable when a saved .keras model that embeds
# the TextVectorization layer is loaded back in evaluate.py / predict.py.
@tf.keras.utils.register_keras_serializable(package="sentiment")
def clean_text(text):
    """Lowercase, drop the <br /> tags IMDB reviews are full of, strip punctuation.

    Runs inside the TextVectorization layer, so it operates on string tensors and
    travels with the saved model - the same cleaning is applied at training time
    and at inference time.
    """
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "<br />", " ")
    text = tf.strings.regex_replace(text, PUNCTUATION_PATTERN, "")
    text = tf.strings.regex_replace(text, NON_ASCII_PATTERN, " ")
    return tf.strings.strip(tf.strings.regex_replace(text, r"\s+", " "))


def clean_text_python(text: str) -> str:
    """Pure-Python mirror of clean_text, for the pandas-based EDA stage."""
    text = text.lower()
    text = text.replace("<br />", " ")
    text = re.sub(PUNCTUATION_PATTERN, "", text)
    text = re.sub(NON_ASCII_PATTERN, " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_datasets(batch_size: int = BATCH_SIZE):
    """Build train / validation / test tf.data pipelines from the folder tree.

    The 25k training reviews are split 80:20 into train and validation using the
    same seed, so no review appears in both. The 25k test reviews are untouched
    and only used for the final evaluation.
    """
    train_ds = tf.keras.utils.text_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.text_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        batch_size=batch_size,
    )
    test_ds = tf.keras.utils.text_dataset_from_directory(
        TEST_DIR,
        batch_size=batch_size,
    )
    return train_ds, val_ds, test_ds


def build_vectorizer(train_ds) -> TextVectorization:
    """Learn the vocabulary from the *training* split only.

    Adapting on validation or test text would leak information about data the
    model is supposed to be judged on.
    """
    vectorizer = TextVectorization(
        standardize=clean_text,
        max_tokens=MAX_FEATURES,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorizer.adapt(train_ds.map(lambda text, label: text))
    return vectorizer


def optimise(dataset):
    """Cache decoded text in memory and overlap batch preparation with training."""
    return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


def main() -> None:
    raw = "I absolutely LOVED this movie! <br /><br /> Amazing... 10/10."
    print("BEFORE:", raw)
    print("AFTER :", clean_text(tf.constant(raw)).numpy().decode())

    train_ds, val_ds, test_ds = load_datasets()
    print("\nLabel order (index = integer label):", train_ds.class_names)
    print(f"Train batches: {train_ds.cardinality().numpy()}")
    print(f"Val batches  : {val_ds.cardinality().numpy()}")
    print(f"Test batches : {test_ds.cardinality().numpy()}")

    print("\nAdapting TextVectorization on the training split ...")
    vectorizer = build_vectorizer(train_ds)
    vocab = vectorizer.get_vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    print("First 15 tokens:", vocab[:15])

    example = tf.constant(["This movie was brilliant and the acting was superb"])
    vectorised = vectorizer(example).numpy()[0]
    print("\nExample sentence ->", example.numpy()[0].decode())
    print("Token ids (first 12):", vectorised[:12].tolist())
    print("Decoded back       :", [vocab[i] for i in vectorised[:12]])
    print("\n0 is padding, 1 is [UNK] (a word outside the 20k vocabulary).")


if __name__ == "__main__":
    main()
