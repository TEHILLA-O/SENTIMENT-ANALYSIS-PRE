"""Stages 7-10 - build, compile, fit and save the classifier.

Two architectures share the same vectorizer + embedding front end:

    pooling : Embedding -> GlobalAveragePooling1D -> Dense -> Dropout -> Sigmoid
    bilstm  : Embedding -> Bidirectional(LSTM)    -> Dense -> Dropout -> Sigmoid

The pooling model treats a review as a bag of words; the BiLSTM reads the words
in order, so it can in principle pick up negation ("not good") and contrast.

    python src/train.py --model pooling --epochs 15
    python src/train.py --model bilstm  --epochs 6
"""

import argparse
import json
import time

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    EMBEDDING_DIM,
    FIGURE_DIR,
    MAX_FEATURES,
    MODEL_DIR,
    REPORT_DIR,
    SEED,
    ensure_dirs,
)
from preprocess import build_vectorizer, load_datasets, optimise


def build_model(name: str, vectorizer) -> tf.keras.Model:
    """Assemble the network. The vectorizer is layer 1, so the model eats raw strings."""
    # shape=() because each dataset element is a single scalar string tensor.
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="review")
    x = vectorizer(inputs)

    # mask_zero=True marks padding tokens so the pooling / LSTM layers ignore them.
    x = layers.Embedding(
        input_dim=MAX_FEATURES,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name="embedding",
    )(x)

    if name == "pooling":
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    elif name == "bilstm":
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
    else:
        raise ValueError(f"Unknown model: {name}")

    # One output neuron + sigmoid squashes the score into (0, 1): P(positive).
    outputs = layers.Dense(1, activation="sigmoid", name="sentiment")(x)
    return tf.keras.Model(inputs, outputs, name=f"sentiment_{name}")


def plot_history(history: dict, model_name: str) -> None:
    """Training vs validation curves - the overfitting check."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["accuracy"], "o-", label="Training accuracy")
    axes[0].plot(epochs, history["val_accuracy"], "s-", label="Validation accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{model_name}: accuracy")

    axes[1].plot(epochs, history["loss"], "o-", label="Training loss")
    axes[1].plot(epochs, history["val_loss"], "s-", label="Validation loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{model_name}: loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(list(epochs))
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = FIGURE_DIR / f"history_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Learning curves -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pooling", "bilstm"], default="pooling")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    matplotlib.use("Agg")  # save figures without needing a GUI window
    ensure_dirs()
    tf.keras.utils.set_random_seed(SEED)

    print("Loading datasets ...")
    train_ds, val_ds, _ = load_datasets(batch_size=args.batch_size)
    class_names = train_ds.class_names
    print(f"Class order: {class_names}  (label 1 = {class_names[1]})")

    print("Adapting TextVectorization on the training split ...")
    vectorizer = build_vectorizer(train_ds)

    train_ds, val_ds = optimise(train_ds), optimise(val_ds)

    model = build_model(args.model, vectorizer)
    model.compile(
        optimizer="adam",
        # Binary cross-entropy: the loss for a single probability output. It
        # punishes confident wrong answers far harder than hesitant ones.
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"\nTraining '{args.model}' for up to {args.epochs} epochs ...")
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stopping],
    )
    elapsed = time.perf_counter() - start
    print(f"Training finished in {elapsed / 60:.1f} min")

    plot_history(history.history, args.model)

    model_path = MODEL_DIR / f"sentiment_model_{args.model}.keras"
    model.save(model_path)
    print(f"Model -> {model_path}")

    best_epoch = int(min(range(len(history.history["val_loss"])),
                         key=lambda i: history.history["val_loss"][i]))
    run = {
        "model": args.model,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best_epoch + 1,
        "train_accuracy": history.history["accuracy"][best_epoch],
        "val_accuracy": history.history["val_accuracy"][best_epoch],
        "train_loss": history.history["loss"][best_epoch],
        "val_loss": history.history["val_loss"][best_epoch],
        "trainable_params": int(sum(v.numpy().size for v in model.trainable_variables)),
        "training_minutes": round(elapsed / 60, 2),
        "history": history.history,
    }
    history_path = REPORT_DIR / f"history_{args.model}.json"
    history_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"History -> {history_path}")

    print(
        f"\nBest epoch {run['best_epoch']}: "
        f"train acc {run['train_accuracy']:.4f} / val acc {run['val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
