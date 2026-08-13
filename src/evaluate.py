"""Stages 11-13 - final test-set evaluation and explainable error analysis.

The test set is touched only here, once the architecture has been chosen on the
validation set. Beyond accuracy it reports precision / recall / F1 / ROC-AUC, a
confusion matrix, and writes every mistake to a CSV for inspection.

    python src/evaluate.py --model pooling
    python src/evaluate.py --model bilstm
    python src/evaluate.py --compare-only   # table + shared-error overlap, no model needed
"""

import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import BATCH_SIZE, FIGURE_DIR, LABEL_NAMES, MODEL_DIR, REPORT_DIR, ensure_dirs
# Importing load_datasets also imports clean_text, whose @register_keras_serializable
# decorator must have run before a saved model containing it can be loaded.
from preprocess import load_datasets


def collect_texts_and_labels(dataset):
    """Flatten a batched dataset into aligned arrays of raw text and label."""
    texts, labels = [], []
    for text_batch, label_batch in dataset:
        texts.extend(t.decode("utf-8") for t in text_batch.numpy())
        labels.extend(label_batch.numpy().tolist())
    return np.array(texts, dtype=object), np.array(labels)


def tune_threshold(model, val_ds) -> tuple[float, float]:
    """Pick the decision threshold that maximises F1 on the *validation* set.

    0.5 is only the natural cut-off if the predicted probabilities happen to be
    well calibrated. Searching for a better one is legitimate tuning - as long as
    it is done on validation data and then applied unchanged to the test set.
    """
    texts, labels = collect_texts_and_labels(val_ds)
    scores = model.predict(texts, verbose=0).ravel()
    candidates = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(labels, (scores >= t).astype(int)) for t in candidates]
    best = int(np.argmax(f1s))
    return float(candidates[best]), float(f1s[best])


def plot_confusion_matrix(matrix, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], [f"Predicted {n}" for n in LABEL_NAMES])
    ax.set_yticks([0, 1], [f"Actual {n}" for n in LABEL_NAMES])
    threshold = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=13,
            )
    ax.set_title(f"Confusion matrix - {model_name}")
    fig.tight_layout()
    path = FIGURE_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Confusion matrix -> {path}")


def plot_score_distribution(scores, labels, model_name: str, tuned_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == 0], bins=50, alpha=0.65, label="Actual negative", color="#d62728")
    ax.hist(scores[labels == 1], bins=50, alpha=0.65, label="Actual positive", color="#2ca02c")
    ax.axvline(0.5, color="black", ls="--", label="threshold 0.5")
    ax.axvline(
        tuned_threshold,
        color="#1f77b4",
        ls="-.",
        label=f"tuned threshold {tuned_threshold:.2f}",
    )
    ax.set_xlabel("Predicted P(positive)")
    ax.set_ylabel("Number of reviews")
    ax.set_title(f"Prediction confidence - {model_name}")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / f"score_distribution_{model_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Score distribution -> {path}")


def error_analysis(texts, labels, scores, model_name: str) -> pd.DataFrame:
    """Write every misclassified review to CSV, worst (most confident) first."""
    predictions = (scores >= 0.5).astype(int)
    df = pd.DataFrame(
        {
            "review": texts,
            "real_label": [LABEL_NAMES[label] for label in labels],
            "prediction": [LABEL_NAMES[p] for p in predictions],
            # Confidence in the answer the model actually gave.
            "confidence": np.where(predictions == 1, scores, 1 - scores),
            "score_positive": scores,
            "word_count": [len(t.split()) for t in texts],
        }
    )
    wrong = df[df.real_label != df.prediction].sort_values("confidence", ascending=False)
    path = REPORT_DIR / f"misclassified_reviews_{model_name}.csv"
    wrong.to_csv(path, index=False)
    print(f"{len(wrong):,} misclassified reviews -> {path}")

    lines = [f"# Error analysis - {model_name}\n"]
    lines.append(f"- Misclassified: {len(wrong):,} of {len(df):,} ({len(wrong) / len(df):.2%})")
    lines.append(
        f"- False positives (neg read as pos): {int(((labels == 0) & (predictions == 1)).sum()):,}"
    )
    lines.append(
        f"- False negatives (pos read as neg): {int(((labels == 1) & (predictions == 0)).sum()):,}"
    )
    lines.append(f"- Mean length, correct: {df[df.real_label == df.prediction].word_count.mean():.0f} words")
    lines.append(f"- Mean length, wrong  : {wrong.word_count.mean():.0f} words")
    lines.append(
        f"- Reviews in the uncertain band 0.4-0.6: "
        f"{int(((scores > 0.4) & (scores < 0.6)).sum()):,}"
    )
    lines.append("\n## Most confidently wrong reviews\n")
    for row in wrong.head(5).itertuples():
        lines.append(
            f"**actual {row.real_label}, predicted {row.prediction} "
            f"(score {row.score_positive:.3f})**\n"
        )
        lines.append("> " + row.review[:600].replace("<br />", " ") + "...\n")

    report_path = REPORT_DIR / f"error_analysis_{model_name}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Error notes -> {report_path}")
    return wrong


def compare_error_overlap() -> None:
    """How many mistakes do the architectures share?

    Errors that every model gets wrong point at the data or the features - label
    noise, sarcasm, mixed verdicts - not at the architecture. That overlap is a
    rough floor on what swapping layers can fix.
    """
    mistakes = {}
    for path in sorted(REPORT_DIR.glob("misclassified_reviews_*.csv")):
        name = path.stem.replace("misclassified_reviews_", "")
        # Keyed on review text, so the corpus's ~400 duplicate reviews collapse and
        # these counts sit slightly below the raw misclassification counts.
        mistakes[name] = set(pd.read_csv(path).review)
    if len(mistakes) < 2:
        return

    print("\nShared mistakes (unique review texts)")
    names = sorted(mistakes)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = mistakes[first] & mistakes[second]
            union = mistakes[first] | mistakes[second]
            print(
                f"  {first} ({len(mistakes[first]):,}) vs {second} ({len(mistakes[second]):,}): "
                f"{len(shared):,} identical reviews wrong "
                f"({len(shared) / len(union):.1%} of their combined errors)"
            )


def compare_saved_runs() -> None:
    """Print a side-by-side table for every model evaluated so far."""
    rows = []
    for path in sorted(REPORT_DIR.glob("metrics_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) < 2:
        return
    table = pd.DataFrame(rows)[
        [
            "model",
            "test_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "test_loss",
            "tuned_threshold",
            "tuned_test_accuracy",
            "tuned_f1",
        ]
    ]
    print("\nModel comparison")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    compare_error_overlap()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pooling", "bilstm"], default="pooling")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="just re-print the comparison of already-evaluated models",
    )
    args = parser.parse_args()

    matplotlib.use("Agg")  # save figures without needing a GUI window
    ensure_dirs()
    if args.compare_only:
        compare_saved_runs()
        return

    model_path = MODEL_DIR / f"sentiment_model_{args.model}.keras"
    if not model_path.exists():
        raise SystemExit(f"No trained model at {model_path} - run src/train.py first")

    print(f"Loading {model_path}")
    model = tf.keras.models.load_model(model_path)

    _, val_ds, test_ds = load_datasets(batch_size=args.batch_size)
    print(f"Test class order: {test_ds.class_names}")

    print("\nSearching for the best decision threshold on the validation set ...")
    threshold, val_f1 = tune_threshold(model, val_ds)
    print(f"Best validation F1 {val_f1:.4f} at threshold {threshold:.2f}")

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss    : {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    texts, labels = collect_texts_and_labels(test_ds)
    scores = model.predict(texts, batch_size=args.batch_size, verbose=1).ravel()
    predictions = (scores >= 0.5).astype(int)
    tuned_predictions = (scores >= threshold).astype(int)

    metrics = {
        "model": args.model,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        # Threshold-free: how well the scores *rank* reviews, whatever the cut-off.
        "roc_auc": float(roc_auc_score(labels, scores)),
        "n_test": int(len(labels)),
        "tuned_threshold": threshold,
        "tuned_val_f1": val_f1,
        "tuned_test_accuracy": float(accuracy_score(labels, tuned_predictions)),
        "tuned_precision": float(precision_score(labels, tuned_predictions)),
        "tuned_recall": float(recall_score(labels, tuned_predictions)),
        "tuned_f1": float(f1_score(labels, tuned_predictions)),
    }
    matrix = confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["tuned_confusion_matrix"] = confusion_matrix(labels, tuned_predictions).tolist()

    print("\nMetrics (positive class = 'pos')")
    print(f"  {'metric':<10} {'@0.50':>8} {'@' + format(threshold, '.2f'):>8}")
    for label, default_key, tuned_key in (
        ("accuracy", "test_accuracy", "tuned_test_accuracy"),
        ("precision", "precision", "tuned_precision"),
        ("recall", "recall", "tuned_recall"),
        ("f1", "f1", "tuned_f1"),
    ):
        print(f"  {label:<10} {metrics[default_key]:>8.4f} {metrics[tuned_key]:>8.4f}")
    print(f"  {'roc_auc':<10} {metrics['roc_auc']:>8.4f}   (threshold-free)")
    print("\n" + classification_report(labels, predictions, target_names=LABEL_NAMES, digits=4))
    print("Confusion matrix at 0.5 [rows = actual neg/pos, cols = predicted neg/pos]")
    print(matrix)

    plot_confusion_matrix(matrix, args.model)
    plot_score_distribution(scores, labels, args.model, threshold)
    error_analysis(texts, labels, scores, args.model)

    metrics_path = REPORT_DIR / f"metrics_{args.model}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics -> {metrics_path}")

    compare_saved_runs()


if __name__ == "__main__":
    main()
