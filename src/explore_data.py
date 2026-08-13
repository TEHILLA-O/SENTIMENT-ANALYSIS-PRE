"""Stages 2 and 4 - exploratory data analysis before any modelling.

Answers: is the dataset balanced, how long are the reviews, and which tokens
actually separate positive from negative reviews?

    python src/explore_data.py
    python src/explore_data.py --min-count 50 --top-n 25

Writes figures to reports/figures/ and a text summary to reports/eda_summary.md.
"""

import argparse
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURE_DIR, IMDB_DIR, REPORT_DIR, SEED, SEQUENCE_LENGTH, ensure_dirs
from preprocess import clean_text_python

# Very common words carry no sentiment; they would dominate every frequency list.
STOPWORDS = {
    "the", "a", "an", "and", "of", "to", "is", "it", "in", "i", "this", "that",
    "was", "as", "for", "with", "but", "on", "movie", "film", "are", "his",
    "have", "be", "one", "you", "at", "all", "he", "her", "she", "they", "by",
    "from", "not", "or", "who", "so", "there", "if", "its", "it's", "out",
    "about", "what", "when", "we", "would", "which", "their", "been", "has",
    "up", "were", "my", "me", "had", "them", "him", "just", "some", "more",
    "very", "no", "only", "than", "into", "do", "does", "did", "can", "could",
    "will", "your", "how", "then", "too", "also", "other", "much", "most",
    "any", "get", "got", "see", "seen", "even", "after", "way", "make", "made",
    "because", "us", "s", "t", "br",
}


def load_split(split: str, limit: int | None = None) -> pd.DataFrame:
    """Read one split into a DataFrame of raw text + integer label.

    Each review is its own small file, so this is pure I/O latency: a thread pool
    turns ~25,000 sequential opens into something that finishes in seconds.
    limit caps the files read per class, which keeps the notebook responsive; the
    scripted EDA always reads everything.
    """
    records = []
    for label_name, label in (("neg", 0), ("pos", 1)):
        paths = sorted((IMDB_DIR / split / label_name).glob("*.txt"))[:limit]
        with ThreadPoolExecutor(max_workers=32) as pool:
            texts = pool.map(lambda p: p.read_text(encoding="utf-8"), paths)
            for path, text in zip(paths, texts):
                records.append(
                    {
                        "file": path.name,
                        "split": split,
                        "label": label,
                        "sentiment": label_name,
                        "text": text,
                    }
                )
    return pd.DataFrame(records)


def report_class_balance(df: pd.DataFrame, lines: list[str]) -> None:
    lines.append("## 3. Is the dataset balanced?\n")
    for split in ("train", "test"):
        subset = df[df.split == split]
        counts = subset.sentiment.value_counts()
        total = len(subset)
        lines.append(f"**{split}** ({total:,} reviews)\n")
        for name in ("pos", "neg"):
            count = int(counts.get(name, 0))
            share = count / total
            bar = "#" * round(share * 40)
            lines.append(f"- {name}: {count:,} ({share:.1%}) `{bar}`")
        lines.append("")

    fig, ax = plt.subplots(figsize=(6, 4))
    pivot = df.pivot_table(index="split", columns="sentiment", values="file", aggfunc="count")
    pivot.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c"], rot=0)
    ax.set_ylabel("Number of reviews")
    ax.set_title("Class balance per split")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "class_balance.png", dpi=120)
    plt.close(fig)


def report_review_lengths(df: pd.DataFrame, lines: list[str]) -> None:
    lengths = df.word_count.to_numpy()
    over_limit = float((lengths > SEQUENCE_LENGTH).mean())

    lines.append("## 4. How long are the reviews?\n")
    lines.append(f"- Mean: {lengths.mean():.1f} words")
    lines.append(f"- Median: {np.median(lengths):.0f} words")
    lines.append(f"- Min / Max: {lengths.min()} / {lengths.max()} words")
    lines.append(f"- 90th percentile: {np.percentile(lengths, 90):.0f} words")
    lines.append(
        f"- Truncated at SEQUENCE_LENGTH={SEQUENCE_LENGTH}: {over_limit:.1%} of reviews"
    )
    for name in ("pos", "neg"):
        lines.append(
            f"- Mean length ({name}): {df[df.sentiment == name].word_count.mean():.1f} words"
        )
    lines.append("")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=60, range=(0, 1200), color="#1f77b4")
    ax.axvline(np.median(lengths), color="black", ls="--", label=f"median {np.median(lengths):.0f}")
    ax.axvline(SEQUENCE_LENGTH, color="red", ls=":", label=f"cut-off {SEQUENCE_LENGTH}")
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Distribution of review length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "review_length.png", dpi=120)
    plt.close(fig)


def count_words(texts) -> Counter:
    counter = Counter()
    for text in texts:
        counter.update(clean_text_python(text).split())
    return counter


def report_word_frequency(
    df: pd.DataFrame, lines: list[str], min_count: int, top_n: int
) -> pd.DataFrame:
    """Rank tokens by how much more often they appear in one class than the other.

    Raw frequency alone is useless here ("the" is everywhere), so each token gets
    a log2 ratio of its *rate* in positive vs negative reviews. Tokens are kept
    only above min_count occurrences so rare words cannot top the list by luck.
    """
    train = df[df.split == "train"]
    pos_counts = count_words(train[train.label == 1].text)
    neg_counts = count_words(train[train.label == 0].text)
    pos_total = sum(pos_counts.values())
    neg_total = sum(neg_counts.values())

    records = []
    for word in set(pos_counts) | set(neg_counts):
        if word in STOPWORDS or len(word) < 3:
            continue
        pos_n, neg_n = pos_counts[word], neg_counts[word]
        if pos_n + neg_n < min_count:
            continue
        # +0.5 smoothing keeps the ratio finite for words absent from one class.
        pos_rate = (pos_n + 0.5) / pos_total
        neg_rate = (neg_n + 0.5) / neg_total
        records.append(
            {
                "word": word,
                "pos_count": pos_n,
                "neg_count": neg_n,
                "total": pos_n + neg_n,
                "log2_pos_over_neg": math.log2(pos_rate / neg_rate),
            }
        )

    scores = pd.DataFrame(records).sort_values("log2_pos_over_neg", ascending=False)
    scores.to_csv(REPORT_DIR / "word_scores.csv", index=False)

    lines.append("## 5. Which tokens appear frequently, and which discriminate?\n")
    lines.append(f"Most frequent non-stopword tokens (train):\n")
    overall = (pos_counts + neg_counts).most_common()
    frequent = [(w, c) for w, c in overall if w not in STOPWORDS and len(w) >= 3][:15]
    lines.append("`" + "`, `".join(f"{w} ({c:,})" for w, c in frequent) + "`\n")

    top_pos = scores.head(top_n)
    top_neg = scores.tail(top_n).iloc[::-1]

    lines.append(f"Top {top_n} positive-associated tokens (min {min_count} occurrences):\n")
    lines.append("| word | pos | neg | log2(pos/neg) |")
    lines.append("| --- | --- | --- | --- |")
    for row in top_pos.itertuples():
        lines.append(
            f"| {row.word} | {row.pos_count:,} | {row.neg_count:,} | {row.log2_pos_over_neg:+.2f} |"
        )
    lines.append("")
    lines.append(f"Top {top_n} negative-associated tokens:\n")
    lines.append("| word | pos | neg | log2(pos/neg) |")
    lines.append("| --- | --- | --- | --- |")
    for row in top_neg.itertuples():
        lines.append(
            f"| {row.word} | {row.pos_count:,} | {row.neg_count:,} | {row.log2_pos_over_neg:+.2f} |"
        )
    lines.append("")

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_df = pd.concat([top_pos.head(15), top_neg.head(15)]).sort_values("log2_pos_over_neg")
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in plot_df.log2_pos_over_neg]
    ax.barh(plot_df.word, plot_df.log2_pos_over_neg, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("log2( rate in positive / rate in negative )")
    ax.set_title("Most sentiment-discriminative tokens (derived from the data)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "discriminative_words.png", dpi=120)
    plt.close(fig)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    # Set here rather than at import time: writing PNGs without a GUI window is
    # what the script wants, but importing this module from a notebook should not
    # silently break inline plotting.
    matplotlib.use("Agg")
    ensure_dirs()
    if not IMDB_DIR.exists():
        raise SystemExit("Dataset missing - run: python src/download_data.py")

    print("Reading all 50,000 review files ...")
    df = pd.concat([load_split("train"), load_split("test")], ignore_index=True)
    print(f"Loaded {len(df):,} reviews. Cleaning text ...")
    df["clean"] = df.text.map(clean_text_python)
    df["word_count"] = df.clean.str.split().str.len()
    print("Computing statistics and word frequencies ...")

    lines: list[str] = ["# EDA summary\n"]
    lines.append("## 2. What does the dataset contain?\n")
    lines.append(f"- {len(df):,} labelled reviews, one plain-text file each")
    lines.append("- Two usable columns: `text` (input X) and `label` (target y: 0=neg, 1=pos)")
    lines.append(f"- Exact duplicate review texts: {int(df.text.duplicated().sum()):,}")
    lines.append("")

    report_class_balance(df, lines)
    report_review_lengths(df, lines)
    report_word_frequency(df, lines, args.min_count, args.top_n)

    lines.append("## Raw examples\n")
    for row in df.sample(2, random_state=SEED).itertuples():
        lines.append(f"**{row.sentiment}** ({row.word_count} words): {row.text[:300]}...\n")

    output = REPORT_DIR / "eda_summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:40]))
    print(f"\nFull summary: {output}")
    print(f"Figures     : {FIGURE_DIR}")


if __name__ == "__main__":
    main()
