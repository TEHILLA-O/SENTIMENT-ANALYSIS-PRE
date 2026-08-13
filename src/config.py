"""Shared paths and hyper-parameters.

Every stage of the pipeline imports from here so that a change to (say)
SEQUENCE_LENGTH is applied consistently to exploration, training and evaluation.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMDB_DIR = DATA_DIR / "aclImdb"
TRAIN_DIR = IMDB_DIR / "train"
TEST_DIR = IMDB_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Vocabulary size: only the 20k most frequent tokens get their own embedding,
# everything else collapses into the [UNK] token.
MAX_FEATURES = 20_000

# Reviews are padded/truncated to this many tokens. The median review is 171
# words and this cut-off truncates 29% of them (see reports/eda_summary.md):
# a deliberate trade-off, since cost grows with sequence length.
SEQUENCE_LENGTH = 250

EMBEDDING_DIM = 128
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

LABEL_NAMES = ("neg", "pos")  # index == integer label assigned by Keras


def ensure_dirs() -> None:
    for directory in (DATA_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
