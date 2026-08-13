"""Stage 1 - download the Large Movie Review Dataset (aclImdb).

50,000 labelled IMDB reviews from Maas et al. (2011), split into 25,000 train
and 25,000 test, plus 50,000 unlabelled reviews in train/unsup.

    python src/download_data.py
    python src/download_data.py --keep-unsup   # keep the unlabelled folder
"""

import argparse
import shutil
import tarfile

import tensorflow as tf

from config import DATA_DIR, DATASET_URL, IMDB_DIR, TRAIN_DIR, ensure_dirs


def download_archive() -> str:
    """Fetch aclImdb_v1.tar.gz into data/ (skipped if already cached)."""
    return tf.keras.utils.get_file(
        fname="aclImdb_v1.tar.gz",
        origin=DATASET_URL,
        cache_dir=str(DATA_DIR),
        cache_subdir="",
        extract=False,
    )


def extract_archive(archive_path: str) -> None:
    if IMDB_DIR.exists():
        print(f"Already extracted: {IMDB_DIR}")
        return

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        # filter="data" blocks absolute paths / symlinks escaping the target dir.
        tar.extractall(path=DATA_DIR, filter="data")


def remove_unsup() -> None:
    """Delete train/unsup.

    text_dataset_from_directory turns every sub-directory into a class, so the
    50k unlabelled reviews would be read as a bogus third class.
    """
    unsup_dir = TRAIN_DIR / "unsup"
    if unsup_dir.exists():
        print(f"Removing unlabelled reviews: {unsup_dir}")
        shutil.rmtree(unsup_dir)
    else:
        print("No train/unsup directory to remove.")


def summarise() -> None:
    print("\nDataset layout")
    print("-" * 40)
    for split in ("train", "test"):
        split_dir = IMDB_DIR / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n_files = sum(1 for _ in class_dir.glob("*.txt"))
            print(f"{split}/{class_dir.name:<6} {n_files:>6,} reviews")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-unsup",
        action="store_true",
        help="keep train/unsup (only useful for unsupervised pre-training)",
    )
    args = parser.parse_args()

    ensure_dirs()
    archive_path = download_archive()
    extract_archive(archive_path)

    if not args.keep_unsup:
        remove_unsup()

    summarise()
    print(f"\nDataset ready at: {IMDB_DIR}")


if __name__ == "__main__":
    main()
