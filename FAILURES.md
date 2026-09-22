# Failure modes, fixes, and results

Honest engineering notes for this project. Nothing here is invented for polish.

## What can go wrong

- **Label leakage through cleaning.** Ratings like `8/10` become tokens `810` and dominate the vocabulary. Impact: inflated accuracy that is not about prose. Mitigation: documented in README error analysis; future work notes dropping digits.
- **Train/test leakage or split drift.** Impact: dishonest metrics. Mitigation: modular pipeline stages; evaluation on held-out test; compare architectures on the same cleaned inputs.
- **Sarcasm and label noise.** Impact: confident wrong predictions. Mitigation: per-review error dumps sorted by confidence; overlap analysis across models.
- **Broken byte sequences / non-ASCII noise in raw files.** Impact: tokenizer chaos. Mitigation: cleaning drops non-ASCII noise and collapses punctuation (README).

## What went wrong

Evidence recorded in README / initial commit (not a production outage):

1. Word-frequency analysis found a **ratings leak** (`810`, `910`, `1010` among strongest positive tokens) after punctuation stripping.
2. **Both** pooling and BiLSTM architectures fail on the **same 2,620** review texts (~55.8% of combined unique errors), so residual error is largely data/feature/sarcasm/label noise rather than architecture choice.
3. Hand-written sarcasm examples fail as expected when sentiment is vocabulary-driven.

## How it was resolved

- Pipeline split into download / explore / preprocess / train / evaluate / predict so the leak is inspectable.
- Error analysis artifacts and compare-only evaluation document shared failures.
- Threshold tuning on validation called out as gaining more than switching architecture; digit-stripping listed as a concrete next fix, not silently claimed done.

## Results

- Metrics and plots live under `reports/` after you run train/evaluate (precision/recall/F1/ROC-AUC). Run the scripts rather than trusting paraphrased numbers here.
- Successful local demo: `pip install -r requirements.txt`, download aclImdb, train pooling, `evaluate.py`, inspect `error_analysis_*.md`.
