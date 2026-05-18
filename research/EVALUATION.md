# Evaluation Methodology — Canonical Reference

> **Required reading before any "this model is N% better than that one" claim.**

## The rule

**Always split the dataset chronologically by `match_id`, train only on the
training side, and evaluate only on the held-out test side.**

This is not optional — it is the difference between honest performance
numbers and self-deceiving inflated ones. We learned this the hard way
(`FAIR_EVALUATION_FINDINGS.md`); the original Phase B "74% top-10" was an
artefact of training on the test set.

## Concrete pipeline

```python
matches = load_matches("data/matches_stratz_enriched.json")
matches.sort(key=lambda m: int(m["match_id"]))   # chronological order

n_test = max(1, len(matches) // 5)               # 20% held out
train_matches = matches[:-n_test]                # oldest 80%
test_matches  = matches[-n_test:]                # newest 20% — DO NOT touch in training

# train any model only on train_matches
# evaluate that model only on test_matches
```

Reference implementations:
- `research/train_fair.py` — trains V8/V9c/V10c on the train side only
- `research/backtest_fair.py` — evaluates four models on the test side only

The current split for our 6282-match dataset is **5026 train / 1256 test**,
persisted to `data/fair_split.json`.

## Why match_id ordering

In Dota, `match_id` increments globally with roughly 30–50 IDs per second.
Sorting by `match_id` gives a temporal ordering. With our current dataset
this corresponds to ~12 hours of real time (very narrow), but the principle
holds for any future dataset where matches span days or weeks.

When the dataset is expanded to a wider temporal window (Plan A), the same
chronological split will give a much more meaningful evaluation — newer
matches will reflect a (potentially) shifted meta, and a model that
overfits to old picks will be penalised by the test set.

## Common anti-patterns to avoid

### 1. Training on all data, then splitting for "evaluation"

```python
# WRONG — model has seen the test set during training
model.fit(all_features, all_labels)
train_idx, test_idx = split_80_20(all_features)
evaluate(model, all_features[test_idx], all_labels[test_idx])
```

This was the root cause of the original V9c/V8 inflation. Always split
first, then train.

### 2. Random shuffle splits

```python
# WRONG — temporal leakage between train and test
train, test = train_test_split(matches, test_size=0.2, shuffle=True)
```

Random shuffling intermixes recent and older matches between train and
test, so any meta drift or temporal pattern (player improvement, hero
rebalances) becomes leakage. Always split chronologically.

### 3. Picking the test set after seeing results

If you tune hyperparameters or features against the test set, the test set
becomes part of your training loop. Use a separate validation split (carve
out 10% of the train set) or simple cross-validation for tuning. Touch the
test set only at the end, once.

### 4. Reusing a stale test set across dataset versions

When new matches are added to the dataset, the chronological split shifts.
A model trained on the new (larger) train set must be re-evaluated on the
new test set — comparing it to the old model's old-test-set number is
apples-to-oranges.

### 5. Quoting backtest numbers without identifying the model file

Every honest number must be traceable to a specific model file on disk
trained by a specific script on a specific split. The pattern:
> "V8 fair (sklearn GBC, 25 features, `data/v8_fair_gbm.joblib`, trained
> by `train_fair.py` on the 5026-match split persisted in
> `data/fair_split.json`) scored 57.5% top-10 on the 1256-test set."

is what we should aim for in any report.

## Useful helpers

- `research/backtest_v9.py::split_chronological(matches, frac_test=0.2)` —
  the canonical split helper. Re-use this instead of writing one inline.
- `research/backtest_v9.py::evaluate_per_position()` — the canonical
  ranking metric helper. Samples 2-3 picks per match for efficiency.
- `research/train_fair.py::build_dataset()` — builds the per-pick feature
  matrix with group counts for lambdarank. Use this when training new
  pairwise rankers.

## Honest numbers — current state (as of this PR)

Evaluated on the same 1256-match held-out set:

| Model | Top-10 |
|---|---:|
| V8 fair | 57.5% (default) |
| V9c fair | 57.3% |
| V10c fair | 57.4% |
| V7e | 55.9% |

Spread is ±1pp, within noise at this sample size. Architectural and
feature-set changes within our current dataset cannot reliably distinguish
between them. See `FAIR_EVALUATION_FINDINGS.md` for the full story.
