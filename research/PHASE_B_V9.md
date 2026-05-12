# Phase B — V9 (LightGBM Ranker on 6282 Divine+ matches)

**Status:** Shipped as new default model.
**Date:** 2026-05.

## TL;DR

Phase B set out to scale Phase A's 25-feature pick-rec model from 1381 → 5000+
Divine+ matches and retrain. The naive approach (same sklearn
GradientBoostingClassifier on the larger dataset) plateaued — V9a matched V8.

The breakthrough came from a **two-pronged** change:

1. **Pairwise ranking objective** (LightGBM `LGBMRanker` with `lambdarank`) instead
   of binary classification.
2. **4.6× more training data** (6282 matches vs 1381) — enough to let the ranker
   leverage richer pairwise comparisons.

Together they delivered **+12pp top-10** (74.0% vs V8's 61.9%) on a fully
chronological held-out test set (newest 20% of matches, never seen during
training).

## Pipeline

### B1 — Data expansion (OpenDota + STRATZ)

- B1a: Scraped 8000 new candidate Divine+ match IDs from OpenDota
  `/publicMatches` (script: `research/opendota_fetch.py`).
- B1b: Enriched all candidate IDs with positions, rank tier, and game mode
  through STRATZ GraphQL (`research/stratz_match_fetch.py`).
- Hit STRATZ hourly rate-limit (2000/hr sliding window) mid-fetch; resumed with
  `sleep=2.0s` (≈30 calls/min) to stay safely under the cap.
- **Final dataset:** 6282 enriched matches with full position data (V8 had 1381).

### B2 — Training variants

| Variant | Objective | Architecture | Trees | Depth/Leaves | LR |
|---------|-----------|--------------|------:|--------------|---:|
| V9a | binary CE | sklearn GBC | 400 | depth=5 | 0.05 |
| V9d | binary CE | sklearn GBC | 400 | depth=4 (V8) | 0.05 |
| **V9c** | **lambdarank** | **LightGBM LGBMRanker** | **400** | **num_leaves=63** | **0.05** |

V9b (winners + losers @ 0.5 weight) was scoped but skipped after V9c clearly
dominated; the binary-CE family had already plateaued.

Features (identical to V8): 25 dims —
- 3 base scalars: `base_wr`, `pos_fit`, `role_gap`
- 10 per-position pair signals: `with_syn_pos1..5`, `vs_adv_pos1..5`
- 6 min/max/spread statistics over team-wide synergy/counter
- 5-dim one-hot of target position
- 1 popularity feature

See `research/train_v8.py:hero_features_v8` for the exact feature builder.

### B3 — Chronological held-out backtest

Split: oldest 80% (5026 matches) → train, newest 20% (1256) → test. Match-id
ordering ensures no temporal leakage; V8 was never trained on these test matches.

**Results** (script: `research/backtest_v9.py`):

| Model | top1 | top5 | top10 | mean rank |
|-------|----:|----:|----:|----:|
| V7e | 18.07% | 41.16% | 55.87% | 12.65 |
| V8 (Phase A) | 20.38% | 44.80% | 61.92% | 10.87 |
| V9a (GBC d=5/n=400) | 18.92% | 43.42% | 61.84% | 10.97 |
| V9d (GBC d=4/n=400) | 18.13% | 41.45% | 59.45% | 11.59 |
| **V9c (LGBMRanker)** | **23.54%** | **57.83%** | **73.99%** | **8.15** |

V9c improves over V8 by:
- **+12pp top-10** (74.0% vs 61.9%)
- **+13pp top-5** (57.8% vs 44.8%)
- **+3pp top-1** (23.5% vs 20.4%)
- **−2.7 mean rank** (8.15 vs 10.87) — true picks are ranked higher on average

The two sklearn variants (V9a, V9d) confirmed our **saturation hypothesis** about
binary classification: even with 4.6× the data and tuned hyperparams, they did
not improve over V8. Pairwise ranking was the missing piece.

### B4 — Win-uplift verification (signal-quality check)

For each match, partition picks into (winners' picks, losers' picks) and measure
the model's recall on each subset separately. Higher recall on winners than
losers means the model is identifying **winning patterns**, not just popularity.

Script: `research/win_uplift_v9.py`.

| Model | top-1 (winners / losers / uplift) | top-10 (winners / losers / uplift) |
|-------|-----|-----|
| **V9c** | **23.24% / 17.38% / +5.86pp (ratio 1.34)** | **73.52% / 56.11% / +17.41pp (ratio 1.31)** |
| V9a | 18.82% / 17.18% / +1.64pp (1.10) | 61.12% / 57.55% / +3.57pp (1.06) |
| V8 | 17.24% / 17.14% / +0.10pp (1.01) | 56.59% / 55.15% / +1.44pp (1.03) |
| V7e | 16.77% / 16.40% / +0.37pp (1.02) | 52.98% / 51.78% / +1.20pp (1.02) |
| M3 | 1.23% / 1.15% / +0.07pp (1.06) | 18.09% / 17.09% / +1.00pp (1.06) |

V9c shows a **+17.4pp top-10 winners-vs-losers uplift** — its recommendations
correlate strongly with the team that actually won. By contrast V8 (+1.4pp) was
mostly a "popularity predictor" with negligible win-signal.

This validates the approach: V9c is not just memorizing what humans pick; it's
picking up real **winning patterns** in the per-position synergy/counter signal.

## Why pairwise ranking worked here when binary didn't

Two reinforcing reasons:

1. **The labels are inherently relative.** When a player picks Juggernaut over
   Ursa in a particular draft context, "Juggernaut > Ursa here" is the ground
   truth — not "Juggernaut is good in the abstract". Binary CE collapses both
   into independent "yes/no" labels and loses the relative information.
2. **Lambdarank optimizes NDCG-like metrics directly.** Our top-K metrics are
   exactly what lambdarank cares about; binary CE is a proxy for log-likelihood
   that does not align cleanly with ranking quality.

The same V9c architecture trained on 1381 matches (Phase A's V8b experiment)
was 0.9pp **worse** than V8a. Why does it work now? Lambdarank needs many pairs
per group to learn effectively — 1381 matches gave ~6900 pick decisions, but
6282 matches give ~31000, ~4.6× more pairwise comparisons.

## Files

- `research/data/v9c_ranker.txt` — LightGBM booster (text format, 2.7 MB)
- `research/data/v9c_model.json` — flattened tree JSON for browser (1.9 MB)
- `data/v9c_model.json` — copy used by `v9.js` at runtime
- `research/data/backtest_v9_results.json` — all backtest numbers
- `research/data/win_uplift_v9.json` — winners-vs-losers alignment
- `v9.js` — browser evaluator (loads `v9c_model.json`, walks flat trees)

## Backwards compatibility

- V8 remains the second option in the dropdown (`Phase A` label).
- V7e and M3 remain as backups.
- M1 stays removed (Phase A cleanup).
- The UX is identical: user still picks only their position; ally/enemy
  positions are inferred from the cache.
