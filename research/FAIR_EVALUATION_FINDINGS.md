# Fair Evaluation — Findings and Methodology Reset

> **TL;DR**: All three of our deployed pick-recommendation models (V8, V9c)
> reported inflated top-10 metrics due to a train-test leak. Under honest
> evaluation (chronological 80/20 split, no leak), every variant we have —
> V7e, V8, V9c, V10c — converges to **55–58% top-10**. The simplest model
> (V8, sklearn GBM, 25 features) is now the new default at **57.5% top-10**.

This document is the source of truth for what our models actually do. Older
reports (`PHASE_A_V8.md`, `PHASE_B_V9.md`, `WIN_UPLIFT_V8.md`) still describe
the inflated numbers — keep them for historical context but trust this file
when assessing performance.

---

## 1. What the leak was

Our prior training pipeline trained on **all** 6282 matches in
`data/matches_stratz_enriched.json`, and the prior backtest **then** split
those same 6282 matches into 80/20 and "evaluated" on the held-out 20%.
The 1256-test subset was in fact part of the training set the model had
already seen.

```
              Phase A / Phase B pipeline (WRONG):
              ┌───────────────────────────────┐
              │ all 6282 matches              │
              │                               │
              │ ──► train model on ALL 6282   │
              │                               │
              │ ──► split 80/20, evaluate     │
              │     on 20% the model SAW      │  ← train-test leak
              └───────────────────────────────┘

              Phase B+A pipeline (CORRECT, this fix):
              ┌───────────────────────────────┐
              │ 6282 matches, sorted by       │
              │ match_id (chronological)      │
              │                               │
              │ ──► OLDEST 5026 → train       │
              │ ──► NEWEST 1256 → held out    │  ← never seen in training
              │     (evaluation only)         │
              └───────────────────────────────┘
```

The leak inflated V9c's top-10 from **57.3% → 74.0%** (+16.7pp) and
V8's from **57.5% → 61.9%** (+4.4pp). LightGBM with `num_leaves=63`
memorises individual training instances much more aggressively than
sklearn GBC with `max_depth=4`, which is why V9c's inflation was much
larger than V8's.

---

## 2. Honest numbers (after Plan B retraining)

All four models below are evaluated on the **same** 1256 newest matches
(`match_id` range `8499886601..8499999909`) which were **never** seen
during training.

Run: `python3 research/backtest_fair.py`

| Model | Architecture | Train set | Top-1 | Top-5 | Top-10 | Mean rank |
|---|---|---:|---:|---:|---:|---:|
| **V8 fair** (new default) | sklearn GBC `n=300 d=4 lr=0.05` (25 features) | 5026 oldest | **17.5%** | **39.1%** | **57.5%** | **12.22** |
| V9c fair | LightGBM lambdarank `n=400 lvs=63` (25 features) | 5026 oldest | 17.4% | 38.6% | 57.3% | 12.35 |
| V10c fair | LightGBM lambdarank `n=400 lvs=63` (39 features, +team-comp) | 5026 oldest | 17.1% | 39.3% | 57.4% | 12.32 |
| V7e | sklearn GBC `n=200 d=3 lr=0.05` (5 features) | 1381 | 18.1% | 41.2% | 55.9% | 12.65 |

For reference (do **not** cite these as model performance):

| Artefact | Reported top-10 | Reality |
|---|---:|---|
| V9c (deployed before this PR) | 74.0% | Train-test leak; honest = 57.3% |
| V8 (deployed before this PR) | 61.9% | Phase A training-set overlap; honest = 57.5% |

The fair top-10 spread across V7e/V8/V9c/V10c is **±1pp** — within noise.
V8 wins on top-10, V7e wins on top-1/top-5, V10c is mid. Differences are
not statistically meaningful at our current dataset size (1256 test).

Random baseline for top-10 across ~80 candidates ≈ 12.5%. Our 57% is
~4.6× over random — meaningful, but not the "breakthrough" earlier metrics
suggested.

---

## 3. What did and didn't matter

- **Architecture change (lambdarank vs sklearn GBC) — no measurable diff.**
  V9c (lambdarank) ties V8 (binary GBC) at ~57% top-10. The "Phase B
  breakthrough" was entirely the leak.
- **More features (V10c: 25 → 39) — no measurable diff.** Team-composition
  features are not noise (we hand-curated them), but the current 12-hour
  snapshot dataset does not contain enough meta variation for those
  features to differentiate winning compositions from non-winning ones.
- **More data (V8 on 1381 vs 5026) — no measurable diff.** V8 honest top-10
  on 1381 = 56.5%, on 5026 = 57.5%. The marginal gain (+1pp) is roughly
  noise. The bottleneck is not training-set size at this regime.
- **What likely matters**: temporal spread of training data, hero metadata
  quality, position-label noise. These need investigation under Plan A.

---

## 4. What the app now ships

After Plan B remediation, the deployed app (https://dota2-drafter-xscnvzaw.devinapps.com):

- **Default model**: V8 fair (`data/v8_model.json`, sklearn GBC 300 trees, 25 features)
- **Dropdown options**:
  1. V8 fair — 57.5% top-10 (default)
  2. V9c fair — 57.3% top-10 (LightGBM Ranker)
  3. V10c fair — 57.4% top-10 (LightGBM Ranker + team-composition)
  4. V7e — 55.9% top-10 (small 5-feature baseline)
- **Removed**: M3 (STRATZ R.O.S.H. clone — outside our ML scope, no longer
  needed).
- **Removed**: original V9c (leaked) — the new `v9c_model.json` is the fair
  retrain.

UX is otherwise unchanged.

---

## 5. Proper evaluation methodology (for future work)

1. **Always split chronologically** by `match_id` (which monotonically
   increases globally in Dota) — do **not** shuffle. We use 80/20:
   oldest 5026 → train, newest 1256 → held-out test.
2. **Train models only on the train set.** Saving training scripts that
   read the full dataset is fine, but the model must be `fit`-ted on the
   train subset only.
3. **Evaluate only on the held-out test set.** The same test set should
   be used across all models compared in a given experiment.
4. **Persist the train/test match-id lists** to a JSON file
   (`data/fair_split.json`) for reproducibility.
5. **When changing the dataset** (new matches added), re-split chrono­
   logically, retrain, and re-evaluate — never reuse old test-set numbers
   with new training data.

The unified pipeline that does all four points: `research/train_fair.py`
+ `research/backtest_fair.py`. Both are in this PR.

See also: `research/EVALUATION.md` (canonical methodology reference).

---

## 6. Plan A — data collection design (next phase)

The current dataset is 6282 matches spanning only ~12 hours of real time
(match_id range `8499403600..8499999909`, ~30–50 IDs/sec global rate).
This is effectively a single snapshot of the meta. To unlock the next
improvement tier we need:

### Target: 15,000–30,000 matches over 2–4 weeks

- **Why temporal spread**: With matches from different days/patches, the
  model can:
  - Learn meta-resilient patterns instead of patch-specific noise
  - Support patch-recency weighting (`exp(-days_old / τ)`) — currently
    pointless since all matches are ~equally old
  - Detect counter-meta picks that don't appear on a single-day snapshot
- **Why 15–30k**: Current 6282 = ~12500 train picks across 25 features.
  Adding ~30 team-comp / interaction features pushes us into the
  ~250 samples/feature range (overfit risk). 20–30k matches restores a
  comfortable 500–800 samples/feature.
- **Why 2–4 weeks (not months)**: Beyond ~4 weeks Dota patches become a
  significant confound (heroes get nerfed/buffed). We want diversity
  within a stable meta — multiple distinct days, not multiple patches.

### Collection script

The simplest path is a daily cron-style fetch using OpenDota's
`/publicMatches` endpoint. We already have `research/opendota_fetch.py`;
it just needs to be invoked across multiple days.

Proposed schedule:
```
day 0   : fetch newest 2000 Divine+ matches
day 3   : fetch newest 2000 (likely overlapping ~30% with day 0)
day 7   : fetch newest 2000
day 10  : fetch newest 2000
day 14  : fetch newest 2000
day 17  : fetch newest 2000
day 21  : fetch newest 2000
```

After dedup by `match_id` and STRATZ enrichment, this gives roughly
12–15k unique matches across 21 days. Doubling the cadence yields 25–30k.

### Storage and pipeline

- Append new matches to `data/matches_stratz_enriched.json` (sorted by
  match_id) — the train scripts already sort, so future-newer matches
  naturally end up in the chronological test set.
- Treat `data/fair_split.json` as ephemeral — regenerate it when the
  dataset grows.

### What this unlocks

- **Phase C3 (patch-recency weighting)**: Currently impossible because
  all matches are within ~12 hours of each other. After 14+ days, we can
  weight matches by `exp(-days_old / τ)` and properly test τ ∈ {7, 15, 30}.
- **Phase C1 (team-composition) re-evaluation**: At 20k+ matches, the
  14 team-comp features may show real uplift instead of the +0.1pp noise
  we saw at 6282 matches.
- **Honest calibration / SHAP**: Larger held-out test (4k+) gives stable
  calibration curves and reliable feature-importance plots.

---

## 7. Files changed in this PR

- **Removed**: `m3.js` (STRATZ R.O.S.H. clone — out of scope).
- **Replaced** (now fair-trained models):
  - `data/v8_model.json` (sklearn GBC, 300 trees, 25 features, on 5026)
  - `data/v9c_model.json` (LightGBM Ranker, 400 trees, 25 features, on 5026)
- **New**:
  - `data/v10c_model.json` (LightGBM Ranker, 400 trees, 39 features, on 5026)
  - `v10.js` (browser inference for V10c)
  - `research/train_fair.py` (unified fair retraining)
  - `research/backtest_fair.py` (unified honest backtest)
  - `research/export_fair_models.py` (booster → JSON for all three)
  - `research/EVALUATION.md` (canonical methodology reference)
  - `data/fair_split.json` (train/test match-id lists)
- **Updated**: `app.js`, `index.html` (V8 default, M3 removed, V10 added,
  modal text updated to honest numbers).
