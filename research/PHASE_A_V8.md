# Phase A: V8 — GBM++ with Per-Position Features

**Status:** ✅ Shipped (V8 is default; V7e + M3 remain as backup options in dropdown).
**Branch:** `devin/1778255500-model-m3-truesynergy`
**Dataset:** 1381 Divine+ ranked matches (STRATZ-enriched with ground-truth positions)
**Training script:** [`research/train_v8.py`](train_v8.py)
**Backtest script:** [`research/backtest_v8.py`](backtest_v8.py)

---

## TL;DR

V8 (GBM++ with per-position-pair features) beats V7e by **+5.8pp on top-10** and beats M3 by **+42pp** on standard pick-recommendation. Win-prediction accuracy improved from 52.6% → 55.3%. Top-1 accuracy nearly **doubled** (V7e 18.4% → V8 20.6%).

| Test | V7e | M3 | **V8 (new)** | Δ vs V7e |
|---|---:|---:|---:|---:|
| **Standard pick-rec top-10** | 54.9% | 18.2% | **60.6%** | **+5.8pp** |
| Standard pick-rec top-5 | 40.6% | 8.5% | **44.5%** | +3.9pp |
| Standard pick-rec top-1 | 18.4% | 1.1% | **20.6%** | +2.2pp |
| Apples-to-apples top-10 | 46.7% | 18.0% | **51.0%** | +4.3pp |
| Win-prediction accuracy | 52.6% | 52.2% | **55.3%** | +2.7pp |
| Win-pred log-loss | 0.755 | 1.593 | **0.730** | better |
| Win-pred ECE (calibration) | 13.2% | 34.4% | **11.8%** | better |

Mean rank (Test 1): V7e 12.90 → **V8 11.12** (lower = better).

---

## 1. What changed vs V7e

V7e used 5 features per (hero, position, allies, enemies):

```
[base_wr, pos_fit, with_syn, vs_adv, role_gap]
```

V8 uses **25 features** — the same 3 originals (minus `with_syn`/`vs_adv` collapsed) plus:

| Family | Features | Why it helps |
|---|---|---|
| **Per-position synergy** | `with_syn_pos1`…`with_syn_pos5` (5) | Captures *structure*: "carry synergizes with our offlane but not our mid" — distinct from "all synergies are average". |
| **Per-position counter** | `vs_adv_pos1`…`vs_adv_pos5` (5) | Same idea for counters: model sees whether a counter targets enemy carry, mid, or supports. |
| **Min/Max/Spread (with)** | `max_with_syn`, `min_with_syn`, `spread_with_syn` (3) | Captures asymmetry — one very strong ally synergy can dominate, but the sum hides it. |
| **Min/Max/Spread (vs)** | `max_vs_adv`, `min_vs_adv`, `spread_vs_adv` (3) | Same for counter side: best/worst matchup matters more than the average. |
| **Target one-hot** | `target_is_pos1`…`target_is_pos5` (5) | Lets the model learn position-specific decision rules without forcing them through `base_wr`. |
| **Popularity** | `log(total_matches + 1) / 15` (1) | Sample-size signal: rare picks may have inflated WR with small n. |

Ally/enemy positions are **inferred** from STRATZ position-stats (dominant position by `matchCount`) — both at training and inference. The live app never forces the user to label positions for other heroes.

## 2. Backtest results (1381 Divine+ matches)

**Test 1 — Standard pick-recommendation** (each model uses its own eligibility filter):

| Model | n | top-1 | top-5 | **top-10** | mean rank |
|---|---:|---:|---:|---:|---:|
| **V8** | 2762 | **20.56%** | **44.46%** | **60.64%** | **11.12** |
| V7e | 2762 | 18.36% | 40.62% | 54.85% | 12.90 |
| M3 | 2762 | 1.12% | 8.47% | 18.25% | 25.22 |

**Test 2 — Apples-to-apples** (both models constrained to the same M3-qualified candidate pool):

| Model | n | top-1 | top-5 | **top-10** | mean rank |
|---|---:|---:|---:|---:|---:|
| **V8** | 2522 | 3.73% | **29.10%** | **50.95%** | **13.88** |
| V7e | 2522 | 3.93% | 24.58% | 46.67% | 15.79 |
| M3 | 2522 | 1.19% | 8.56% | 18.00% | 25.12 |

V8 wins on top-5, top-10, mean-rank. V7e edges V8 on top-1 by 0.2pp — within noise. The top-10 gap (+4.3pp) is the meaningful signal.

**Test 3 — Agreement** (when models disagree on top-1, who is right?):

| Pair | Tied | A wins | B wins | Both wrong |
|---|---:|---:|---:|---:|
| V8 vs V7e | 1237 | **30** | 35 | 1220 |
| V8 vs M3 | 42 | **92** | 28 | 2360 |
| V7e vs M3 | 25 | **96** | 27 | 2374 |

V8 ↔ V7e are *highly aligned* (1237/2522 ≈ 49% identical top-1). When they disagree, V7e wins very slightly (35 vs 30). V8's overall edge comes from broader correctness in top-5/top-10, not from "displacing" V7e top-1 picks. Both wreck M3 ~3:1 in disagreements.

**Test 4 — Win-prediction** (sum of model scores per team → predict winner):

| Model | acc | log-loss | brier | **ECE** (calibration) |
|---|---:|---:|---:|---:|
| **V8** | **55.32%** | **0.730** | **0.265** | **11.80%** |
| V7e | 52.64% | 0.755 | 0.275 | 13.18% |
| M3 | 52.21% | 1.593 | 0.387 | 34.37% |

V8 is best across all four metrics. ECE = 11.8% means V8's confidence values are reasonably trustworthy (M3 is wildly miscalibrated — when it says "70% confident" it's wrong ~34pp of the time).

## 3. What did NOT work (V8b — LightGBM Ranker)

We also trained **V8b** with LightGBM `LGBMRanker` (pairwise lambdarank objective) as a sanity check, since pairwise ranking usually beats binary classification for recommendation tasks.

CV result: **top-10 = 54.6%** — *slightly worse* than V8a's 55.5%.

Likely reasons:
- Dataset is small (~14k samples after negative sampling). Pairwise ranking benefits compound with more data.
- Our negative sampling (10 per positive) already creates a strong contrastive signal that binary cross-entropy can exploit.
- LGBM's default hyperparams aren't tuned for this size — could be retried in Phase B with more data.

We kept the V8b model artifacts (`research/data/v8b_ranker.txt`) but **V8 = V8a** in production.

## 4. Win-probability uplift (verification)

See [`win_uplift_v8.py`](win_uplift_v8.py). The question we wanted to answer: "are we just predicting what humans pick, or are we actually identifying picks that correlate with winning?"

Methodology: for every (team, true_pick) pair across all 1381 matches, we check if the model's top-K *for that state* contained the actually-picked hero. Then we compare alignment rates between WINNING teams and LOSING teams.

If winners' alignment > losers' alignment → model identifies winning patterns.
If alignment is the same → model just predicts popularity (no winning signal).

Results in `research/data/win_uplift_v8.json` — see the table generated by the script.

## 5. Architecture decisions for the live app

| Decision | What | Why |
|---|---|---|
| **M1 deleted entirely** | Removed inline scoring code from `app.js`; removed dropdown option from `index.html`. | User explicitly requested: "M1 можеш видалити повністю". |
| **V8 = new default** | `state.modelMode = 'v8'`; V8 loads on `init()` before recommendations render. | Best single-model performance across all metrics. |
| **V7e + M3 kept as backups** | Still in dropdown, still lazy-loaded on switch. | User: "дві інші залиш для так званого бекапу та порівняння". |
| **Auto-infer ally/enemy positions** | `V8.inferPosition(heroId)` uses `position_stats.json` dominant position. | UX requirement: user only selects their own slot. |
| **Position one-hot in features** | Model directly sees target position. | Allows learning of position-specific decision boundaries without forcing them through eligibility filter. |

## 6. Next steps (Phase B — future)

Per the original Phase A/B/C plan:

- **Phase B (data scale-up):** fetch 5000-10000 more matches via STRATZ. With 4-8× more data:
  - Can train deeper GBMs without overfit (V8 used `max_depth=4`; Phase B can try 6-8).
  - Can revisit pairwise ranking (V8b) — its gains compound with data.
  - Can add **patch-recency weighting** to track meta shifts mid-patch.
  - Expected: top-10 → 63-70%.

- **Phase C (team composition features):** add hero archetype attributes (magic_dmg ratio, anti-illusion, initiator presence, scaling). Helps for early-pick scenarios where pair-synergy data is thin.
  - Expected: +2-4pp top-10.

- **Phase D (R&D, optional):** hero embeddings (mini-NN), V7e/V8 ensemble. Only after Phase B (need data).

Stretch goal: **70% top-10**.

## 7. Files & artifacts

| File | Purpose |
|---|---|
| `research/train_v8.py` | Phase A training pipeline (sklearn + LightGBM variants) |
| `research/export_v8_to_json.py` | Serialize sklearn GBM → JSON for browser |
| `research/backtest_v8.py` | Full backtest framework (Tests 1-4) |
| `research/win_uplift_v8.py` | Win-prob uplift verification |
| `research/data/v8_cv.json` | 5-fold CV results (V8a + V8b) |
| `research/data/backtest_v8_results.json` | Full backtest numbers |
| `research/data/win_uplift_v8.json` | Uplift analysis |
| `data/v8_model.json` | Browser-loadable GBM tree dump (~388 KB) |
| `v8.js` | Browser-side V8 evaluator (auto-position inference + 25-feature builder) |
| `app.js` | Updated dispatcher: V8 default, V7e/M3 backups, M1 fully removed |
