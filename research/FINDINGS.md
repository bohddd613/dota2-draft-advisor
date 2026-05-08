# Algorithm Research & Backtest Findings

## TL;DR

The new **V7e** model (Gradient Boosting trained directly for pick-recommendation) is **2× better than the production M1 model** on the metric users actually care about (top-10 recall) and also wins win-prediction by 7 points.

| Task | M1 (current prod) | **V7e (new)** | Improvement |
|---|---|---|---|
| Pick top-10 recall | 24.1% | **48.2%** | **+24 pp (2.0×)** |
| Pick top-5 recall | 6.1% | **32.8%** | **+27 pp (5.4×)** |
| Pick top-1 recall | 0.2% | **9.3%** | **+9 pp (52×)** |
| Pick mean rank (of true hero) | 28.5 / ~52 candidates | **16.8** | −41% |
| Win-pred accuracy | 50.0% | **57.1%** | +7 pp |
| Win-pred log-loss | 0.703 | **0.682** | −3% |

## What I tried

### Data sources
- **OpenDota** `/api/heroStats`, `/api/heroes/<id>/matchups`, `/api/publicMatches` — pure pubs, free, ~10–60 sec rate-limit
- **STRATZ GraphQL** (with API token) — Divine+ ranked, true `with`-synergy, true position labels per match. Way better signal than OpenDota.
- Ground-truth dataset: **1381 Divine+ ranked matches** with full position labels for all 10 players in each match (STRATZ `match.players[].position`)

### Models tested

| ID | Description | Key Idea |
|---|---|---|
| M0 | Role-weight matrix (old V0 baseline) | Naive role-mapping heuristic |
| **M1** | Curated `HERO_POSITIONS` map (current production) | Hand-curated eligibility, role-diversity synergy |
| M2 | STRATZ data-driven positions | Auto eligibility from STRATZ `matchCount` |
| M3 | M2 + true `with`-synergy from STRATZ | Replaces role-diversity heuristic |
| M4 | M3 + role-gap detection bonus | Bonus for filling missing key roles |
| M5 | Logistic regression on engineered features (default weights) | Linear in [base_wr, pos_fit, with_syn, vs_adv, role_gap] |
| M5* | Logistic regression (weights trained from data) | Same model, MLE-fitted to win outcomes |
| V6 | HistGradientBoosting on win-prediction | Nonlinear, trained on team-strength deltas |
| **V7e** | GradientBoosting trained for **pick-recommendation** | Directly optimizes the ranking task we care about |

### Key insights

1. **The training task matters more than the algorithm.** V6 (GBM) trained for win-prediction performs WORSE than M5 logistic on pick-recommendation. V7e (same algorithm class, trained for pick-rec) crushes everything. Always train for the task you're going to evaluate.

2. **Logistic regression is too weak for pick-rec.** V7 with logistic regression only got 24.8% top-10 (similar to M1). The same features fed into a GBM jumped to **52.7% top-10 in CV** (48.2% in final eval). Tree models capture feature interactions that linear models miss.

3. **Counter-advantage is the dominant signal.** Trained M5* coefficients: `vs_adv=+2.49` is by far the largest. Counter-pick info is more predictive than overall hero winrate.

4. **True position labels matter for backtesting, not for ranking quality.** The greedy position assignment that M1 used in the older backtest was already near-optimal at the team-strength level; the real noise was in pick-recommendation evaluation, where knowing the true position of the held-out hero is essential.

## How V7e is built

**Features (5 per hero/position/state):**
1. `base_wr` — Bayesian-shrunk per-position winrate from STRATZ Divine+ pool (subtract 0.5)
2. `pos_fit` — STRATZ-derived eligibility decay (1.0 / 0.7 / 0.5 / 0.35 / 0.2 by rank)
3. `with_syn` — average `with`-synergy with current allies (subtract 0.5)
4. `vs_adv` — average matchup advantage vs current enemies (subtract 0.5)
5. `role_gap` — fraction of missing key roles (Initiator, Disabler, Support, Durable, Nuker) the hero fills

**Training task:**
For each match, for each hero in the winning team, the (state, true hero) is a positive sample. Generate 10 negative samples by picking other eligible heroes at the same position that were NOT picked. Binary classification: P(this hero is the real pick | state).

**Algorithm:**
- sklearn `GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)`
- Trained on 1381 matches, ~70k samples
- 5-fold cross-validation on **matches** (not samples) to prevent leakage
- CV top-10 = **52.7% ± 1.9%** — robust across folds

**Browser deployment:**
- Trees exported to JSON (~250 KB) via custom serializer
- Pure-JS tree walker in `v7e.js` produces predictions identical to sklearn (diff = 0)
- All STRATZ data bundled as static JSON — no backend required, no API keys leak to client

## Files in this PR

```
research/
  models.py                       — M0..M5 model definitions
  train_v2.py                     — V5/V6 training (win-prediction)
  train_v7.py                     — V7/V7g training (pick-recommendation)
  train_v7_exportable.py          — V7e training (sklearn GBC, exportable)
  export_gbm_to_json.py           — Tree → JSON exporter
  backtest.py                     — Original backtest (greedy position assignment)
  backtest_v2.py                  — Improved backtest (TRUE position labels)
  build_frontend_data.py          — Compact data bundle builder for browser
  stratz_fetch.py                 — STRATZ position-stats and matchup fetch
  stratz_match_fetch.py           — STRATZ per-match enrichment (positions per player)
  opendota_fetch.py               — OpenDota /publicMatches fetch
  data/                           — All collected & trained artifacts (cached)
data/                             — Frontend-loaded JSON bundles
v7e.js                            — Browser-side V7e implementation
```

## Limitations & future work

- **Pro/Captain's Mode logic not modeled.** Ban-phase ordering is a separate problem.
- **No item/skill-build context.** Drafting is just one factor in match outcomes.
- **MMR brackets not fully separable.** Trained on Divine+ but evaluated on the same pool.
- **No live meta updates.** Snapshot of current 7.x meta — would need scheduled retraining as patches change hero strength.
- **Hand-engineered features still dominate.** Could try a small MLP over hero-id embeddings (would need ~10× more data).
