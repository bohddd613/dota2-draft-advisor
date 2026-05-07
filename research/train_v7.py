"""
V7: Pick-prediction model trained DIRECTLY for the pick-recommendation task.

For each (state, true_picked_hero) tuple from training matches, we generate
negative samples: 10 eligible heroes that were NOT picked. Then train a
binary classifier where positive = "this is the actual pick" and negative =
"this was not picked but was eligible".

Outputs:
  - research/data/v7_lr_weights.json — logistic regression weights
  - research/data/v7_gbm.joblib — gradient boosting model

Both are evaluated by 5-fold CV.
"""
from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap, M1_Curated, HERO_POSITIONS_M1, POSITION_KEYS  # noqa: E402
from train_v2 import hero_features, FEATURE_NAMES, sigmoid, kfold_split  # noqa: E402

POS_TO_INT = {f"POSITION_{i}": i for i in range(1, 6)}


def load_matches(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    out = []
    for m in raw:
        rad, dire = [], []
        for p in m.get("players", []):
            pos = POS_TO_INT.get(p["position"])
            if pos is None: continue
            tup = (p["hero_id"], pos)
            (rad if p["is_radiant"] else dire).append(tup)
        if len(rad) != 5 or len(dire) != 5:
            continue
        out.append({"match_id": m["match_id"], "radiant_win": m["radiant_win"], "radiant": rad, "dire": dire})
    return out


def build_pick_dataset(matches, ctx, m2, m4, all_hero_ids, neg_per_pos: int = 10, seed: int = 42):
    """For each (team, hero, position), positive sample for that hero,
    negatives sampled from other eligible heroes at the same position."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    used_count = 0
    for m in matches:
        # Use winning team picks (these are 'good' picks)
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        for true_hero, pos in winners:
            allies = [h for h in team_ids if h != true_hero]
            taken = set(team_ids) | set(enemy_ids)
            # Eligible candidates at this position (using M2 data-driven eligibility)
            eligible = [h for h in all_hero_ids if h not in taken and pos in (m2.eligible.get(h) or [])]
            if true_hero not in eligible:
                eligible.append(true_hero)
            # Positive
            f = hero_features(m4, m2, true_hero, pos, allies, enemy_ids)
            X.append(f); y.append(1.0)
            # Negatives
            negatives = [h for h in eligible if h != true_hero]
            if not negatives:
                continue
            sample = rng.choice(negatives, size=min(neg_per_pos, len(negatives)), replace=False)
            for h in sample:
                f = hero_features(m4, m2, int(h), pos, allies, enemy_ids)
                X.append(f); y.append(0.0)
            used_count += 1
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), used_count


def train_lr(X, y, l2: float = 0.01):
    from scipy.optimize import minimize
    n_features = X.shape[1]

    def loss(w):
        b = w[0]; coef = w[1:]
        z = b + X @ coef
        p = 1 / (1 + np.exp(-z))
        eps = 1e-12
        nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return nll + l2 * np.sum(coef**2)

    w0 = np.zeros(n_features + 1)
    res = minimize(loss, w0, method="L-BFGS-B")
    return float(res.x[0]), res.x[1:].astype(float)


def evaluate_pick_rec(scorer, matches, ctx, m2, m4, all_hero_ids, samples_per_match=2):
    """Use the trained scorer (callable: scorer(features) -> score) to rank candidates."""
    rng = np.random.default_rng(42)
    ranks = []
    top1 = top5 = top10 = 0
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = rng.choice(len(winners), size=min(samples_per_match, len(winners)), replace=False)
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [h for h in team_ids if h != true_hero]
            taken = set(team_ids) | set(enemy_ids)
            eligible = [h for h in all_hero_ids if h not in taken and target_pos in (m2.eligible.get(h) or [])]
            if true_hero not in eligible:
                eligible.append(true_hero)
            scored = []
            for h in eligible:
                f = hero_features(m4, m2, h, target_pos, allies, enemy_ids)
                scored.append((h, scorer(f)))
            scored.sort(key=lambda kv: kv[1], reverse=True)
            rank = next((i + 1 for i, (h, _) in enumerate(scored) if h == true_hero), len(scored))
            ranks.append(rank)
            if rank == 1: top1 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1
    n = len(ranks)
    return {
        "n": n,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "median_rank": float(np.median(ranks)) if ranks else 0.0,
        "top1": top1 / n if n else 0.0,
        "top5": top5 / n if n else 0.0,
        "top10": top10 / n if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-per-pos", type=int, default=10)
    args = ap.parse_args()

    matches = load_matches(Path(args.matches))
    if args.limit:
        matches = matches[: args.limit]
    print(f"[v7] loaded {len(matches)} matches with TRUE position labels")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    # Build train/test split FIRST on matches (not on samples) — avoid leakage
    print(f"[v7] {args.folds}-fold CV (split on matches, not samples)...")
    accs = []
    pick_top10s = []
    last_intercept, last_coef = None, None

    for fi, (tr_idx, te_idx) in enumerate(kfold_split(len(matches), k=args.folds)):
        tr_matches = [matches[i] for i in tr_idx]
        te_matches = [matches[i] for i in te_idx]

        # Build training set
        X_tr, y_tr, n_used = build_pick_dataset(
            tr_matches, ctx, m2, m4, all_hero_ids, neg_per_pos=args.neg_per_pos, seed=42 + fi
        )
        print(f"  fold{fi}: train {X_tr.shape[0]} samples ({int(y_tr.sum())} positives)")

        # Train LR
        intercept, coef = train_lr(X_tr, y_tr)

        # Evaluate on test matches via pick-recommendation rank
        def scorer(f):
            return float(intercept + np.dot(coef, f))

        pick_metrics = evaluate_pick_rec(scorer, te_matches, ctx, m2, m4, all_hero_ids, samples_per_match=2)
        pick_top10s.append(pick_metrics["top10"])
        print(f"    LR weights: int={intercept:+.3f} {dict(zip(FEATURE_NAMES, [round(c, 3) for c in coef]))}")
        print(f"    pick_rec: mean={pick_metrics['mean_rank']:.2f} top1={pick_metrics['top1']:.4f} top5={pick_metrics['top5']:.4f} top10={pick_metrics['top10']:.4f}")
        last_intercept = intercept
        last_coef = coef

    avg_top10 = float(np.mean(pick_top10s))
    print(f"\n[v7] CV average top10 = {avg_top10:.4f} ± {float(np.std(pick_top10s)):.4f}")

    # Refit on full data
    X, y, _ = build_pick_dataset(matches, ctx, m2, m4, all_hero_ids, neg_per_pos=args.neg_per_pos)
    intercept, coef = train_lr(X, y)
    out = {
        "intercept": float(intercept),
        **dict(zip(FEATURE_NAMES, [float(c) for c in coef])),
        "n_train_samples": int(X.shape[0]),
        "n_train_matches": len(matches),
        "feature_names": FEATURE_NAMES,
        "cv_top10_mean": avg_top10,
        "cv_top10_std": float(np.std(pick_top10s)),
    }
    (DATA_DIR / "v7_lr_weights.json").write_text(json.dumps(out, indent=2))
    print(f"[v7] saved -> v7_lr_weights.json")
    print(f"[v7] full-data weights: intercept={intercept:+.3f} {dict(zip(FEATURE_NAMES, [round(float(c), 3) for c in coef]))}")

    # Also try GBM with same setup
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        print(f"\n[v7] training GBM (V7g)...")
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=42)
        gbm.fit(X, y)
        joblib.dump(gbm, DATA_DIR / "v7_gbm.joblib")

        # CV evaluation
        gbm_top10s = []
        for fi, (tr_idx, te_idx) in enumerate(kfold_split(len(matches), k=args.folds)):
            tr_matches = [matches[i] for i in tr_idx]
            te_matches = [matches[i] for i in te_idx]
            X_tr, y_tr, _ = build_pick_dataset(tr_matches, ctx, m2, m4, all_hero_ids, neg_per_pos=args.neg_per_pos, seed=42 + fi)
            gbm_fold = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, random_state=42)
            gbm_fold.fit(X_tr, y_tr)
            def scorer(f):
                return float(gbm_fold.predict_proba(f.reshape(1, -1))[0, 1])
            r = evaluate_pick_rec(scorer, te_matches, ctx, m2, m4, all_hero_ids, samples_per_match=2)
            gbm_top10s.append(r["top10"])
            print(f"  fold{fi}: top10={r['top10']:.4f}, mean_rank={r['mean_rank']:.2f}")
        print(f"[v7g] CV top10 = {float(np.mean(gbm_top10s)):.4f} ± {float(np.std(gbm_top10s)):.4f}")
    except Exception as e:
        print(f"[v7g] failed: {e}")


if __name__ == "__main__":
    main()
