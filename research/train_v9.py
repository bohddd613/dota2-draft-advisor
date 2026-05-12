"""
V9: Phase B model — V8 architecture trained on expanded dataset (+ Phase B2 ideas).

Key Phase B changes over V8:
  - **More data**: ~5000-8000 Divine+ matches (vs V8's 1381)
  - **B2 — loser-team picks**: V8 trained only on winner team's picks ("if winners
    picked X, X was a good choice"). V9 ALSO uses loser-team picks at half weight
    — losers' picks are still rational signal about "what a player would pick in
    this situation", just with reduced positivity since the outcome was negative.
  - **Deeper trees**: max_depth 4 → 5 (more data tolerates deeper trees without overfit)

Features (25, unchanged from V8) — see train_v8.py for full breakdown.

Variants trained:
  - V9a: sklearn GBC on winners only (apples-to-apples vs V8a on expanded data)
  - V9b: sklearn GBC on winners + losers@0.5 (B2)
  - V9c: LightGBM LGBMRanker on winners only (revisit V8b with more data)

Evaluation: 5-fold CV on the same dataset + chronological held-out 20% backtest.
The chronological held-out is the fair shootout against V8 (which was trained on
older data only — none of which leaked into the test set).

Outputs:
  - research/data/v9{a,b,c}_gbm.joblib (or lgb_model.txt for c)
  - research/data/v9{a,b,c}_model.json
  - research/data/v9_cv.json
"""
from __future__ import annotations
import json
import math
import sys
import argparse
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v2 import kfold_split  # noqa: E402
from train_v8 import (  # noqa: E402
    FEATURE_NAMES_V8, hero_features_v8, load_matches, POS_TO_INT,
    evaluate_pick_rec_v8,
)


# ----------------------------- dataset builder ------------------------------

def build_pick_dataset_v9(
    matches, ctx, m2, m4, all_hero_ids,
    neg_per_pos: int = 10, seed: int = 42,
    include_losers: bool = False, loser_weight: float = 0.5,
):
    """
    Like build_pick_dataset_v8 but:
      - optionally includes loser-team picks as positives with reduced weight
      - returns sample weights alongside X, y
    """
    rng = np.random.default_rng(seed)
    X, y, w, groups = [], [], [], []

    def emit_team(true_team, opp_team, group_weight, taken):
        team_ids = [h for h, _ in true_team]
        for true_hero, pos in true_team:
            allies = [(h, p) for h, p in true_team if h != true_hero]
            enemies = opp_team
            eligible = [h for h in all_hero_ids
                        if h not in taken and pos in (m2.eligible.get(h) or [])]
            if true_hero not in eligible:
                eligible.append(true_hero)
            negatives = [h for h in eligible if h != true_hero]
            if not negatives:
                continue
            sample = rng.choice(
                negatives,
                size=min(neg_per_pos, len(negatives)),
                replace=False,
            )
            # Positive (weighted)
            f_pos = hero_features_v8(m4, m2, true_hero, pos, allies, enemies)
            X.append(f_pos); y.append(1.0); w.append(group_weight)
            # Negatives (full weight — they are random across all matches)
            for h in sample:
                f_neg = hero_features_v8(m4, m2, int(h), pos, allies, enemies)
                X.append(f_neg); y.append(0.0); w.append(1.0)
            groups.append(1 + len(sample))

    for m in matches:
        radiant = m["radiant"]
        dire = m["dire"]
        winners = radiant if m["radiant_win"] else dire
        losers = dire if m["radiant_win"] else radiant
        taken = {h for h, _ in radiant} | {h for h, _ in dire}
        emit_team(winners, losers, 1.0, taken)
        if include_losers:
            emit_team(losers, winners, loser_weight, taken)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(w, dtype=np.float32),
        np.array(groups, dtype=np.int32),
    )


# ----------------------------- train ---------------------------------------

def cv_train_sklearn(matches, ctx, m2, m4, all_hero_ids,
                     n_folds: int, neg_per_pos: int,
                     include_losers: bool, loser_weight: float,
                     n_estimators: int, max_depth: int, learning_rate: float,
                     label: str):
    """Sklearn GBC 5-fold CV."""
    from sklearn.ensemble import GradientBoostingClassifier
    print(f"\n[{label}] sklearn GradientBoostingClassifier "
          f"(n={n_estimators}, depth={max_depth}, lr={learning_rate}, "
          f"losers={include_losers}@{loser_weight if include_losers else '-'})")
    fold_results = []
    for fi, (tr_idx, te_idx) in enumerate(kfold_split(len(matches), k=n_folds)):
        tr = [matches[i] for i in tr_idx]; te = [matches[i] for i in te_idx]
        X_tr, y_tr, w_tr, _ = build_pick_dataset_v9(
            tr, ctx, m2, m4, all_hero_ids,
            neg_per_pos=neg_per_pos, seed=42 + fi,
            include_losers=include_losers, loser_weight=loser_weight,
        )
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        def s(F, m=clf): return m.predict_proba(F)[:, 1]
        r = evaluate_pick_rec_v8(s, te, ctx, m2, m4, all_hero_ids)
        fold_results.append(r)
        print(f"  fold{fi}: top10={r['top10']:.4f}, top5={r['top5']:.4f}, "
              f"top1={r['top1']:.4f}, mean={r['mean_rank']:.2f}")
    avg = {k: float(np.mean([r[k] for r in fold_results]))
           for k in ["top1", "top5", "top10", "mean_rank"]}
    print(f"[{label}] CV: top10={avg['top10']:.4f}  top5={avg['top5']:.4f}  "
          f"top1={avg['top1']:.4f}  mean={avg['mean_rank']:.2f}")
    return avg, fold_results


def cv_train_lgbm_ranker(matches, ctx, m2, m4, all_hero_ids,
                         n_folds: int, neg_per_pos: int,
                         n_estimators: int, learning_rate: float, num_leaves: int,
                         label: str):
    """LightGBM LGBMRanker (lambdarank) 5-fold CV."""
    import lightgbm as lgb
    print(f"\n[{label}] LightGBM LGBMRanker (n={n_estimators}, "
          f"leaves={num_leaves}, lr={learning_rate})")
    fold_results = []
    for fi, (tr_idx, te_idx) in enumerate(kfold_split(len(matches), k=n_folds)):
        tr = [matches[i] for i in tr_idx]; te = [matches[i] for i in te_idx]
        X_tr, y_tr, _, g_tr = build_pick_dataset_v9(
            tr, ctx, m2, m4, all_hero_ids,
            neg_per_pos=neg_per_pos, seed=42 + fi,
            include_losers=False,
        )
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=20,
            random_state=42,
            verbosity=-1,
        )
        ranker.fit(X_tr, y_tr.astype(int), group=g_tr)
        def s(F, m=ranker): return m.predict(F)
        r = evaluate_pick_rec_v8(s, te, ctx, m2, m4, all_hero_ids)
        fold_results.append(r)
        print(f"  fold{fi}: top10={r['top10']:.4f}, top5={r['top5']:.4f}, "
              f"top1={r['top1']:.4f}, mean={r['mean_rank']:.2f}")
    avg = {k: float(np.mean([r[k] for r in fold_results]))
           for k in ["top1", "top5", "top10", "mean_rank"]}
    print(f"[{label}] CV: top10={avg['top10']:.4f}  top5={avg['top5']:.4f}  "
          f"top1={avg['top1']:.4f}  mean={avg['mean_rank']:.2f}")
    return avg, fold_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["v9a", "v9b", "v9c"],
        choices=["v9a", "v9b", "v9c"],
        help="Which variants to train",
    )
    ap.add_argument("--final-only", action="store_true",
                    help="Skip CV — fit final model on all data only")
    args = ap.parse_args()

    matches = load_matches(Path(args.matches))
    print(f"[v9] loaded {len(matches)} matches with valid positions")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    report = {
        "n_matches": len(matches),
        "feature_names": FEATURE_NAMES_V8,
        "variants": {},
    }

    # ----- V9a: sklearn GBC, winners only, depth=5 -----
    if "v9a" in args.variants:
        if not args.final_only:
            avg, folds = cv_train_sklearn(
                matches, ctx, m2, m4, all_hero_ids,
                n_folds=args.folds, neg_per_pos=args.neg_per_pos,
                include_losers=False, loser_weight=0.0,
                n_estimators=400, max_depth=5, learning_rate=0.05,
                label="v9a",
            )
            report["variants"]["v9a"] = {"cv": avg, "folds": folds}

        # Final fit
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib
        X, y, w, _ = build_pick_dataset_v9(
            matches, ctx, m2, m4, all_hero_ids,
            neg_per_pos=args.neg_per_pos, seed=42,
            include_losers=False, loser_weight=0.0,
        )
        print(f"[v9a] fit on full dataset (X={X.shape})")
        clf = GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42,
        )
        clf.fit(X, y, sample_weight=w)
        joblib.dump(clf, DATA_DIR / "v9a_gbm.joblib")
        print(f"[v9a] saved → research/data/v9a_gbm.joblib")

    # ----- V9b: sklearn GBC, winners + losers@0.5 (B2 idea) -----
    if "v9b" in args.variants:
        if not args.final_only:
            avg, folds = cv_train_sklearn(
                matches, ctx, m2, m4, all_hero_ids,
                n_folds=args.folds, neg_per_pos=args.neg_per_pos,
                include_losers=True, loser_weight=0.5,
                n_estimators=400, max_depth=5, learning_rate=0.05,
                label="v9b",
            )
            report["variants"]["v9b"] = {"cv": avg, "folds": folds}

        from sklearn.ensemble import GradientBoostingClassifier
        import joblib
        X, y, w, _ = build_pick_dataset_v9(
            matches, ctx, m2, m4, all_hero_ids,
            neg_per_pos=args.neg_per_pos, seed=42,
            include_losers=True, loser_weight=0.5,
        )
        print(f"[v9b] fit on full dataset (X={X.shape}, winners+losers@0.5)")
        clf = GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42,
        )
        clf.fit(X, y, sample_weight=w)
        joblib.dump(clf, DATA_DIR / "v9b_gbm.joblib")
        print(f"[v9b] saved → research/data/v9b_gbm.joblib")

    # ----- V9c: LightGBM LGBMRanker (lambdarank) on bigger data -----
    if "v9c" in args.variants:
        if not args.final_only:
            avg, folds = cv_train_lgbm_ranker(
                matches, ctx, m2, m4, all_hero_ids,
                n_folds=args.folds, neg_per_pos=args.neg_per_pos,
                n_estimators=400, learning_rate=0.05, num_leaves=63,
                label="v9c",
            )
            report["variants"]["v9c"] = {"cv": avg, "folds": folds}

        import lightgbm as lgb
        X, y, _, g = build_pick_dataset_v9(
            matches, ctx, m2, m4, all_hero_ids,
            neg_per_pos=args.neg_per_pos, seed=42,
            include_losers=False,
        )
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=400, learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            random_state=42, verbosity=-1,
        )
        ranker.fit(X, y.astype(int), group=g)
        ranker.booster_.save_model(str(DATA_DIR / "v9c_ranker.txt"))
        (DATA_DIR / "v9c_model.json").write_text(
            json.dumps(ranker.booster_.dump_model())
        )
        print(f"[v9c] saved → research/data/v9c_ranker.txt + v9c_model.json")

    (DATA_DIR / "v9_cv.json").write_text(json.dumps(report, indent=2))
    print(f"\n[v9] CV report → research/data/v9_cv.json")


if __name__ == "__main__":
    main()
