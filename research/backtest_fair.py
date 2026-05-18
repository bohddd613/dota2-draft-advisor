"""
Unified honest backtest of V7e, V8 fair, V9c fair, V10c fair.

All four models evaluated on the same 1256 newest matches (truly held-out —
never seen during training). Compare top-1 / top-5 / top-10 / mean rank.

This replaces the old backtests which used models with train-test leak.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import joblib
import lightgbm as lgb

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import hero_features_v8, load_matches  # noqa: E402
from train_v10 import hero_features_v10  # noqa: E402
from train_v2 import hero_features as hero_features_v7e  # noqa: E402
from backtest_v9 import split_chronological, evaluate_per_position  # noqa: E402


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"[bt-fair] loaded {len(matches)} matches")
    _, test_idx = split_chronological(matches, frac_test=0.2)
    test = [matches[i] for i in test_idx]
    print(f"[bt-fair] test={len(test)} (truly held-out from training)")

    ctx = Context(); m2 = M2_DataPositions(ctx); m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    results = {"n_test": len(test), "models": {}}

    # V7e (already fair — small training set, evaluated on Phase B newest)
    v7e = joblib.load(DATA_DIR / "v7e_gbm.joblib")
    def v7e_builder(h, p, a, e):
        return hero_features_v7e(m4, m2, h, p, [x for x, _ in a], [x for x, _ in e])
    r = evaluate_per_position(
        lambda F: v7e.predict_proba(F)[:, 1],
        test, ctx, m2, m4, all_hero_ids,
        feat_builder=v7e_builder, samples_per_match=3,
    )
    results["models"]["v7e"] = r
    print(f"  V7e fair         top10={r['top10']*100:5.1f}%  top5={r['top5']*100:5.1f}%  "
          f"top1={r['top1']*100:5.1f}%  mean={r['mean_rank']:5.2f}")

    # V8 fair
    v8 = joblib.load(DATA_DIR / "v8_fair_gbm.joblib")
    r = evaluate_per_position(
        lambda F: v8.predict_proba(F)[:, 1],
        test, ctx, m2, m4, all_hero_ids,
        feat_builder=None, samples_per_match=3,
    )
    results["models"]["v8_fair"] = r
    print(f"  V8 fair          top10={r['top10']*100:5.1f}%  top5={r['top5']*100:5.1f}%  "
          f"top1={r['top1']*100:5.1f}%  mean={r['mean_rank']:5.2f}")

    # V9c fair (lambdarank, 25 features)
    v9c = lgb.Booster(model_file=str(DATA_DIR / "v9c_fair_ranker.txt"))
    r = evaluate_per_position(
        lambda F: v9c.predict(F),
        test, ctx, m2, m4, all_hero_ids,
        feat_builder=None, samples_per_match=3,
    )
    results["models"]["v9c_fair"] = r
    print(f"  V9c fair (rank)  top10={r['top10']*100:5.1f}%  top5={r['top5']*100:5.1f}%  "
          f"top1={r['top1']*100:5.1f}%  mean={r['mean_rank']:5.2f}")

    # V10c fair (lambdarank, 39 features)
    v10c = lgb.Booster(model_file=str(DATA_DIR / "v10c_fair_ranker.txt"))
    v10_builder = lambda h, p, a, e: hero_features_v10(m4, m2, h, p, a, e)
    r = evaluate_per_position(
        lambda F: v10c.predict(F),
        test, ctx, m2, m4, all_hero_ids,
        feat_builder=v10_builder, samples_per_match=3,
    )
    results["models"]["v10c_fair"] = r
    print(f"  V10c fair (+comp) top10={r['top10']*100:5.1f}%  top5={r['top5']*100:5.1f}% "
          f"top1={r['top1']*100:5.1f}%  mean={r['mean_rank']:5.2f}")

    # For posterity: ORIGINAL (leaked) V9c and V8 — they reported inflated scores
    print()
    print("--- ARTIFACTS (kept for transparency; do NOT cite these as model performance) ---")
    try:
        v9c_old = lgb.Booster(model_file=str(DATA_DIR / "v9c_ranker.txt"))
        r = evaluate_per_position(
            lambda F: v9c_old.predict(F),
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None, samples_per_match=3,
        )
        results["models"]["v9c_LEAKED_DO_NOT_USE"] = r
        print(f"  V9c LEAKED (orig.) top10={r['top10']*100:5.1f}%  (was claimed as 74% — INFLATED due to train-test leak)")
    except Exception:
        pass
    try:
        v8_old = joblib.load(DATA_DIR / "v8a_gbm.joblib")
        r = evaluate_per_position(
            lambda F: v8_old.predict_proba(F)[:, 1],
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None, samples_per_match=3,
        )
        results["models"]["v8_LEAKED_DO_NOT_USE"] = r
        print(f"  V8  LEAKED (orig.) top10={r['top10']*100:5.1f}%  (was claimed as 61.9% — INFLATED due to Phase A training-set overlap)")
    except Exception:
        pass

    out = DATA_DIR / "backtest_fair_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[bt-fair] saved → {out}")


if __name__ == "__main__":
    main()
