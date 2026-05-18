"""
Phase C1 backtest: V10c (V9 + team composition) vs V9c vs V8 on the same
chronological hold-out 20% (newest 1256 matches).

V10c was trained on the oldest 80% (5026) — same split as V9c. The hold-out
is identical, so V10c vs V9c is apples-to-apples.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import (  # noqa: E402
    hero_features_v8, load_matches, POS_TO_INT,
)
from train_v10 import hero_features_v10  # noqa: E402
from backtest_v9 import split_chronological, evaluate_per_position  # noqa: E402
from train_v2 import hero_features as hero_features_v7e  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--frac-test", type=float, default=0.2)
    ap.add_argument("--samples-per-match", type=int, default=3)
    args = ap.parse_args()

    matches = load_matches(Path(args.matches))
    print(f"[bt10] loaded {len(matches)} matches")
    train_idx, test_idx = split_chronological(matches, frac_test=args.frac_test)
    test = [matches[i] for i in test_idx]
    train = [matches[i] for i in train_idx]
    print(f"[bt10] train={len(train)} test={len(test)} "
          f"(test newest match_id: {test[-1]['match_id']})")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    results = {
        "n_train": len(train),
        "n_test": len(test),
        "models": {},
    }

    # V10c (LightGBM Ranker, 39 features)
    import lightgbm as lgb
    try:
        v10c = lgb.Booster(model_file=str(DATA_DIR / "v10c_ranker.txt"))
        print("\n[bt10] V10c (LightGBM Ranker, 39 feat) on held-out…")
        v10_builder = lambda h, pos, allies, enemies: hero_features_v10(m4, m2, h, pos, allies, enemies)
        r = evaluate_per_position(
            lambda F: v10c.predict(F),
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=v10_builder,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v10c"] = r
        print(f"  V10c: top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt10] v10c_ranker.txt not found — skip")

    # V9c (LightGBM Ranker, 25 features)
    try:
        v9c = lgb.Booster(model_file=str(DATA_DIR / "v9c_ranker.txt"))
        print("\n[bt10] V9c (LightGBM Ranker, 25 feat) on held-out…")
        r = evaluate_per_position(
            lambda F: v9c.predict(F),
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v9c"] = r
        print(f"  V9c : top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt10] v9c_ranker.txt not found — skip")

    # V8 (sklearn GBC, 25 features)
    try:
        v8 = joblib.load(DATA_DIR / "v8a_gbm.joblib")
        print("\n[bt10] V8 on held-out…")
        r = evaluate_per_position(
            lambda F: v8.predict_proba(F)[:, 1],
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v8"] = r
        print(f"  V8  : top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt10] v8a_gbm.joblib not found — skip")

    # V7e
    try:
        v7e = joblib.load(DATA_DIR / "v7e_gbm.joblib")
        print("\n[bt10] V7e on held-out…")
        def v7e_features(h, pos, allies, enemies):
            ally_ids = [hid for hid, _ in allies]
            enemy_ids = [hid for hid, _ in enemies]
            return hero_features_v7e(m4, m2, h, pos, ally_ids, enemy_ids)
        r = evaluate_per_position(
            lambda F: v7e.predict_proba(F)[:, 1],
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=v7e_features,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v7e"] = r
        print(f"  V7e : top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt10] v7e_gbm.joblib not found — skip")
    except Exception as e:
        print(f"[bt10] V7e eval failed: {e}")

    out_path = DATA_DIR / "backtest_v10_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[bt10] saved → {out_path}")

    # Summary table
    print("\n=== Summary (chronological held-out 20%) ===")
    print(f"{'model':<6} {'top1':>7} {'top5':>7} {'top10':>7} {'mean':>7}")
    for name in ["v10c", "v9c", "v8", "v7e"]:
        if name in results["models"]:
            r = results["models"][name]
            print(f"{name:<6} {r['top1']*100:>6.1f}% {r['top5']*100:>6.1f}% "
                  f"{r['top10']*100:>6.1f}% {r['mean_rank']:>7.2f}")


if __name__ == "__main__":
    main()
