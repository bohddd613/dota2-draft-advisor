"""
Phase B fair shootout: V9 (trained on expanded data) vs V8 (existing weights)
vs V7e on a held-out test set drawn only from the NEW Phase B matches.

Why this is fair:
  - V8 was trained on the original 1381-match dataset
  - Phase B added ~6000 NEW matches via fresh STRATZ enrichment
  - We hold out 20% of the NEW matches (matches V8 has never seen) as test
  - V9 is trained on (old 1446 + 80% of new) — V9 also doesn't see the test set
  - Both models evaluated on the same held-out tail → apples-to-apples

Outputs research/data/backtest_v9_results.json with top-1/5/10, mean rank,
and per-position breakdown for V9a, V9b, V8, V7e.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import (  # noqa: E402
    FEATURE_NAMES_V8, hero_features_v8, load_matches, evaluate_pick_rec_v8,
    POS_TO_INT,
)
# V7e helper (5 features) — uses train_v2's hero_features
from train_v2 import hero_features as hero_features_v7e  # noqa: E402


def split_chronological(matches, frac_test=0.2, seed=42):
    """Split matches by match_id (older→newer). Returns (train_idx, test_idx)."""
    sorted_by_id = sorted(range(len(matches)), key=lambda i: matches[i]["match_id"])
    n_test = int(round(frac_test * len(matches)))
    test_idx = sorted_by_id[-n_test:]
    train_idx = sorted_by_id[:-n_test]
    return train_idx, test_idx


def evaluate_per_position(scorer, matches, ctx, m2, m4, all_hero_ids,
                          feat_builder=None, samples_per_match=2):
    """Evaluate with per-position breakdown.

    feat_builder(h, pos, allies, enemies) → np.ndarray. If None, uses V8's.
    """
    rng = np.random.default_rng(42)
    ranks_by_pos = defaultdict(list)
    top1_by_pos = defaultdict(int)
    top5_by_pos = defaultdict(int)
    top10_by_pos = defaultdict(int)
    count_by_pos = defaultdict(int)
    all_ranks = []
    top1 = top5 = top10 = 0

    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = rng.choice(
            len(winners),
            size=min(samples_per_match, len(winners)),
            replace=False,
        )
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [(h, p) for h, p in winners if h != true_hero]
            enemies = losers
            taken = set(team_ids) | set(enemy_ids)
            eligible = [h for h in all_hero_ids
                        if h not in taken and target_pos in (m2.eligible.get(h) or [])]
            if true_hero not in eligible:
                eligible.append(true_hero)
            if feat_builder is None:
                feats = np.vstack([
                    hero_features_v8(m4, m2, h, target_pos, allies, enemies)
                    for h in eligible
                ])
            else:
                feats = np.vstack([
                    feat_builder(h, target_pos, allies, enemies)
                    for h in eligible
                ])
            scores = scorer(feats)
            order = np.argsort(-scores)
            ranked = [eligible[i] for i in order]
            try:
                rank = ranked.index(true_hero) + 1
            except ValueError:
                rank = len(ranked)
            all_ranks.append(rank)
            ranks_by_pos[target_pos].append(rank)
            count_by_pos[target_pos] += 1
            if rank == 1:
                top1 += 1; top1_by_pos[target_pos] += 1
            if rank <= 5:
                top5 += 1; top5_by_pos[target_pos] += 1
            if rank <= 10:
                top10 += 1; top10_by_pos[target_pos] += 1

    n = len(all_ranks)
    per_pos = {}
    for p in sorted(count_by_pos.keys()):
        c = count_by_pos[p]
        per_pos[f"pos{p}"] = {
            "n": c,
            "top1": top1_by_pos[p] / c,
            "top5": top5_by_pos[p] / c,
            "top10": top10_by_pos[p] / c,
            "mean_rank": float(np.mean(ranks_by_pos[p])),
        }
    return {
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "top10": top10 / n,
        "mean_rank": float(np.mean(all_ranks)),
        "per_position": per_pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--frac-test", type=float, default=0.2)
    ap.add_argument("--samples-per-match", type=int, default=3)
    args = ap.parse_args()

    matches = load_matches(Path(args.matches))
    print(f"[bt9] loaded {len(matches)} matches")

    # Chronological split — newest 20% as test
    train_idx, test_idx = split_chronological(matches, frac_test=args.frac_test)
    test = [matches[i] for i in test_idx]
    train = [matches[i] for i in train_idx]
    print(f"[bt9] train={len(train)} test={len(test)} "
          f"(test match_id range: {min(m['match_id'] for m in test)} → {max(m['match_id'] for m in test)})")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    results = {
        "n_train": len(train),
        "n_test": len(test),
        "models": {},
    }

    # ----- V9a -----
    try:
        v9a = joblib.load(DATA_DIR / "v9a_gbm.joblib")
        print("\n[bt9] V9a on held-out…")
        r = evaluate_per_position(
            lambda F: v9a.predict_proba(F)[:, 1],
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v9a"] = r
        print(f"  V9a: top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt9] v9a_gbm.joblib not found — skip")

    # ----- V9b -----
    try:
        v9b = joblib.load(DATA_DIR / "v9b_gbm.joblib")
        print("\n[bt9] V9b on held-out…")
        r = evaluate_per_position(
            lambda F: v9b.predict_proba(F)[:, 1],
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v9b"] = r
        print(f"  V9b: top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt9] v9b_gbm.joblib not found — skip")

    # ----- V9d (d=4, n=400 on full 6282) -----
    try:
        v9d = joblib.load(DATA_DIR / "v9d_gbm.joblib")
        print("\n[bt9] V9d (d=4 n=400) on held-out…")
        r = evaluate_per_position(
            lambda F: v9d.predict_proba(F)[:, 1],
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v9d"] = r
        print(f"  V9d: top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt9] v9d_gbm.joblib not found — skip")

    # ----- V9c (LightGBM Ranker) -----
    try:
        import lightgbm as lgb
        v9c_booster = lgb.Booster(model_file=str(DATA_DIR / "v9c_ranker.txt"))
        print("\n[bt9] V9c (LightGBM Ranker) on held-out…")
        r = evaluate_per_position(
            lambda F: v9c_booster.predict(F),
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v9c"] = r
        print(f"  V9c: top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt9] v9c_ranker.txt not found — skip")
    except Exception as e:
        print(f"[bt9] V9c eval failed: {e}")

    # ----- V8 (existing) -----
    try:
        v8 = joblib.load(DATA_DIR / "v8a_gbm.joblib")
        print("\n[bt9] V8 (existing) on held-out…")
        r = evaluate_per_position(
            lambda F: v8.predict_proba(F)[:, 1],
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=None,
            samples_per_match=args.samples_per_match,
        )
        results["models"]["v8"] = r
        print(f"  V8 : top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt9] v8a_gbm.joblib not found — skip")

    # ----- V7e (existing) — use its 5-feature builder -----
    try:
        v7e = joblib.load(DATA_DIR / "v7e_gbm.joblib")
        print("\n[bt9] V7e (existing) on held-out…")
        # V7e takes allies/enemies as list[int] (just IDs)
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
        print(f"  V7e: top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}  n={r['n']}")
    except FileNotFoundError:
        print("[bt9] v7e_gbm.joblib not found — skip")
    except Exception as e:
        print(f"[bt9] V7e eval failed: {e}")

    # ----- Summary table -----
    print("\n=== Phase B held-out shootout ===")
    print(f"{'Model':<6}  {'top1':>7}  {'top5':>7}  {'top10':>7}  {'mean':>7}  n")
    for label in ["v7e", "v8", "v9a", "v9b", "v9c", "v9d"]:
        if label in results["models"]:
            r = results["models"][label]
            print(f"{label:<6}  {r['top1']*100:>6.2f}%  {r['top5']*100:>6.2f}%  "
                  f"{r['top10']*100:>6.2f}%  {r['mean_rank']:>7.2f}  {r['n']}")

    (DATA_DIR / "backtest_v9_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[bt9] saved → research/data/backtest_v9_results.json")


if __name__ == "__main__":
    main()
