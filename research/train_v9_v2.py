"""
V9 second attempt:
  - v9d_gbm: sklearn GBC d=4, n=400, lr=0.05 (V8 architecture + 4.6× data)
  - v9c_ranker: LightGBM LGBMRanker pairwise

Both on full 6282 matches, final fit only (no CV).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import load_matches  # noqa: E402
from train_v9 import build_pick_dataset_v9  # noqa: E402


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"loaded {len(matches)} matches")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    # === V9d: sklearn GBC d=4, n=400 ===
    from sklearn.ensemble import GradientBoostingClassifier
    X, y, w, groups = build_pick_dataset_v9(
        matches, ctx, m2, m4, all_hero_ids,
        neg_per_pos=10, seed=42, include_losers=False,
    )
    print(f"[v9d] fit on {X.shape} with d=4 n=400 lr=0.05")
    clf = GradientBoostingClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42,
    )
    clf.fit(X, y, sample_weight=w)
    joblib.dump(clf, DATA_DIR / "v9d_gbm.joblib")
    print(f"[v9d] saved → research/data/v9d_gbm.joblib")

    # === V9c: LightGBM LGBMRanker ===
    try:
        import lightgbm as lgb
    except ImportError:
        print("[v9c] lightgbm not installed; skipping")
        return

    X, y, w, groups = build_pick_dataset_v9(
        matches, ctx, m2, m4, all_hero_ids,
        neg_per_pos=10, seed=42, include_losers=False,
    )
    # `groups` is already an array of group SIZES (one entry per pick decision,
    # each = 1 + neg_per_pos). LightGBM expects these as the `group` argument.
    print(f"[v9c] fit ranker on {X.shape} with n_groups={len(groups)} group_size={groups[0]}")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=400, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbosity=-1,
    )
    ranker.fit(X, y.astype(int), group=groups)
    ranker.booster_.save_model(str(DATA_DIR / "v9c_ranker.txt"))
    print(f"[v9c] saved → research/data/v9c_ranker.txt")


if __name__ == "__main__":
    main()
