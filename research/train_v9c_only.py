"""V9c only — LightGBM Ranker fit on full 6282 matches."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import load_matches  # noqa: E402
from train_v9 import build_pick_dataset_v9  # noqa: E402
import lightgbm as lgb


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"loaded {len(matches)} matches")
    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    X, y, w, groups = build_pick_dataset_v9(
        matches, ctx, m2, m4, all_hero_ids,
        neg_per_pos=10, seed=42, include_losers=False,
    )
    print(f"[v9c] X.shape={X.shape}  groups.shape={groups.shape}  groups[0]={groups[0]}")
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
