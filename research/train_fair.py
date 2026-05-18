"""
Fair retraining of V8 / V9c / V10c — chronological 80/20 split, NO train-test leak.

All three models train on the same 5026 oldest matches; backtest evaluates on
the same 1256 newest matches (which are NEVER seen during training).

Architectures:
  - V8 fair: sklearn GradientBoostingClassifier(n_estimators=300, max_depth=4)
             on the original 25 features.
  - V9c fair: LightGBM LGBMRanker(lambdarank, n_estimators=400, num_leaves=63)
              on the original 25 features.
  - V10c fair: LightGBM LGBMRanker(lambdarank, n_estimators=400, num_leaves=63)
               on 39 features (25 base + 14 team-composition).

Outputs:
  - data/v8_fair_gbm.joblib
  - data/v9c_fair_ranker.txt
  - data/v10c_fair_ranker.txt
  - data/fair_split.json (train/test match_id lists for reproducibility)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import hero_features_v8, load_matches  # noqa: E402
from train_v10 import hero_features_v10  # noqa: E402
from train_v10_variants import build_dataset  # noqa: E402
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
import joblib


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    matches.sort(key=lambda m: int(m["match_id"]))
    n_test = max(1, len(matches) // 5)
    train_matches = matches[:-n_test]
    test_matches = matches[-n_test:]
    print(f"[fair] train={len(train_matches)} test={len(test_matches)}")
    print(f"[fair] train match_id range: "
          f"{train_matches[0]['match_id']}..{train_matches[-1]['match_id']}")
    print(f"[fair] test  match_id range: "
          f"{test_matches[0]['match_id']}..{test_matches[-1]['match_id']}")

    ctx = Context(); m2 = M2_DataPositions(ctx); m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    # 25-feature dataset (shared by V8 and V9c)
    print("\n[fair] building 25-feature dataset…")
    X25, y25, g25 = build_dataset(
        train_matches, ctx, m2, m4, all_hero_ids,
        lambda m4, m2, h, p, a, e: hero_features_v8(m4, m2, h, p, a, e),
        25,
    )
    print(f"  X25={X25.shape} groups25={g25.shape}")

    # 39-feature dataset (V10c)
    print("\n[fair] building 39-feature dataset…")
    X39, y39, g39 = build_dataset(
        train_matches, ctx, m2, m4, all_hero_ids,
        lambda m4, m2, h, p, a, e: hero_features_v10(m4, m2, h, p, a, e),
        39,
    )
    print(f"  X39={X39.shape} groups39={g39.shape}")

    # V8 fair (sklearn GBC, 300 trees, depth 4)
    print("\n[fair] training V8 fair (sklearn GBC n=300 d=4 on 25 features)…")
    v8 = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42,
    )
    v8.fit(X25, y25)
    joblib.dump(v8, DATA_DIR / "v8_fair_gbm.joblib")
    print(f"  saved → data/v8_fair_gbm.joblib")

    # V9c fair (LightGBM Ranker, 400 trees, 63 leaves, 25 features)
    print("\n[fair] training V9c fair (LightGBM lambdarank on 25 features)…")
    v9c = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=400, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbosity=-1,
    )
    v9c.fit(X25, y25.astype(int), group=g25)
    v9c.booster_.save_model(str(DATA_DIR / "v9c_fair_ranker.txt"))
    print(f"  saved → data/v9c_fair_ranker.txt")

    # V10c fair (LightGBM Ranker, 400 trees, 63 leaves, 39 features)
    print("\n[fair] training V10c fair (LightGBM lambdarank on 39 features)…")
    v10c = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=400, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbosity=-1,
    )
    v10c.fit(X39, y39.astype(int), group=g39)
    v10c.booster_.save_model(str(DATA_DIR / "v10c_fair_ranker.txt"))
    print(f"  saved → data/v10c_fair_ranker.txt")

    # Save split
    (DATA_DIR / "fair_split.json").write_text(json.dumps({
        "train_match_ids": [int(m["match_id"]) for m in train_matches],
        "test_match_ids": [int(m["match_id"]) for m in test_matches],
        "n_train": len(train_matches),
        "n_test": len(test_matches),
    }))
    print(f"\n[fair] split saved → data/fair_split.json")


if __name__ == "__main__":
    main()
