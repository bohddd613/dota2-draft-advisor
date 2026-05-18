"""
Fair V9 retraining — proper train/test split (no leak).

Variants:
  - v9c_fair: same as original V9c (lambdarank, n=400, leaves=63), but trained on
              the same 5026 train matches V10 was trained on. Already exists as
              v10_repro_v9.txt (see diagnose_v10.py). Listed here for completeness.
  - v9_fair_sklearn: sklearn GradientBoostingClassifier on 5026 train. Same arch
                     as V8 but with 3.6× more data. Tests whether sklearn arch
                     scales with more data.
  - v9c_fair_strict: LightGBM Ranker with strict regularization (num_leaves=15,
                     min_child_samples=100). Tests if V9c needs regularization.
  - v9c_fair_binary: LightGBM with binary objective (not lambdarank).
                     Same hyperparams as V9c. Tests if lambdarank is the issue.

All trained on 5026 oldest matches, evaluated on 1256 newest.
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
from train_v10_variants import build_dataset  # noqa: E402
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
import joblib


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    matches.sort(key=lambda m: int(m["match_id"]))
    n_test = max(1, len(matches) // 5)
    train = matches[:-n_test]
    print(f"[fair_v9] train={len(train)} (will eval on remaining {n_test})")

    ctx = Context(); m2 = M2_DataPositions(ctx); m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    def feat(m4, m2, h, p, a, e): return hero_features_v8(m4, m2, h, p, a, e)
    X, y, groups = build_dataset(train, ctx, m2, m4, all_hero_ids, feat, 25)
    print(f"[fair_v9] dataset: X={X.shape} groups={groups.shape}")

    # --- v9_fair_sklearn (same arch as V8 but on 5026)
    print("\n[fair] sklearn GBC (V8 arch, depth=4, n=400) on 5026")
    clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05,
                                      max_depth=4, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, DATA_DIR / "v9_fair_sklearn.joblib")
    print("[fair] saved → v9_fair_sklearn.joblib")

    # --- v9c_fair_strict (LightGBM Ranker with stricter regularization)
    print("\n[fair] LightGBM Ranker STRICT (leaves=15, min_child=100)")
    r = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=400, learning_rate=0.05,
        num_leaves=15, min_child_samples=100,
        random_state=42, verbosity=-1,
    )
    r.fit(X, y.astype(int), group=groups)
    r.booster_.save_model(str(DATA_DIR / "v9c_fair_strict.txt"))
    print("[fair] saved → v9c_fair_strict.txt")

    # --- v9c_fair_binary (LightGBM with binary objective)
    print("\n[fair] LightGBM Binary (default hyperparams, no groups)")
    rb = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbosity=-1,
    )
    rb.fit(X, y.astype(int))
    rb.booster_.save_model(str(DATA_DIR / "v9c_fair_binary.txt"))
    print("[fair] saved → v9c_fair_binary.txt")


if __name__ == "__main__":
    main()
