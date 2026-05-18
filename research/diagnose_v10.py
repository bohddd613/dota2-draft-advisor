"""
Diagnostic — train a few variants to isolate WHY V10 regresses:

A) v10_repro_v9: train V9c-like (25 features) with my pipeline. Should ~= 74%.
B) v10_v9_plus_dummy: V9c features + 5 dummy zero features. Tests feature-count effect.
C) v10c_with_es: V10c features but with early stopping on a val split.
"""
from __future__ import annotations
import sys
import math
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import hero_features_v8, load_matches  # noqa: E402
from train_v10 import hero_features_v10  # noqa: E402
from train_v10_variants import build_dataset  # noqa: E402
import lightgbm as lgb


def hero_features_v9_pure(m4, m2, hid, target_pos, allies, enemies):
    """V8/V9 features identically."""
    return hero_features_v8(m4, m2, hid, target_pos, allies, enemies)


def hero_features_v9_plus_dummy(m4, m2, hid, target_pos, allies, enemies):
    """V8/V9 features + 5 zero-padding features (dummy)."""
    v8 = hero_features_v8(m4, m2, hid, target_pos, allies, enemies)
    return np.concatenate([v8, np.zeros(5, dtype=np.float32)]).astype(np.float32)


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    matches.sort(key=lambda m: int(m["match_id"]))
    n_test = max(1, len(matches) // 5)
    train = matches[:-n_test]
    test = matches[-n_test:]
    print(f"[diag] train={len(train)} test={len(test)}")

    ctx = Context(); m2 = M2_DataPositions(ctx); m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    hp = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20)

    print("\n[A] Repro V9c with own pipeline (25 features)")
    X, y, groups = build_dataset(train, ctx, m2, m4, all_hero_ids,
                                  hero_features_v9_pure, 25)
    r = lgb.LGBMRanker(objective="lambdarank", random_state=42, verbosity=-1, **hp)
    r.fit(X, y.astype(int), group=groups)
    r.booster_.save_model(str(DATA_DIR / "v10_repro_v9.txt"))
    print("[A] saved → v10_repro_v9.txt")

    print("\n[B] V9c features + 5 zero-dummy features (30 features)")
    X, y, groups = build_dataset(train, ctx, m2, m4, all_hero_ids,
                                  hero_features_v9_plus_dummy, 30)
    r = lgb.LGBMRanker(objective="lambdarank", random_state=42, verbosity=-1, **hp)
    r.fit(X, y.astype(int), group=groups)
    r.booster_.save_model(str(DATA_DIR / "v10_v9_dummy.txt"))
    print("[B] saved → v10_v9_dummy.txt")

    print("\n[C] V10c features with random_state different (test seed sensitivity)")
    X, y, groups = build_dataset(train, ctx, m2, m4, all_hero_ids,
                                  hero_features_v10, 39)
    r = lgb.LGBMRanker(objective="lambdarank", random_state=123, verbosity=-1, **hp)
    r.fit(X, y.astype(int), group=groups)
    r.booster_.save_model(str(DATA_DIR / "v10c_seed123.txt"))
    print("[C] saved → v10c_seed123.txt")


if __name__ == "__main__":
    main()
