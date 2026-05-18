"""
V10: Phase C1 — V9 architecture + team composition features.

Adds 14 team-composition features (TEAM_COMP_FEATURE_NAMES) on top of V8's 25.
Total: 39 features. Same LightGBM LGBMRanker (lambdarank) architecture as V9c.

Hypothesis: V9 saturates pairwise-interaction signal but cannot represent team
archetype reasoning ("we already have 3 carries, pick a support"; "enemy has 3
illusion heroes, pick AoE"). Team-composition features fill that gap.

Variants:
  - V10c: LightGBM LGBMRanker, same hyperparams as V9c, expanded feature set.
    (Chosen as primary since V9c was breakthrough; we extend that architecture.)

Evaluation: chronological 80/20 hold-out (5026 train, 1256 test newest matches),
identical split used by V9 backtest. Test set unchanged → direct V9c vs V10c comparison.

Outputs:
  - research/data/v10c_ranker.txt
  - research/data/v10c_model.json (for browser)
"""
from __future__ import annotations
import sys
import math
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import (  # noqa: E402
    FEATURE_NAMES_V8, hero_features_v8, load_matches, POS_TO_INT,
)
from hero_attrs import (  # noqa: E402
    TEAM_COMP_FEATURE_NAMES, team_comp_features,
)
import lightgbm as lgb


# ----------------------------- features ------------------------------------

FEATURE_NAMES_V10 = FEATURE_NAMES_V8 + TEAM_COMP_FEATURE_NAMES
assert len(FEATURE_NAMES_V10) == 25 + 14 == 39


def hero_features_v10(
    m4: M4_RoleGap, m2: M2_DataPositions,
    hid: int, target_pos: int,
    allies: list[tuple[int, int]], enemies: list[tuple[int, int]],
) -> np.ndarray:
    """V8 features (25) + team-comp features (14) → 39-dim vector."""
    v8 = hero_features_v8(m4, m2, hid, target_pos, allies, enemies)
    comp = team_comp_features(hid, allies, enemies)
    out = np.concatenate([v8, np.array(comp, dtype=np.float32)]).astype(np.float32)
    assert out.shape == (39,), out.shape
    return out


# ----------------------------- dataset builder ------------------------------

def build_pick_dataset_v10(
    matches, ctx, m2, m4, all_hero_ids,
    neg_per_pos: int = 10, seed: int = 42,
):
    """Mirror of build_pick_dataset_v9 (winners only) but uses V10 features."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []

    for m in matches:
        radiant = m["radiant"]; dire = m["dire"]
        winners = radiant if m["radiant_win"] else dire
        losers = dire if m["radiant_win"] else radiant
        taken = {h for h, _ in radiant} | {h for h, _ in dire}
        for true_hero, pos in winners:
            allies = [(h, p) for h, p in winners if h != true_hero]
            enemies = losers
            eligible = [h for h in all_hero_ids
                        if h not in taken and pos in (m2.eligible.get(h) or [])]
            if true_hero not in eligible:
                eligible.append(true_hero)
            negatives = [h for h in eligible if h != true_hero]
            if not negatives:
                continue
            sample = rng.choice(negatives, size=min(neg_per_pos, len(negatives)), replace=False)
            X.append(hero_features_v10(m4, m2, true_hero, pos, allies, enemies))
            y.append(1.0)
            for h in sample:
                X.append(hero_features_v10(m4, m2, int(h), pos, allies, enemies))
                y.append(0.0)
            groups.append(1 + len(sample))

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(groups, dtype=np.int32),
    )


# ----------------------------- train ---------------------------------------

def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"[v10] loaded {len(matches)} matches")

    # Chronological split: 80% train (oldest), 20% test (newest)
    matches.sort(key=lambda m: int(m["match_id"]))
    n_test = max(1, len(matches) // 5)
    train_matches = matches[:-n_test]
    test_matches = matches[-n_test:]
    print(f"[v10] train: {len(train_matches)}  test: {len(test_matches)}  "
          f"(test newest match_id: {test_matches[-1]['match_id']})")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    X, y, groups = build_pick_dataset_v10(
        train_matches, ctx, m2, m4, all_hero_ids,
        neg_per_pos=10, seed=42,
    )
    print(f"[v10c] X.shape={X.shape}  groups.shape={groups.shape}  groups[0]={groups[0]}")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=400, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        random_state=42, verbosity=-1,
    )
    ranker.fit(X, y.astype(int), group=groups)
    ranker.booster_.save_model(str(DATA_DIR / "v10c_ranker.txt"))
    print(f"[v10c] saved → research/data/v10c_ranker.txt")

    # Save train/test split for backtest_v10 to use the same one.
    import json
    (DATA_DIR / "v10_split.json").write_text(json.dumps({
        "train_match_ids": [int(m["match_id"]) for m in train_matches],
        "test_match_ids": [int(m["match_id"]) for m in test_matches],
    }))
    print(f"[v10c] saved → research/data/v10_split.json")


if __name__ == "__main__":
    main()
