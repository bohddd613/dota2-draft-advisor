"""
V10d / V10e: variants to fix V10c regression.

V10d — only 5 most informative team-comp features (drop noisy/constant-in-group ones)
V10e — same as V10c but n_estimators=600 to compensate split dilution
V10f — only "candidate role flags" (5 features), simpler than counts
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
from hero_attrs import hero_attrs  # noqa: E402
import lightgbm as lgb


# ----------------------------- V10d features (slim subset) -----

V10D_EXTRA = [
    "team_init_count",
    "team_disabler_count",
    "team_nuker_count",
    "team_has_illusions",
    "enemy_has_illusions",
]
FEATURE_NAMES_V10D = FEATURE_NAMES_V8 + V10D_EXTRA


def hero_features_v10d(m4, m2, hid, target_pos, allies, enemies):
    v8 = hero_features_v8(m4, m2, hid, target_pos, allies, enemies)
    team_with = [hid] + [a for a, _ in allies]
    enemies_only = [e for e, _ in enemies]
    ta = [hero_attrs(h) for h in team_with]
    ea = [hero_attrs(h) for h in enemies_only]
    extra = np.array([
        sum(a["is_initiator"] for a in ta),
        sum(a["is_disabler"] for a in ta),
        sum(a["is_nuker"] for a in ta),
        1.0 if any(a["has_illusions"] for a in ta) else 0.0,
        1.0 if any(a["has_illusions"] for a in ea) else 0.0,
    ], dtype=np.float32)
    out = np.concatenate([v8, extra]).astype(np.float32)
    assert out.shape == (30,), out.shape
    return out


# ----------------------------- V10f features (candidate-only flags) -----
# These are pure candidate-attribute flags — varies only by candidate. Simpler than
# team-aggregates but captures the same "is candidate filling a role?" info if model
# can interact with the context.

V10F_EXTRA = [
    "cand_is_initiator",
    "cand_is_disabler",
    "cand_is_nuker",
    "cand_is_pusher",
    "cand_is_durable",
    "cand_is_int",      # int primary → magic-dmg-leaning
    "cand_is_agi",      # agi primary → phys-dmg-leaning
    "cand_has_illusions",
]
FEATURE_NAMES_V10F = FEATURE_NAMES_V8 + V10F_EXTRA


def hero_features_v10f(m4, m2, hid, target_pos, allies, enemies):
    v8 = hero_features_v8(m4, m2, hid, target_pos, allies, enemies)
    a = hero_attrs(hid)
    extra = np.array([
        a["is_initiator"], a["is_disabler"], a["is_nuker"],
        a["is_pusher"], a["is_durable"],
        1 if a["primary_attr"] == "int" else 0,
        1 if a["primary_attr"] == "agi" else 0,
        a["has_illusions"],
    ], dtype=np.float32)
    out = np.concatenate([v8, extra]).astype(np.float32)
    assert out.shape == (33,), out.shape
    return out


# ----------------------------- builder helper -----

def build_dataset(matches, ctx, m2, m4, all_hero_ids, feat_fn, n_features,
                  neg_per_pos: int = 10, seed: int = 42):
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
            X.append(feat_fn(m4, m2, true_hero, pos, allies, enemies)); y.append(1.0)
            for h in sample:
                X.append(feat_fn(m4, m2, int(h), pos, allies, enemies)); y.append(0.0)
            groups.append(1 + len(sample))
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(groups, dtype=np.int32),
    )


def train_variant(label, feat_fn, n_feat, hp, matches, ctx, m2, m4, all_hero_ids, out_path):
    X, y, groups = build_dataset(matches, ctx, m2, m4, all_hero_ids, feat_fn, n_feat)
    print(f"[{label}] X.shape={X.shape}  groups={groups.shape}  hp={hp}")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        random_state=42, verbosity=-1,
        **hp,
    )
    ranker.fit(X, y.astype(int), group=groups)
    ranker.booster_.save_model(str(out_path))
    print(f"[{label}] saved → {out_path}")


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    matches.sort(key=lambda m: int(m["match_id"]))
    n_test = max(1, len(matches) // 5)
    train = matches[:-n_test]
    print(f"[v10_var] train={len(train)} test={n_test}")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    hp_base = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20)
    hp_more = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20)
    hp_reg  = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30)

    # V10d — slim 5 team-comp features
    train_variant("v10d", hero_features_v10d, 30, hp_base,
                  train, ctx, m2, m4, all_hero_ids,
                  DATA_DIR / "v10d_ranker.txt")

    # V10f — candidate-flag features (8)
    train_variant("v10f", hero_features_v10f, 33, hp_base,
                  train, ctx, m2, m4, all_hero_ids,
                  DATA_DIR / "v10f_ranker.txt")

    # V10e — V10c features but more trees (n=600)
    from train_v10 import hero_features_v10
    train_variant("v10e", hero_features_v10, 39, hp_more,
                  train, ctx, m2, m4, all_hero_ids,
                  DATA_DIR / "v10e_ranker.txt")

    # V10g — V10c features but tighter regularization
    train_variant("v10g", hero_features_v10, 39, hp_reg,
                  train, ctx, m2, m4, all_hero_ids,
                  DATA_DIR / "v10g_ranker.txt")

    print("\nDONE. Now run backtest_v10_variants.py")


if __name__ == "__main__":
    main()
