"""
Win-probability uplift analysis for V9 (Phase B verification).

Identical methodology to win_uplift_v8.py — adds V9a/V9b to the comparison.
Question: do V9's top-N recommendations correlate with match outcomes
better than V8's? (Verification only, not production logic.)

Output: research/data/win_uplift_v9.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from backtest_v8 import load_matches, ModelV8, ModelV7e, ModelM3  # noqa: E402
from win_uplift_v8 import uplift_for_model  # noqa: E402


class ModelV9(ModelV8):
    """V9a — same arch as V8, different weights."""
    name = "V9a"


class ModelV9b(ModelV8):
    name = "V9b"


class ModelV9c:
    """LightGBM Ranker — wraps booster.predict(raw_score=True)."""
    name = "V9c"

    def __init__(self, ctx, m2, m4, booster):
        self.ctx = ctx; self.m2 = m2; self.m4 = m4; self.booster = booster

    def score_many(self, candidates, target_pos, allies_pos, enemies_pos):
        import numpy as np
        from train_v8 import hero_features_v8
        feats = np.vstack([
            hero_features_v8(self.m4, self.m2, h, target_pos, allies_pos, enemies_pos)
            for h in candidates
        ])
        return self.booster.predict(feats, raw_score=True)

    def eligible(self, h, pos):
        return pos in (self.m2.eligible.get(h) or [])


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"[uplift_v9] {len(matches)} matches")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)

    clf_v8 = joblib.load(DATA_DIR / "v8a_gbm.joblib")
    clf_v7e = joblib.load(DATA_DIR / "v7e_gbm.joblib")

    models = []
    # V9c — the winner from backtest
    try:
        import lightgbm as lgb
        booster_v9c = lgb.Booster(model_file=str(DATA_DIR / "v9c_ranker.txt"))
        models.append(ModelV9c(ctx, m2, m4, booster_v9c))
    except FileNotFoundError:
        print("[uplift_v9] v9c_ranker.txt not found — skip")
    try:
        clf_v9a = joblib.load(DATA_DIR / "v9a_gbm.joblib")
        models.append(ModelV9(ctx, m2, m4, clf_v9a))
    except FileNotFoundError:
        print("[uplift_v9] v9a_gbm.joblib not found — skip")
    try:
        clf_v9b = joblib.load(DATA_DIR / "v9b_gbm.joblib")
        models.append(ModelV9b(ctx, m2, m4, clf_v9b))
    except FileNotFoundError:
        print("[uplift_v9] v9b_gbm.joblib not found — skip")
    models.append(ModelV8(ctx, m2, m4, clf_v8))
    models.append(ModelV7e(ctx, m2, m4, clf_v7e))
    models.append(ModelM3(ctx, m2, m4))

    results = {}
    for mdl in models:
        print(f"\n=== {mdl.name} ===")
        r = uplift_for_model(mdl, matches, ctx)
        results[mdl.name] = r
        print(f"  n_winners={r['n_winners']}  n_losers={r['n_losers']}")
        for k, m_ in r["metrics"].items():
            sign = "+" if m_["uplift_pp"] >= 0 else ""
            ratio = m_["ratio"] if m_["ratio"] is not None else float("nan")
            print(f"  {k}: winners {m_['winners_aligned_rate']*100:.2f}% vs losers "
                  f"{m_['losers_aligned_rate']*100:.2f}%  → uplift {sign}{m_['uplift_pp']:.2f}pp  ratio {ratio:.3f}")

    out = DATA_DIR / "win_uplift_v9.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[uplift_v9] saved → {out}")


if __name__ == "__main__":
    main()
