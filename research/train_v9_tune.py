"""
Quick hyperparam sweep for V9: single fold + evaluation on different hyperparams.
Uses the same 4310 matches, tests 4 configs on a single random fold.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v2 import kfold_split  # noqa: E402
from train_v8 import (FEATURE_NAMES_V8, hero_features_v8, load_matches,
                       evaluate_pick_rec_v8)  # noqa: E402
from train_v9 import build_pick_dataset_v9  # noqa: E402

from sklearn.ensemble import GradientBoostingClassifier
import joblib


def main():
    matches = load_matches(Path(DATA_DIR / "matches_stratz_enriched.json"))
    print(f"loaded {len(matches)} matches")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    # Single fold (first fold)
    tr_idx, te_idx = next(iter(kfold_split(len(matches), k=5)))
    tr = [matches[i] for i in tr_idx]
    te = [matches[i] for i in te_idx]

    configs = [
        {"depth": 4, "n": 300, "lr": 0.05, "label": "d4_n300"},
        {"depth": 4, "n": 400, "lr": 0.05, "label": "d4_n400"},
        {"depth": 5, "n": 300, "lr": 0.05, "label": "d5_n300"},
        {"depth": 5, "n": 400, "lr": 0.05, "label": "d5_n400"},
        {"depth": 4, "n": 500, "lr": 0.03, "label": "d4_n500_lr03"},
        {"depth": 3, "n": 600, "lr": 0.03, "label": "d3_n600_lr03"},
    ]

    results = {}
    for c in configs:
        print(f"\n--- {c['label']}: depth={c['depth']} n={c['n']} lr={c['lr']} ---")
        X_tr, y_tr, w_tr, _ = build_pick_dataset_v9(
            tr, ctx, m2, m4, all_hero_ids,
            neg_per_pos=10, seed=42,
            include_losers=False,
        )
        clf = GradientBoostingClassifier(
            n_estimators=c["n"], max_depth=c["depth"],
            learning_rate=c["lr"], random_state=42,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        def s(F, m=clf): return m.predict_proba(F)[:, 1]
        r = evaluate_pick_rec_v8(s, te, ctx, m2, m4, all_hero_ids)
        print(f"  top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
              f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}")
        results[c["label"]] = r

    # Also eval existing V8 on same test fold (its weights, different data)
    v8 = joblib.load(DATA_DIR / "v8a_gbm.joblib")
    def v8_scorer(F): return v8.predict_proba(F)[:, 1]
    r = evaluate_pick_rec_v8(v8_scorer, te, ctx, m2, m4, all_hero_ids)
    print(f"\n--- V8 (existing weights) ---")
    print(f"  top10={r['top10']:.4f}  top5={r['top5']:.4f}  "
          f"top1={r['top1']:.4f}  mean={r['mean_rank']:.2f}")
    results["v8_existing"] = r

    print("\n=== Summary ===")
    print(f"{'config':<16} {'top1':>7} {'top5':>7} {'top10':>7} {'mean':>7}")
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["top10"]):
        print(f"{label:<16} {r['top1']*100:>6.2f}% {r['top5']*100:>6.2f}% "
              f"{r['top10']*100:>6.2f}% {r['mean_rank']:>7.2f}")


if __name__ == "__main__":
    main()
