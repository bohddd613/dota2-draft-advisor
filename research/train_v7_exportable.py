"""
V7e: Train GradientBoostingClassifier (sklearn, exportable) for browser deployment.

Same features and training procedure as train_v7.py, but uses
GradientBoostingClassifier instead of HistGradientBoostingClassifier so we can
export to JavaScript via m2cgen.

Outputs:
  - research/data/v7e_gbm.joblib
  - research/data/v7e_model.js (pure JS scoring function)
  - research/data/v7e_cv.json (cross-validation metrics)
"""
from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v2 import hero_features, FEATURE_NAMES, kfold_split  # noqa: E402
from train_v7 import load_matches, build_pick_dataset, evaluate_pick_rec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    matches = load_matches(Path(args.matches))
    if args.limit:
        matches = matches[: args.limit]
    print(f"[v7e] loaded {len(matches)} matches with TRUE position labels")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    from sklearn.ensemble import GradientBoostingClassifier
    import joblib

    print(f"[v7e] {args.folds}-fold CV...")
    fold_top10 = []
    for fi, (tr_idx, te_idx) in enumerate(kfold_split(len(matches), k=args.folds)):
        tr_matches = [matches[i] for i in tr_idx]
        te_matches = [matches[i] for i in te_idx]
        X_tr, y_tr, _ = build_pick_dataset(
            tr_matches, ctx, m2, m4, all_hero_ids,
            neg_per_pos=args.neg_per_pos, seed=42 + fi,
        )
        clf = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)

        def scorer(f):
            return float(clf.predict_proba(f.reshape(1, -1))[0, 1])

        r = evaluate_pick_rec(scorer, te_matches, ctx, m2, m4, all_hero_ids, samples_per_match=2)
        fold_top10.append(r["top10"])
        print(f"  fold{fi}: top10={r['top10']:.4f}, top5={r['top5']:.4f}, top1={r['top1']:.4f}, mean_rank={r['mean_rank']:.2f}")

    avg = float(np.mean(fold_top10))
    sd = float(np.std(fold_top10))
    print(f"\n[v7e] CV top10 = {avg:.4f} ± {sd:.4f}")

    # Refit on all data
    print("[v7e] refitting on all data...")
    X, y, _ = build_pick_dataset(matches, ctx, m2, m4, all_hero_ids, neg_per_pos=args.neg_per_pos)
    clf_all = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=42,
    )
    clf_all.fit(X, y)
    joblib.dump(clf_all, DATA_DIR / "v7e_gbm.joblib")

    # Export to JavaScript
    print("[v7e] exporting to JavaScript via m2cgen...")
    import m2cgen as m2c
    js_code = m2c.export_to_javascript(clf_all)
    out_js = DATA_DIR / "v7e_model.js"
    out_js.write_text(js_code)

    cv_report = {
        "model": "V7e_gbm_exportable",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_matches": len(matches),
        "feature_names": FEATURE_NAMES,
        "cv_top10_mean": avg,
        "cv_top10_std": sd,
        "n_train_samples": int(X.shape[0]),
    }
    (DATA_DIR / "v7e_cv.json").write_text(json.dumps(cv_report, indent=2))
    print(f"[v7e] saved -> v7e_gbm.joblib, v7e_model.js ({out_js.stat().st_size} bytes), v7e_cv.json")


if __name__ == "__main__":
    main()
