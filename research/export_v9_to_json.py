"""
Export V9 sklearn GradientBoostingClassifier to JSON for browser eval.

Uses the same format as V8 (init_log_odds + per-tree feature/threshold/left/right/value).
Selectable variant: --variant v9a (winners only) or v9b (winners + losers@0.5).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from train_v8 import FEATURE_NAMES_V8  # noqa: E402


def export_tree(tree):
    return {
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        "value": tree.value.reshape(-1).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v9a", "v9b"], default="v9a")
    ap.add_argument("--output", default="v9_model.json",
                    help="filename in research/data and root /data (default v9_model.json)")
    args = ap.parse_args()

    clf = joblib.load(DATA_DIR / f"{args.variant}_gbm.joblib")
    init = float(clf.init_.class_prior_[1])
    init_log_odds = float(np.log(init / (1 - init)))
    lr = float(clf.learning_rate)
    trees = [export_tree(est_arr[0].tree_) for est_arr in clf.estimators_]
    out = {
        "init_log_odds": init_log_odds,
        "learning_rate": lr,
        "feature_names": FEATURE_NAMES_V8,
        "n_features": len(FEATURE_NAMES_V8),
        "n_trees": len(trees),
        "trees": trees,
        "variant": args.variant,
    }
    dst = DATA_DIR / args.output
    dst.write_text(json.dumps(out))
    print(f"exported {len(trees)} trees ({dst.stat().st_size/1024:.1f} KB)  → {dst}")
    # Copy to /data for browser
    APP_DATA = Path(__file__).parent.parent / "data"
    (APP_DATA / args.output).write_text(json.dumps(out))
    print(f"copied to {APP_DATA/args.output} (variant={args.variant}, "
          f"n_features={out['n_features']}, init_log_odds={init_log_odds:+.4f}, lr={lr})")


if __name__ == "__main__":
    main()
