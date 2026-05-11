"""
Export V8 sklearn GradientBoostingClassifier to JSON for browser eval.

Uses the same format as V7e (init_log_odds + per-tree feature/threshold/left/right/value),
plus the FEATURE_NAMES_V8 ordering for the 25-feature V8 model.
"""
import json
import sys
from pathlib import Path
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from train_v8 import FEATURE_NAMES_V8  # noqa: E402


def export_tree(tree):
    feat = tree.feature.tolist()
    thr = tree.threshold.tolist()
    left = tree.children_left.tolist()
    right = tree.children_right.tolist()
    val = tree.value.reshape(-1).tolist()
    return {"feature": feat, "threshold": thr, "left": left, "right": right, "value": val}


def main():
    clf = joblib.load(DATA_DIR / "v8a_gbm.joblib")
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
    }
    dst = DATA_DIR / "v8_model.json"
    dst.write_text(json.dumps(out))
    print(f"exported {len(trees)} trees ({dst.stat().st_size/1024:.1f} KB)  → {dst}")
    # Copy to /data for browser
    APP_DATA = Path(__file__).parent.parent / "data"
    (APP_DATA / "v8_model.json").write_text(json.dumps(out))
    print(f"copied to {APP_DATA/'v8_model.json'} (n_features={out['n_features']}, init_log_odds={init_log_odds:+.4f}, lr={lr})")


if __name__ == "__main__":
    main()
