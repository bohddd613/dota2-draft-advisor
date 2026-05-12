"""
Export V9c (LightGBM LGBMRanker) to JSON for browser eval.

LightGBM dumps nested tree dicts; we flatten them to the same
{feature, threshold, left, right, value} flat-array format used by V8/V9a,
so the browser tree walker can be reused.

V9c is a ranker — output is a raw score (sum of leaf values), not a
probability. The browser uses raw scores to rank candidates (no sigmoid).
"""
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from train_v8 import FEATURE_NAMES_V8  # noqa: E402
import lightgbm as lgb


def flatten_tree(root):
    """
    Convert LightGBM nested tree dict to flat arrays.
    Returns dict with feature[], threshold[], left[], right[], value[].
    feature[i] = -2 if node i is a leaf.
    """
    feature, threshold, left, right, value = [], [], [], [], []

    def walk(node):
        idx = len(feature)
        # Reserve a slot
        feature.append(0); threshold.append(0.0); left.append(-1); right.append(-1); value.append(0.0)
        if "leaf_value" in node:
            feature[idx] = -2
            threshold[idx] = 0.0
            left[idx] = -1
            right[idx] = -1
            value[idx] = float(node["leaf_value"])
            return idx
        # Internal node
        feature[idx] = int(node["split_feature"])
        threshold[idx] = float(node["threshold"])
        # LightGBM uses "<= threshold" → left; sklearn convention matches.
        l = walk(node["left_child"])
        r = walk(node["right_child"])
        left[idx] = l
        right[idx] = r
        value[idx] = 0.0  # internal nodes have no value
        return idx

    walk(root)
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "value": value,
    }


def main():
    src = DATA_DIR / "v9c_ranker.txt"
    booster = lgb.Booster(model_file=str(src))
    m = booster.dump_model()

    trees = []
    for t in m["tree_info"]:
        flat = flatten_tree(t["tree_structure"])
        trees.append(flat)

    out = {
        # ranker outputs raw scores summed across trees; no init_log_odds and
        # no learning_rate (LightGBM bakes shrinkage into leaf values).
        "init_score": 0.0,
        "learning_rate": 1.0,
        "objective": "ranker",
        "feature_names": FEATURE_NAMES_V8,
        "n_features": len(FEATURE_NAMES_V8),
        "n_trees": len(trees),
        "trees": trees,
        "variant": "v9c",
    }
    dst = DATA_DIR / "v9c_model.json"
    dst.write_text(json.dumps(out))
    print(f"exported {len(trees)} trees ({dst.stat().st_size/1024:.1f} KB)  → {dst}")
    APP_DATA = Path(__file__).parent.parent / "data"
    (APP_DATA / "v9c_model.json").write_text(json.dumps(out))
    print(f"copied to {APP_DATA/'v9c_model.json'} (variant=v9c)")


if __name__ == "__main__":
    main()
