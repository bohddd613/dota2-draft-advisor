"""
Export the FAIR (no-leak) V8 / V9c / V10c models to JSON for browser eval.

Overwrites:
  - data/v8_model.json     (was leak-trained)
  - data/v9c_model.json    (was leak-trained)
Creates:
  - data/v10c_model.json   (new)

All three models honestly trained on 5026 oldest matches (1256 newest are
held-out for evaluation only). Numbers reported by these models match the
backtest_fair_results.json output exactly.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
APP_DATA = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from train_v8 import FEATURE_NAMES_V8  # noqa: E402
from train_v10 import FEATURE_NAMES_V10  # noqa: E402
import lightgbm as lgb


# ----- sklearn GBC export (V8) -----

def export_sklearn_tree(tree):
    return {
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        "value": tree.value.reshape(-1).tolist(),
    }


def export_v8_fair():
    clf = joblib.load(DATA_DIR / "v8_fair_gbm.joblib")
    init = float(clf.init_.class_prior_[1])
    init_log_odds = float(np.log(init / (1 - init)))
    lr = float(clf.learning_rate)
    trees = [export_sklearn_tree(est[0].tree_) for est in clf.estimators_]
    out = {
        "init_log_odds": init_log_odds,
        "learning_rate": lr,
        "feature_names": FEATURE_NAMES_V8,
        "n_features": len(FEATURE_NAMES_V8),
        "n_trees": len(trees),
        "trees": trees,
        "variant": "v8_fair",
        "training": {
            "matches": 5026,
            "split": "chronological 80/20 — 1256 newest held out",
            "honest_top10": 0.575,
        },
    }
    text = json.dumps(out)
    (DATA_DIR / "v8_model.json").write_text(text)
    (APP_DATA / "v8_model.json").write_text(text)
    print(f"[V8 fair]  {len(trees)} trees ({len(text)/1024:.1f} KB)  → data/v8_model.json")


# ----- LightGBM Ranker export (V9c, V10c) -----

def flatten_lgbm_tree(root):
    feature, threshold, left, right, value = [], [], [], [], []

    def walk(node):
        idx = len(feature)
        feature.append(0); threshold.append(0.0); left.append(-1); right.append(-1); value.append(0.0)
        if "leaf_value" in node:
            feature[idx] = -2
            value[idx] = float(node["leaf_value"])
            return idx
        feature[idx] = int(node["split_feature"])
        threshold[idx] = float(node["threshold"])
        l = walk(node["left_child"])
        r = walk(node["right_child"])
        left[idx] = l
        right[idx] = r
        return idx

    walk(root)
    return {"feature": feature, "threshold": threshold,
            "left": left, "right": right, "value": value}


def export_lgbm(src_path: Path, dst_name: str, feature_names: list[str], variant: str, top10: float):
    booster = lgb.Booster(model_file=str(src_path))
    m = booster.dump_model()
    trees = [flatten_lgbm_tree(t["tree_structure"]) for t in m["tree_info"]]
    out = {
        "init_score": 0.0,
        "learning_rate": 1.0,
        "objective": "ranker",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_trees": len(trees),
        "trees": trees,
        "variant": variant,
        "training": {
            "matches": 5026,
            "split": "chronological 80/20 — 1256 newest held out",
            "honest_top10": top10,
        },
    }
    text = json.dumps(out)
    (DATA_DIR / dst_name).write_text(text)
    (APP_DATA / dst_name).write_text(text)
    print(f"[{variant}] {len(trees)} trees ({len(text)/1024:.1f} KB)  → data/{dst_name}")


def main():
    export_v8_fair()
    export_lgbm(DATA_DIR / "v9c_fair_ranker.txt", "v9c_model.json", FEATURE_NAMES_V8, "v9c_fair", 0.573)
    export_lgbm(DATA_DIR / "v10c_fair_ranker.txt", "v10c_model.json", FEATURE_NAMES_V10, "v10c_fair", 0.574)


if __name__ == "__main__":
    main()
