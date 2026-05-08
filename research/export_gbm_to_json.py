"""
Export sklearn GradientBoostingClassifier as JSON tree dump for browser eval.

Output format:
{
  "init_log_odds": float,            # initial score (log-odds, before sigmoid)
  "learning_rate": float,
  "feature_names": [...],
  "n_features": int,
  "trees": [
    {
      "feature": [...],       # length = n_nodes; -2 for leaf
      "threshold": [...],     # threshold to compare X[feature] <=
      "left": [...],          # left child idx
      "right": [...],         # right child idx
      "value": [...]          # leaf value (0 for non-leaves)
    },
    ...
  ]
}

Browser evaluation:
  score = init + lr * sum(tree_i.predict(features))
  proba = sigmoid(score)
"""
import json
import sys
from pathlib import Path
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"


def export_tree(tree):
    """tree is a sklearn.tree.Tree (low-level)"""
    feat = tree.feature.tolist()
    thr = tree.threshold.tolist()
    left = tree.children_left.tolist()
    right = tree.children_right.tolist()
    # value is shape (n_nodes, 1, 1) for regression
    val = tree.value.reshape(-1).tolist()
    return {
        "feature": feat,
        "threshold": thr,
        "left": left,
        "right": right,
        "value": val,
    }


def main(model_path: Path, out_path: Path, feature_names: list[str]):
    clf = joblib.load(model_path)
    init = float(clf.init_.class_prior_[1])
    init_log_odds = float(np.log(init / (1 - init)))
    lr = float(clf.learning_rate)
    trees = []
    for est_arr in clf.estimators_:
        est = est_arr[0]  # binary classification: 1 estimator per stage
        trees.append(export_tree(est.tree_))
    out = {
        "init_log_odds": init_log_odds,
        "learning_rate": lr,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_trees": len(trees),
        "trees": trees,
    }
    out_path.write_text(json.dumps(out))
    size_kb = out_path.stat().st_size / 1024
    print(f"exported {len(trees)} trees, {size_kb:.1f} KB -> {out_path}")
    print(f"init_log_odds={init_log_odds:+.4f} lr={lr}")


if __name__ == "__main__":
    feat_names = ["base_wr", "pos_fit", "with_syn", "vs_adv", "role_gap"]
    src = DATA_DIR / "v7e_gbm.joblib"
    dst = DATA_DIR / "v7e_model.json"
    main(src, dst, feat_names)
