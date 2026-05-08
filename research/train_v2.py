"""
Improved training: uses STRATZ-enriched matches with TRUE position labels.

Trains four models on the same features for direct comparison:
  - V5_lr     — logistic regression (linear, interpretable)
  - V6_gbm    — gradient boosted trees (HistGradientBoosting, captures interactions)
  - V7_pick   — pick-prediction logistic (P(hero | partial draft state, slot))

K-fold cross-validation (5 folds) for V5/V6 win-prediction.

Output:
  - research/data/v5_weights.json
  - research/data/v6_gbm.joblib
  - research/data/v7_pick.joblib
  - research/data/training_report.json
"""
from __future__ import annotations
import json
import math
import sys
import argparse
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

sys.path.insert(0, str(Path(__file__).parent))
from models import Context, M2_DataPositions, M4_RoleGap, HERO_POSITIONS_M1, POSITION_KEYS  # noqa: E402

POS_TO_INT = {f"POSITION_{i}": i for i in range(1, 6)}


def load_enriched_matches(path: Path) -> list[dict]:
    """Returns list of {match_id, radiant_win, radiant: [(hero_id, pos)], dire: [(hero_id, pos)]}."""
    raw = json.loads(path.read_text())
    out = []
    for m in raw:
        rad, dire = [], []
        for p in m.get("players", []):
            pos = POS_TO_INT.get(p["position"])
            if pos is None:
                pos = 0  # unknown → 0
            tup = (p["hero_id"], pos)
            (rad if p["is_radiant"] else dire).append(tup)
        if len(rad) != 5 or len(dire) != 5:
            continue
        # Need positions for everyone; if any 0 then skip this match
        if any(t[1] == 0 for t in rad + dire):
            continue
        out.append({"match_id": m["match_id"], "radiant_win": m["radiant_win"], "radiant": rad, "dire": dire})
    return out


def hero_features(m4: M4_RoleGap, m2: M2_DataPositions, hid: int, pos: int, allies: list[int], enemies: list[int]) -> np.ndarray:
    base_wr = m2.base_wr(hid, pos) - 0.5
    pos_fit = m2.position_fit(hid, pos)
    with_syn = m4.synergy(hid, allies) - 0.5
    vs_adv = m4.counter(hid, enemies) - 0.5
    KEY_ROLES = m4.KEY_ROLES
    ally_roles = m4._team_roles(allies)
    cand_roles = {r["roleId"] for r in m4.ctx.heroes.get(hid, {}).get("roles", []) if r["level"] >= 2}
    missing = KEY_ROLES - ally_roles
    role_gap = len(cand_roles & missing) / max(1, len(KEY_ROLES))
    return np.array([base_wr, pos_fit, with_syn, vs_adv, role_gap], dtype=np.float32)


FEATURE_NAMES = ["base_wr", "pos_fit", "with_syn", "vs_adv", "role_gap"]


def team_features_delta(m4, m2, radiant: list[tuple[int, int]], dire: list[tuple[int, int]]) -> np.ndarray:
    """Sum hero features for radiant minus same for dire — the input vector."""
    rad_ids = [h for h, _ in radiant]
    dire_ids = [h for h, _ in dire]

    rad_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for hid, pos in radiant:
        allies = [h for h in rad_ids if h != hid]
        enemies = dire_ids
        rad_sum += hero_features(m4, m2, hid, pos, allies, enemies)

    dire_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for hid, pos in dire:
        allies = [h for h in dire_ids if h != hid]
        enemies = rad_ids
        dire_sum += hero_features(m4, m2, hid, pos, allies, enemies)

    return rad_sum - dire_sum


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_lr(X: np.ndarray, y: np.ndarray, l2: float = 0.01) -> tuple[np.ndarray, float, dict]:
    """Train logistic regression via scipy.optimize."""
    from scipy.optimize import minimize
    n_features = X.shape[1]

    def loss(w):
        b = w[0]
        coef = w[1:]
        z = b + X @ coef
        # numerically stable
        p = sigmoid(z)
        eps = 1e-12
        nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = l2 * np.sum(coef**2)
        return nll + reg

    w0 = np.zeros(n_features + 1, dtype=np.float64)
    res = minimize(loss, w0, method="L-BFGS-B")
    w = res.x
    intercept = float(w[0])
    coef = w[1:].astype(np.float64)
    z = intercept + X @ coef
    p = sigmoid(z)
    pred = (p >= 0.5).astype(int)
    acc = float(np.mean(pred == y))
    eps = 1e-12
    nll = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    brier = float(np.mean((p - y) ** 2))
    return intercept, coef, {"accuracy": acc, "log_loss": nll, "brier": brier}


def kfold_split(n: int, k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    matches = load_enriched_matches(Path(args.matches))
    if args.limit:
        matches = matches[: args.limit]
    print(f"[train] loaded {len(matches)} matches with full position labels")
    if not matches:
        print("error: no matches with positions to train on")
        return

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)

    # Build feature matrix for win-prediction task
    print("[train] building feature matrix...")
    X = np.zeros((len(matches), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(len(matches), dtype=np.float32)
    for i, m in enumerate(matches):
        X[i] = team_features_delta(m4, m2, m["radiant"], m["dire"])
        y[i] = 1.0 if m["radiant_win"] else 0.0
        if (i + 1) % 200 == 0:
            print(f"  built features for {i + 1}/{len(matches)}")
    print(f"[train] X shape={X.shape}, mean(y)={y.mean():.4f}")

    # ==== V5 LR with K-fold CV ====
    print(f"[train] V5 logistic regression with {args.folds}-fold CV...")
    accs, lls, briers = [], [], []
    last_intercept, last_coef = None, None
    for fi, (tr, te) in enumerate(kfold_split(len(matches), k=args.folds)):
        intercept, coef, mtrain = train_lr(X[tr], y[tr])
        z = intercept + X[te] @ coef
        p = sigmoid(z)
        pred = (p >= 0.5).astype(int)
        acc = float(np.mean(pred == y[te]))
        eps = 1e-12
        nll = float(-np.mean(y[te] * np.log(p + eps) + (1 - y[te]) * np.log(1 - p + eps)))
        brier = float(np.mean((p - y[te]) ** 2))
        print(f"  fold{fi}: train_acc={mtrain['accuracy']:.4f}, test_acc={acc:.4f}, ll={nll:.4f}, brier={brier:.4f}")
        accs.append(acc); lls.append(nll); briers.append(brier)
        last_intercept, last_coef = intercept, coef
    v5_results = {
        "model": "V5_lr",
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_log_loss_mean": float(np.mean(lls)),
        "cv_brier_mean": float(np.mean(briers)),
        "intercept": float(last_intercept),
        "coef": dict(zip(FEATURE_NAMES, [float(c) for c in last_coef])),
    }
    print(f"[V5] CV acc={v5_results['cv_accuracy_mean']:.4f}±{v5_results['cv_accuracy_std']:.4f}")

    # Also fit on all data for production use
    intercept_all, coef_all, _ = train_lr(X, y)
    v5_weights_full = {
        "intercept": float(intercept_all),
        **dict(zip(FEATURE_NAMES, [float(c) for c in coef_all])),
    }
    (DATA_DIR / "v5_weights.json").write_text(json.dumps(v5_weights_full, indent=2))
    print(f"[V5] full-data weights: {v5_weights_full}")

    # ==== V6 GBM ====
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        import joblib
        print(f"[train] V6 gradient boosting with {args.folds}-fold CV...")
        accs, lls, briers = [], [], []
        for fi, (tr, te) in enumerate(kfold_split(len(matches), k=args.folds)):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            pred = (p >= 0.5).astype(int)
            acc = float(np.mean(pred == y[te]))
            eps = 1e-12
            nll = float(-np.mean(y[te] * np.log(p + eps) + (1 - y[te]) * np.log(1 - p + eps)))
            brier = float(np.mean((p - y[te]) ** 2))
            print(f"  fold{fi}: test_acc={acc:.4f}, ll={nll:.4f}, brier={brier:.4f}")
            accs.append(acc); lls.append(nll); briers.append(brier)
        v6_results = {
            "model": "V6_gbm",
            "cv_accuracy_mean": float(np.mean(accs)),
            "cv_accuracy_std": float(np.std(accs)),
            "cv_log_loss_mean": float(np.mean(lls)),
            "cv_brier_mean": float(np.mean(briers)),
        }
        clf_all = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42)
        clf_all.fit(X, y)
        joblib.dump(clf_all, DATA_DIR / "v6_gbm.joblib")
        print(f"[V6] CV acc={v6_results['cv_accuracy_mean']:.4f}±{v6_results['cv_accuracy_std']:.4f}")
    except ImportError as e:
        print(f"[V6] sklearn not available: {e}")
        v6_results = None

    # ==== Save report ====
    report = {
        "n_matches": len(matches),
        "feature_names": FEATURE_NAMES,
        "y_mean": float(y.mean()),
        "V5": v5_results,
        "V6": v6_results,
    }
    (DATA_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"[train] saved -> training_report.json")


if __name__ == "__main__":
    main()
