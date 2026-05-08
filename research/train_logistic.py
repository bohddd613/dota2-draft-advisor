"""
Train logistic-regression weights for M5 using historical match outcomes.

The training task is: given a complete draft (10 heroes, with greedy-assigned
positions), predict which side wins by computing a feature vector for each
"team-hero-in-context" and aggregating to a team-strength delta.

We learn weights {base_wr, with_syn, vs_adv, pos_fit, role_gap} such that
sigmoid(team_strength_radiant - team_strength_dire) matches the actual outcome.

Solver: scipy.optimize.minimize with logistic loss. No external ML deps.

Usage:
  python3 research/train_logistic.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from models import (
    Context, M2_DataPositions, M3_TrueSynergy, M4_RoleGap,
    assign_positions, HERO_POSITIONS_M1,
)


def hero_features(m4: M4_RoleGap, m2: M2_DataPositions, hid: int, pos: int,
                  allies: list[int], enemies: list[int]) -> np.ndarray:
    """5-dim feature vector for one hero."""
    base_wr = m2.base_wr(hid, pos) - 0.5
    pos_fit = m2.position_fit(hid, pos)
    with_syn = m4.synergy(hid, allies) - 0.5
    vs_adv = m4.counter(hid, enemies) - 0.5
    # role-gap: number of new key roles this hero adds to the team
    ally_roles = m4._team_roles(allies)
    cand_roles = {r["roleId"] for r in m4.ctx.heroes.get(hid, {}).get("roles", []) if r["level"] >= 2}
    missing = m4.KEY_ROLES - ally_roles
    role_gap = len(cand_roles & missing) / max(1, len(m4.KEY_ROLES))
    return np.array([base_wr, pos_fit, with_syn, vs_adv, role_gap])


def team_features_delta(m4: M4_RoleGap, m2: M2_DataPositions,
                         radiant: list[int], dire: list[int]) -> np.ndarray:
    def elig(h):
        return m2.eligible.get(h) or list(range(1, 6))

    rad_pos = assign_positions(radiant, elig)
    dir_pos = assign_positions(dire, elig)

    rad_sum = np.zeros(5)
    for h in radiant:
        pos = rad_pos.get(h, 1)
        allies = [a for a in radiant if a != h]
        rad_sum += hero_features(m4, m2, h, pos, allies, dire)

    dir_sum = np.zeros(5)
    for h in dire:
        pos = dir_pos.get(h, 1)
        allies = [a for a in dire if a != h]
        dir_sum += hero_features(m4, m2, h, pos, allies, radiant)

    return rad_sum - dir_sum  # 5-dim


def main():
    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)

    matches = json.loads(Path("research/data/matches_public_divine.json").read_text())
    print(f"Computing features for {len(matches)} matches…")

    X_list = []
    y_list = []
    for i, m in enumerate(matches):
        try:
            x = team_features_delta(m4, m2, m["radiant"], m["dire"])
            X_list.append(x)
            y_list.append(1.0 if m["radiant_win"] else 0.0)
        except Exception as e:
            continue
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(matches)}")
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Feature matrix: X={X.shape}, y={y.shape}")
    print(f"Class balance: P(radiant_win) = {y.mean():.4f}")

    # Train/test split: last 20% holdout
    n = len(X)
    split = int(n * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    feature_names = ["base_wr", "pos_fit", "with_syn", "vs_adv", "role_gap"]

    # Logistic regression: minimize negative log-likelihood
    def loss(w, X, y):
        intercept = w[0]
        coef = w[1:]
        z = X @ coef + intercept
        # Numerically stable logistic loss
        ll = np.sum(np.log1p(np.exp(-y * z + (1 - y) * (-z))))
        # Equivalent to standard binary cross-entropy
        # but with explicit numerical care
        eps = 1e-9
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    w0 = np.zeros(6)
    res = minimize(loss, w0, args=(X_tr, y_tr), method="L-BFGS-B")
    print(f"\nTraining converged: {res.success}, iters={res.nit}, train_loss={res.fun:.4f}")

    intercept = res.x[0]
    coef = res.x[1:]
    print(f"Intercept: {intercept:+.4f}")
    for name, c in zip(feature_names, coef):
        print(f"  {name:10s}: {c:+.4f}")

    # Eval on holdout
    z_te = X_te @ coef + intercept
    p_te = 1 / (1 + np.exp(-z_te))
    pred = (p_te >= 0.5).astype(float)
    acc = (pred == y_te).mean()
    log_loss = -np.mean(y_te * np.log(np.clip(p_te, 1e-9, 1 - 1e-9)) +
                       (1 - y_te) * np.log(np.clip(1 - p_te, 1e-9, 1 - 1e-9)))
    brier = ((p_te - y_te) ** 2).mean()
    print(f"\nHoldout: n={len(y_te)}, acc={acc:.4f}, log_loss={log_loss:.4f}, brier={brier:.4f}")

    weights = {
        "intercept": float(intercept),
        **{name: float(c) for name, c in zip(feature_names, coef)},
    }
    out_path = Path("research/data/m5_trained_weights.json")
    out_path.write_text(json.dumps(weights, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
