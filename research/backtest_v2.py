"""
Improved backtest using TRUE position labels from STRATZ-enriched matches.

Evaluates all 7 models on win-prediction and pick-recommendation.

Models:
  M0 — role-baseline (V0)
  M1 — curated HERO_POSITIONS (current production)
  M2 — STRATZ data-driven positions
  M3 — M2 + true `with`-synergy
  M4 — M3 + role-gap bonus
  M5 — logistic regression (default weights)
  M5* — logistic regression (trained weights from train_v2.py)
  M6 — gradient boosting (HistGradientBoostingClassifier)

Pick-recommendation evaluation:
  For each match, we know the actual position of every hero. For each hero,
  hide it from its team, then ask "what's the best hero to pick at <position>
  given <allies> vs <enemies>?". Compute rank of the actual hero in the
  ranking returned by the model.

Output: research/data/backtest_v2.json
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

from models import (  # noqa: E402
    Context, Baseline, M1_Curated, M2_DataPositions,
    M3_TrueSynergy, M4_RoleGap, M5_Logistic,
    HERO_POSITIONS_M1, POSITION_KEYS,
)

POS_TO_INT = {f"POSITION_{i}": i for i in range(1, 6)}


def load_matches(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    out = []
    for m in raw:
        rad, dire = [], []
        for p in m.get("players", []):
            pos = POS_TO_INT.get(p["position"])
            tup = (p["hero_id"], pos)
            (rad if p["is_radiant"] else dire).append(tup)
        if len(rad) != 5 or len(dire) != 5:
            continue
        if any(t[1] is None for t in rad + dire):
            continue
        out.append({
            "match_id": m["match_id"],
            "radiant_win": m["radiant_win"],
            "radiant": rad,
            "dire": dire,
        })
    return out


def true_team_strength(model, team: list[tuple[int, int]], enemy_ids: list[int]) -> float:
    """Sum of per-hero scores using TRUE positions."""
    total = 0.0
    team_ids = [h for h, _ in team]
    for hid, pos in team:
        allies = [h for h in team_ids if h != hid]
        s = model.score(hid, pos, allies, enemy_ids)
        total += s
    return total


def evaluate_win_pred(model, matches):
    correct, n = 0, 0
    log_loss = 0.0
    brier = 0.0
    for m in matches:
        rad_strength = true_team_strength(model, m["radiant"], [h for h, _ in m["dire"]])
        dire_strength = true_team_strength(model, m["dire"], [h for h, _ in m["radiant"]])
        # Convert to probability via softmax-like
        diff = rad_strength - dire_strength
        p_rad = 1 / (1 + math.exp(-diff))
        y = 1.0 if m["radiant_win"] else 0.0
        if (p_rad >= 0.5) == m["radiant_win"]:
            correct += 1
        eps = 1e-12
        log_loss += -(y * math.log(p_rad + eps) + (1 - y) * math.log(1 - p_rad + eps))
        brier += (p_rad - y) ** 2
        n += 1
    return {
        "name": model.name,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "log_loss": log_loss / n if n else float("inf"),
        "brier": brier / n if n else 1.0,
    }


def evaluate_pick_rec(model, matches, all_hero_ids: list[int], samples_per_match: int = 2):
    """For each match, sample hero positions from winning team, hide one,
    rank candidates eligible at that position, measure rank of true hero."""
    ranks = []
    top1 = top5 = top10 = 0
    np.random.seed(42)

    def model_eligible(h, pos):
        # M1 uses HERO_POSITIONS_M1
        if isinstance(model, M1_Curated):
            return pos in (HERO_POSITIONS_M1.get(h) or [])
        if hasattr(model, "eligible") and isinstance(model.eligible, dict):
            return pos in (model.eligible.get(h) or [])
        if hasattr(model, "m2"):
            return pos in (model.m2.eligible.get(h) or [])
        # M0: check role_weights coverage
        return model.position_fit(h, pos) > 0

    for m in matches:
        # Use winning team only, since picks of losers may be "bad" picks
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        # Pick `samples_per_match` random hero positions from winners
        idx_choices = np.random.choice(len(winners), size=min(samples_per_match, len(winners)), replace=False)
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [h for h in team_ids if h != true_hero]
            # Candidate set: any hero eligible at target_pos, excluding heroes already in either team
            taken = set(team_ids) | set(enemy_ids)
            candidates = [h for h in all_hero_ids if h not in taken and model_eligible(h, target_pos)]
            # Always include true_hero (it was in team but we're hiding it for this evaluation)
            if true_hero not in candidates:
                candidates.append(true_hero)
            scored = [(h, model.score(h, target_pos, allies, enemy_ids)) for h in candidates]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            rank = next((i + 1 for i, (h, _) in enumerate(scored) if h == true_hero), len(scored))
            ranks.append(rank)
            if rank == 1: top1 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1

    n = len(ranks)
    return {
        "name": model.name,
        "n": n,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "median_rank": float(np.median(ranks)) if ranks else 0.0,
        "top1": top1 / n if n else 0.0,
        "top5": top5 / n if n else 0.0,
        "top10": top10 / n if n else 0.0,
    }


class V7e_GBM:
    """Wrapper around exportable GBM (sklearn GradientBoostingClassifier).

    Trained for pick-prediction (not win-prediction). Use score() for ranking.
    """
    name = "V7e_gbm_pickrec"

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.m2 = M2_DataPositions(ctx)
        self.m4 = M4_RoleGap(ctx)
        try:
            import joblib
            self.model = joblib.load(DATA_DIR / "v7e_gbm.joblib")
        except Exception:
            self.model = None

    def position_fit(self, hid, pos):
        return self.m2.position_fit(hid, pos)

    def score(self, hid, pos, allies, enemies):
        if self.position_fit(hid, pos) == 0:
            return 0.0
        if self.model is None:
            return 0.5
        from train_v2 import hero_features as hf
        f = hf(self.m4, self.m2, hid, pos, allies, enemies)
        try:
            return float(self.model.predict_proba(f.reshape(1, -1))[0, 1])
        except Exception:
            return 0.5


class V6_GBM:
    """Wrapper around trained GBM that exposes score(hid,pos,allies,enemies)."""
    name = "V6_gbm"

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.m2 = M2_DataPositions(ctx)
        self.m4 = M4_RoleGap(ctx)
        try:
            import joblib
            self.model = joblib.load(DATA_DIR / "v6_gbm.joblib")
        except Exception:
            self.model = None
        # GBM was trained on team-delta features, but for hero ranking we use
        # the per-hero feature, mapped through a calibrated unary classifier.
        # We approximate it by computing the hero's contribution via a single
        # feature vector of the same dim and using model.predict_proba.

    def position_fit(self, hid, pos):
        return self.m2.position_fit(hid, pos)

    def hero_features(self, hid, pos, allies, enemies):
        from train_v2 import hero_features as hf
        return hf(self.m4, self.m2, hid, pos, allies, enemies)

    def score(self, hid, pos, allies, enemies):
        if self.position_fit(hid, pos) == 0:
            return 0.0
        if self.model is None:
            return 0.5
        f = self.hero_features(hid, pos, allies, enemies)
        try:
            p = float(self.model.predict_proba(f.reshape(1, -1))[0, 1])
        except Exception:
            p = 0.5
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples-per-match", type=int, default=2)
    args = ap.parse_args()

    matches = load_matches(Path(args.matches))
    if args.limit:
        matches = matches[: args.limit]
    print(f"[backtest] loaded {len(matches)} matches with TRUE position labels")

    if not matches:
        print("error: no enriched matches available")
        return

    ctx = Context()
    all_hero_ids = list(ctx.heroes.keys())

    # Load trained weights for M5*
    m5_default = M5_Logistic(ctx)
    m5_trained_weights = None
    twp = DATA_DIR / "v5_weights.json"
    if twp.exists():
        m5_trained_weights = json.loads(twp.read_text())
    m5_trained = M5_Logistic(ctx, weights=m5_trained_weights) if m5_trained_weights else None
    if m5_trained is not None:
        m5_trained.name = "M5*_logistic_trained"

    models = [
        Baseline(ctx),
        M1_Curated(ctx),
        M2_DataPositions(ctx),
        M3_TrueSynergy(ctx),
        M4_RoleGap(ctx),
        m5_default,
    ]
    if m5_trained is not None:
        models.append(m5_trained)
    # GBM if available
    v6 = V6_GBM(ctx)
    if v6.model is not None:
        models.append(v6)
    v7e = V7e_GBM(ctx)
    if v7e.model is not None:
        models.append(v7e)

    print(f"[backtest] evaluating {len(models)} models...\n")
    print("=== WIN PREDICTION ===")
    win_results = []
    for m in models:
        r = evaluate_win_pred(m, matches)
        print(f"  {r['name']:30s}  acc={r['accuracy']:.4f}  ll={r['log_loss']:.4f}  brier={r['brier']:.4f}")
        win_results.append(r)

    print("\n=== PICK RECOMMENDATION ===")
    pick_results = []
    for m in models:
        r = evaluate_pick_rec(m, matches, all_hero_ids, samples_per_match=args.samples_per_match)
        print(f"  {r['name']:30s}  mean={r['mean_rank']:.2f}  top1={r['top1']:.4f}  top5={r['top5']:.4f}  top10={r['top10']:.4f}")
        pick_results.append(r)

    out = {
        "n_matches": len(matches),
        "win_prediction": win_results,
        "pick_recommendation": pick_results,
    }
    (DATA_DIR / "backtest_v2.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved → backtest_v2.json")


if __name__ == "__main__":
    main()
