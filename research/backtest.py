"""
Backtest framework: compare M0..M5 head-to-head on historical match outcomes.

Two evaluation methods:

1) Win-prediction accuracy (discriminative):
   For each match, compute team_strength for both sides given the actual
   drafted heroes. Predict winner = side with higher strength. Compute:
     - Accuracy
     - Log-loss (using sigmoid of strength delta)
     - Brier score

2) Pick-recommendation rank (supervised):
   For each match, simulate a sequential draft. At each step, hide one of
   the actual hero picks and ask the model to recommend a hero for that
   position given the partial state. Compute:
     - Mean rank of the actual pick in the model's ranking
     - Top-1 / Top-5 / Top-10 accuracy

Method (2) requires position labels. We infer them via greedy eligibility.

Usage:
  python3 research/backtest.py [--matches research/data/matches_public_divine.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from models import (
    Baseline, M1_Curated, M2_DataPositions, M3_TrueSynergy, M4_RoleGap,
    M5_Logistic, Context, assign_positions, HERO_POSITIONS_M1, team_strength,
)


def evaluate_win_prediction(model, matches: list[dict]) -> dict:
    correct = 0
    log_loss = 0.0
    brier = 0.0
    n = 0
    deltas = []
    for m in matches:
        try:
            sr = team_strength(model, m["radiant"], m["dire"])
            sd = team_strength(model, m["dire"], m["radiant"])
        except Exception:
            continue
        delta = sr - sd
        deltas.append(delta)
        # Sigmoid w/ scale 1.0 (strength is unitless sum of 0..1 scores per hero)
        p_radiant_win = 1 / (1 + math.exp(-delta))
        truth = 1 if m["radiant_win"] else 0
        pred = 1 if p_radiant_win >= 0.5 else 0
        if pred == truth:
            correct += 1
        # log-loss
        eps = 1e-9
        p = max(eps, min(1 - eps, p_radiant_win))
        log_loss += -(truth * math.log(p) + (1 - truth) * math.log(1 - p))
        brier += (p - truth) ** 2
        n += 1
    if n == 0:
        return {"name": model.name, "n": 0}
    return {
        "name": getattr(model, "name", type(model).__name__),
        "n": n,
        "accuracy": correct / n,
        "log_loss": log_loss / n,
        "brier": brier / n,
        "delta_mean": statistics.mean(deltas) if deltas else 0,
        "delta_std": statistics.pstdev(deltas) if deltas else 0,
    }


def evaluate_pick_recommendation(model, matches: list[dict], samples_per_match: int = 1) -> dict:
    """For each match, hide one random hero and check rank in model's recs.

    We use the WINNING team only (so that the actual pick is presumably good).
    """
    import random

    rng = random.Random(42)
    ranks = []
    top1 = 0
    top5 = 0
    top10 = 0
    n = 0

    def elig(h):
        if hasattr(model, "eligible"):
            return model.eligible.get(h) or list(range(1, 6))
        if hasattr(model, "m2"):
            return model.m2.eligible.get(h) or list(range(1, 6))
        if isinstance(model, M1_Curated):
            return HERO_POSITIONS_M1.get(h) or list(range(1, 6))
        return list(range(1, 6))

    for m in matches:
        winning = m["radiant"] if m["radiant_win"] else m["dire"]
        losing = m["dire"] if m["radiant_win"] else m["radiant"]
        # Assign positions to winning team
        pos_map = assign_positions(winning, elig)
        for _ in range(samples_per_match):
            target_idx = rng.randrange(5)
            target_hero = winning[target_idx]
            target_pos = pos_map.get(target_hero)
            if target_pos is None:
                continue
            allies = [h for i, h in enumerate(winning) if i != target_idx]
            enemies = losing
            # Score every candidate hero at target_pos
            scores = []
            for hid in range(1, 200):
                if hid == target_hero:
                    s = model.score(hid, target_pos, allies, enemies)
                    scores.append((hid, s))
                    continue
                if hid in allies or hid in enemies:
                    continue
                pf = 0.0
                if hasattr(model, "position_fit"):
                    pf = model.position_fit(hid, target_pos)
                elif hasattr(model, "m2"):
                    pf = model.m2.position_fit(hid, target_pos)
                if pf <= 0:
                    continue
                s = model.score(hid, target_pos, allies, enemies)
                scores.append((hid, s))
            scores.sort(key=lambda x: -x[1])
            rank = next((i for i, (h, _) in enumerate(scores, 1) if h == target_hero), None)
            if rank is None:
                # Hero not eligible per this model — skip this sample
                continue
            ranks.append(rank)
            if rank == 1:
                top1 += 1
            if rank <= 5:
                top5 += 1
            if rank <= 10:
                top10 += 1
            n += 1
    if n == 0:
        return {"name": getattr(model, "name", type(model).__name__), "n": 0}
    return {
        "name": getattr(model, "name", type(model).__name__),
        "n": n,
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
        "top1": top1 / n,
        "top5": top5 / n,
        "top10": top10 / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="research/data/matches_public_divine.json")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--rec-samples", type=int, default=1)
    args = ap.parse_args()

    matches = json.loads(Path(args.matches).read_text())[: args.limit]
    print(f"Loaded {len(matches)} matches for backtest")

    ctx = Context()
    models = [
        Baseline(ctx),
        M1_Curated(ctx),
        M2_DataPositions(ctx),
        M3_TrueSynergy(ctx),
        M4_RoleGap(ctx),
        M5_Logistic(ctx),
    ]

    print("\n=== Win-prediction backtest ===")
    win_results = []
    for m in models:
        r = evaluate_win_prediction(m, matches)
        win_results.append(r)
        print(f"  {r['name']:30s}  acc={r.get('accuracy', 0):.4f}  log_loss={r.get('log_loss', 0):.4f}  brier={r.get('brier', 0):.4f}  Δμ={r.get('delta_mean', 0):+.4f}  Δσ={r.get('delta_std', 0):.4f}")

    print("\n=== Pick-recommendation backtest (winner-team) ===")
    rec_results = []
    for m in models:
        r = evaluate_pick_recommendation(m, matches[:500], samples_per_match=args.rec_samples)
        rec_results.append(r)
        print(f"  {r['name']:30s}  n={r['n']:5d}  mean_rank={r.get('mean_rank', 0):.2f}  top1={r.get('top1', 0):.3f}  top5={r.get('top5', 0):.3f}  top10={r.get('top10', 0):.3f}")

    Path("research/data/backtest_results.json").write_text(json.dumps({
        "win_prediction": win_results,
        "pick_recommendation": rec_results,
    }, indent=2))
    print("\nWrote research/data/backtest_results.json")


if __name__ == "__main__":
    main()
