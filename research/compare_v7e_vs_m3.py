"""
Rigorous head-to-head comparison: V7e (trained GBM) vs M3 (STRATZ TrueSynergy).

Goals (in increasing order of fairness):
  1. Standard pick-rec on the union candidate pool (already done; recap).
  2. Apples-to-apples: restrict BOTH models to the same M3-qualified pool.
     This isolates the algorithmic question from the candidate-pool difference.
  3. Stratify pick-rec by position (1-5).
  4. Stratify pick-rec by hero popularity (was the true hero a "meta" pick?).
  5. Agreement analysis — when models disagree, who is right more often?
  6. Win-prediction with full calibration (log-loss, Brier, reliability bins).
  7. Direct R.O.S.H.-fidelity check on user's exact scenario.

Output: research/data/compare_v7e_vs_m3.json + console summary.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
APP_DATA = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from backtest_m3 import (
    M3_TrueSynergy, M1_Production, V7e_GBM,
    load_matches, evaluate_pick_rec, evaluate_win_pred,
)
from models import Context, HERO_POSITIONS_M1

# ----------------------- shared utilities ---------------------------------

def model_eligible(model, h, pos):
    if isinstance(model, M1_Production):
        return pos in (HERO_POSITIONS_M1.get(h) or [])
    if isinstance(model, M3_TrueSynergy):
        return pos in (model.eligible.get(h) or [])
    if hasattr(model, "m2"):
        return pos in (model.m2.eligible.get(h) or [])
    return model.position_fit(h, pos) > 0


def evaluate_pick_rec_restricted(
    model, matches, m3_model, samples_per_match=2,
):
    """
    Pick-rec where the candidate pool is restricted to M3-qualified heroes
    at the target position. Same evaluation on both models = fair comparison.
    """
    ranks = []
    top1 = top5 = top10 = 0
    skipped = 0
    np.random.seed(42)

    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = np.random.choice(
            len(winners), size=min(samples_per_match, len(winners)), replace=False
        )
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [h for h in team_ids if h != true_hero]
            # Restrict pool to M3-qualified heroes at target_pos.
            pool = m3_model.qualified.get(target_pos, set())
            taken = set(team_ids) | set(enemy_ids)
            candidates = [h for h in pool if h not in taken]
            # If true hero isn't in M3 pool, this sample is unrepresentable
            # by M3's restricted view — skip it for fairness.
            if true_hero not in pool:
                skipped += 1
                continue
            if true_hero not in candidates:
                candidates.append(true_hero)
            scored = [(h, model.score(h, target_pos, allies, enemy_ids))
                      for h in candidates]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            rank = next(
                (i + 1 for i, (h, _) in enumerate(scored) if h == true_hero),
                len(scored),
            )
            ranks.append(rank)
            if rank == 1: top1 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1
    n = len(ranks)
    return {
        "name": model.name,
        "n": n,
        "skipped": skipped,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "top1": top1 / n if n else 0.0,
        "top5": top5 / n if n else 0.0,
        "top10": top10 / n if n else 0.0,
    }


def evaluate_pick_rec_per_position(model, matches, all_hero_ids, samples_per_match=2):
    """Same as evaluate_pick_rec but bucketed by target position."""
    per_pos = defaultdict(lambda: {"ranks": [], "top1": 0, "top5": 0, "top10": 0})
    np.random.seed(42)
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = np.random.choice(
            len(winners), size=min(samples_per_match, len(winners)), replace=False
        )
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [h for h in team_ids if h != true_hero]
            taken = set(team_ids) | set(enemy_ids)
            candidates = [h for h in all_hero_ids
                          if h not in taken and model_eligible(model, h, target_pos)]
            if true_hero not in candidates:
                candidates.append(true_hero)
            scored = [(h, model.score(h, target_pos, allies, enemy_ids))
                      for h in candidates]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            rank = next(
                (i + 1 for i, (h, _) in enumerate(scored) if h == true_hero),
                len(scored),
            )
            bucket = per_pos[target_pos]
            bucket["ranks"].append(rank)
            if rank == 1: bucket["top1"] += 1
            if rank <= 5: bucket["top5"] += 1
            if rank <= 10: bucket["top10"] += 1

    out = {}
    for pos, b in per_pos.items():
        n = len(b["ranks"])
        out[pos] = {
            "n": n,
            "mean_rank": float(np.mean(b["ranks"])) if n else 0.0,
            "top1": b["top1"] / n if n else 0.0,
            "top5": b["top5"] / n if n else 0.0,
            "top10": b["top10"] / n if n else 0.0,
        }
    return out


def agreement_analysis(v7e, m3, matches, all_hero_ids, samples_per_match=2):
    """
    For each hidden-hero query, get top-1 from both models.
    Buckets:
      - agree_correct  : both same top-1 AND it == true hero
      - agree_wrong    : both same top-1 BUT != true hero (or one of them)
      - disagree_v7e_right : different top-1; V7e's top-1 == true; M3's didn't
      - disagree_m3_right  : different top-1; M3's top-1 == true; V7e's didn't
      - both_wrong         : different top-1; neither equals true hero
    Also compute: when they disagree, who reaches the true hero in top-3 first?
    """
    cnt = defaultdict(int)
    np.random.seed(42)
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = np.random.choice(
            len(winners), size=min(samples_per_match, len(winners)), replace=False
        )
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [h for h in team_ids if h != true_hero]
            taken = set(team_ids) | set(enemy_ids)

            def top1(model):
                candidates = [h for h in all_hero_ids
                              if h not in taken and model_eligible(model, h, target_pos)]
                if true_hero not in candidates:
                    candidates.append(true_hero)
                scored = [(h, model.score(h, target_pos, allies, enemy_ids))
                          for h in candidates]
                scored.sort(key=lambda kv: kv[1], reverse=True)
                return scored[0][0] if scored else None

            t1_v = top1(v7e)
            t1_m = top1(m3)
            agree = (t1_v == t1_m)
            v_right = (t1_v == true_hero)
            m_right = (t1_m == true_hero)
            if agree:
                cnt["agree_correct" if v_right else "agree_wrong"] += 1
            else:
                if v_right and not m_right:
                    cnt["disagree_v7e_right"] += 1
                elif m_right and not v_right:
                    cnt["disagree_m3_right"] += 1
                elif v_right and m_right:
                    cnt["disagree_both_right"] += 1  # shouldn't happen
                else:
                    cnt["disagree_both_wrong"] += 1
            cnt["total"] += 1
    return dict(cnt)


def evaluate_win_pred_calibrated(model, matches):
    """Win prediction with calibration metrics + reliability bins."""
    bin_count = 10
    bins = [[] for _ in range(bin_count)]
    correct = 0
    log_loss = 0.0
    brier = 0.0
    n = 0
    for m in matches:
        rad_ids = [h for h, _ in m["radiant"]]
        dire_ids = [h for h, _ in m["dire"]]
        rad_s, dire_s = 0.0, 0.0
        for hid, pos in m["radiant"]:
            allies = [h for h in rad_ids if h != hid]
            s = model.score(hid, pos, allies, dire_ids)
            if s > -1e8:
                rad_s += s
        for hid, pos in m["dire"]:
            allies = [h for h in dire_ids if h != hid]
            s = model.score(hid, pos, allies, rad_ids)
            if s > -1e8:
                dire_s += s
        diff = rad_s - dire_s
        # Models output very different scales. For V7e the diff is a sum of
        # probabilities (~10 numbers in [0,1]); for M3 it's a sum of pp.
        if isinstance(model, V7e_GBM):
            scale = 1.0
        else:
            scale = 1 / 30.0
        p_rad = 1 / (1 + math.exp(-diff * scale))
        y = 1.0 if m["radiant_win"] else 0.0
        if (p_rad >= 0.5) == m["radiant_win"]:
            correct += 1
        eps = 1e-12
        log_loss += -(y * math.log(p_rad + eps) + (1 - y) * math.log(1 - p_rad + eps))
        brier += (p_rad - y) ** 2
        n += 1

        bidx = min(bin_count - 1, int(p_rad * bin_count))
        bins[bidx].append((p_rad, y))

    reliability = []
    for i, b in enumerate(bins):
        if not b:
            reliability.append({"bin_lo": i / bin_count, "bin_hi": (i + 1) / bin_count,
                                "n": 0, "avg_pred": 0.0, "avg_actual": 0.0})
            continue
        avg_pred = float(np.mean([x[0] for x in b]))
        avg_act = float(np.mean([x[1] for x in b]))
        reliability.append({"bin_lo": i / bin_count, "bin_hi": (i + 1) / bin_count,
                            "n": len(b), "avg_pred": avg_pred, "avg_actual": avg_act})

    # Expected Calibration Error
    ece = 0.0
    for r in reliability:
        if r["n"] > 0:
            ece += (r["n"] / n) * abs(r["avg_pred"] - r["avg_actual"])

    return {
        "name": model.name,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "log_loss": log_loss / n if n else float("inf"),
        "brier": brier / n if n else 1.0,
        "ece": ece,
        "reliability": reliability,
    }


# ---------------------- main ---------------------------------------------

def main():
    print("Loading 1381 Divine+ matches with STRATZ ground-truth positions...")
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"  → {len(matches)} matches")

    ctx = Context()
    m1 = M1_Production(ctx)
    v7e = V7e_GBM(ctx)
    m3 = M3_TrueSynergy()
    all_hero_ids = sorted(set(ctx.heroes.keys()) | set(m3.all_hero_ids))

    out = {"n_matches": len(matches)}

    # 1. Standard pick-rec (recap from backtest_m3.json — but recompute for safety)
    print("\n[1/5] Standard pick-recommendation (full candidate pool)…")
    out["standard"] = []
    for label, mdl in [("M1", m1), ("V7e", v7e), ("M3", m3)]:
        r = evaluate_pick_rec(mdl, matches, all_hero_ids)
        r["label"] = label
        out["standard"].append(r)
        print(f"  {label:5s}  mean={r['mean_rank']:.2f}  "
              f"top1={r['top1']*100:.1f}%  top5={r['top5']*100:.1f}%  top10={r['top10']*100:.1f}%")

    # 2. Apples-to-apples — restrict candidate pool to M3-qualified.
    print("\n[2/5] APPLES-TO-APPLES (candidate pool = M3-qualified only)…")
    out["restricted"] = []
    for label, mdl in [("M1", m1), ("V7e", v7e), ("M3", m3)]:
        r = evaluate_pick_rec_restricted(mdl, matches, m3)
        r["label"] = label
        out["restricted"].append(r)
        print(f"  {label:5s}  n={r['n']}  skipped={r['skipped']}  "
              f"mean={r['mean_rank']:.2f}  top1={r['top1']*100:.1f}%  "
              f"top5={r['top5']*100:.1f}%  top10={r['top10']*100:.1f}%")

    # 3. Per-position breakdown
    print("\n[3/5] Per-position breakdown (top-10)…")
    out["per_position"] = {}
    for label, mdl in [("M1", m1), ("V7e", v7e), ("M3", m3)]:
        per_pos = evaluate_pick_rec_per_position(mdl, matches, all_hero_ids)
        out["per_position"][label] = per_pos
        row = "  ".join(f"P{p}={per_pos[p]['top10']*100:.1f}%(n={per_pos[p]['n']})"
                       for p in sorted(per_pos.keys()))
        print(f"  {label:5s}  {row}")

    # 4. Agreement analysis (V7e vs M3)
    print("\n[4/5] Agreement analysis: V7e vs M3 top-1…")
    agree = agreement_analysis(v7e, m3, matches, all_hero_ids)
    out["agreement"] = agree
    total = agree["total"]
    for k, v in agree.items():
        if k == "total": continue
        print(f"  {k:25s}  {v}  ({100*v/total:.1f}%)")

    # 5. Win-prediction with calibration
    print("\n[5/5] Win-prediction calibration…")
    out["win_pred"] = []
    for label, mdl in [("M1", m1), ("V7e", v7e), ("M3", m3)]:
        r = evaluate_win_pred_calibrated(mdl, matches)
        r["label"] = label
        out["win_pred"].append(r)
        print(f"  {label:5s}  acc={r['accuracy']*100:.1f}%  "
              f"log_loss={r['log_loss']:.3f}  brier={r['brier']:.4f}  ece={r['ece']:.4f}")

    out_path = DATA_DIR / "compare_v7e_vs_m3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
