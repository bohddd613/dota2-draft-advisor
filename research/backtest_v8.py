"""
V8 vs V7e vs M3 backtest on 1381 Divine+ STRATZ-enriched matches.

Reuses the same pick-rec / win-pred / restricted-pool / agreement methodology
from research/compare_v7e_vs_m3.py.

Outputs research/data/backtest_v8_results.json + console summary.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap, POSITION_KEYS
from train_v8 import hero_features_v8, FEATURE_NAMES_V8


POS_TO_INT = {f"POSITION_{i}": i for i in range(1, 6)}


def load_matches(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    out = []
    for m in raw:
        rad, dire = [], []
        for p in m.get("players", []):
            pos = POS_TO_INT.get(p["position"])
            if pos is None: continue
            (rad if p["is_radiant"] else dire).append((p["hero_id"], pos))
        if len(rad) != 5 or len(dire) != 5: continue
        out.append({"match_id": m["match_id"], "radiant_win": m["radiant_win"],
                    "radiant": rad, "dire": dire})
    return out


def infer_position(ctx, hid: int) -> int:
    best = (0, 0)
    for p in range(1, 6):
        rec = ctx.pos_stats[POSITION_KEYS[p]].get(hid)
        if rec and rec["matchCount"] > best[1]:
            best = (p, rec["matchCount"])
    return best[0]


# ----------------------------- model wrappers -----------------------------

class ModelV8:
    name = "V8"
    def __init__(self, ctx, m2, m4, clf):
        self.ctx = ctx; self.m2 = m2; self.m4 = m4; self.clf = clf
    def score_many(self, candidates, target_pos, allies_pos, enemies_pos):
        feats = np.vstack([
            hero_features_v8(self.m4, self.m2, h, target_pos, allies_pos, enemies_pos)
            for h in candidates
        ])
        return self.clf.predict_proba(feats)[:, 1]
    def eligible(self, h, pos):
        return pos in (self.m2.eligible.get(h) or [])


class ModelV7e:
    name = "V7e"
    def __init__(self, ctx, m2, m4, clf):
        self.ctx = ctx; self.m2 = m2; self.m4 = m4; self.clf = clf
    def score_many(self, candidates, target_pos, allies_pos, enemies_pos):
        # V7e uses 5-feature representation (ids only, no positions).
        ally_ids = [h for h, _ in allies_pos]
        enemy_ids = [h for h, _ in enemies_pos]
        feats = []
        for h in candidates:
            base_wr = self.m2.base_wr(h, target_pos) - 0.5
            pos_fit = self.m2.position_fit(h, target_pos)
            with_syn = self.m4.synergy(h, ally_ids) - 0.5
            vs_adv = self.m4.counter(h, enemy_ids) - 0.5
            ally_roles = self.m4._team_roles(ally_ids)
            cand_roles = {r["roleId"] for r in self.ctx.heroes.get(h, {}).get("roles", []) if r["level"] >= 2}
            missing = self.m4.KEY_ROLES - ally_roles
            role_gap = len(cand_roles & missing) / max(1, len(self.m4.KEY_ROLES))
            feats.append([base_wr, pos_fit, with_syn, vs_adv, role_gap])
        return self.clf.predict_proba(np.array(feats, dtype=np.float32))[:, 1]
    def eligible(self, h, pos):
        return pos in (self.m2.eligible.get(h) or [])


class ModelM3:
    """STRATZ R.O.S.H.-equivalent additive TS formula."""
    name = "M3"
    def __init__(self, ctx, m2, m4):
        self.ctx = ctx; self.m2 = m2; self.m4 = m4
        # Position-qualified set: hero matches at pos ≥ 200 AND 10% of hero's total
        MIN_POS_MATCHES = 200
        MIN_POS_PR = 0.10
        self.qualified: dict[int, set] = {p: set() for p in range(1, 6)}
        total_per_hero = defaultdict(int)
        for p_int, pkey in POSITION_KEYS.items():
            for hid, rec in ctx.pos_stats[pkey].items():
                total_per_hero[hid] += rec["matchCount"]
        for p_int, pkey in POSITION_KEYS.items():
            for hid, rec in ctx.pos_stats[pkey].items():
                mc = rec["matchCount"]
                t = total_per_hero[hid]
                if mc >= MIN_POS_MATCHES and t > 0 and mc / t >= MIN_POS_PR:
                    self.qualified[p_int].add(hid)
    def score_many(self, candidates, target_pos, allies_pos, enemies_pos):
        ally_ids = [h for h, _ in allies_pos]
        enemy_ids = [h for h, _ in enemies_pos]
        scores = []
        for h in candidates:
            base = self.m2.base_wr(h, target_pos) * 100 - 50  # pp
            syn = sum((self.m4.synergy(h, [a]) - 0.5) * 100 for a in ally_ids)
            ctr = sum((self.m4.counter(h, [e]) - 0.5) * 100 for e in enemy_ids)
            scores.append(base + syn + ctr)
        return np.array(scores)
    def eligible(self, h, pos):
        return h in self.qualified.get(pos, set())


# ----------------------------- evaluation ---------------------------------

def evaluate_unrestricted(model, matches, ctx, samples_per_match=2):
    """Standard pick-rec: each model uses its own eligibility."""
    rng = np.random.default_rng(42)
    ranks = []
    top1 = top5 = top10 = 0
    by_pos = defaultdict(lambda: [0, 0, 0, 0])  # [n, t1, t5, t10]
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = rng.choice(len(winners), size=min(samples_per_match, len(winners)), replace=False)
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies_pos = [(h, p) for h, p in winners if h != true_hero]
            enemies_pos = losers
            taken = set(team_ids) | set(enemy_ids)
            all_heroes = list(ctx.heroes.keys())
            eligible = [h for h in all_heroes if h not in taken and model.eligible(h, target_pos)]
            if true_hero not in eligible:
                eligible.append(true_hero)
            scores = model.score_many(eligible, target_pos, allies_pos, enemies_pos)
            order = np.argsort(-scores)
            ranked = [eligible[i] for i in order]
            try:
                rank = ranked.index(true_hero) + 1
            except ValueError:
                rank = len(ranked)
            ranks.append(rank)
            stat = by_pos[target_pos]
            stat[0] += 1
            if rank == 1: top1 += 1; stat[1] += 1
            if rank <= 5: top5 += 1; stat[2] += 1
            if rank <= 10: top10 += 1; stat[3] += 1
    n = len(ranks)
    pos_breakdown = {}
    for p, (np_, t1, t5, t10) in sorted(by_pos.items()):
        pos_breakdown[str(p)] = {"n": np_, "top1": t1/np_, "top5": t5/np_, "top10": t10/np_}
    return {
        "n": n, "mean_rank": float(np.mean(ranks)),
        "top1": top1/n, "top5": top5/n, "top10": top10/n,
        "by_pos": pos_breakdown,
    }


def evaluate_restricted(model, m3, matches, ctx, samples_per_match=2):
    """Apples-to-apples: candidate pool = M3.qualified[target_pos]."""
    rng = np.random.default_rng(42)
    ranks = []
    top1 = top5 = top10 = 0
    skipped = 0
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = rng.choice(len(winners), size=min(samples_per_match, len(winners)), replace=False)
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies_pos = [(h, p) for h, p in winners if h != true_hero]
            enemies_pos = losers
            pool = m3.qualified.get(target_pos, set())
            if true_hero not in pool:
                skipped += 1; continue
            taken = set(team_ids) | set(enemy_ids)
            candidates = [h for h in pool if h not in taken]
            if true_hero not in candidates:
                candidates.append(true_hero)
            scores = model.score_many(candidates, target_pos, allies_pos, enemies_pos)
            order = np.argsort(-scores)
            ranked = [candidates[i] for i in order]
            try:
                rank = ranked.index(true_hero) + 1
            except ValueError:
                rank = len(ranked)
            ranks.append(rank)
            if rank == 1: top1 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1
    n = len(ranks)
    return {
        "n": n, "mean_rank": float(np.mean(ranks)),
        "top1": top1/n, "top5": top5/n, "top10": top10/n,
        "skipped": skipped,
    }


def agreement_analysis(modelA, modelB, m3, matches, ctx, samples_per_match=2):
    """When A and B disagree on top-1 in restricted pool, who is right?"""
    rng = np.random.default_rng(42)
    A_wins = B_wins = ties = both_wrong = total = 0
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = rng.choice(len(winners), size=min(samples_per_match, len(winners)), replace=False)
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies_pos = [(h, p) for h, p in winners if h != true_hero]
            enemies_pos = losers
            pool = m3.qualified.get(target_pos, set())
            if true_hero not in pool: continue
            taken = set(team_ids) | set(enemy_ids)
            candidates = [h for h in pool if h not in taken]
            if true_hero not in candidates:
                candidates.append(true_hero)
            sA = modelA.score_many(candidates, target_pos, allies_pos, enemies_pos)
            sB = modelB.score_many(candidates, target_pos, allies_pos, enemies_pos)
            topA = candidates[int(np.argmax(sA))]
            topB = candidates[int(np.argmax(sB))]
            total += 1
            if topA == topB:
                ties += 1
            else:
                a_hit = (topA == true_hero)
                b_hit = (topB == true_hero)
                if a_hit and not b_hit: A_wins += 1
                elif b_hit and not a_hit: B_wins += 1
                else: both_wrong += 1
    return {
        "total": total, "tied_top1": ties,
        "A_wins": A_wins, "B_wins": B_wins, "both_wrong": both_wrong,
        "disagreements": total - ties,
    }


def evaluate_win_pred(model, matches, ctx):
    """Win-prob from sum-of-scores on actual picks."""
    preds = []; labels = []
    for m in matches:
        rad = m["radiant"]; dire = m["dire"]
        # Score each picked hero "as if" we were ranking it; sum gives a team-strength proxy.
        rad_sum = 0.0; dire_sum = 0.0
        rad_ids = [h for h, _ in rad]
        dire_ids = [h for h, _ in dire]
        for h, p in rad:
            allies_pos = [(hh, pp) for hh, pp in rad if hh != h]
            s = model.score_many([h], p, allies_pos, dire)[0]
            rad_sum += s
        for h, p in dire:
            allies_pos = [(hh, pp) for hh, pp in dire if hh != h]
            s = model.score_many([h], p, allies_pos, rad)[0]
            dire_sum += s
        diff = rad_sum - dire_sum
        # Sigmoid-style — but scores already in [0,1] (V8/V7e) or pp (M3); normalize.
        if model.name == "M3":
            # M3 scores are TrueSynergy in pp; diff in pp / 10 roughly maps to [-1,1]
            p_rad = 1 / (1 + math.exp(-diff / 10))
        else:
            p_rad = 1 / (1 + math.exp(-diff))
        preds.append(p_rad)
        labels.append(1 if m["radiant_win"] else 0)
    preds = np.array(preds); labels = np.array(labels)
    pred_class = (preds >= 0.5).astype(int)
    acc = float((pred_class == labels).mean())
    # log-loss (clip to avoid inf)
    eps = 1e-12
    pclip = np.clip(preds, eps, 1 - eps)
    ll = float(-np.mean(labels * np.log(pclip) + (1 - labels) * np.log(1 - pclip)))
    brier = float(np.mean((preds - labels) ** 2))
    # ECE (10 bins)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (preds >= lo) & (preds < hi if i < 9 else preds <= hi)
        if mask.sum() > 0:
            avg_p = preds[mask].mean()
            avg_y = labels[mask].mean()
            ece += abs(avg_p - avg_y) * mask.sum() / len(preds)
    return {"n": len(preds), "acc": acc, "log_loss": ll, "brier": brier, "ece": float(ece)}


# ----------------------------- main ---------------------------------------

def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"[backtest] {len(matches)} matches")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)

    clf_v8 = joblib.load(DATA_DIR / "v8a_gbm.joblib")
    clf_v7e = joblib.load(DATA_DIR / "v7e_gbm.joblib")

    v8 = ModelV8(ctx, m2, m4, clf_v8)
    v7e = ModelV7e(ctx, m2, m4, clf_v7e)
    m3 = ModelM3(ctx, m2, m4)

    report = {"n_matches": len(matches)}

    # Test 1: standard pick-rec
    print("\n=== Test 1: Standard pick-rec (own eligibility) ===")
    for mdl in [v8, v7e, m3]:
        r = evaluate_unrestricted(mdl, matches, ctx)
        print(f"  {mdl.name:5s}  n={r['n']}  top1={r['top1']:.4f}  top5={r['top5']:.4f}  top10={r['top10']:.4f}  mean_rank={r['mean_rank']:.2f}")
        report.setdefault("test1_standard", {})[mdl.name] = r

    # Test 2: apples-to-apples (M3-restricted pool)
    print("\n=== Test 2: Apples-to-apples (M3-restricted pool) ===")
    for mdl in [v8, v7e, m3]:
        r = evaluate_restricted(mdl, m3, matches, ctx)
        print(f"  {mdl.name:5s}  n={r['n']}  top1={r['top1']:.4f}  top5={r['top5']:.4f}  top10={r['top10']:.4f}  mean_rank={r['mean_rank']:.2f}")
        report.setdefault("test2_restricted", {})[mdl.name] = r

    # Test 3: pairwise agreement (V8 vs V7e, V8 vs M3, V7e vs M3)
    print("\n=== Test 3: Agreement (top-1 disagreement → who's right) ===")
    pairs = [("V8", v8, "V7e", v7e), ("V8", v8, "M3", m3), ("V7e", v7e, "M3", m3)]
    for nA, mA, nB, mB in pairs:
        r = agreement_analysis(mA, mB, m3, matches, ctx)
        print(f"  {nA} vs {nB}:  tied={r['tied_top1']}  {nA}_wins={r['A_wins']}  {nB}_wins={r['B_wins']}  both_wrong={r['both_wrong']}")
        report.setdefault("test3_agreement", {})[f"{nA}_vs_{nB}"] = r

    # Test 4: win-prediction
    print("\n=== Test 4: Win-prediction (full draft, team-score diff) ===")
    for mdl in [v8, v7e, m3]:
        r = evaluate_win_pred(mdl, matches, ctx)
        print(f"  {mdl.name:5s}  acc={r['acc']:.4f}  log_loss={r['log_loss']:.4f}  brier={r['brier']:.4f}  ece={r['ece']:.4f}")
        report.setdefault("test4_winpred", {})[mdl.name] = r

    out = DATA_DIR / "backtest_v8_results.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[backtest] saved → {out}")


if __name__ == "__main__":
    main()
