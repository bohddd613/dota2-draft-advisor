"""
Standalone backtest of M3 (TrueSynergy / R.O.S.H.-equivalent) on the same
1381 Divine+ matches used for V7e evaluation.

Reproduces M3 exactly as implemented in m3.js:
  TrueSynergy(h, position, allies, enemies)
    = (winrate(h, position) - 50)
    + Σ synergy(h, ally) over allies
    + Σ counter(h, enemy) over enemies

Position qualification:
  - matches at position ≥ MIN_POSITION_MATCHES
  - matches at position / total hero matches ≥ MIN_POSITION_PR
  - hero total matches ≥ small floor (avoid empty heroes)

Evaluates side-by-side with M1 and V7e using the existing backtest framework.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
APP_DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M1_Curated, M2_DataPositions, M4_RoleGap, HERO_POSITIONS_M1  # noqa: E402

POS_TO_INT = {f"POSITION_{i}": i for i in range(1, 6)}

# M3 thresholds — mirror m3.js exactly.
MIN_POSITION_MATCHES = 200
MIN_PAIR_MATCHES = 30
MIN_HERO_PR = 0.005
MIN_POSITION_PR = 0.10


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


class M3_TrueSynergy:
    """Pure STRATZ R.O.S.H.-style TrueSynergy. No training, no weights."""

    name = "M3_truesynergy"

    def __init__(self):
        # Load production data files (the same files frontend uses)
        pos_stats_raw = json.loads((APP_DATA_DIR / "position_stats.json").read_text())
        matchups_raw = json.loads((APP_DATA_DIR / "matchups.json").read_text())

        # Reshape position_stats: {hid: {pos_int: {matchCount, winCount}}}
        self.pos_stats: dict[int, dict[int, dict]] = {}
        for pkey, rows in pos_stats_raw.items():
            pos = int(pkey.split("_")[1])
            for r in rows:
                hid = r["heroId"]
                self.pos_stats.setdefault(hid, {})[pos] = {
                    "matchCount": r["matchCount"], "winCount": r["winCount"]}

        # Reshape matchups: {hid: {"vs": {oid: row}, "with": {oid: row}}}
        self.matchups: dict[int, dict] = {}
        for hid_str, mu in matchups_raw.items():
            hid = int(hid_str)
            self.matchups[hid] = {
                "vs": {e["id"]: e for e in mu.get("vs", [])},
                "with": {e["id"]: e for e in mu.get("with", [])},
            }

        # Pre-compute hero match totals + qualification sets.
        self.hero_total_matches: dict[int, int] = {
            hid: sum(r["matchCount"] for r in posmap.values())
            for hid, posmap in self.pos_stats.items()
        }
        total = sum(self.hero_total_matches.values())
        n_heroes = max(1, len(self.hero_total_matches))
        avg_hero = total / n_heroes
        floor = MIN_HERO_PR * avg_hero * 5  # rough hero PR proxy

        self.qualified: dict[int, set[int]] = {p: set() for p in range(1, 6)}
        for hid, posmap in self.pos_stats.items():
            ht = self.hero_total_matches.get(hid, 0)
            if ht < floor:
                continue
            for pos, r in posmap.items():
                if r["matchCount"] < MIN_POSITION_MATCHES:
                    continue
                if r["matchCount"] / max(1, ht) < MIN_POSITION_PR:
                    continue
                self.qualified[pos].add(hid)

    @property
    def all_hero_ids(self) -> list[int]:
        return list(self.pos_stats.keys())

    def position_fit(self, hid: int, pos: int) -> float:
        return 1.0 if hid in self.qualified.get(pos, set()) else 0.0

    @property
    def eligible(self) -> dict[int, list[int]]:
        # {hid: list of qualified positions}
        out: dict[int, list[int]] = {}
        for pos, hids in self.qualified.items():
            for h in hids:
                out.setdefault(h, []).append(pos)
        return out

    def base_winrate(self, hid: int, pos: int) -> float | None:
        r = self.pos_stats.get(hid, {}).get(pos)
        if not r or r["matchCount"] < MIN_POSITION_MATCHES:
            return None
        return 100.0 * r["winCount"] / r["matchCount"]

    def synergy_with(self, hid: int, ally: int) -> float:
        ent = self.matchups.get(hid, {}).get("with", {}).get(ally)
        if not ent or ent["m"] < MIN_PAIR_MATCHES:
            return 0.0
        return ent["s"]

    def counter_vs(self, hid: int, enemy: int) -> float:
        ent = self.matchups.get(hid, {}).get("vs", {}).get(enemy)
        if not ent or ent["m"] < MIN_PAIR_MATCHES:
            return 0.0
        return ent["s"]

    def score(self, hid: int, pos: int, allies: list[int], enemies: list[int]) -> float:
        if hid not in self.qualified.get(pos, set()):
            return -1e9  # disqualified -> bottom of ranking
        wr = self.base_winrate(hid, pos)
        if wr is None:
            return -1e9
        base = wr - 50.0
        syn = sum(self.synergy_with(hid, a) for a in allies)
        ctr = sum(self.counter_vs(hid, e) for e in enemies)
        return base + syn + ctr


class M1_Production:
    """M1 = curated HERO_POSITIONS — current production model.

    Matches scoring used in app.js: weighted (baseWR, posFit, counter, synergy).
    For backtest fidelity we reuse models.M1_Curated.
    """
    name = "M1_curated_production"

    def __init__(self, ctx):
        self._inner = M1_Curated(ctx)

    def position_fit(self, hid, pos):
        return self._inner.position_fit(hid, pos)

    def score(self, hid, pos, allies, enemies):
        return self._inner.score(hid, pos, allies, enemies)


class V7e_GBM:
    """V7e GBM (pick-rec trained). Mirrors backtest_v2.V7e_GBM."""
    name = "V7e_gbm"

    def __init__(self, ctx):
        self.ctx = ctx
        self.m2 = M2_DataPositions(ctx)
        self.m4 = M4_RoleGap(ctx)
        import joblib
        self.model = joblib.load(DATA_DIR / "v7e_gbm.joblib")

    def position_fit(self, hid, pos):
        return self.m2.position_fit(hid, pos)

    def score(self, hid, pos, allies, enemies):
        if self.position_fit(hid, pos) == 0:
            return -1e9
        from train_v2 import hero_features as hf
        f = hf(self.m4, self.m2, hid, pos, allies, enemies)
        return float(self.model.predict_proba(f.reshape(1, -1))[0, 1])


def evaluate_pick_rec(model, matches, all_hero_ids, samples_per_match=2):
    ranks = []
    top1 = top5 = top10 = 0
    np.random.seed(42)

    def model_eligible(h, pos):
        if isinstance(model, M1_Production):
            return pos in (HERO_POSITIONS_M1.get(h) or [])
        if isinstance(model, M3_TrueSynergy):
            return pos in (model.eligible.get(h) or [])
        if hasattr(model, "m2"):
            return pos in (model.m2.eligible.get(h) or [])
        return model.position_fit(h, pos) > 0

    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = np.random.choice(len(winners), size=min(samples_per_match, len(winners)), replace=False)
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [h for h in team_ids if h != true_hero]
            taken = set(team_ids) | set(enemy_ids)
            candidates = [h for h in all_hero_ids if h not in taken and model_eligible(h, target_pos)]
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


def evaluate_win_pred(model, matches):
    correct = 0; n = 0; log_loss = 0.0; brier = 0.0
    for m in matches:
        rad_team_ids = [h for h, _ in m["radiant"]]
        dire_team_ids = [h for h, _ in m["dire"]]
        rad_strength = 0.0
        for hid, pos in m["radiant"]:
            allies = [h for h in rad_team_ids if h != hid]
            s = model.score(hid, pos, allies, dire_team_ids)
            if s > -1e8:
                rad_strength += s
        dire_strength = 0.0
        for hid, pos in m["dire"]:
            allies = [h for h in dire_team_ids if h != hid]
            s = model.score(hid, pos, allies, rad_team_ids)
            if s > -1e8:
                dire_strength += s
        diff = rad_strength - dire_strength
        # diff is in pp for M3; scale to log-odds via a soft scaler
        scale = 1 / 30.0  # heuristic — diffs of ±60 → strong signal
        p_rad = 1 / (1 + math.exp(-diff * scale))
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


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"Loaded {len(matches)} matches with full STRATZ position labels.")

    ctx = Context()
    print(f"Heroes in context: {len(ctx.heroes)}")

    m1 = M1_Production(ctx)
    v7e = V7e_GBM(ctx)
    m3 = M3_TrueSynergy()

    all_hero_ids = sorted(set(ctx.heroes.keys()) | set(m3.all_hero_ids))
    print(f"All hero ids: {len(all_hero_ids)}")
    print(f"M3 qualified counts: " + ", ".join(f"P{p}={len(m3.qualified[p])}" for p in range(1, 6)))

    print("\n--- PICK-RECOMMENDATION ---")
    pick_results = []
    for label, model in [("M1", m1), ("V7e", v7e), ("M3", m3)]:
        r = evaluate_pick_rec(model, matches, all_hero_ids)
        r["label"] = label
        pick_results.append(r)
        print(f"{label:5s}  mean_rank={r['mean_rank']:.2f}  top1={r['top1']*100:.1f}%  top5={r['top5']*100:.1f}%  top10={r['top10']*100:.1f}%  (n={r['n']})")

    print("\n--- WIN-PREDICTION (Team strength differential) ---")
    win_results = []
    for label, model in [("M1", m1), ("V7e", v7e), ("M3", m3)]:
        r = evaluate_win_pred(model, matches)
        r["label"] = label
        win_results.append(r)
        print(f"{label:5s}  acc={r['accuracy']*100:.1f}%  log_loss={r['log_loss']:.3f}  brier={r['brier']:.4f}  (n={r['n']})")

    # Persist combined report
    out = {
        "n_matches": len(matches),
        "pick_recommendation": pick_results,
        "win_prediction": win_results,
        "thresholds": {
            "M3_MIN_POSITION_MATCHES": MIN_POSITION_MATCHES,
            "M3_MIN_PAIR_MATCHES": MIN_PAIR_MATCHES,
            "M3_MIN_HERO_PR": MIN_HERO_PR,
            "M3_MIN_POSITION_PR": MIN_POSITION_PR,
        },
    }
    out_path = DATA_DIR / "backtest_m3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
