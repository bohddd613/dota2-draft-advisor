"""
Algorithm models for the Dota 2 draft advisor.

All models implement:
  - score(hero_id, position, allies, enemies) -> float in [0,1]
  - position_fit(hero_id, position) -> float in [0,1]

Models:
  M0 — legacy role-based heuristic (V0 baseline, before HERO_POSITIONS)
  M1 — current production (HERO_POSITIONS curated map + role-derived synergy)
  M2 — STRATZ data-driven positions (eligibility from matchCount thresholds)
  M3 — M2 + STRATZ "with" synergy (true team-pair winrates)
  M4 — M3 + opponent role-gap detection bonus
  M5 — Logistic regression on hand-engineered features (data-driven weights)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_heroes() -> dict[int, dict]:
    raw = json.loads((DATA_DIR / "heroes.json").read_text())
    return {h["id"]: h for h in raw}


def load_position_stats(bracket: str = "DIVINE_IMMORTAL") -> dict[str, dict[int, dict]]:
    """Returns {position_name: {hero_id: {matchCount, winCount, ...}}}."""
    raw = json.loads((DATA_DIR / f"position_stats_{bracket}.json").read_text())
    out = {}
    for pos, rows in raw.items():
        out[pos] = {r["heroId"]: r for r in rows}
    return out


def load_matchups(bracket: str = "DIVINE_IMMORTAL") -> dict[int, dict]:
    """Returns {hero_id: {vs: {hid2: entry}, with: {hid2: entry}}}."""
    raw = json.loads((DATA_DIR / f"matchups_{bracket}.json").read_text())
    out = {}
    for hid, mu in raw.items():
        if mu is None:
            continue
        vs = {e["heroId2"]: e for e in mu.get("vs") or []}
        wi = {e["heroId2"]: e for e in mu.get("with") or []}
        out[int(hid)] = {"vs": vs, "with": wi, "heroId": int(hid)}
    return out


POSITION_KEYS = {1: "POSITION_1", 2: "POSITION_2", 3: "POSITION_3", 4: "POSITION_4", 5: "POSITION_5"}

POSITION_RANK_DECAY = [1.0, 0.7, 0.5, 0.35]

# Same curated map as in app.js (V1 production).
HERO_POSITIONS_M1 = {
    1: [1, 2], 2: [3, 1], 3: [5, 4], 4: [1, 5], 5: [5, 4], 6: [1, 2], 7: [3, 4],
    8: [4, 1], 10: [3, 1], 11: [3, 1], 12: [3, 1], 13: [4, 5], 14: [3, 4],
    16: [3, 4], 17: [1, 2], 18: [3, 4], 19: [4, 5], 20: [4, 5], 21: [4, 5],
    22: [2, 1], 23: [3, 1], 25: [4, 5], 26: [5, 4], 27: [5, 4], 28: [3, 4],
    29: [3, 1], 30: [5, 4], 31: [5, 4], 32: [4, 1], 33: [3, 4], 34: [2, 1],
    35: [4, 5], 36: [4, 3], 37: [5, 4], 38: [4, 5], 39: [4, 5], 40: [4, 5],
    41: [3, 1], 42: [1, 3], 43: [3, 1], 44: [1, 2], 45: [4, 5], 46: [3, 4],
    47: [4, 5], 48: [3, 1], 49: [3, 1], 50: [4, 5], 51: [4, 5], 52: [4, 5],
    53: [3, 4], 54: [3, 1], 55: [4, 5], 56: [3, 4], 57: [3, 4], 58: [4, 5],
    59: [3, 1], 60: [4, 5], 61: [4, 5], 62: [4, 1], 63: [4, 5], 64: [5, 4],
    65: [2, 3], 66: [4, 5], 67: [4, 3], 68: [3, 4], 69: [3, 4], 70: [4, 5],
    71: [3, 1], 72: [3, 4], 73: [1, 2], 74: [4, 5], 75: [4, 5], 76: [3, 4],
    77: [3, 4], 78: [3, 1], 79: [4, 5], 80: [3, 1], 81: [4, 5], 82: [4, 5],
    83: [4, 5], 84: [4, 5], 85: [4, 5], 86: [4, 5], 87: [4, 5], 88: [3, 1],
    89: [1, 3], 90: [4, 5], 91: [4, 5], 92: [3, 4], 93: [4, 5], 94: [1, 2],
    95: [3, 1], 96: [3, 4], 97: [3, 4], 98: [3, 1], 99: [1, 4], 100: [4, 5],
    101: [3, 4], 102: [3, 5, 1], 103: [4, 5], 104: [3, 4], 105: [4, 5],
    106: [4, 5], 107: [3, 4], 108: [3, 1], 109: [3, 1], 110: [4, 5],
    111: [4, 5], 112: [4, 5], 113: [2, 1], 114: [4, 5], 119: [3, 4],
    120: [3, 1], 121: [4, 5], 123: [4, 5], 126: [3, 4], 128: [3, 1],
    129: [3, 1], 131: [3, 4], 135: [4, 5], 136: [3, 1], 137: [4, 5],
    138: [3, 4], 145: [3, 4], 149: [3, 1],
}

# Legacy V0 role-weight matrix (only for ablation, not active in production).
POSITION_ROLE_WEIGHTS = {
    1: {"CARRY": 4, "ESCAPE": 1, "PUSHER": 1},
    2: {"NUKER": 3, "CARRY": 2, "ESCAPE": 1, "DISABLER": 1},
    3: {"INITIATOR": 3, "DURABLE": 3, "DISABLER": 1, "NUKER": 1},
    4: {"SUPPORT": 2, "INITIATOR": 2, "DISABLER": 2, "NUKER": 1, "ESCAPE": 1},
    5: {"SUPPORT": 4, "DISABLER": 1, "DURABLE": 1},
}


@dataclass
class Context:
    heroes: dict[int, dict] = field(default_factory=load_heroes)
    pos_stats: dict[str, dict[int, dict]] = field(default_factory=load_position_stats)
    matchups: dict[int, dict] = field(default_factory=load_matchups)


def _wr(entry: dict | None) -> float:
    if not entry or not entry.get("matchCount"):
        return 0.5
    return entry["winCount"] / entry["matchCount"]


def _normalized(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# ----- Models ----------------------------------------------------------------


class Baseline:
    """V0 — naive role-based heuristic (the FIRST shipped version, before
    HERO_POSITIONS). Buggy by design — it's the baseline we're improving on.
    """

    name = "M0_role_baseline"

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def position_fit(self, hero_id: int, pos: int) -> float:
        h = self.ctx.heroes.get(hero_id)
        if not h:
            return 0.0
        roles = {r["roleId"] for r in h.get("roles", [])}
        weights = POSITION_ROLE_WEIGHTS[pos]
        score = sum(w for r, w in weights.items() if r in roles)
        return score / sum(weights.values())

    def base_wr(self, hero_id: int, pos: int) -> float:
        rec = self.ctx.pos_stats[POSITION_KEYS[pos]].get(hero_id)
        return _wr(rec)

    def counter(self, hero_id: int, enemies: list[int]) -> float:
        if not enemies:
            return 0.5
        mu = self.ctx.matchups.get(hero_id, {}).get("vs", {})
        adv = []
        for e in enemies:
            ent = mu.get(e)
            if not ent or ent["matchCount"] < 30:
                continue
            adv.append(0.5 + ent["synergy"] / 100.0)
        if not adv:
            return 0.5
        return sum(adv) / len(adv)

    def synergy(self, hero_id: int, allies: list[int]) -> float:
        if not allies:
            return 0.5
        roles_have = set()
        for a in allies:
            for r in self.ctx.heroes.get(a, {}).get("roles", []):
                roles_have.add(r["roleId"])
        new = {r["roleId"] for r in self.ctx.heroes.get(hero_id, {}).get("roles", [])} - roles_have
        return min(1.0, 0.5 + 0.05 * len(new))

    def score(self, hero_id: int, pos: int, allies: list[int], enemies: list[int]) -> float:
        if enemies:
            w = (0.25, 0.20, 0.40, 0.15)
        else:
            w = (0.45, 0.40, 0.00, 0.15)
        return (
            w[0] * _normalized(self.base_wr(hero_id, pos), 0.40, 0.60)
            + w[1] * self.position_fit(hero_id, pos)
            + w[2] * self.counter(hero_id, enemies)
            + w[3] * self.synergy(hero_id, allies)
        )


class M1_Curated(Baseline):
    """V1 — HERO_POSITIONS curated + role-derived synergy."""

    name = "M1_curated_positions"

    def position_fit(self, hero_id: int, pos: int) -> float:
        positions = HERO_POSITIONS_M1.get(hero_id)
        if not positions or pos not in positions:
            return 0.0
        idx = positions.index(pos)
        return POSITION_RANK_DECAY[idx] if idx < len(POSITION_RANK_DECAY) else 0.2


class M2_DataPositions(Baseline):
    """V2 — DATA-DRIVEN position eligibility from STRATZ matchCount.

    Hero is eligible at position iff:
      matchCount >= MIN_MATCHES AND >= MIN_PCT_OF_TOP * top_position_matches.
    """

    name = "M2_data_positions"
    MIN_PCT_OF_TOP = 0.20
    MIN_MATCHES = 200

    def __init__(self, ctx: Context):
        super().__init__(ctx)
        self.hero_pos_matches: dict[int, dict[int, int]] = {}
        for pos in range(1, 6):
            for hid, rec in ctx.pos_stats[POSITION_KEYS[pos]].items():
                self.hero_pos_matches.setdefault(hid, {})[pos] = rec["matchCount"]

        self.eligible: dict[int, list[int]] = {}
        for hid, by_pos in self.hero_pos_matches.items():
            sorted_pos = sorted(by_pos.items(), key=lambda kv: -kv[1])
            top_matches = sorted_pos[0][1] if sorted_pos else 0
            elig = []
            for pos, m in sorted_pos:
                if m >= self.MIN_MATCHES and (top_matches == 0 or m / top_matches >= self.MIN_PCT_OF_TOP):
                    elig.append(pos)
            self.eligible[hid] = elig

    def position_fit(self, hero_id: int, pos: int) -> float:
        elig = self.eligible.get(hero_id, [])
        if pos not in elig:
            return 0.0
        idx = elig.index(pos)
        return POSITION_RANK_DECAY[idx] if idx < len(POSITION_RANK_DECAY) else 0.2

    def base_wr(self, hero_id: int, pos: int) -> float:
        # Bayesian shrinkage toward 0.5 with pseudocount 200.
        rec = self.ctx.pos_stats[POSITION_KEYS[pos]].get(hero_id)
        if not rec or not rec["matchCount"]:
            return 0.5
        wins = rec["winCount"] + 100
        total = rec["matchCount"] + 200
        return wins / total


class M3_TrueSynergy(M2_DataPositions):
    """V3 — adds STRATZ true `with`-synergy data."""

    name = "M3_true_synergy"

    def synergy(self, hero_id: int, allies: list[int]) -> float:
        if not allies:
            return 0.5
        mu = self.ctx.matchups.get(hero_id, {}).get("with", {})
        vals = []
        for a in allies:
            ent = mu.get(a)
            if not ent or ent["matchCount"] < 30:
                continue
            vals.append(0.5 + ent["synergy"] / 100.0)
        if not vals:
            return 0.5
        return sum(vals) / len(vals)


class M4_RoleGap(M3_TrueSynergy):
    """V4 — adds role-gap detection bonus."""

    name = "M4_role_gap"
    KEY_ROLES = {"INITIATOR", "DISABLER", "SUPPORT", "DURABLE", "NUKER"}
    GAP_BONUS = 0.05

    def _team_roles(self, heroes: list[int]) -> set[str]:
        out = set()
        for h in heroes:
            for r in self.ctx.heroes.get(h, {}).get("roles", []):
                if r["level"] >= 2:
                    out.add(r["roleId"])
        return out

    def score(self, hero_id, pos, allies, enemies):
        base = super().score(hero_id, pos, allies, enemies)
        ally_roles = self._team_roles(allies)
        cand_roles = {r["roleId"] for r in self.ctx.heroes.get(hero_id, {}).get("roles", []) if r["level"] >= 2}
        missing = self.KEY_ROLES - ally_roles
        bonus = self.GAP_BONUS * len(cand_roles & missing) / max(1, len(self.KEY_ROLES))
        return min(1.0, base + bonus)


class M5_Logistic:
    """V5 — logistic regression on engineered features (weights fit on data)."""

    name = "M5_logistic"

    def __init__(self, ctx: Context, weights: dict | None = None):
        self.ctx = ctx
        self.m2 = M2_DataPositions(ctx)
        self.m4 = M4_RoleGap(ctx)
        self.weights = weights or {
            "intercept": 0.0,
            "base_wr": 4.0,
            "with_syn": 4.0,
            "vs_adv": 4.0,
            "pos_fit": 1.5,
            "role_gap": 1.0,
        }

    def features(self, hero_id, pos, allies, enemies):
        return {
            "base_wr": self.m2.base_wr(hero_id, pos) - 0.5,
            "with_syn": self.m4.synergy(hero_id, allies) - 0.5,
            "vs_adv": self.m4.counter(hero_id, enemies) - 0.5,
            "pos_fit": self.m2.position_fit(hero_id, pos),
            "role_gap": 0.0,
        }

    def position_fit(self, hero_id, pos):
        return self.m2.position_fit(hero_id, pos)

    def score(self, hero_id, pos, allies, enemies):
        if self.m2.position_fit(hero_id, pos) == 0:
            return 0.0
        f = self.features(hero_id, pos, allies, enemies)
        z = self.weights["intercept"]
        for k, v in f.items():
            z += self.weights.get(k, 0.0) * v
        return 1 / (1 + math.exp(-z))


# ----- Team-strength predictor for outcome backtest --------------------------


def assign_positions(team: list[int], elig_lookup) -> dict[int, int]:
    """Greedy hero→position assignment (Hungarian-lite).

    For each hero, prefer their primary eligible slot. If conflicts, fall back
    to next preference. Returns {hero_id: position}.
    """
    used = set()
    out = {}
    # Sort heroes by *fewest* eligible positions first (tighter constraint)
    sorted_heroes = sorted(team, key=lambda h: len(elig_lookup(h)))
    for hid in sorted_heroes:
        for pos in elig_lookup(hid):
            if pos not in used:
                out[hid] = pos
                used.add(pos)
                break
        if hid not in out:
            # Forced fallback — pick any free slot
            for pos in range(1, 6):
                if pos not in used:
                    out[hid] = pos
                    used.add(pos)
                    break
    return out


def team_strength(model, team: list[int], enemies: list[int]) -> float:
    """Sum of per-hero scores at greedy-assigned positions."""

    def elig(h):
        if hasattr(model, "eligible"):
            return model.eligible.get(h) or list(range(1, 6))
        if hasattr(model, "m2"):
            return model.m2.eligible.get(h) or list(range(1, 6))
        # M0/M1: use HERO_POSITIONS_M1 if available, else any
        if isinstance(model, M1_Curated):
            return HERO_POSITIONS_M1.get(h) or list(range(1, 6))
        return list(range(1, 6))

    pos_map = assign_positions(team, elig)
    total = 0.0
    for hid in team:
        pos = pos_map.get(hid, 1)
        allies = [a for a in team if a != hid]
        s = model.score(hid, pos, allies, enemies)
        total += s
    return total
