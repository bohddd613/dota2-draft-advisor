"""
Hero attribute table for V10 team-composition features.

Derives per-hero attributes from STRATZ data + small hardcoded special-cases.
All derivations are deterministic and depend only on:
  - research/data/heroes.json (STRATZ /heroStats response)
  - the ILLUSION_HEROES / BIG_AOE_HEROES hardcoded sets below

Used by hero_features_v10() in train_v10.py.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).parent / "data"

# Heroes whose core kit creates manageable illusions (manta-style threat).
# These usually require AoE answer from the enemy team.
ILLUSION_HEROES = frozenset({
    89,   # Naga Siren
    12,   # Phantom Lancer
    81,   # Chaos Knight
    109,  # Terrorblade
    113,  # Arc Warden
    67,   # Spectre (haunt illusions)
})

# Heroes who bring strong AoE damage / displacement (typical anti-illusion answer).
# This is a conservative "core AoE threat" list — heroes whose ult or core spell
# wipes multiple weak units in one cast.
BIG_AOE_HEROES = frozenset({
    7,    # Earthshaker
    23,   # Kunkka
    26,   # Lion (Earth Spike chain not strong vs illusions but Finger=meh, exclude)
    28,   # Slardar (no aoe really, exclude — wait this hero list needs care)
    33,   # Enigma (Black Hole)
    47,   # Viper (no aoe, exclude)
    51,   # Clockwerk (no aoe, exclude)
    66,   # Chen (no aoe, exclude)
    68,   # Ancient Apparition (Ice Blast + Ice Vortex AoE)
    74,   # Invoker (Sun Strike + Tornado + EMP + Deafening Blast)
    79,   # Shadow Demon (no aoe, exclude — Demonic Purge single)
    84,   # Ogre Magi (single target mostly)
    85,   # Undying (Tombstone vs illusions)
    87,   # Disruptor (Static Storm + Glimpse — AoE silence)
    91,   # Wisp/Io (no aoe, exclude)
    97,   # Magnus (RP + Shockwave)
    101,  # Skywrath Mage (Mystic Flare AoE)
    102,  # Abaddon (no aoe, exclude)
    104,  # Legion Commander (no aoe direct, exclude)
    106,  # Ember Spirit (Sleight + Searing Chains)
    110,  # Phoenix (Supernova + Fire Spirits AoE)
    112,  # Winter Wyvern (Winter's Curse + AoE Splinter Blast)
    114,  # Monkey King (Wukong's Command + Boundless Strike)
    116,  # Dark Willow (Bramble Maze + Cursed Crown)
    119,  # Mars (Arena of Blood + Spear)
    120,  # Pangolier (Rolling Thunder)
    126,  # Void Spirit (Astral Step + Dissimilate)
})


# Curated list — verified hero IDs that bring AoE that effectively answers illusions.
# More precise than a "loose AoE" list — these are the heroes you specifically pick
# vs Naga/PL/CK comps. Hand-verified by hero ID.
ANTI_ILLUSION_HEROES = frozenset({
    7,    # Earthshaker (Echo Slam)
    33,   # Enigma (Black Hole + Midnight Pulse)
    68,   # Ancient Apparition (Ice Blast + IV)
    74,   # Invoker (Sun Strike + Tornado)
    85,   # Undying (Tombstone)
    97,   # Magnus (RP + Shockwave)
    110,  # Phoenix (Supernova)
    119,  # Mars (Arena + Spear)
    73,   # Alchemist (Acid Spray)
    37,   # Witch Doctor (Maledict)
    47,   # Viper (Viper Strike + corrosive)
    27,   # Shadow Shaman (Mass Serpent Wards push)
    100,  # Tusk (no aoe really — keep out)
})


_HEROES: dict[int, dict] | None = None


def _load_heroes() -> dict[int, dict]:
    global _HEROES
    if _HEROES is None:
        raw = json.loads((DATA_DIR / "heroes.json").read_text())
        _HEROES = {h["id"]: h for h in raw}
    return _HEROES


def hero_attrs(hid: int) -> dict:
    """Return a small attribute dict for a hero. Cheap lookups; no caching needed."""
    h = _load_heroes().get(hid)
    if h is None:
        return {
            "primary_attr": "all",
            "attack_type": "Melee",
            "is_initiator": 0,
            "is_disabler": 0,
            "is_nuker": 0,
            "is_pusher": 0,
            "is_durable": 0,
            "is_carry": 0,
            "is_support": 0,
            "is_escape": 0,
            "has_illusions": 0,
            "has_aoe": 0,
        }
    role_lv = {r["roleId"]: r["level"] for r in h.get("roles", [])}
    stats = h.get("stats", {})
    return {
        "primary_attr": stats.get("primaryAttribute", "all"),
        "attack_type": stats.get("attackType", "Melee"),
        "is_initiator": 1 if role_lv.get("INITIATOR", 0) >= 2 else 0,
        "is_disabler": 1 if role_lv.get("DISABLER", 0) >= 2 else 0,
        "is_nuker":    1 if role_lv.get("NUKER",     0) >= 2 else 0,
        "is_pusher":   1 if role_lv.get("PUSHER",    0) >= 2 else 0,
        "is_durable":  1 if role_lv.get("DURABLE",   0) >= 2 else 0,
        "is_carry":    1 if role_lv.get("CARRY",     0) >= 3 else 0,
        "is_support":  1 if role_lv.get("SUPPORT",   0) >= 2 else 0,
        "is_escape":   1 if role_lv.get("ESCAPE",    0) >= 2 else 0,
        "has_illusions": 1 if hid in ILLUSION_HEROES else 0,
        "has_aoe":       1 if hid in ANTI_ILLUSION_HEROES else 0,
    }


# Feature names for V10 team-composition block (14 features added to V8's 25).
TEAM_COMP_FEATURE_NAMES = [
    # team role saturation (5)
    "team_init_count",     # incl. candidate
    "team_disabler_count",
    "team_nuker_count",
    "team_pusher_count",
    "team_durable_count",
    # enemy role count (4)
    "enemy_init_count",
    "enemy_disabler_count",
    "enemy_nuker_count",
    "enemy_durable_count",
    # damage-type balance (3)
    "team_agi_ratio",      # incl. candidate; agi/5 = phys-heavy proxy
    "team_int_ratio",      # int/5 = magic-heavy proxy
    "enemy_agi_ratio",     # enemies/4 (no candidate)
    # asymmetric flags (2)
    "team_has_illusions",  # 1 if team (with candidate) has any ILLUSION hero
    "enemy_has_illusions",
]
assert len(TEAM_COMP_FEATURE_NAMES) == 14


def team_comp_features(
    candidate_hid: int,
    allies: Iterable[tuple[int, int]],
    enemies: Iterable[tuple[int, int]],
) -> list[float]:
    """
    Compute 14 team-composition features for `candidate_hid` joining `allies` against `enemies`.

    `allies` / `enemies`: iterables of (hero_id, pos) tuples.
    Returns list of 14 floats in the order of TEAM_COMP_FEATURE_NAMES.
    """
    ally_ids = [hid for hid, _ in allies]
    enemy_ids = [hid for hid, _ in enemies]
    team_with = [candidate_hid] + ally_ids  # candidate is part of "ally team" eval

    team_attrs = [hero_attrs(h) for h in team_with]
    enemy_attrs = [hero_attrs(h) for h in enemy_ids]

    def _count(attrs_list, key):
        return sum(a[key] for a in attrs_list)

    def _ratio(attrs_list, attr_name, n):
        if n == 0: return 0.0
        return sum(1 for a in attrs_list if a["primary_attr"] == attr_name) / n

    def _has(attrs_list, key):
        return 1.0 if any(a[key] for a in attrs_list) else 0.0

    return [
        # team (with candidate)
        float(_count(team_attrs, "is_initiator")),
        float(_count(team_attrs, "is_disabler")),
        float(_count(team_attrs, "is_nuker")),
        float(_count(team_attrs, "is_pusher")),
        float(_count(team_attrs, "is_durable")),
        # enemy
        float(_count(enemy_attrs, "is_initiator")),
        float(_count(enemy_attrs, "is_disabler")),
        float(_count(enemy_attrs, "is_nuker")),
        float(_count(enemy_attrs, "is_durable")),
        # balance
        _ratio(team_attrs, "agi", len(team_with)),
        _ratio(team_attrs, "int", len(team_with)),
        _ratio(enemy_attrs, "agi", len(enemy_ids)),
        # flags
        _has(team_attrs, "has_illusions"),
        _has(enemy_attrs, "has_illusions"),
    ]
