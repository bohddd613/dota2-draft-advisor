"""
Build compact frontend data bundles from STRATZ research data.

Outputs to /data/ (next to index.html):
  - data/heroes_v2.json       — hero metadata (id, displayName, roles)
  - data/position_stats.json  — per-position win/match counts
  - data/matchups.json        — compact vs/with table per hero
  - data/v7e_model.json       — GBM trees for browser
  - data/eligibility.json     — STRATZ data-driven position eligibility per hero
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)


# 1) Heroes — keep id, name, displayName, shortName, roles, primaryAttribute
heroes_raw = json.loads((DATA_DIR / "heroes.json").read_text())
heroes_compact = []
for h in heroes_raw:
    heroes_compact.append({
        "id": h["id"],
        "name": h["name"],
        "displayName": h["displayName"],
        "shortName": h["shortName"],
        "roles": h.get("roles") or [],
        "primaryAttribute": (h.get("stats") or {}).get("primaryAttribute") or "",
    })
(OUT / "heroes_v2.json").write_text(json.dumps(heroes_compact))

# 2) Position stats
pos_stats_raw = json.loads((DATA_DIR / "position_stats_DIVINE_IMMORTAL.json").read_text())
# Keep only essential fields
pos_compact = {}
for pos, rows in pos_stats_raw.items():
    pos_compact[pos] = [{"heroId": r["heroId"], "matchCount": r["matchCount"], "winCount": r["winCount"]} for r in rows]
(OUT / "position_stats.json").write_text(json.dumps(pos_compact))

# 3) Matchups — for each hero, drop matchCount<30 entries, keep heroId2, matchCount, winCount, synergy
mu_raw = json.loads((DATA_DIR / "matchups_DIVINE_IMMORTAL.json").read_text())
mu_compact = {}
total_kept = 0
total_drop = 0
for hid, mu in mu_raw.items():
    if not mu:
        continue
    vs = []
    for e in mu.get("vs") or []:
        if e["matchCount"] < 30:
            total_drop += 1
            continue
        vs.append({"id": e["heroId2"], "m": e["matchCount"], "w": e["winCount"], "s": round(e["synergy"], 3)})
        total_kept += 1
    wi = []
    for e in mu.get("with") or []:
        if e["matchCount"] < 30:
            total_drop += 1
            continue
        wi.append({"id": e["heroId2"], "m": e["matchCount"], "w": e["winCount"], "s": round(e["synergy"], 3)})
        total_kept += 1
    mu_compact[int(hid)] = {"vs": vs, "with": wi}
(OUT / "matchups.json").write_text(json.dumps(mu_compact))
print(f"matchup entries: {total_kept} kept, {total_drop} dropped (matchCount<30)")

# 4) V7e model trees
v7e = json.loads((DATA_DIR / "v7e_model.json").read_text())
(OUT / "v7e_model.json").write_text(json.dumps(v7e))

# 5) Eligibility — derived from position stats + thresholds
MIN_PCT_OF_TOP = 0.20
MIN_MATCHES = 200
hero_pos_matches = {}
for pos, rows in pos_compact.items():
    for r in rows:
        hero_pos_matches.setdefault(r["heroId"], {})[pos] = r["matchCount"]
elig = {}
for hid, by_pos in hero_pos_matches.items():
    sorted_pos = sorted(by_pos.items(), key=lambda kv: -kv[1])
    top_matches = sorted_pos[0][1] if sorted_pos else 0
    out_list = []
    for pos, m in sorted_pos:
        if m >= MIN_MATCHES and (top_matches == 0 or m / top_matches >= MIN_PCT_OF_TOP):
            out_list.append(int(pos.split("_")[1]))  # POSITION_3 -> 3
    elig[hid] = out_list
(OUT / "eligibility.json").write_text(json.dumps(elig))

# Print sizes
print()
print("frontend data:")
for f in sorted(OUT.glob("*.json")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
