"""
Fetch high-quality data from STRATZ GraphQL API for backtesting and frontend use.

Outputs JSON files into research/data/:
  - heroes.json: hero metadata (id, displayName, roles)
  - position_stats.json: per-position stats for each hero (matchCount, winrate)
  - matchups.json: per-hero matchup data (vs / with)

Designed to respect rate limits (~500 req/day free tier). Re-runs are idempotent
because results are cached on disk.

Usage:
  STRATZ_API_KEY=... python3 research/stratz_fetch.py [--bracket DIVINE_IMMORTAL]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

STRATZ_URL = "https://api.stratz.com/graphql"


def gql(query: str, variables: dict | None = None) -> dict:
    key = os.environ["STRATZ_API_KEY"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API",
    }
    body = {"query": query}
    if variables:
        body["variables"] = variables
    for attempt in range(4):
        try:
            r = requests.post(STRATZ_URL, headers=headers, json=body, timeout=60)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"[stratz] rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL errors: {data['errors']}")
            return data["data"]
        except Exception as e:
            print(f"[stratz] attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError("STRATZ request failed after 4 attempts")


def fetch_heroes() -> list[dict]:
    print("[stratz] fetching hero constants…")
    q = """
    query {
      constants {
        heroes(language: ENGLISH) {
          id
          name
          displayName
          shortName
          roles { roleId level }
          stats { primaryAttribute attackType }
        }
      }
    }
    """
    data = gql(q)
    return data["constants"]["heroes"]


POSITIONS = ["POSITION_1", "POSITION_2", "POSITION_3", "POSITION_4", "POSITION_5"]


def fetch_position_stats(bracket: str) -> dict:
    print(f"[stratz] fetching position stats (bracket={bracket})…")
    out = {}
    q = """
    query Pos($pos: [MatchPlayerPositionType!], $bracket: [RankBracketBasicEnum!]) {
      heroStats {
        stats(positionIds: $pos, bracketBasicIds: $bracket) {
          heroId
          position
          winCount
          matchCount
        }
      }
    }
    """
    for pos in POSITIONS:
        d = gql(q, {"pos": [pos], "bracket": [bracket]})
        rows = d["heroStats"]["stats"]
        out[pos] = rows
        print(f"  {pos}: {len(rows)} entries")
        time.sleep(0.5)
    return out


def fetch_matchups(hero_ids: list[int], bracket: str) -> dict:
    """For each hero, get top vs/with matchups.

    STRATZ matchUp endpoint returns top entries by sample size.
    """
    print(f"[stratz] fetching matchups for {len(hero_ids)} heroes (bracket={bracket})…")
    out = {}
    q = """
    query MU($id: Short!, $bracket: [RankBracketBasicEnum!]) {
      heroStats {
        matchUp(heroId: $id, bracketBasicIds: $bracket, take: 200) {
          heroId
          vs { heroId2 winCount matchCount synergy }
          with { heroId2 winCount matchCount synergy }
        }
      }
    }
    """
    for i, hid in enumerate(hero_ids):
        try:
            d = gql(q, {"id": hid, "bracket": [bracket]})
            mu = d["heroStats"]["matchUp"]
            out[hid] = mu[0] if mu else None
        except Exception as e:
            print(f"  hero {hid}: error {e}")
            out[hid] = None
        if (i + 1) % 10 == 0:
            print(f"  fetched {i + 1}/{len(hero_ids)}")
        time.sleep(0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracket", default="DIVINE_IMMORTAL", help="One of HERALD_GUARDIAN, CRUSADER_ARCHON, LEGEND_ANCIENT, DIVINE_IMMORTAL")
    ap.add_argument("--skip-matchups", action="store_true", help="Skip matchup fetch (fast mode)")
    ap.add_argument("--limit-heroes", type=int, default=0, help="Only fetch matchups for N heroes (testing)")
    args = ap.parse_args()

    if "STRATZ_API_KEY" not in os.environ:
        print("error: STRATZ_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    heroes_path = DATA_DIR / "heroes.json"
    if not heroes_path.exists():
        heroes = fetch_heroes()
        heroes_path.write_text(json.dumps(heroes, indent=2))
    else:
        heroes = json.loads(heroes_path.read_text())
    print(f"[stratz] {len(heroes)} heroes")

    pos_path = DATA_DIR / f"position_stats_{args.bracket}.json"
    if not pos_path.exists():
        pos_stats = fetch_position_stats(args.bracket)
        pos_path.write_text(json.dumps(pos_stats, indent=2))
    else:
        pos_stats = json.loads(pos_path.read_text())
        print(f"[stratz] using cached {pos_path.name}")

    if not args.skip_matchups:
        mu_path = DATA_DIR / f"matchups_{args.bracket}.json"
        if not mu_path.exists():
            ids = [h["id"] for h in heroes]
            if args.limit_heroes:
                ids = ids[: args.limit_heroes]
            mus = fetch_matchups(ids, args.bracket)
            mu_path.write_text(json.dumps(mus, indent=2))
        else:
            print(f"[stratz] using cached {mu_path.name}")

    print("[stratz] done. data in", DATA_DIR)


if __name__ == "__main__":
    main()
