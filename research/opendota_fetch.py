"""
Fetch ground-truth matches from OpenDota for backtesting.

We use /publicMatches (recent ranked matches) which exposes:
  - radiant_team / dire_team: comma-separated hero IDs
  - radiant_win: bool
  - avg_rank_tier: average rank tier of the match
  - lobby_type / game_mode

We filter to:
  - lobby_type=7 (ranked all pick) and game_mode=22 (all pick ranked)
  - avg_rank_tier >= 60 (Divine 1+) for high-quality drafts

Outputs research/data/matches.json — list of match dicts for backtesting.

Usage:
  python3 research/opendota_fetch.py [--target 5000] [--min-rank 70]
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

BASE = "https://api.opendota.com/api"


def get(path: str, params: dict | None = None) -> list | dict:
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}{path}", params=params or {}, timeout=30)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"[od] 429, sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[od] attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError("OpenDota request failed")


def fetch_public_matches(target: int, min_rank: int, start_match_id: int | None) -> list[dict]:
    """Walk /publicMatches backward via less_than_match_id.

    OpenDota only fills hero IDs for matches a few hours/days old, so callers
    should pass an older `start_match_id` (e.g. 8_800_000_000) for backtesting.
    """
    out = []
    less_than = start_match_id
    iterations = 0
    while len(out) < target:
        iterations += 1
        params = {}
        if less_than:
            params["less_than_match_id"] = less_than
        batch = get("/publicMatches", params)
        if not batch:
            break
        less_than = min(m["match_id"] for m in batch)
        kept = 0
        for m in batch:
            if m.get("avg_rank_tier") is None or m["avg_rank_tier"] < min_rank:
                continue
            radiant = m.get("radiant_team")
            dire = m.get("dire_team")
            if isinstance(radiant, str):
                radiant = [int(x) for x in radiant.split(",") if x]
            if isinstance(dire, str):
                dire = [int(x) for x in dire.split(",") if x]
            if not radiant or not dire:
                continue
            if len(radiant) != 5 or len(dire) != 5:
                continue
            # Filter unparsed matches (all zeros)
            if all(h == 0 for h in radiant) or all(h == 0 for h in dire):
                continue
            out.append({
                "match_id": m["match_id"],
                "radiant_win": bool(m.get("radiant_win")),
                "radiant": list(radiant),
                "dire": list(dire),
                "avg_rank_tier": m["avg_rank_tier"],
                "duration": m.get("duration"),
                "start_time": m.get("start_time"),
            })
            kept += 1
        if iterations % 10 == 0 or kept > 0:
            print(f"[od] iter={iterations} batch={len(batch)} kept={kept} cumulative={len(out)} (less_than={less_than})")
        if len(out) >= target:
            break
        time.sleep(0.6)
    return out[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--min-rank", type=int, default=60, help="60=Divine, 70=Immortal")
    ap.add_argument("--start-match-id", type=int, default=8_800_000_000,
                    help="Use older match IDs (most recent matches lack parsed hero IDs)")
    ap.add_argument("--out", default="matches_public.json")
    args = ap.parse_args()
    print(f"[od] target={args.target} min_rank_tier={args.min_rank} start={args.start_match_id}")
    matches = fetch_public_matches(args.target, args.min_rank, args.start_match_id)
    out_path = DATA_DIR / args.out
    out_path.write_text(json.dumps(matches))
    print(f"[od] wrote {len(matches)} matches to {out_path}")


if __name__ == "__main__":
    main()
