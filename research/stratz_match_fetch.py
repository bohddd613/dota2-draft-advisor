"""
Enrich existing OpenDota match IDs with STRATZ position labels.

Output: research/data/matches_stratz_enriched.json
Format: list of {match_id, radiant_win, players: [{hero_id, is_radiant, position, role, lane}], rank, bracket, gameMode}

Rate limits: STRATZ allows 20/sec, 250/min, 2000/hr, 10000/day. We sleep 0.3s between calls.

Usage:
  STRATZ_API_KEY=... python3 research/stratz_match_fetch.py [--limit N] [--max-calls N]
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

QUERY = """
query MM($id: Long!) {
  match(id: $id) {
    id
    didRadiantWin
    rank
    bracket
    averageRank
    gameMode
    durationSeconds
    players {
      heroId
      isRadiant
      position
      role
      lane
    }
  }
}
"""


def gql(mid: int) -> dict | None:
    key = os.environ["STRATZ_API_KEY"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "STRATZ_API",
    }
    body = {"query": QUERY, "variables": {"id": mid}}
    for attempt in range(4):
        try:
            r = requests.post(STRATZ_URL, headers=headers, json=body, timeout=60)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"[stratz] rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"[stratz] http {r.status_code} for match {mid}")
                return None
            data = r.json()
            if "errors" in data:
                # treat as missing match
                return None
            return data["data"]["match"]
        except Exception as e:
            print(f"[stratz] attempt {attempt + 1} failed for {mid}: {e}")
            time.sleep(5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="research/data/matches_public_divine.json")
    ap.add_argument("--output", default="research/data/matches_stratz_enriched.json")
    ap.add_argument("--limit", type=int, default=0, help="Max matches to fetch (0 = all)")
    ap.add_argument("--max-calls", type=int, default=2500, help="Hard cap on API calls per run")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    if "STRATZ_API_KEY" not in os.environ:
        print("error: STRATZ_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    inputs = json.loads(Path(args.input).read_text())
    out_path = Path(args.output)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = []
    seen = {m["match_id"] for m in existing}

    target_ids = [m["match_id"] for m in inputs]
    if args.limit:
        target_ids = target_ids[: args.limit]
    todo = [mid for mid in target_ids if mid not in seen]
    print(f"[stratz] {len(existing)} cached, {len(todo)} to fetch (cap {args.max_calls})")
    todo = todo[: args.max_calls]

    calls = 0
    started = time.time()
    for i, mid in enumerate(todo):
        m = gql(mid)
        calls += 1
        if m is None:
            time.sleep(args.sleep)
            continue
        players = m.get("players") or []
        # Sanity: must have 10 players, all with heroId and position
        if len(players) != 10 or any(p.get("heroId") in (None, 0) for p in players):
            time.sleep(args.sleep)
            continue
        radiant = [p["heroId"] for p in players if p["isRadiant"]]
        dire = [p["heroId"] for p in players if not p["isRadiant"]]
        if len(radiant) != 5 or len(dire) != 5:
            time.sleep(args.sleep)
            continue
        record = {
            "match_id": m["id"],
            "radiant_win": m["didRadiantWin"],
            "rank": m["rank"],
            "bracket": m["bracket"],
            "average_rank": m["averageRank"],
            "duration": m["durationSeconds"],
            "game_mode": m["gameMode"],
            "radiant": radiant,
            "dire": dire,
            "players": [
                {
                    "hero_id": p["heroId"],
                    "is_radiant": p["isRadiant"],
                    "position": p["position"],
                    "role": p["role"],
                    "lane": p["lane"],
                }
                for p in players
            ],
        }
        existing.append(record)
        seen.add(m["id"])
        if (i + 1) % 50 == 0:
            elapsed = time.time() - started
            rate = (i + 1) / max(1, elapsed)
            eta = (len(todo) - (i + 1)) / max(0.01, rate)
            print(f"  fetched {i + 1}/{len(todo)} | {rate:.2f}/s | eta {eta/60:.1f}m | total cached {len(existing)}")
            # Periodic save in case of interruption
            out_path.write_text(json.dumps(existing))
        time.sleep(args.sleep)

    out_path.write_text(json.dumps(existing))
    print(f"[stratz] done. {len(existing)} total enriched matches saved -> {out_path}")
    print(f"[stratz] used {calls} API calls this run")


if __name__ == "__main__":
    main()
