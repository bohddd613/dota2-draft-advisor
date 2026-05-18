"""
Parity test: Python V10 feature vector vs JS V10 feature vector for the same input.

Generates a small fixture, runs Python feature extraction, and prints the vector
plus the model's prediction. The same fixture is documented for manual JS
re-running (open dev console on the deployed app).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v10 import hero_features_v10  # noqa: E402


def main():
    ctx = Context(); m2 = M2_DataPositions(ctx); m4 = M4_RoleGap(ctx)

    # Fixture: candidate Enigma (33) at pos 4 with this team and enemy
    # heroes match v9.js logic style.
    cases = [
        {"cand": 33, "pos": 4,
         "allies": [(12, 1), (5, 3), (8, 2)],
         "enemies": [(7, 3), (110, 5), (1, 1), (74, 2)]},
        {"cand": 1, "pos": 1,
         "allies": [(33, 4), (5, 5), (8, 2)],
         "enemies": [(7, 3), (110, 5), (89, 4), (74, 2)]},
        {"cand": 89, "pos": 1,
         "allies": [(11, 2), (5, 5), (8, 3), (88, 4)],
         "enemies": [(7, 3), (110, 5), (12, 1), (74, 2)]},
    ]

    booster = lgb.Booster(model_file=str(DATA_DIR / "v10c_fair_ranker.txt"))

    print("Python parity fixture for V10:")
    for i, c in enumerate(cases):
        f = hero_features_v10(m4, m2, c["cand"], c["pos"], c["allies"], c["enemies"])
        score = booster.predict(np.array([f]))[0]
        print(f"\nCase {i+1}: candidate={c['cand']} pos={c['pos']}")
        print(f"  allies={c['allies']} enemies={c['enemies']}")
        print(f"  vec[0..2]:   {f[:3].tolist()}")
        print(f"  vec[25..38]: {f[25:].tolist()}")
        print(f"  raw_score: {score:.6f}")

    print("\n--- JS replication ---")
    print("Open https://dota2-drafter-xscnvzaw.devinapps.com and run in console:")
    print(("  await V10.init();" if False else "  // ensure V10 selected"))
    print("  V10.features(33, 4, [{id:12,position:1},{id:5,position:3},{id:8,position:2}], "
          "[{id:7,position:3},{id:110,position:5},{id:1,position:1},{id:74,position:2}])")


if __name__ == "__main__":
    main()
