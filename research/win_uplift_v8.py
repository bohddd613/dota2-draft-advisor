"""
Win-probability uplift analysis (Phase A verification).

Question: do the model's top-N recommendations actually correlate with match
outcomes? Or are we just predicting "what humans pick" (which could be
a meta-following proxy, not a winning-proxy)?

Methodology (for each model):
  For every winning team's pick, we count if model's top-K contained
  the true_hero. Then for every LOSING team's pick, same.
  If winners' picks are model-aligned more often than losers' picks,
  the model is identifying winning patterns (not just popularity).

Output: research/data/win_uplift_v8.json + console table.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap
from train_v8 import hero_features_v8
from backtest_v8 import load_matches, ModelV8, ModelV7e, ModelM3


def uplift_for_model(model, matches, ctx, K_values=(1, 5, 10)):
    """
    For every (team, true_pick, position) triple:
      - team_alignment = was true_pick in model's top-K for that team's state?
    Then compare: winners' alignment rate vs losers' alignment rate.

    Higher alignment rate on winners than losers → model identifies winning patterns.
    Same rate → model just predicts "what humans pick" (no winning signal).
    """
    rng = np.random.default_rng(42)
    win_hits = {k: 0 for k in K_values}
    lose_hits = {k: 0 for k in K_values}
    win_n = lose_n = 0
    for m in matches:
        for team_label in ("radiant", "dire"):
            team = m[team_label]
            other = m["dire"] if team_label == "radiant" else m["radiant"]
            team_won = (team_label == "radiant") == m["radiant_win"]
            team_ids = [h for h, _ in team]
            other_ids = [h for h, _ in other]
            # Randomly sample 2 picks from this team
            n_picks = min(2, len(team))
            idx_choices = rng.choice(len(team), size=n_picks, replace=False)
            for idx in idx_choices:
                true_hero, target_pos = team[idx]
                allies_pos = [(h, p) for h, p in team if h != true_hero]
                enemies_pos = other
                taken = set(team_ids) | set(other_ids)
                eligible = [h for h in ctx.heroes.keys()
                            if h not in taken and model.eligible(h, target_pos)]
                if true_hero not in eligible:
                    eligible.append(true_hero)
                scores = model.score_many(eligible, target_pos, allies_pos, enemies_pos)
                order = np.argsort(-scores)
                ranked = [eligible[i] for i in order]
                try:
                    rank = ranked.index(true_hero) + 1
                except ValueError:
                    rank = len(ranked) + 1
                bucket = win_hits if team_won else lose_hits
                for k in K_values:
                    if rank <= k:
                        bucket[k] += 1
                if team_won: win_n += 1
                else: lose_n += 1
    out = {}
    for k in K_values:
        wr = win_hits[k] / win_n if win_n else 0
        lr = lose_hits[k] / lose_n if lose_n else 0
        out[f"top{k}"] = {
            "winners_aligned_rate": float(wr),
            "losers_aligned_rate":  float(lr),
            "uplift_pp": float((wr - lr) * 100),  # percentage points
            "ratio": float(wr / lr) if lr else None,
        }
    return {"n_winners": win_n, "n_losers": lose_n, "metrics": out}


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"[uplift] {len(matches)} matches")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)

    clf_v8 = joblib.load(DATA_DIR / "v8a_gbm.joblib")
    clf_v7e = joblib.load(DATA_DIR / "v7e_gbm.joblib")

    v8 = ModelV8(ctx, m2, m4, clf_v8)
    v7e = ModelV7e(ctx, m2, m4, clf_v7e)
    m3 = ModelM3(ctx, m2, m4)

    results = {}
    for mdl in [v8, v7e, m3]:
        print(f"\n=== {mdl.name} ===")
        r = uplift_for_model(mdl, matches, ctx)
        results[mdl.name] = r
        print(f"  n_winners={r['n_winners']}  n_losers={r['n_losers']}")
        for k, m_ in r["metrics"].items():
            sign = "+" if m_["uplift_pp"] >= 0 else ""
            print(f"  {k}: winners {m_['winners_aligned_rate']*100:.2f}% vs losers {m_['losers_aligned_rate']*100:.2f}%  → uplift {sign}{m_['uplift_pp']:.2f}pp  ratio {m_['ratio']:.3f}")

    out = DATA_DIR / "win_uplift_v8.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[uplift] saved → {out}")


if __name__ == "__main__":
    main()
