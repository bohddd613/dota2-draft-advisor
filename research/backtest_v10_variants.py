"""Quick backtest of V10 variants (d, e, f, g) vs V9c baseline."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v8 import hero_features_v8, load_matches  # noqa: E402
from train_v10 import hero_features_v10  # noqa: E402
from train_v10_variants import hero_features_v10d, hero_features_v10f  # noqa: E402
from backtest_v9 import split_chronological, evaluate_per_position  # noqa: E402
import lightgbm as lgb


def main():
    matches = load_matches(DATA_DIR / "matches_stratz_enriched.json")
    print(f"[bt10v] loaded {len(matches)} matches")
    _, test_idx = split_chronological(matches, frac_test=0.2)
    test = [matches[i] for i in test_idx]
    print(f"[bt10v] test={len(test)} (newest match_id: {max(m['match_id'] for m in test)})")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    results = {"models": {}}

    variants = [
        ("v9c", "v9c_ranker.txt", None),
        ("v10c", "v10c_ranker.txt", lambda h, p, a, e: hero_features_v10(m4, m2, h, p, a, e)),
        ("v10d", "v10d_ranker.txt", lambda h, p, a, e: hero_features_v10d(m4, m2, h, p, a, e)),
        ("v10e", "v10e_ranker.txt", lambda h, p, a, e: hero_features_v10(m4, m2, h, p, a, e)),
        ("v10f", "v10f_ranker.txt", lambda h, p, a, e: hero_features_v10f(m4, m2, h, p, a, e)),
        ("v10g", "v10g_ranker.txt", lambda h, p, a, e: hero_features_v10(m4, m2, h, p, a, e)),
    ]

    for label, model_file, builder in variants:
        try:
            booster = lgb.Booster(model_file=str(DATA_DIR / model_file))
        except Exception as ex:
            print(f"[{label}] missing/error: {ex}")
            continue
        print(f"\n[{label}] evaluating…")
        r = evaluate_per_position(
            lambda F: booster.predict(F),
            test, ctx, m2, m4, all_hero_ids,
            feat_builder=builder,
            samples_per_match=3,
        )
        results["models"][label] = r
        print(f"  {label}: top10={r['top10']*100:.1f}%  top5={r['top5']*100:.1f}%  "
              f"top1={r['top1']*100:.1f}%  mean={r['mean_rank']:.2f}  n={r['n']}")

    (DATA_DIR / "backtest_v10_variants.json").write_text(json.dumps(results, indent=2))

    print("\n=== Summary ===")
    print(f"{'model':<6} {'top1':>7} {'top5':>7} {'top10':>7} {'mean':>7}")
    for name in ["v9c", "v10c", "v10d", "v10e", "v10f", "v10g"]:
        if name in results["models"]:
            r = results["models"][name]
            print(f"{name:<6} {r['top1']*100:>6.1f}% {r['top5']*100:>6.1f}% "
                  f"{r['top10']*100:>6.1f}% {r['mean_rank']:>7.2f}")


if __name__ == "__main__":
    main()
