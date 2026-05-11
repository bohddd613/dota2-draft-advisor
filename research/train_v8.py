"""
V8: improved pick-recommendation model.

Feature additions vs V7e (5 → 25 features):
  - 5 per-position synergy: with_syn_pos1..5 (synergy with ally at that position)
  - 5 per-position counter: vs_adv_pos1..5 (counter vs enemy at that position)
  - 3 with-stats: max_with_syn, min_with_syn, spread_with_syn
  - 3 vs-stats:  max_vs_adv,  min_vs_adv,  spread_vs_adv
  - 5 target one-hot: target_is_pos1..5
  - 1 popularity:  log(hero_total_matches + 1)
  (plus 3 originals: base_wr, pos_fit, role_gap)

Trains TWO variants for comparison:
  - V8a: sklearn GradientBoostingClassifier (binary; same objective as V7e)
         → exportable via m2cgen (pure JS)
  - V8b: LightGBM LGBMRanker (lambdarank pairwise objective)
         → exported via booster_.dump_model() to JSON (custom JS evaluator)

5-fold CV on the same 1381 Divine+ matches with STRATZ-verified positions.

Outputs:
  - research/data/v8a_gbm.joblib
  - research/data/v8a_model.js          (m2cgen output)
  - research/data/v8b_ranker.txt        (LightGBM text model)
  - research/data/v8b_model.json        (JSON-serialized booster)
  - research/data/v8_cv.json
"""
from __future__ import annotations
import json
import math
import sys
import argparse
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).parent))

from models import Context, M2_DataPositions, M4_RoleGap  # noqa: E402
from train_v2 import kfold_split  # noqa: E402

POS_TO_INT = {f"POSITION_{i}": i for i in range(1, 6)}


# ----------------------------- features ------------------------------------

FEATURE_NAMES_V8 = (
    ["base_wr", "pos_fit", "role_gap"]
    + [f"with_syn_pos{p}" for p in range(1, 6)]
    + [f"vs_adv_pos{p}"   for p in range(1, 6)]
    + ["max_with_syn", "min_with_syn", "spread_with_syn"]
    + ["max_vs_adv",   "min_vs_adv",   "spread_vs_adv"]
    + [f"target_is_pos{p}" for p in range(1, 6)]
    + ["log_total_matches"]
)
assert len(FEATURE_NAMES_V8) == 25


_HERO_TOTAL_MATCHES_CACHE: dict[int, int] = {}


def _hero_total_matches(ctx, hid: int) -> int:
    if hid in _HERO_TOTAL_MATCHES_CACHE:
        return _HERO_TOTAL_MATCHES_CACHE[hid]
    total = 0
    for pos_key, by_hero in ctx.pos_stats.items():
        rec = by_hero.get(hid)
        if rec:
            total += rec["matchCount"]
    _HERO_TOTAL_MATCHES_CACHE[hid] = total
    return total


def _pair_value(ctx, hid: int, other: int, kind: str) -> float:
    """Return synergy/counter advantage in pp/100 (i.e. [-0.5..+0.5]).

    kind: 'with' for synergy, 'vs' for counter.
    Returns 0.0 when sample size < 30 or no entry exists.
    """
    mu = ctx.matchups.get(hid, {}).get(kind, {})
    ent = mu.get(other)
    if not ent or ent.get("matchCount", 0) < 30:
        return 0.0
    return ent["synergy"] / 100.0


def hero_features_v8(
    m4: M4_RoleGap, m2: M2_DataPositions,
    hid: int, target_pos: int,
    allies: list[tuple[int, int]], enemies: list[tuple[int, int]],
) -> np.ndarray:
    """
    allies/enemies: list of (hero_id, position) tuples.
    """
    ctx = m4.ctx
    # Original 3 features
    base_wr = m2.base_wr(hid, target_pos) - 0.5
    pos_fit = m2.position_fit(hid, target_pos)
    KEY_ROLES = m4.KEY_ROLES
    ally_ids = [h for h, _ in allies]
    ally_roles = m4._team_roles(ally_ids)
    cand_roles = {r["roleId"] for r in ctx.heroes.get(hid, {}).get("roles", []) if r["level"] >= 2}
    missing = KEY_ROLES - ally_roles
    role_gap = len(cand_roles & missing) / max(1, len(KEY_ROLES))

    # Per-position synergy (with allies)
    with_per_pos = np.zeros(5, dtype=np.float32)
    for a_hid, a_pos in allies:
        if 1 <= a_pos <= 5:
            with_per_pos[a_pos - 1] = _pair_value(ctx, hid, a_hid, "with")

    # Per-position counter (vs enemies)
    vs_per_pos = np.zeros(5, dtype=np.float32)
    for e_hid, e_pos in enemies:
        if 1 <= e_pos <= 5:
            vs_per_pos[e_pos - 1] = _pair_value(ctx, hid, e_hid, "vs")

    # Stats over with/vs (only over non-zero entries — empty allies/enemies → 0)
    def stats(arr):
        nz = arr[arr != 0]
        if len(nz) == 0:
            return 0.0, 0.0, 0.0
        return float(np.max(nz)), float(np.min(nz)), float(np.std(nz))

    w_max, w_min, w_std = stats(with_per_pos)
    v_max, v_min, v_std = stats(vs_per_pos)

    # Target one-hot
    target_oh = np.zeros(5, dtype=np.float32)
    if 1 <= target_pos <= 5:
        target_oh[target_pos - 1] = 1.0

    # Popularity proxy
    total_matches = _hero_total_matches(ctx, hid)
    log_total = math.log1p(total_matches) / 15.0  # ~[0..1.2]

    vec = np.concatenate([
        np.array([base_wr, pos_fit, role_gap], dtype=np.float32),
        with_per_pos,
        vs_per_pos,
        np.array([w_max, w_min, w_std, v_max, v_min, v_std], dtype=np.float32),
        target_oh,
        np.array([log_total], dtype=np.float32),
    ]).astype(np.float32)
    assert vec.shape == (25,), vec.shape
    return vec


# ----------------------------- data ----------------------------------------

def load_matches(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    out = []
    for m in raw:
        rad, dire = [], []
        for p in m.get("players", []):
            pos = POS_TO_INT.get(p["position"])
            if pos is None: continue
            (rad if p["is_radiant"] else dire).append((p["hero_id"], pos))
        if len(rad) != 5 or len(dire) != 5: continue
        out.append({"match_id": m["match_id"], "radiant_win": m["radiant_win"],
                    "radiant": rad, "dire": dire})
    return out


def build_pick_dataset_v8(matches, ctx, m2, m4, all_hero_ids,
                          neg_per_pos: int = 10, seed: int = 42):
    """
    For each (state, true_hero) pair from winning team's picks, generate
    1 positive + N negatives. Returns X, y, group (group sizes for LGBMRanker).
    """
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        for true_hero, pos in winners:
            allies = [(h, p) for h, p in winners if h != true_hero]
            enemies = losers
            taken = set(team_ids) | set(enemy_ids)
            eligible = [h for h in all_hero_ids
                        if h not in taken and pos in (m2.eligible.get(h) or [])]
            if true_hero not in eligible:
                eligible.append(true_hero)
            negatives = [h for h in eligible if h != true_hero]
            if not negatives:
                continue
            sample = rng.choice(negatives, size=min(neg_per_pos, len(negatives)), replace=False)
            # Positive
            f_pos = hero_features_v8(m4, m2, true_hero, pos, allies, enemies)
            X.append(f_pos); y.append(1.0)
            # Negatives
            for h in sample:
                f_neg = hero_features_v8(m4, m2, int(h), pos, allies, enemies)
                X.append(f_neg); y.append(0.0)
            groups.append(1 + len(sample))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(groups, dtype=np.int32)


def evaluate_pick_rec_v8(scorer, matches, ctx, m2, m4, all_hero_ids, samples_per_match=2):
    rng = np.random.default_rng(42)
    ranks = []
    top1 = top5 = top10 = 0
    for m in matches:
        winners = m["radiant"] if m["radiant_win"] else m["dire"]
        losers = m["dire"] if m["radiant_win"] else m["radiant"]
        team_ids = [h for h, _ in winners]
        enemy_ids = [h for h, _ in losers]
        idx_choices = rng.choice(len(winners), size=min(samples_per_match, len(winners)), replace=False)
        for idx in idx_choices:
            true_hero, target_pos = winners[idx]
            allies = [(h, p) for h, p in winners if h != true_hero]
            enemies = losers
            taken = set(team_ids) | set(enemy_ids)
            eligible = [h for h in all_hero_ids
                        if h not in taken and target_pos in (m2.eligible.get(h) or [])]
            if true_hero not in eligible:
                eligible.append(true_hero)
            feats = np.vstack([
                hero_features_v8(m4, m2, h, target_pos, allies, enemies)
                for h in eligible
            ])
            scores = scorer(feats)
            order = np.argsort(-scores)
            ranked = [eligible[i] for i in order]
            try:
                rank = ranked.index(true_hero) + 1
            except ValueError:
                rank = len(ranked)
            ranks.append(rank)
            if rank == 1: top1 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1
    n = len(ranks)
    return {
        "n": n,
        "mean_rank": float(np.mean(ranks)),
        "top1": top1 / n, "top5": top5 / n, "top10": top10 / n,
    }


# ----------------------------- train ---------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default=str(DATA_DIR / "matches_stratz_enriched.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument("--variant", choices=["v8a", "v8b", "both"], default="both")
    args = ap.parse_args()

    matches = load_matches(Path(args.matches))
    print(f"[v8] loaded {len(matches)} matches")

    ctx = Context()
    m2 = M2_DataPositions(ctx)
    m4 = M4_RoleGap(ctx)
    all_hero_ids = list(ctx.heroes.keys())

    report = {
        "n_matches": len(matches),
        "feature_names": FEATURE_NAMES_V8,
        "variants": {},
    }

    # ------ V8a: sklearn GBC (binary) — exportable via m2cgen ------
    if args.variant in ("v8a", "both"):
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib

        print("\n[v8a] sklearn GradientBoostingClassifier, 5-fold CV…")
        fold_results = []
        for fi, (tr_idx, te_idx) in enumerate(kfold_split(len(matches), k=args.folds)):
            tr = [matches[i] for i in tr_idx]; te = [matches[i] for i in te_idx]
            X_tr, y_tr, _ = build_pick_dataset_v8(
                tr, ctx, m2, m4, all_hero_ids,
                neg_per_pos=args.neg_per_pos, seed=42 + fi,
            )
            clf = GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42,
            )
            clf.fit(X_tr, y_tr)
            def s(F): return clf.predict_proba(F)[:, 1]
            r = evaluate_pick_rec_v8(s, te, ctx, m2, m4, all_hero_ids)
            fold_results.append(r)
            print(f"  fold{fi}: top10={r['top10']:.4f}, top5={r['top5']:.4f}, "
                  f"top1={r['top1']:.4f}, mean={r['mean_rank']:.2f}")

        avg = {k: float(np.mean([r[k] for r in fold_results]))
               for k in ["top1", "top5", "top10", "mean_rank"]}
        print(f"[v8a] CV: top10={avg['top10']:.4f}  top5={avg['top5']:.4f}  "
              f"top1={avg['top1']:.4f}  mean={avg['mean_rank']:.2f}")

        # Refit on all data + export
        X, y, _ = build_pick_dataset_v8(matches, ctx, m2, m4, all_hero_ids,
                                        neg_per_pos=args.neg_per_pos)
        clf_all = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42,
        )
        clf_all.fit(X, y)
        joblib.dump(clf_all, DATA_DIR / "v8a_gbm.joblib")
        import m2cgen as m2c
        js_code = m2c.export_to_javascript(clf_all)
        (DATA_DIR / "v8a_model.js").write_text(js_code)
        report["variants"]["v8a"] = {"cv": avg, "folds": fold_results}

    # ------ V8b: LightGBM Ranker (lambdarank) ------
    if args.variant in ("v8b", "both"):
        import lightgbm as lgb

        print("\n[v8b] LightGBM LGBMRanker (lambdarank), 5-fold CV…")
        fold_results = []
        for fi, (tr_idx, te_idx) in enumerate(kfold_split(len(matches), k=args.folds)):
            tr = [matches[i] for i in tr_idx]; te = [matches[i] for i in te_idx]
            X_tr, y_tr, g_tr = build_pick_dataset_v8(
                tr, ctx, m2, m4, all_hero_ids,
                neg_per_pos=args.neg_per_pos, seed=42 + fi,
            )
            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                min_child_samples=20,
                random_state=42,
                verbosity=-1,
            )
            ranker.fit(X_tr, y_tr.astype(int), group=g_tr)
            def s(F, m=ranker): return m.predict(F)
            r = evaluate_pick_rec_v8(s, te, ctx, m2, m4, all_hero_ids)
            fold_results.append(r)
            print(f"  fold{fi}: top10={r['top10']:.4f}, top5={r['top5']:.4f}, "
                  f"top1={r['top1']:.4f}, mean={r['mean_rank']:.2f}")

        avg = {k: float(np.mean([r[k] for r in fold_results]))
               for k in ["top1", "top5", "top10", "mean_rank"]}
        print(f"[v8b] CV: top10={avg['top10']:.4f}  top5={avg['top5']:.4f}  "
              f"top1={avg['top1']:.4f}  mean={avg['mean_rank']:.2f}")

        # Refit on all data + export
        X, y, g = build_pick_dataset_v8(matches, ctx, m2, m4, all_hero_ids,
                                        neg_per_pos=args.neg_per_pos)
        ranker_all = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbosity=-1,
        )
        ranker_all.fit(X, y.astype(int), group=g)
        ranker_all.booster_.save_model(str(DATA_DIR / "v8b_ranker.txt"))
        model_dict = ranker_all.booster_.dump_model()
        (DATA_DIR / "v8b_model.json").write_text(json.dumps(model_dict))
        report["variants"]["v8b"] = {"cv": avg, "folds": fold_results}

    (DATA_DIR / "v8_cv.json").write_text(json.dumps(report, indent=2))
    print(f"\n[v8] saved → research/data/v8_cv.json")


if __name__ == "__main__":
    main()
