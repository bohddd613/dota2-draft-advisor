/**
 * V9 — Phase B model (LightGBM LGBMRanker, pairwise lambdarank).
 *
 * Architecture: pairwise ranking on 25 features (same feature set as V8).
 * Training: 400 trees, num_leaves=63, lr=0.05
 * Dataset: 6282 Divine+ matches (V8 was 1381 — 4.6× more data)
 *
 * Model file: data/v9c_model.json
 * The ranker outputs raw scores (not probabilities). Higher = better pick.
 * No sigmoid applied — scores are directly comparable for ranking.
 *
 * Held-out backtest (vs V8 trained on its own 1381):
 *   - V9: top10 = 74.0%, top5 = 57.8%, top1 = 23.5%
 *   - V8: top10 = 61.9%, top5 = 44.8%, top1 = 20.4%
 *   - V7e: top10 = 55.9%, top5 = 41.2%, top1 = 18.1%
 *
 * +12pp top-10 over V8 (Phase A); the gain comes from BOTH more data
 * AND pairwise ranking objective (sklearn binary classifier on the same
 * data only matches V8).
 */
(function () {
  'use strict';

  const DATA_BASE = './data';
  const MIN_PAIR_MATCHES = 30;

  const V9 = {
    ready: false,
    error: null,

    heroes: {},            // {id: hero}
    posStatsByHero: {},    // {heroId: {1..5: {matchCount, winCount}}}
    heroTotalMatches: {},  // {heroId: int}
    matchups: {},          // {heroId: {vs: {id->row}, with: {id->row}}}
    eligibility: {},       // {heroId: [pos1, pos2, ...]}
    model: null,           // {init_log_odds, learning_rate, trees, feature_names}

    async init() {
      try {
        const [heroes, posStats, matchups, eligibility, model] = await Promise.all([
          fetch(`${DATA_BASE}/heroes_v2.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/position_stats.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/matchups.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/eligibility.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/v9c_model.json`).then(r => r.json()),
        ]);

        this.heroes = {};
        for (const h of heroes) this.heroes[h.id] = h;

        this.posStatsByHero = {};
        this.heroTotalMatches = {};
        for (const [pkey, rows] of Object.entries(posStats)) {
          const posInt = parseInt(pkey.split('_')[1]);
          for (const r of rows) {
            if (!this.posStatsByHero[r.heroId]) this.posStatsByHero[r.heroId] = {};
            this.posStatsByHero[r.heroId][posInt] = { matchCount: r.matchCount, winCount: r.winCount };
            this.heroTotalMatches[r.heroId] = (this.heroTotalMatches[r.heroId] || 0) + r.matchCount;
          }
        }

        this.matchups = {};
        for (const [hid, mu] of Object.entries(matchups)) {
          const vs = {};
          for (const e of mu.vs || []) vs[e.id] = e;
          const wi = {};
          for (const e of mu.with || []) wi[e.id] = e;
          this.matchups[parseInt(hid)] = { vs, with: wi };
        }

        this.eligibility = {};
        for (const [hid, list] of Object.entries(eligibility)) {
          this.eligibility[parseInt(hid)] = list;
        }

        this.model = model;
        this.ready = true;
      } catch (e) {
        this.error = e;
        console.error('[V9] init failed:', e);
      }
    },

    isEligible(heroId, position) {
      const elig = this.eligibility[heroId];
      return Array.isArray(elig) && elig.includes(position);
    },

    positionFit(heroId, position) {
      const elig = this.eligibility[heroId] || [];
      if (!elig.includes(position)) return 0;
      const idx = elig.indexOf(position);
      const decay = [1.0, 0.7, 0.5, 0.35];
      return idx < decay.length ? decay[idx] : 0.2;
    },

    baseWr(heroId, position) {
      const r = (this.posStatsByHero[heroId] || {})[position];
      if (!r || !r.matchCount) return 0.5;
      const wins = r.winCount + 100;
      const total = r.matchCount + 200;
      return wins / total;
    },

    pairAdvantage(heroId, other, kind /* 'with' | 'vs' */) {
      const mu = (this.matchups[heroId] || {})[kind] || {};
      const ent = mu[other];
      if (!ent || ent.m < MIN_PAIR_MATCHES) return 0;
      return ent.s / 100.0;
    },

    teamRoles(heroIds) {
      const out = new Set();
      for (const h of heroIds) {
        const hero = this.heroes[h];
        if (!hero) continue;
        for (const r of hero.roles || []) if (r.level >= 2) out.add(r.roleId);
      }
      return out;
    },

    KEY_ROLES: ['INITIATOR', 'DISABLER', 'SUPPORT', 'DURABLE', 'NUKER'],

    roleGap(heroId, allies) {
      const allyRoles = this.teamRoles(allies);
      const candHero = this.heroes[heroId];
      if (!candHero) return 0;
      const candRoles = new Set();
      for (const r of candHero.roles || []) if (r.level >= 2) candRoles.add(r.roleId);
      const missing = this.KEY_ROLES.filter(r => !allyRoles.has(r));
      let overlap = 0;
      for (const r of missing) if (candRoles.has(r)) overlap++;
      return overlap / Math.max(1, this.KEY_ROLES.length);
    },

    /**
     * Infer dominant position for a hero from cached position_stats.
     * Returns position 1-5 with the most matches; 0 if no data.
     */
    inferPosition(heroId) {
      const stats = this.posStatsByHero[heroId] || {};
      let bestPos = 0, bestM = 0;
      for (let p = 1; p <= 5; p++) {
        const r = stats[p];
        if (r && r.matchCount > bestM) {
          bestM = r.matchCount;
          bestPos = p;
        }
      }
      return bestPos;
    },

    /**
     * Coerce ally/enemy into {id, position} pairs. If position is missing,
     * infer from data. This makes inference work for the live app where the
     * user only specifies their own slot.
     */
    _withPositions(arr) {
      return arr.map(a => {
        if (typeof a === 'object' && a !== null && a.position != null) return a;
        const id = typeof a === 'object' ? a.id : a;
        return { id, position: this.inferPosition(id) };
      });
    },

    /**
     * Build the 25-feature vector — must mirror Python hero_features_v8 exactly.
     * allies / enemies: array of {id, position} objects OR plain id list (positions auto-inferred).
     */
    features(heroId, position, allies, enemies) {
      const baseWr = this.baseWr(heroId, position) - 0.5;
      const posFit = this.positionFit(heroId, position);
      const allyIds = allies.map(a => typeof a === 'object' ? a.id : a);
      const roleGap = this.roleGap(heroId, allyIds);

      const withPerPos = [0, 0, 0, 0, 0];
      for (const a of allies) {
        const aid = typeof a === 'object' ? a.id : a;
        const apos = typeof a === 'object' ? a.position : 0;
        if (apos >= 1 && apos <= 5) {
          withPerPos[apos - 1] = this.pairAdvantage(heroId, aid, 'with');
        }
      }

      const vsPerPos = [0, 0, 0, 0, 0];
      for (const e of enemies) {
        const eid = typeof e === 'object' ? e.id : e;
        const epos = typeof e === 'object' ? e.position : 0;
        if (epos >= 1 && epos <= 5) {
          vsPerPos[epos - 1] = this.pairAdvantage(heroId, eid, 'vs');
        }
      }

      const stats = arr => {
        const nz = arr.filter(x => x !== 0);
        if (!nz.length) return [0, 0, 0];
        const max = Math.max(...nz);
        const min = Math.min(...nz);
        const mean = nz.reduce((s, x) => s + x, 0) / nz.length;
        const variance = nz.reduce((s, x) => s + (x - mean) * (x - mean), 0) / nz.length;
        return [max, min, Math.sqrt(variance)];
      };
      const [wMax, wMin, wStd] = stats(withPerPos);
      const [vMax, vMin, vStd] = stats(vsPerPos);

      const targetOH = [0, 0, 0, 0, 0];
      if (position >= 1 && position <= 5) targetOH[position - 1] = 1;

      const totalMatches = this.heroTotalMatches[heroId] || 0;
      const logTotal = Math.log1p(totalMatches) / 15.0;

      return [
        baseWr, posFit, roleGap,
        ...withPerPos,
        ...vsPerPos,
        wMax, wMin, wStd,
        vMax, vMin, vStd,
        ...targetOH,
        logTotal,
      ];
    },

    _walkTree(tree, features) {
      const { feature, threshold, left, right, value } = tree;
      let node = 0;
      while (feature[node] !== -2) {
        const f = feature[node];
        node = features[f] <= threshold[node] ? left[node] : right[node];
      }
      return value[node];
    },

    _sigmoid(x) { return 1 / (1 + Math.exp(-x)); },

    /**
     * Raw ranker score from the LightGBM model.
     * Higher = better recommendation. Range typically -5..+5.
     */
    _rawScore(f) {
      let z = this.model.init_score || 0;
      for (const tree of this.model.trees) {
        z += this._walkTree(tree, f);  // LightGBM bakes shrinkage into leaves
      }
      return z;
    },

    score(heroId, position, allies, enemies) {
      if (!this.ready) return 0;
      if (!this.isEligible(heroId, position)) return 0;
      const allyPairs = this._withPositions(allies);
      const enemyPairs = this._withPositions(enemies);
      const f = this.features(heroId, position, allyPairs, enemyPairs);
      // Map raw score to (0, 1) via sigmoid for UI consistency.
      return this._sigmoid(this._rawScore(f));
    },

    /**
     * Rank all eligible heroes for (position, allies, enemies).
     * allies/enemies should be arrays of {id, position} for the V9 model to use
     * per-position pair features. If positions are missing, the model still scores
     * but loses per-position resolution (max/min/spread features still work).
     *
     * Returns: [{heroId, score, components}] sorted descending by score.
     */
    rank(position, allies, enemies, excludeIds = []) {
      // Auto-infer ally/enemy positions if not provided (live app passes IDs only).
      const allyPairs = this._withPositions(allies);
      const enemyPairs = this._withPositions(enemies);
      const allyIds = allyPairs.map(a => a.id);
      const enemyIds = enemyPairs.map(e => e.id);
      const exclude = new Set([...allyIds, ...enemyIds, ...excludeIds]);
      const candidates = [];
      for (const hidStr of Object.keys(this.heroes)) {
        const hid = Number(hidStr);
        if (exclude.has(hid)) continue;
        if (!this.isEligible(hid, position)) continue;
        const f = this.features(hid, position, allyPairs, enemyPairs);
        const rawScore = this._rawScore(f);
        const score = this._sigmoid(rawScore);

        // Aggregate "explainable" components for the Why-modal.
        const withSum = f[3] + f[4] + f[5] + f[6] + f[7];   // with_syn_pos1..5
        const vsSum   = f[8] + f[9] + f[10] + f[11] + f[12]; // vs_adv_pos1..5
        candidates.push({
          heroId: hid,
          score,
          components: {
            base_wr: f[0] + 0.5,
            pos_fit: f[1],
            role_gap: f[2],
            with_syn_total: withSum,
            vs_adv_total: vsSum,
            with_max: f[13],
            vs_max: f[16],
            features: f,  // raw vector for Why-modal
          },
        });
      }
      candidates.sort((a, b) => b.score - a.score);
      return candidates;
    },
  };

  window.V9 = V9;
})();
