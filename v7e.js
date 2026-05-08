/**
 * V7e Model — Gradient Boosted pick-prediction trained on STRATZ data.
 *
 * Trained via 5-fold CV (research/train_v7_exportable.py):
 *   CV top-10 = ~52.7% on Divine+ ranked matches
 *   vs M1 (current production): top-10 ~ 24.6%
 *
 * Loads data from /data/*.json and applies the GBM tree ensemble for scoring.
 */
(function () {
  'use strict';

  const DATA_BASE = './data';

  const V7e = {
    ready: false,
    error: null,

    // loaded data
    heroes: {},          // {id: {id, displayName, roles, primaryAttribute}}
    posStats: {},        // {POSITION_1..5: [{heroId, matchCount, winCount}]}
    posStatsByHero: {},  // {heroId: {1..5: {matchCount, winCount}}}
    matchups: {},        // {heroId: {vs: [{id,m,w,s}], with: [...]}}
    eligibility: {},     // {heroId: [pos1, pos2, ...]}
    model: null,         // {init_log_odds, learning_rate, trees}

    async init() {
      try {
        const [heroes, posStats, matchups, eligibility, model] = await Promise.all([
          fetch(`${DATA_BASE}/heroes_v2.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/position_stats.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/matchups.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/eligibility.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/v7e_model.json`).then(r => r.json()),
        ]);

        this.heroes = {};
        for (const h of heroes) this.heroes[h.id] = h;

        this.posStats = posStats;
        this.posStatsByHero = {};
        for (const [pkey, rows] of Object.entries(posStats)) {
          const posInt = parseInt(pkey.split('_')[1]);
          for (const r of rows) {
            if (!this.posStatsByHero[r.heroId]) this.posStatsByHero[r.heroId] = {};
            this.posStatsByHero[r.heroId][posInt] = { matchCount: r.matchCount, winCount: r.winCount };
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
        console.error('[V7e] init failed:', e);
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
      // Bayesian shrinkage to 0.5 with pseudocount 200
      const r = (this.posStatsByHero[heroId] || {})[position];
      if (!r || !r.matchCount) return 0.5;
      const wins = r.winCount + 100;
      const total = r.matchCount + 200;
      return wins / total;
    },

    counterAvg(heroId, enemies) {
      if (!enemies || !enemies.length) return 0.5;
      const mu = (this.matchups[heroId] || {}).vs || {};
      let sum = 0, n = 0;
      for (const e of enemies) {
        const ent = mu[e];
        if (!ent || ent.m < 30) continue;
        // synergy is in percentage points (e.g. +9.285 means hero has +9.3% advantage)
        sum += 0.5 + ent.s / 100.0;
        n++;
      }
      return n > 0 ? sum / n : 0.5;
    },

    synergyAvg(heroId, allies) {
      if (!allies || !allies.length) return 0.5;
      const mu = (this.matchups[heroId] || {}).with || {};
      let sum = 0, n = 0;
      for (const a of allies) {
        const ent = mu[a];
        if (!ent || ent.m < 30) continue;
        sum += 0.5 + ent.s / 100.0;
        n++;
      }
      return n > 0 ? sum / n : 0.5;
    },

    teamRoles(heroIds) {
      const out = new Set();
      for (const h of heroIds) {
        const hero = this.heroes[h];
        if (!hero) continue;
        for (const r of hero.roles || []) {
          if (r.level >= 2) out.add(r.roleId);
        }
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

    features(heroId, position, allies, enemies) {
      // Match the Python hero_features() function exactly.
      // Order: [base_wr, pos_fit, with_syn, vs_adv, role_gap]
      return [
        this.baseWr(heroId, position) - 0.5,
        this.positionFit(heroId, position),
        this.synergyAvg(heroId, allies) - 0.5,
        this.counterAvg(heroId, enemies) - 0.5,
        this.roleGap(heroId, allies),
      ];
    },

    _walkTree(tree, features) {
      // Iterative tree walk
      const { feature, threshold, left, right, value } = tree;
      let node = 0;
      while (feature[node] !== -2) {
        const f = feature[node];
        if (features[f] <= threshold[node]) {
          node = left[node];
        } else {
          node = right[node];
        }
      }
      return value[node];
    },

    _sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
    },

    score(heroId, position, allies, enemies) {
      if (!this.ready) return 0;
      if (!this.isEligible(heroId, position)) return 0;
      const f = this.features(heroId, position, allies, enemies);
      let z = this.model.init_log_odds;
      for (const tree of this.model.trees) {
        z += this.model.learning_rate * this._walkTree(tree, f);
      }
      return this._sigmoid(z);
    },

    /**
     * Return ranked candidate list for given (position, allies, enemies).
     * Each entry: {heroId, score, components: {base_wr, pos_fit, with_syn, vs_adv, role_gap}}
     */
    rank(position, allies, enemies, excludeIds = []) {
      const exclude = new Set([...allies, ...enemies, ...excludeIds]);
      const candidates = [];
      for (const hid of Object.keys(this.heroes).map(Number)) {
        if (exclude.has(hid)) continue;
        if (!this.isEligible(hid, position)) continue;
        const f = this.features(hid, position, allies, enemies);
        let z = this.model.init_log_odds;
        for (const tree of this.model.trees) {
          z += this.model.learning_rate * this._walkTree(tree, f);
        }
        candidates.push({
          heroId: hid,
          score: this._sigmoid(z),
          components: {
            base_wr: f[0] + 0.5,
            pos_fit: f[1],
            with_syn: f[2] + 0.5,
            vs_adv: f[3] + 0.5,
            role_gap: f[4],
          },
        });
      }
      candidates.sort((a, b) => b.score - a.score);
      return candidates;
    },
  };

  window.V7e = V7e;
})();
