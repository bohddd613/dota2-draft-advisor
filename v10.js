/**
 * V10 — Phase C1 model (LightGBM LGBMRanker + team-composition features).
 *
 * Architecture: pairwise ranking on 39 features (V9c's 25 + 14 team-comp).
 * Training: 400 trees, num_leaves=63, lr=0.05
 * Dataset: 5026 oldest Divine+ matches (chronological 80/20; 1256 newest held out)
 *
 * Model file: data/v10c_model.json
 *
 * Honest held-out backtest (1256 newest matches, never seen during training):
 *   - V10c: top10 = 57.4%, top5 = 39.3%, top1 = 17.1%
 *   - V9c fair: top10 = 57.3%, top5 = 38.6%, top1 = 17.4%
 *   - V8 fair: top10 = 57.5%, top5 = 39.1%, top1 = 17.5%
 *   - V7e:    top10 = 55.9%, top5 = 41.2%, top1 = 18.1%
 *
 * Team composition gives ≈0pp uplift over V9c with the current snapshot dataset.
 * The new features explore the right space (team archetypes), but on a 12-hour
 * data window without temporal/meta variation, they add no measurable signal.
 *
 * Use this model if you want to ship the broadest feature set we have; for
 * pure performance pick V8 fair (sklearn GBC) which is simpler and ties V10c.
 */
(function () {
  'use strict';

  const DATA_BASE = './data';
  const MIN_PAIR_MATCHES = 30;

  // Illusion-heavy heroes (used by team_has_illusions / enemy_has_illusions).
  // Must mirror research/hero_attrs.py ILLUSION_HEROES.
  const ILLUSION_HEROES = new Set([89, 12, 81, 109, 113, 67]);

  const V10 = {
    ready: false,
    error: null,

    heroes: {},
    posStatsByHero: {},
    heroTotalMatches: {},
    matchups: {},
    eligibility: {},
    model: null,

    async init() {
      try {
        const [heroes, posStats, matchups, eligibility, model] = await Promise.all([
          fetch(`${DATA_BASE}/heroes_v2.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/position_stats.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/matchups.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/eligibility.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/v10c_model.json`).then(r => r.json()),
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
        console.error('[V10] init failed:', e);
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

    pairAdvantage(heroId, other, kind) {
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

    _withPositions(arr) {
      return arr.map(a => {
        if (typeof a === 'object' && a !== null && a.position != null) return a;
        const id = typeof a === 'object' ? a.id : a;
        return { id, position: this.inferPosition(id) };
      });
    },

    // ---- Team composition helpers (mirror research/hero_attrs.py) ----

    heroAttrs(heroId) {
      const h = this.heroes[heroId];
      if (!h) {
        return { primaryAttribute: 'all', init: 0, disabler: 0, nuker: 0,
                 pusher: 0, durable: 0, illusion: 0 };
      }
      const lv = {};
      for (const r of h.roles || []) lv[r.roleId] = r.level;
      return {
        primaryAttribute: h.primaryAttribute || 'all',
        init:     (lv.INITIATOR || 0) >= 2 ? 1 : 0,
        disabler: (lv.DISABLER  || 0) >= 2 ? 1 : 0,
        nuker:    (lv.NUKER     || 0) >= 2 ? 1 : 0,
        pusher:   (lv.PUSHER    || 0) >= 2 ? 1 : 0,
        durable:  (lv.DURABLE   || 0) >= 2 ? 1 : 0,
        illusion: ILLUSION_HEROES.has(heroId) ? 1 : 0,
      };
    },

    teamCompFeatures(candHid, allyIds, enemyIds) {
      const team = [candHid, ...allyIds].map(h => this.heroAttrs(h));
      const enemy = enemyIds.map(h => this.heroAttrs(h));
      const cnt = (arr, k) => arr.reduce((s, a) => s + a[k], 0);
      const ratio = (arr, attr) => arr.length
        ? arr.filter(a => a.primaryAttribute === attr).length / arr.length
        : 0;
      const any = (arr, k) => arr.some(a => a[k]) ? 1 : 0;
      return [
        cnt(team, 'init'),
        cnt(team, 'disabler'),
        cnt(team, 'nuker'),
        cnt(team, 'pusher'),
        cnt(team, 'durable'),
        cnt(enemy, 'init'),
        cnt(enemy, 'disabler'),
        cnt(enemy, 'nuker'),
        cnt(enemy, 'durable'),
        ratio(team, 'agi'),
        ratio(team, 'int'),
        ratio(enemy, 'agi'),
        any(team, 'illusion'),
        any(enemy, 'illusion'),
      ];
    },

    /**
     * Build the 39-feature vector — must mirror Python hero_features_v10 exactly.
     */
    features(heroId, position, allies, enemies) {
      const baseWr = this.baseWr(heroId, position) - 0.5;
      const posFit = this.positionFit(heroId, position);
      const allyIds = allies.map(a => typeof a === 'object' ? a.id : a);
      const enemyIds = enemies.map(e => typeof e === 'object' ? e.id : e);
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

      const comp = this.teamCompFeatures(heroId, allyIds, enemyIds);

      return [
        baseWr, posFit, roleGap,
        ...withPerPos,
        ...vsPerPos,
        wMax, wMin, wStd,
        vMax, vMin, vStd,
        ...targetOH,
        logTotal,
        ...comp,
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

    _rawScore(f) {
      let z = this.model.init_score || 0;
      for (const tree of this.model.trees) z += this._walkTree(tree, f);
      return z;
    },

    score(heroId, position, allies, enemies) {
      if (!this.ready) return 0;
      if (!this.isEligible(heroId, position)) return 0;
      const allyPairs = this._withPositions(allies);
      const enemyPairs = this._withPositions(enemies);
      const f = this.features(heroId, position, allyPairs, enemyPairs);
      return this._sigmoid(this._rawScore(f));
    },

    rank(position, allies, enemies, excludeIds = []) {
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

        const withSum = f[3] + f[4] + f[5] + f[6] + f[7];
        const vsSum   = f[8] + f[9] + f[10] + f[11] + f[12];
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
            // team composition aggregates (indices 25..38 in feature vector)
            team_init: f[25],
            team_disabler: f[26],
            team_nuker: f[27],
            team_pusher: f[28],
            team_durable: f[29],
            team_agi_ratio: f[34],
            team_int_ratio: f[35],
            team_has_illusions: f[37],
            enemy_has_illusions: f[38],
            features: f,
          },
        });
      }
      candidates.sort((a, b) => b.score - a.score);
      return candidates;
    },
  };

  window.V10 = V10;
})();
