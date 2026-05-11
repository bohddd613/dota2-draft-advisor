/**
 * M3 — TrueSynergy (R.O.S.H.-equivalent)
 *
 * Replicates the algorithm STRATZ documents publicly for their R.O.S.H.
 * Alpha Draft Assistant (https://stratz.com/rosh/analysis):
 *
 *   TrueSynergy(hero) = base_winrate(hero, position)
 *                     + Σ_a∈allies   synergy(hero, ally)
 *                     + Σ_e∈enemies  counter(hero, enemy)
 *
 * All three components are STRATZ data we already cache in /data/.
 *
 * Position qualification (per ROSH settings):
 *   - Hero overall pick rate ≥ MIN_HERO_PR (default 1%)
 *   - Hero's % of matches at this position ≥ MIN_POSITION_PR (default 20%)
 *
 * Sample-size thresholds (per STRATZ blog 2023-06):
 *   - 1000+ matches for base win-rate at position
 *   - 100+ matches for synergy/counter pairs
 *
 * Source: https://medium.com/stratz/rosh-the-alpha-draft-assistant-587afabc8048
 */
(function () {
  'use strict';

  const DATA_BASE = './data';

  // Sample-size thresholds (STRATZ-documented).
  const MIN_POSITION_MATCHES = 200;   // we relax 1000 -> 200 for our smaller cached dataset
  const MIN_PAIR_MATCHES = 30;        // we relax 100 -> 30 for the same reason

  // Position-qualification thresholds (ROSH default settings).
  const MIN_HERO_PR = 0.005;          // 0.5% — generous, very rare heroes excluded
  const MIN_POSITION_PR = 0.10;       // 10% — must be played at position in ≥10% of hero's matches

  const M3 = {
    ready: false,
    error: null,

    heroes: {},          // {id: hero}
    posStatsByHero: {},  // {heroId: {1..5: {matchCount, winCount}}}
    matchups: {},        // {heroId: {vs: {id->row}, with: {id->row}}}

    // derived
    heroMatchCount: {},  // {heroId: total matches across all positions}
    totalMatchesAcrossAll: 0,
    qualifiedAtPos: {},  // {position: Set(heroId)}

    async init() {
      try {
        const [heroes, posStats, matchups] = await Promise.all([
          fetch(`${DATA_BASE}/heroes_v2.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/position_stats.json`).then(r => r.json()),
          fetch(`${DATA_BASE}/matchups.json`).then(r => r.json()),
        ]);

        this.heroes = {};
        for (const h of heroes) this.heroes[h.id] = h;

        this.posStatsByHero = {};
        for (const [pkey, rows] of Object.entries(posStats)) {
          const pos = parseInt(pkey.split('_')[1]);
          for (const r of rows) {
            if (!this.posStatsByHero[r.heroId]) this.posStatsByHero[r.heroId] = {};
            this.posStatsByHero[r.heroId][pos] = { matchCount: r.matchCount, winCount: r.winCount };
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

        // Pre-compute hero total match counts and qualification sets.
        this.heroMatchCount = {};
        let total = 0;
        for (const [hidStr, posMap] of Object.entries(this.posStatsByHero)) {
          const hid = parseInt(hidStr);
          let sum = 0;
          for (const r of Object.values(posMap)) sum += (r.matchCount || 0);
          this.heroMatchCount[hid] = sum;
          total += sum;
        }
        this.totalMatchesAcrossAll = total;

        this.qualifiedAtPos = { 1: new Set(), 2: new Set(), 3: new Set(), 4: new Set(), 5: new Set() };
        for (const [hidStr, posMap] of Object.entries(this.posStatsByHero)) {
          const hid = parseInt(hidStr);
          const heroTotal = this.heroMatchCount[hid] || 1;
          // Hero PR check uses a population denominator (10×total / 5 positions ~ rough heuristic);
          // since position_stats is per-position, the hero's "global pickrate" is hard to compute
          // exactly. Use heroTotal vs an avg-hero baseline instead.
          const avgHero = total / Math.max(1, Object.keys(this.posStatsByHero).length);
          if (heroTotal < MIN_HERO_PR * avgHero * 5) continue;  // hero too rare overall
          for (const [posStr, r] of Object.entries(posMap)) {
            const p = parseInt(posStr);
            if ((r.matchCount || 0) < MIN_POSITION_MATCHES) continue;
            if (r.matchCount / heroTotal < MIN_POSITION_PR) continue;
            this.qualifiedAtPos[p].add(hid);
          }
        }

        this.ready = true;
      } catch (e) {
        this.error = e;
        console.error('[M3] init failed:', e);
      }
    },

    // -- Components --

    /** Base win-rate at position (as percentage 0-100). null if not enough data. */
    baseWinrate(heroId, position) {
      const r = (this.posStatsByHero[heroId] || {})[position];
      if (!r || (r.matchCount || 0) < MIN_POSITION_MATCHES) return null;
      return 100.0 * r.winCount / r.matchCount;
    },

    /** Synergy: ally's "with" advantage in pp. 0 if not enough sample. */
    synergyWith(heroId, allyId) {
      const mu = (this.matchups[heroId] || {}).with || {};
      const ent = mu[allyId];
      if (!ent || (ent.m || 0) < MIN_PAIR_MATCHES) return { value: 0, games: ent ? ent.m : 0, qualified: false };
      return { value: ent.s, games: ent.m, qualified: true };
    },

    /** Counter: hero's advantage vs enemy in pp. 0 if not enough sample. */
    counterVs(heroId, enemyId) {
      const mu = (this.matchups[heroId] || {}).vs || {};
      const ent = mu[enemyId];
      if (!ent || (ent.m || 0) < MIN_PAIR_MATCHES) return { value: 0, games: ent ? ent.m : 0, qualified: false };
      return { value: ent.s, games: ent.m, qualified: true };
    },

    isQualified(heroId, position) {
      return (this.qualifiedAtPos[position] || new Set()).has(heroId);
    },

    /**
     * Compute TrueSynergy + breakdown for a single hero.
     * Returns null when hero doesn't qualify or lacks data.
     */
    score(heroId, position, allies, enemies) {
      if (!this.ready) return null;
      if (!this.isQualified(heroId, position)) return null;

      const wr = this.baseWinrate(heroId, position);
      if (wr === null) return null;

      let synSum = 0;
      const synPerAlly = [];
      for (const a of allies) {
        const s = this.synergyWith(heroId, a);
        synSum += s.value;
        synPerAlly.push({ heroId: a, ...s });
      }

      let ctrSum = 0;
      const ctrPerEnemy = [];
      for (const e of enemies) {
        const c = this.counterVs(heroId, e);
        ctrSum += c.value;
        ctrPerEnemy.push({ heroId: e, ...c });
      }

      // TrueSynergy: pp above 50% baseline.
      const baseAdv = wr - 50.0;
      const trueSynergy = baseAdv + synSum + ctrSum;

      return {
        heroId,
        trueSynergy,            // raw TS in pp (e.g. +16.88)
        baseWinrate: wr,        // % (e.g. 55.0)
        baseAdvantage: baseAdv, // pp above 50 (e.g. +5.0)
        synergyTotal: synSum,
        counterTotal: ctrSum,
        synergyPerAlly: synPerAlly,
        counterPerEnemy: ctrPerEnemy,
        // 0..1 normalized score for UI consistency (sigmoid-like).
        // Most TS values in [-30, +50]; sigmoid of TS/15 maps to ~0.05 to ~0.95.
        scoreNormalized: 1 / (1 + Math.exp(-trueSynergy / 15)),
      };
    },

    /**
     * Rank all qualified candidates for a slot.
     * Returns array sorted by trueSynergy descending.
     */
    rank(position, allies, enemies, excludeIds = []) {
      const exclude = new Set([...allies, ...enemies, ...excludeIds]);
      const out = [];
      for (const hidStr of Object.keys(this.heroes)) {
        const hid = parseInt(hidStr);
        if (exclude.has(hid)) continue;
        const s = this.score(hid, position, allies, enemies);
        if (s) out.push(s);
      }
      out.sort((a, b) => b.trueSynergy - a.trueSynergy);
      return out;
    },

    /**
     * Team TrueSynergy: sum of TS for all heroes on a side.
     * Useful for the "Team TS Differential" display.
     */
    teamTrueSynergy(side, allyIds, enemyIds, positionMap) {
      // side: 'ally' or 'enemy'
      // positionMap: { heroId: position }
      let total = 0;
      const team = side === 'ally' ? allyIds : enemyIds;
      const oppos = side === 'ally' ? enemyIds : allyIds;
      for (const h of team) {
        const pos = positionMap[h];
        if (!pos) continue;
        const others = team.filter(x => x !== h);
        const s = this.score(h, pos, others, oppos);
        if (s) total += s.trueSynergy;
      }
      return total;
    },
  };

  window.M3 = M3;
})();
