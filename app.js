/**
 * Dota 2 Draft Advisor
 * Uses OpenDota public API for hero stats and matchup data
 * to recommend optimal hero picks based on draft context.
 */

// ============================================================
// Constants & Config
// ============================================================

const API_BASE = 'https://api.opendota.com/api';
const CDN_BASE = 'https://cdn.cloudflare.steamstatic.com';
const CACHE_TTL = 30 * 60 * 1000; // 30 min

const POSITIONS = {
  1: { name: 'Carry', lane: 'Safe Lane' },
  2: { name: 'Mid', lane: 'Mid Lane' },
  3: { name: 'Offlane', lane: 'Off Lane' },
  4: { name: 'Soft Support', lane: 'Off Lane / Roam' },
  5: { name: 'Hard Support', lane: 'Safe Lane' },
};

// Curated hero-position eligibility map (current 7.36+ meta).
// Per hero ID: ordered list of positions the hero plays.
// First entry is the PRIMARY position; subsequent are flex/secondary.
// If a position is not in the list, the hero is filtered out as candidate.
// Source: high-MMR pickrate data (Dotabuff/Stratz public stats).
const HERO_POSITIONS = {
  102: [3, 5, 1],     // Abaddon
  73:  [1, 2],        // Alchemist
  68:  [5, 4],        // Ancient Apparition
  1:   [1],           // Anti-Mage
  113: [1, 2],        // Arc Warden
  2:   [3, 4],        // Axe
  3:   [5, 4],        // Bane
  65:  [3, 4],        // Batrider
  38:  [3],           // Beastmaster
  4:   [3, 1],        // Bloodseeker
  62:  [4],           // Bounty Hunter
  78:  [3, 2],        // Brewmaster
  99:  [3, 1],        // Bristleback
  61:  [1, 3],        // Broodmother
  96:  [3, 1],        // Centaur Warrunner
  81:  [1, 3],        // Chaos Knight
  66:  [4, 5],        // Chen
  56:  [1, 2],        // Clinkz
  51:  [3, 4],        // Clockwerk
  5:   [5, 4],        // Crystal Maiden
  55:  [3],           // Dark Seer
  119: [4, 5],        // Dark Willow
  135: [3, 2],        // Dawnbreaker
  50:  [5, 4],        // Dazzle
  43:  [2, 3],        // Death Prophet
  87:  [5, 4],        // Disruptor
  69:  [3, 2],        // Doom
  49:  [2, 3],        // Dragon Knight
  6:   [1],           // Drow Ranger
  107: [4, 3],        // Earth Spirit
  7:   [4, 5, 3],     // Earthshaker
  103: [3, 4],        // Elder Titan
  106: [2, 1],        // Ember Spirit
  58:  [4, 5],        // Enchantress
  33:  [3, 5],        // Enigma
  41:  [1],           // Faceless Void
  121: [4, 5],        // Grimstroke
  72:  [1, 4],        // Gyrocopter
  123: [4, 5],        // Hoodwink
  59:  [1, 2],        // Huskar
  74:  [2, 1],        // Invoker
  91:  [5, 4],        // Io
  64:  [5, 4],        // Jakiro
  8:   [1],           // Juggernaut
  90:  [4, 5],        // Keeper of the Light
  145: [1, 2],        // Kez
  23:  [3, 2, 4],     // Kunkka
  155: [5],           // Largo
  104: [3, 1, 4],     // Legion Commander
  52:  [3, 2],        // Leshrac
  31:  [5],           // Lich
  54:  [1, 3],        // Lifestealer
  25:  [4, 2],        // Lina
  26:  [5, 4],        // Lion
  80:  [1],           // Lone Druid
  48:  [1],           // Luna
  77:  [1, 3],        // Lycan
  97:  [3, 4, 2],     // Magnus
  136: [4, 3],        // Marci
  129: [3],           // Mars
  94:  [1],           // Medusa
  82:  [2, 1],        // Meepo
  9:   [4, 2],        // Mirana
  114: [1, 4],        // Monkey King
  10:  [1, 2],        // Morphling
  138: [1, 2],        // Muerta
  89:  [1, 5],        // Naga Siren
  53:  [3, 4, 1, 2],  // Nature's Prophet (flex god)
  36:  [3, 2],        // Necrophos
  60:  [3, 1],        // Night Stalker
  88:  [4],           // Nyx Assassin
  84:  [4, 5],        // Ogre Magi
  57:  [5, 1, 3],     // Omniknight
  111: [5, 4],        // Oracle
  76:  [2, 1],        // Outworld Devourer
  120: [3, 4],        // Pangolier
  44:  [1],           // Phantom Assassin
  12:  [1],           // Phantom Lancer
  110: [4, 5, 3],     // Phoenix
  137: [3],           // Primal Beast
  13:  [2],           // Puck
  14:  [4, 5, 3],     // Pudge
  45:  [2, 5],        // Pugna
  39:  [2, 4],        // Queen of Pain
  15:  [2, 3],        // Razor
  32:  [4, 1],        // Riki
  131: [4, 5],        // Ring Master
  86:  [4, 5],        // Rubick
  16:  [4, 3],        // Sand King
  79:  [4, 5],        // Shadow Demon
  11:  [2],           // Shadow Fiend
  27:  [5, 4],        // Shadow Shaman
  75:  [5, 2],        // Silencer
  101: [4, 5],        // Skywrath Mage
  28:  [3, 4],        // Slardar
  93:  [1],           // Slark
  128: [4, 5],        // Snapfire
  35:  [1, 2],        // Sniper
  67:  [1],           // Spectre
  71:  [4, 3],        // Spirit Breaker
  17:  [2],           // Storm Spirit
  18:  [1, 3],        // Sven
  105: [4, 5],        // Techies
  46:  [2, 1],        // Templar Assassin
  109: [1],           // Terrorblade
  29:  [3],           // Tidehunter
  98:  [3],           // Timbersaw
  34:  [2],           // Tinker
  19:  [4, 1, 3],     // Tiny
  83:  [5],           // Treant Protector
  95:  [1],           // Troll Warlord
  100: [3, 4],        // Tusk
  108: [3],           // Underlord
  85:  [5, 3],        // Undying
  70:  [1],           // Ursa
  20:  [4, 5],        // Vengeful Spirit
  40:  [3, 4],        // Venomancer
  47:  [3, 2],        // Viper
  92:  [3, 4],        // Visage
  126: [2],           // Void Spirit
  37:  [5],           // Warlock
  63:  [1, 4],        // Weaver
  21:  [4, 2, 5],     // Windranger
  112: [5],           // Winter Wyvern
  30:  [4, 5],        // Witch Doctor
  42:  [1, 3],        // Wraith King
  22:  [2, 4],        // Zeus
};

// Position rank decay: 1st=primary 1.0, 2nd=secondary 0.7, etc.
const POSITION_RANK_DECAY = [1.0, 0.7, 0.5, 0.35];

// Minimum sample size for OpenDota matchup data to be considered statistically meaningful.
const MATCHUP_MIN_GAMES = 50;

// Scoring weights — when enemies are picked, counter matters more
const SCORE_WEIGHTS_NO_ENEMIES = {
  baseWinrate: 0.45,
  positionFit: 0.40,
  counterScore: 0.00,
  synergyScore: 0.15,
};
const SCORE_WEIGHTS_WITH_ENEMIES = {
  baseWinrate: 0.25,
  positionFit: 0.20,
  counterScore: 0.40,
  synergyScore: 0.15,
};

const MAX_ALLIES = 4;
const MAX_ENEMIES = 5;
const TOP_RECOMMENDATIONS = 10;
const MATCHUP_FETCH_DELAY = 120; // ms between requests to be polite

// ============================================================
// State
// ============================================================

const state = {
  heroes: [],           // All hero data from API
  heroMap: {},          // id -> hero
  selectedPosition: null,
  allies: [],           // hero ids
  enemies: [],          // hero ids
  addMode: 'ally',      // 'ally' | 'enemy'
  matchupCache: {},     // hero_id -> { data, timestamp }
  searchQuery: '',
  attrFilter: 'all',
  loading: true,
  lastRecs: [],         // cached recommendations for modal lookup
  modelMode: 'm1',      // 'm1' (curated, default) | 'v7e' (gradient boosting)
};

// ============================================================
// API Layer with Retry + Cache
// ============================================================

class ApiClient {
  constructor(baseUrl, cacheTtl) {
    this.baseUrl = baseUrl;
    this.cacheTtl = cacheTtl;
    this.cache = {};
  }

  async fetch(endpoint, retries = 3) {
    const url = `${this.baseUrl}${endpoint}`;
    const cached = this.cache[url];
    if (cached && Date.now() - cached.ts < this.cacheTtl) {
      return cached.data;
    }

    let lastError;
    for (let i = 0; i < retries; i++) {
      try {
        const resp = await fetch(url, {
          signal: AbortSignal.timeout(15000),
        });
        if (resp.status === 429) {
          // Rate limited — wait and retry
          await sleep(2000 * (i + 1));
          continue;
        }
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
        const data = await resp.json();
        this.cache[url] = { data, ts: Date.now() };
        return data;
      } catch (err) {
        lastError = err;
        if (i < retries - 1) await sleep(1000 * (i + 1));
      }
    }
    throw lastError;
  }
}

const api = new ApiClient(API_BASE, CACHE_TTL);

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================
// Data Loading
// ============================================================

async function loadHeroes() {
  try {
    const heroStats = await api.fetch('/heroStats');
    state.heroes = heroStats
      .filter(h => h.id > 0)
      .map(h => ({
        id: h.id,
        name: h.localized_name,
        internalName: h.name.replace('npc_dota_hero_', ''),
        attr: h.primary_attr === 'all' ? 'all' : h.primary_attr,
        attackType: h.attack_type,
        roles: h.roles || [],
        img: `${CDN_BASE}${h.img}`,
        icon: `${CDN_BASE}${h.icon}`,
        // Use brackets 4-7 (Archon to Divine) for balanced winrate data
        winrate: computeWinrate(h),
        picks: computePicks(h),
        proPick: h.pro_pick || 0,
        proWin: h.pro_win || 0,
      }));

    state.heroes.sort((a, b) => a.name.localeCompare(b.name));
    state.heroMap = {};
    state.heroes.forEach(h => { state.heroMap[h.id] = h; });

    return true;
  } catch (err) {
    showToast(`Помилка завантаження героїв: ${err.message}`, 'error');
    return false;
  }
}

function computeWinrate(h) {
  let wins = 0, picks = 0;
  // Brackets 4-7 (Archon, Legend, Ancient, Divine)
  for (let b = 4; b <= 7; b++) {
    wins += h[`${b}_win`] || 0;
    picks += h[`${b}_pick`] || 0;
  }
  return picks > 0 ? wins / picks : 0.5;
}

function computePicks(h) {
  let picks = 0;
  for (let b = 1; b <= 8; b++) {
    picks += h[`${b}_pick`] || 0;
  }
  return picks;
}

async function loadMatchups(heroId) {
  const cached = state.matchupCache[heroId];
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }

  try {
    const data = await api.fetch(`/heroes/${heroId}/matchups`);
    const matchupMap = {};
    data.forEach(m => {
      matchupMap[m.hero_id] = {
        games: m.games_played,
        wins: m.wins,
        winrate: m.games_played > 0 ? m.wins / m.games_played : 0.5,
      };
    });
    state.matchupCache[heroId] = { data: matchupMap, timestamp: Date.now() };
    return matchupMap;
  } catch (err) {
    console.warn(`Failed to load matchups for hero ${heroId}:`, err);
    return null;
  }
}

async function loadMatchupsForPicks() {
  const heroIds = [...state.enemies, ...state.allies];
  const toLoad = heroIds.filter(id => {
    const cached = state.matchupCache[id];
    return !cached || Date.now() - cached.timestamp > CACHE_TTL;
  });

  for (let i = 0; i < toLoad.length; i++) {
    await loadMatchups(toLoad[i]);
    if (i < toLoad.length - 1) await sleep(MATCHUP_FETCH_DELAY);
  }
}

// ============================================================
// Scoring Algorithm
// ============================================================

function computePositionFit(hero, position) {
  const positions = HERO_POSITIONS[hero.id];
  if (!positions || !positions.includes(position)) return 0;
  const idx = positions.indexOf(position);
  return POSITION_RANK_DECAY[idx] ?? 0.2;
}

function getPositionRank(heroId, position) {
  const positions = HERO_POSITIONS[heroId];
  if (!positions) return null;
  const idx = positions.indexOf(position);
  if (idx < 0) return null;
  return idx === 0 ? 'primary' : 'flex';
}

function computeCounterScore(heroId, enemies) {
  const breakdown = computeCounterBreakdown(heroId, enemies);
  return breakdown.score;
}

function computeCounterBreakdown(heroId, enemies) {
  if (enemies.length === 0) return { score: 0, perEnemy: [] };

  let totalAdvantage = 0;
  let count = 0;
  const perEnemy = [];

  for (const enemyId of enemies) {
    const enemyMatchups = state.matchupCache[enemyId]?.data;
    const enemy = state.heroMap[enemyId];
    const entry = { enemy, advantage: null, games: 0 };
    if (!enemyMatchups || !enemyMatchups[heroId]) {
      perEnemy.push(entry);
      continue;
    }

    const m = enemyMatchups[heroId];
    entry.games = m.games;
    if (m.games < MATCHUP_MIN_GAMES) {
      perEnemy.push(entry);
      continue;
    }

    // If enemy's WR vs us is LOW, we counter them (positive advantage)
    const ourAdvantage = 0.5 - m.winrate;
    entry.advantage = ourAdvantage;
    totalAdvantage += ourAdvantage;
    count++;
    perEnemy.push(entry);
  }

  return {
    score: count > 0 ? totalAdvantage / count : 0,
    perEnemy,
    samples: count,
  };
}

function computeSynergyScore(heroId, allies) {
  return computeSynergyBreakdown(heroId, allies).score;
}

function computeSynergyBreakdown(heroId, allies) {
  if (allies.length === 0) return { score: 0, newRoles: [], allRoles: [] };

  const hero = state.heroMap[heroId];
  if (!hero) return { score: 0, newRoles: [], allRoles: [] };

  const allyRoles = new Set();
  allies.forEach(aId => {
    const ally = state.heroMap[aId];
    if (ally) ally.roles.forEach(r => allyRoles.add(r));
  });

  const newRoles = hero.roles.filter(r => !allyRoles.has(r));
  const roleDiversityBonus = newRoles.length * 0.05;
  return {
    score: Math.min(roleDiversityBonus, 0.3),
    newRoles,
    allRoles: hero.roles,
  };
}

function computeRecommendations() {
  if (!state.selectedPosition) return [];

  // V7e model dispatch — uses STRATZ data + GBM trees for ranking.
  if (state.modelMode === 'v7e' && window.V7e && window.V7e.ready) {
    return computeRecommendationsV7e();
  }

  const pickedIds = new Set([...state.allies, ...state.enemies]);
  const candidates = state.heroes.filter(h => !pickedIds.has(h.id));

  const eligibleCandidates = candidates.filter(
    h => computePositionFit(h, state.selectedPosition) > 0
  );

  const scored = eligibleCandidates.map(hero => {
    const positionFit = computePositionFit(hero, state.selectedPosition);
    const baseWinrate = hero.winrate;
    const counterBreakdown = computeCounterBreakdown(hero.id, state.enemies);
    const synergyBreakdown = computeSynergyBreakdown(hero.id, state.allies);
    const counterScore = counterBreakdown.score;
    const synergyScore = synergyBreakdown.score;

    // Normalize scores to 0-1 range
    const normalizedWr = (baseWinrate - 0.40) / 0.20; // 40%-60% → 0-1
    const normalizedCounter = (counterScore + 0.10) / 0.20; // -10% to +10% → 0-1
    const normalizedSynergy = synergyScore / 0.30; // 0-30% → 0-1

    const w = state.enemies.length > 0 ? SCORE_WEIGHTS_WITH_ENEMIES : SCORE_WEIGHTS_NO_ENEMIES;
    const contributions = {
      baseWinrate: w.baseWinrate * clamp(normalizedWr),
      positionFit: w.positionFit * clamp(positionFit),
      counterScore: w.counterScore * clamp(normalizedCounter),
      synergyScore: w.synergyScore * clamp(normalizedSynergy),
    };
    const finalScore =
      contributions.baseWinrate +
      contributions.positionFit +
      contributions.counterScore +
      contributions.synergyScore;

    // Generate explanation tags
    const tags = [];
    tags.push({ type: 'wr', text: `WR ${(baseWinrate * 100).toFixed(1)}%` });
    const rank = getPositionRank(hero.id, state.selectedPosition);
    if (rank === 'primary') {
      tags.push({ type: 'fit', text: `Pos ${state.selectedPosition}` });
    } else if (rank === 'flex') {
      tags.push({ type: 'fit-flex', text: `Pos ${state.selectedPosition} (flex)` });
    }
    if (state.enemies.length > 0 && Math.abs(counterScore) > 0.003) {
      const sign = counterScore > 0 ? '+' : '';
      tags.push({ type: 'counter', text: `vs ворогів ${sign}${(counterScore * 100).toFixed(1)}%` });
    }
    if (synergyScore > 0.05 && state.allies.length > 0) {
      tags.push({ type: 'synergy', text: 'Synergy' });
    }

    return {
      hero,
      score: finalScore,
      tags,
      components: { positionFit, baseWinrate, counterScore, synergyScore },
      breakdown: {
        weights: w,
        contributions,
        counter: counterBreakdown,
        synergy: synergyBreakdown,
        positionRank: rank,
        normalizedWr,
        normalizedCounter,
        normalizedSynergy,
      },
    };
  });

  // Filter out heroes with very low position fit (unless no better options)
  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, TOP_RECOMMENDATIONS);
}

function clamp(v, min = 0, max = 1) {
  return Math.max(min, Math.min(max, v));
}

// ============================================================
// V7e Model Recommendations
// Uses STRATZ-derived data + Gradient Boosting trees.
// ============================================================
function computeRecommendationsV7e() {
  const v7 = window.V7e;
  const pos = state.selectedPosition;
  const ranked = v7.rank(pos, state.allies, state.enemies);
  const top = ranked.slice(0, TOP_RECOMMENDATIONS);

  // Map V7e ranked entries to UI rec format compatible with score modal
  return top.map(entry => {
    const hero = state.heroMap[entry.heroId];
    if (!hero) return null;
    const elig = v7.eligibility[entry.heroId] || [];
    const isPrimary = elig[0] === pos;
    const tags = [];
    const wrComp = entry.components.base_wr;
    tags.push({ type: 'wr', text: `WR ${(wrComp * 100).toFixed(1)}%` });
    if (isPrimary) tags.push({ type: 'fit', text: `Pos ${pos}` });
    else tags.push({ type: 'fit-flex', text: `Pos ${pos} (flex)` });
    if (state.enemies.length > 0) {
      const advPct = (entry.components.vs_adv - 0.5) * 100;
      if (Math.abs(advPct) >= 0.5) {
        const sign = advPct >= 0 ? '+' : '';
        tags.push({ type: 'counter', text: `vs ${sign}${advPct.toFixed(1)}%` });
      }
    }
    if (state.allies.length > 0) {
      const synPct = (entry.components.with_syn - 0.5) * 100;
      if (synPct >= 0.5) tags.push({ type: 'synergy', text: `synergy +${synPct.toFixed(1)}%` });
    }
    tags.push({ type: 'model', text: 'V7e' });

    return {
      hero,
      score: entry.score,
      tags,
      components: entry.components,
      breakdown: {
        model: 'v7e',
        positionRank: isPrimary ? 'primary' : 'flex',
        components: entry.components,
        // For Why? modal compatibility:
        weights: null,
        contributions: null,
        counter: null,
        synergy: null,
      },
    };
  }).filter(Boolean);
}

// ============================================================
// UI Rendering
// ============================================================

function renderHeroGrid() {
  const grid = document.getElementById('heroGrid');

  if (state.loading) {
    grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>Завантаження героїв...</p></div>';
    return;
  }

  const pickedIds = new Set([...state.allies, ...state.enemies]);
  let filtered = state.heroes;

  // Apply attribute filter
  if (state.attrFilter !== 'all') {
    const filterVal = state.attrFilter === 'all_attr' ? 'all' : state.attrFilter;
    filtered = filtered.filter(h => h.attr === filterVal);
  }

  // Apply search filter
  if (state.searchQuery) {
    const q = state.searchQuery.toLowerCase();
    filtered = filtered.filter(h => h.name.toLowerCase().includes(q));
  }

  grid.innerHTML = filtered.map(hero => {
    const isPicked = pickedIds.has(hero.id);
    const isAlly = state.allies.includes(hero.id);
    const isEnemy = state.enemies.includes(hero.id);
    let cls = 'hero-card';
    if (isPicked) cls += ' disabled';
    if (isAlly) cls += ' ally-pick';
    if (isEnemy) cls += ' enemy-pick';

    return `
      <div class="${cls}" data-hero-id="${hero.id}" title="${hero.name}">
        <img src="${hero.img}" alt="${hero.name}" loading="lazy">
        <div class="hero-attr-dot ${hero.attr}"></div>
        <div class="hero-name-overlay">${hero.name}</div>
      </div>
    `;
  }).join('');

  // Attach click handlers
  grid.querySelectorAll('.hero-card:not(.disabled)').forEach(card => {
    card.addEventListener('click', () => {
      const heroId = parseInt(card.dataset.heroId);
      addHeroToDraft(heroId);
    });
  });
}

function renderTeamSlots() {
  renderSlots('allySlots', state.allies, MAX_ALLIES, 'Союзник', 'radiant');
  renderSlots('enemySlots', state.enemies, MAX_ENEMIES, 'Ворог', 'dire');
  document.getElementById('allyCount').textContent = `${state.allies.length}/${MAX_ALLIES}`;
  document.getElementById('enemyCount').textContent = `${state.enemies.length}/${MAX_ENEMIES}`;
}

function renderSlots(containerId, heroIds, maxSlots, labelPrefix, team) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  for (let i = 0; i < maxSlots; i++) {
    const slot = document.createElement('div');
    const heroId = heroIds[i];

    if (heroId) {
      const hero = state.heroMap[heroId];
      slot.className = 'hero-slot filled';
      slot.innerHTML = `<img src="${hero.img}" alt="${hero.name}" title="${hero.name}">`;
      slot.addEventListener('click', () => removeHeroFromDraft(heroId, team));
    } else {
      slot.className = 'hero-slot empty';
      slot.innerHTML = `<span class="slot-label">${labelPrefix} ${i + 1}</span>`;
    }

    container.appendChild(slot);
  }
}

function renderRecommendations() {
  const grid = document.getElementById('recGrid');
  const subtitle = document.getElementById('recSubtitle');

  if (!state.selectedPosition) {
    subtitle.textContent = 'Обери позицію для початку';
    grid.innerHTML = '<div class="rec-placeholder"><p>Обери свою позицію та додай героїв ворожої/союзної команди для отримання рекомендацій</p></div>';
    state.lastRecs = [];
    return;
  }

  const recs = computeRecommendations();
  state.lastRecs = recs;

  if (recs.length === 0) {
    subtitle.textContent = 'Немає даних';
    grid.innerHTML = '<div class="rec-placeholder"><p>Не вдалося обчислити рекомендації</p></div>';
    return;
  }

  const posName = POSITIONS[state.selectedPosition].name;
  const parts = [];
  if (state.enemies.length > 0) parts.push(`vs ${state.enemies.length} ворогів`);
  if (state.allies.length > 0) parts.push(`з ${state.allies.length} союзниками`);
  subtitle.textContent = `Позиція ${state.selectedPosition} (${posName})${parts.length ? ' | ' + parts.join(', ') : ''}`;

  grid.innerHTML = recs.map((rec, idx) => {
    const scorePercent = (rec.score * 100).toFixed(0);
    const scoreClass = rec.score >= 0.65 ? 'high' : rec.score >= 0.45 ? 'mid' : 'low';

    return `
      <div class="rec-card" data-hero-id="${rec.hero.id}">
        <div class="rec-rank">${idx + 1}</div>
        <img class="rec-hero-img" src="${rec.hero.img}" alt="${rec.hero.name}">
        <div class="rec-info">
          <div class="rec-hero-name">${rec.hero.name}</div>
          <div class="rec-details">
            ${rec.tags.map(t => `<span class="rec-tag ${t.type}">${t.text}</span>`).join('')}
          </div>
        </div>
        <div class="rec-score-bar">
          <div class="rec-score-value ${scoreClass}">${scorePercent}</div>
          <div class="rec-score-label">Score</div>
        </div>
      </div>
    `;
  }).join('');

  // Click recommendation to open detailed score breakdown
  grid.querySelectorAll('.rec-card').forEach(card => {
    card.addEventListener('click', () => {
      const heroId = parseInt(card.dataset.heroId);
      const rec = state.lastRecs.find(r => r.hero.id === heroId);
      if (rec) showScoreModal(rec);
    });
  });
}

function showScoreModal(rec) {
  const backdrop = document.getElementById('modalBackdrop');
  const titleEl = document.getElementById('modalTitle');
  const subtitleEl = document.getElementById('modalSubtitle');
  const imgEl = document.getElementById('modalHeroImg');
  const scoreEl = document.getElementById('modalScore');
  const bodyEl = document.getElementById('modalBody');

  const { hero, score, breakdown, components } = rec;
  const posName = POSITIONS[state.selectedPosition].name;
  imgEl.src = hero.img;
  imgEl.alt = hero.name;
  titleEl.textContent = hero.name;
  scoreEl.innerHTML = `${(score * 100).toFixed(0)}`;

  // V7e modal: shows feature contributions instead of weighted components
  if (breakdown && breakdown.model === 'v7e') {
    const c = components;
    const elig = (window.V7e?.eligibility[hero.id]) || [];
    const allPosNames = elig.map(p => `Pos ${p}`).join(' / ');
    subtitleEl.textContent = `${allPosNames || 'без даних'} — обрано Pos ${state.selectedPosition} (${posName}), ${breakdown.positionRank === 'primary' ? 'основна' : 'флекс'}. Модель: V7e (GBM)`;
    const advPct = (c.vs_adv - 0.5) * 100;
    const synPct = (c.with_syn - 0.5) * 100;
    bodyEl.innerHTML = `
      <div class="score-section-title">Вхідні фічі моделі V7e</div>
      <div class="score-detail"><span>Base WR (Pos ${state.selectedPosition}, Bayesian)</span><span>${(c.base_wr * 100).toFixed(2)}%</span></div>
      <div class="score-detail"><span>Position Fit (rank ${elig.indexOf(state.selectedPosition) + 1}/${elig.length})</span><span>${(c.pos_fit * 100).toFixed(0)}%</span></div>
      <div class="score-detail"><span>Counter Adv vs ворогів${state.enemies.length ? '' : ' (n/a)'}</span><span class="${advPct > 0 ? 'pos-val' : advPct < 0 ? 'neg-val' : ''}">${advPct > 0 ? '+' : ''}${advPct.toFixed(2)}%</span></div>
      <div class="score-detail"><span>Synergy with союзниками${state.allies.length ? '' : ' (n/a)'}</span><span class="${synPct > 0 ? 'pos-val' : ''}">${synPct > 0 ? '+' : ''}${synPct.toFixed(2)}%</span></div>
      <div class="score-detail"><span>Role Gap fill</span><span>${(c.role_gap * 100).toFixed(0)}%</span></div>

      <div class="score-section-title">Як працює V7e</div>
      <div class="score-detail-text">Gradient Boosting (200 дерев) натренований на 1381 Divine+ матчах: для кожного з 5 фіч модель будує дерево вирішень, фінальний скор — sigmoid від суми внесків дерев. CV top-10 = ${((window.V7e?.model?.trees?.length || 0) > 0 ? '52.7%' : 'n/a')}, що ~2.1× краще за M1 (24.6%).</div>
    `;
    backdrop.hidden = false;
    return;
  }

  // M1 modal: original breakdown
  const { weights, contributions, counter, synergy, positionRank } = breakdown;
  const positions = HERO_POSITIONS[hero.id] || [];
  const allPosNames = positions.map(p => `Pos ${p}`).join(' / ');
  subtitleEl.textContent = `${allPosNames || 'без даних'} — для тебе обрано Pos ${state.selectedPosition} (${posName})${positionRank ? `, ${positionRank === 'primary' ? 'основна' : 'флекс'}` : ''}`;

  // Build the body HTML
  const fmtPct = (v) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;
  const fmtBar = (raw, max) => {
    const ratio = max === 0 ? 0 : Math.max(0, Math.min(1, raw / max));
    return `<div class="score-component-bar"><div class="score-component-bar-fill" style="transform: scaleX(${ratio})"></div></div>`;
  };

  const counterRows = counter.perEnemy && counter.perEnemy.length
    ? counter.perEnemy.map(e => {
        if (!e.enemy) return '';
        if (e.advantage === null) {
          return `<div class="score-detail"><span>vs ${e.enemy.name}</span><span>${e.games < MATCHUP_MIN_GAMES && e.games > 0 ? `мало даних (${e.games})` : 'дані відсутні'}</span></div>`;
        }
        const cls = e.advantage > 0 ? 'pos-val' : 'neg-val';
        return `<div class="score-detail"><span>vs ${e.enemy.name}</span><span class="${cls}">${fmtPct(e.advantage)} (n=${e.games})</span></div>`;
      }).join('')
    : '';

  const synergyRows = synergy.newRoles && synergy.newRoles.length
    ? `<div class="score-detail"><span>Нові ролі для команди</span><span class="pos-val">${synergy.newRoles.join(', ')}</span></div>`
    : (state.allies.length > 0 ? '<div class="score-detail"><span>Нові ролі</span><span>команда вже покриває всі ролі цього героя</span></div>' : '');

  bodyEl.innerHTML = `
    <div class="score-section-title">Внески у фінальний скор</div>
    <div class="score-component">
      <div class="score-component-label">Base WR</div>
      ${fmtBar(contributions.baseWinrate, weights.baseWinrate)}
      <div class="score-component-value">${(contributions.baseWinrate * 100).toFixed(1)}/${(weights.baseWinrate * 100).toFixed(0)}</div>
    </div>
    <div class="score-detail"><span>Загальний WR (Archon-Divine)</span><span>${(components.baseWinrate * 100).toFixed(2)}%</span></div>

    <div class="score-component">
      <div class="score-component-label">Position Fit</div>
      ${fmtBar(contributions.positionFit, weights.positionFit)}
      <div class="score-component-value">${(contributions.positionFit * 100).toFixed(1)}/${(weights.positionFit * 100).toFixed(0)}</div>
    </div>
    <div class="score-detail"><span>Грає на цій позиції</span><span class="pos-val">${positionRank === 'primary' ? 'основна' : 'флекс'} (${(components.positionFit * 100).toFixed(0)}%)</span></div>

    <div class="score-component">
      <div class="score-component-label">Counter Score</div>
      ${fmtBar(contributions.counterScore, weights.counterScore || 0.0001)}
      <div class="score-component-value">${(contributions.counterScore * 100).toFixed(1)}/${(weights.counterScore * 100).toFixed(0)}</div>
    </div>
    ${state.enemies.length === 0 ? '<div class="score-detail"><span>Counter</span><span>немає ворогів — не враховується</span></div>' : counterRows || '<div class="score-detail"><span>Counter</span><span>matchup-дані не завантажені</span></div>'}

    <div class="score-component">
      <div class="score-component-label">Synergy</div>
      ${fmtBar(contributions.synergyScore, weights.synergyScore)}
      <div class="score-component-value">${(contributions.synergyScore * 100).toFixed(1)}/${(weights.synergyScore * 100).toFixed(0)}</div>
    </div>
    ${synergyRows || '<div class="score-detail"><span>Synergy</span><span>немає союзників — не враховується</span></div>'}

    <div class="score-section-title">Ваги цієї конфігурації</div>
    <div class="score-detail"><span>Base WR</span><span>${(weights.baseWinrate * 100).toFixed(0)}%</span></div>
    <div class="score-detail"><span>Position Fit</span><span>${(weights.positionFit * 100).toFixed(0)}%</span></div>
    <div class="score-detail"><span>Counter</span><span>${(weights.counterScore * 100).toFixed(0)}%</span></div>
    <div class="score-detail"><span>Synergy</span><span>${(weights.synergyScore * 100).toFixed(0)}%</span></div>
  `;

  backdrop.hidden = false;
}

function closeScoreModal() {
  document.getElementById('modalBackdrop').hidden = true;
}

// ============================================================
// Draft Actions
// ============================================================

function addHeroToDraft(heroId) {
  const pickedIds = new Set([...state.allies, ...state.enemies]);
  if (pickedIds.has(heroId)) return;

  if (state.addMode === 'ally') {
    if (state.allies.length >= MAX_ALLIES) {
      showToast('Максимум 4 союзника (5-й — ти)', 'error');
      return;
    }
    state.allies.push(heroId);
  } else {
    if (state.enemies.length >= MAX_ENEMIES) {
      showToast('Максимум 5 ворогів', 'error');
      return;
    }
    state.enemies.push(heroId);
  }

  // Fire and forget but ensure recommendations render after matchups load
  updateDraftWithLoading();
}

function removeHeroFromDraft(heroId, team) {
  if (team === 'radiant') {
    state.allies = state.allies.filter(id => id !== heroId);
  } else {
    state.enemies = state.enemies.filter(id => id !== heroId);
  }
  updateDraftWithLoading();
}

async function updateDraftWithLoading() {
  renderTeamSlots();
  renderHeroGrid();

  // Show loading in recommendations while fetching matchups
  if (state.enemies.length > 0 || state.allies.length > 0) {
    const recGrid = document.getElementById('recGrid');
    if (state.selectedPosition) {
      recGrid.innerHTML = '<div class="rec-placeholder"><div class="spinner" style="margin:0 auto"></div><p style="margin-top:8px">Завантаження матчапів...</p></div>';
    }
    await loadMatchupsForPicks();
  }

  renderRecommendations();
}



function selectPosition(pos) {
  state.selectedPosition = pos;

  document.querySelectorAll('.pos-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.pos) === pos);
  });

  renderRecommendations();
}

function setAddMode(mode) {
  state.addMode = mode;
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
}

function resetAll() {
  state.selectedPosition = null;
  state.allies = [];
  state.enemies = [];
  state.searchQuery = '';
  state.attrFilter = 'all';

  document.querySelectorAll('.pos-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('heroSearch').value = '';
  document.querySelectorAll('.attr-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.attr === 'all');
  });

  renderTeamSlots();
  renderHeroGrid();
  renderRecommendations();
}

// ============================================================
// Toast Notifications
// ============================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(30px)';
    toast.style.transition = 'all 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ============================================================
// Event Binding
// ============================================================

function bindEvents() {
  // Position buttons
  document.getElementById('positionButtons').addEventListener('click', e => {
    const btn = e.target.closest('.pos-btn');
    if (btn) selectPosition(parseInt(btn.dataset.pos));
  });

  // Mode toggle
  document.getElementById('modeToggle').addEventListener('click', e => {
    const btn = e.target.closest('.mode-btn');
    if (btn) setAddMode(btn.dataset.mode);
  });

  // Search
  document.getElementById('heroSearch').addEventListener('input', e => {
    state.searchQuery = e.target.value;
    renderHeroGrid();
  });

  // Attribute filter
  document.getElementById('attrFilters').addEventListener('click', e => {
    const btn = e.target.closest('.attr-btn');
    if (!btn) return;
    state.attrFilter = btn.dataset.attr;
    document.querySelectorAll('.attr-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.attr === state.attrFilter);
    });
    renderHeroGrid();
  });

  // Reset
  document.getElementById('resetBtn').addEventListener('click', resetAll);

  // Model selector
  const modelSel = document.getElementById('modelSelect');
  if (modelSel) {
    modelSel.addEventListener('change', async e => {
      const newMode = e.target.value;
      if (newMode === 'v7e' && (!window.V7e || !window.V7e.ready)) {
        showToast('Завантажую V7e дані…', 'info');
        await window.V7e.init();
        if (!window.V7e.ready) {
          showToast('Не вдалося завантажити V7e', 'error');
          modelSel.value = 'm1';
          state.modelMode = 'm1';
          return;
        }
      }
      state.modelMode = newMode;
      showToast(newMode === 'v7e' ? 'Перемкнено на V7e (GBM)' : 'Перемкнено на M1 (Curated)', 'info');
      renderRecommendations();
    });
  }

  // Modal close
  document.getElementById('modalClose').addEventListener('click', closeScoreModal);
  document.getElementById('modalBackdrop').addEventListener('click', e => {
    if (e.target.id === 'modalBackdrop') closeScoreModal();
  });

  // Keyboard shortcut: 'q' to toggle mode, 'r' to reset, 'Esc' to close modal
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeScoreModal();
      return;
    }
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'q') setAddMode(state.addMode === 'ally' ? 'enemy' : 'ally');
    if (e.key === 'r') resetAll();
    if (e.key >= '1' && e.key <= '5') selectPosition(parseInt(e.key));
  });
}

// ============================================================
// Init
// ============================================================

async function init() {
  bindEvents();
  state.loading = true;
  renderHeroGrid();

  const success = await loadHeroes();
  state.loading = false;

  if (success) {
    renderHeroGrid();
    renderTeamSlots();
    renderRecommendations();
  }
}

document.addEventListener('DOMContentLoaded', init);
