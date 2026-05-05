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

// Role weights for position affinity scoring
const POSITION_ROLE_WEIGHTS = {
  1: { Carry: 4, Escape: 1, Pusher: 1 },
  2: { Nuker: 3, Carry: 2, Escape: 1, Disabler: 1 },
  3: { Initiator: 3, Durable: 3, Disabler: 1, Nuker: 1 },
  4: { Support: 2, Initiator: 2, Disabler: 2, Nuker: 1, Escape: 1 },
  5: { Support: 4, Disabler: 1, Durable: 1 },
};

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
  const weights = POSITION_ROLE_WEIGHTS[position];
  if (!weights) return 0;

  let score = 0;
  let maxPossible = 0;
  for (const [role, weight] of Object.entries(weights)) {
    maxPossible += weight;
    if (hero.roles.includes(role)) {
      score += weight;
    }
  }

  return maxPossible > 0 ? score / maxPossible : 0;
}

function computeCounterScore(heroId, enemies) {
  if (enemies.length === 0) return 0;

  let totalAdvantage = 0;
  let count = 0;

  for (const enemyId of enemies) {
    const enemyMatchups = state.matchupCache[enemyId]?.data;
    if (!enemyMatchups || !enemyMatchups[heroId]) continue;

    const m = enemyMatchups[heroId];
    if (m.games < 10) continue; // Need enough data

    // Enemy's winrate against our hero candidate
    // If enemy has LOW winrate vs us, that's good (we counter them)
    const enemyWr = m.winrate;
    const ourAdvantage = 0.5 - enemyWr; // positive = we counter them
    totalAdvantage += ourAdvantage;
    count++;
  }

  return count > 0 ? totalAdvantage / count : 0;
}

function computeSynergyScore(heroId, allies) {
  if (allies.length === 0) return 0;

  let totalSynergy = 0;
  let count = 0;

  // For synergy, we use the candidate hero's matchup data.
  // If ally hero often appears against the same enemies and both win,
  // that's a rough proxy for synergy.
  // Simpler approach: use role diversity as synergy indicator
  const hero = state.heroMap[heroId];
  if (!hero) return 0;

  const allyRoles = new Set();
  allies.forEach(aId => {
    const ally = state.heroMap[aId];
    if (ally) ally.roles.forEach(r => allyRoles.add(r));
  });

  // Bonus for bringing roles the team doesn't have
  const newRoles = hero.roles.filter(r => !allyRoles.has(r));
  const roleDiversityBonus = newRoles.length * 0.05;

  // Also check matchup-based synergy (if we have ally matchup data)
  for (const allyId of allies) {
    const allyMatchups = state.matchupCache[allyId]?.data;
    if (!allyMatchups || !allyMatchups[heroId]) continue;

    const m = allyMatchups[heroId];
    if (m.games < 10) continue;

    // This is how the ally performs AGAINST our candidate
    // If ally has a HIGH winrate when facing this hero as opponent,
    // it doesn't directly indicate synergy.
    // But if the matchup is close to 50%, they're neutral.
    // For synergy, we'd need "with" data which isn't available directly.
    // Skip this for now and rely on role diversity.
  }

  return Math.min(roleDiversityBonus, 0.3);
}

function computeRecommendations() {
  if (!state.selectedPosition) return [];

  const pickedIds = new Set([...state.allies, ...state.enemies]);
  const candidates = state.heroes.filter(h => !pickedIds.has(h.id));

  const scored = candidates.map(hero => {
    const positionFit = computePositionFit(hero, state.selectedPosition);
    const baseWinrate = hero.winrate;
    const counterScore = computeCounterScore(hero.id, state.enemies);
    const synergyScore = computeSynergyScore(hero.id, state.allies);

    // Normalize scores to 0-1 range
    const normalizedWr = (baseWinrate - 0.40) / 0.20; // 40%-60% → 0-1
    const normalizedCounter = (counterScore + 0.10) / 0.20; // -10% to +10% → 0-1
    const normalizedSynergy = synergyScore / 0.30; // 0-30% → 0-1

    const w = state.enemies.length > 0 ? SCORE_WEIGHTS_WITH_ENEMIES : SCORE_WEIGHTS_NO_ENEMIES;
    const finalScore =
      w.baseWinrate * clamp(normalizedWr) +
      w.positionFit * clamp(positionFit) +
      w.counterScore * clamp(normalizedCounter) +
      w.synergyScore * clamp(normalizedSynergy);

    // Generate explanation tags
    const tags = [];
    tags.push({ type: 'wr', text: `WR ${(baseWinrate * 100).toFixed(1)}%` });
    if (positionFit >= 0.5) {
      tags.push({ type: 'fit', text: `Pos ${state.selectedPosition}` });
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
    return;
  }

  const recs = computeRecommendations();

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

  // Click recommendation to add to allies
  grid.querySelectorAll('.rec-card').forEach(card => {
    card.addEventListener('click', () => {
      const heroId = parseInt(card.dataset.heroId);
      showToast(`${state.heroMap[heroId].name} — рекомендований пік!`, 'info');
    });
  });
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

  // Keyboard shortcut: 'q' to toggle mode, 'r' to reset
  document.addEventListener('keydown', e => {
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
