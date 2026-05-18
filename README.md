# Dota 2 Draft Advisor

Web-додаток для прогнозування найкращого вибору героїв на стадії драфту у Dota 2.

## Можливості

- **Вибір позиції (1-5)**: Carry, Mid, Offlane, Soft Support, Hard Support
- **Чотири моделі рекомендацій** (вибір у заголовку):
  - **V8 fair — sklearn GBM** (default): 300 дерев, 25 фіч.
  - **V9c fair — LightGBM lambdarank**: 400 дерев, 25 фіч.
  - **V10c fair — LightGBM lambdarank + team-composition**: 400 дерев, 39 фіч.
  - **V7e — GBM**: 5-фічний baseline.
- **Драфт-панель**: додавайте союзних та ворожих героїв.
- **Рекомендації в реальному часі**: алгоритм аналізує та рекомендує топ-10 героїв.
- **Counter-pick аналіз**: враховує матчапи проти ворожих героїв.
- **Фільтри**: пошук за назвою, фільтрація за атрибутом (STR/AGI/INT/UNI).
- **Гарячі клавіші**: `1-5` позиція, `Q` перемикання режиму, `R` скидання.

## Honest backtest (1256 newest matches held out)

Усі моделі натреновані на найстаріших 5026 матчах, оцінені на 1256 найновіших
(які жодна модель ніколи не бачила під час тренування):

| Модель | Архітектура | top-1 | top-5 | top-10 | mean rank |
|---|---|---:|---:|---:|---:|
| **V8 fair** (default) | sklearn GBM, 25 features | **17.5%** | **39.1%** | **57.5%** | **12.22** |
| V9c fair | LightGBM lambdarank, 25 features | 17.4% | 38.6% | 57.3% | 12.35 |
| V10c fair | LightGBM lambdarank + team-comp, 39 features | 17.1% | 39.3% | 57.4% | 12.32 |
| V7e | sklearn GBM, 5 features | 18.1% | 41.2% | 55.9% | 12.65 |

> Note: previously reported V9c "74% top-10" and V8 "62% top-10" were
> inflated due to a train-test leak. See
> [research/FAIR_EVALUATION_FINDINGS.md](research/FAIR_EVALUATION_FINDINGS.md)
> for the full explanation and
> [research/EVALUATION.md](research/EVALUATION.md) for proper methodology.

## Джерело даних

Використовує [OpenDota API](https://docs.opendota.com/) (безкоштовний, без API ключа)
+ [STRATZ GraphQL](https://stratz.com/api) для enrichment:
- `/api/heroStats` — статистика героїв
- STRATZ GraphQL — рангові матчі, позиційні winrate'и, синергії, counter-pair дані

## Алгоритм

V8/V9/V10 — gradient boosted trees, натреновані на 5026 Divine+ матчах
(чесний chronological 80/20 split). Кожен кандидат-герой описується 25
фічами (per-position синергія/counter, min/max/spread, one-hot позиції,
popularity, role-gap). V10 додає 14 додаткових team-composition фіч
(role counts, magic/agi/int ratios, illusion flags).

Інференс у браузері: модель завантажується як JSON, ~400 дерев обходяться
sub-millisecond. Вихід — score 0..1 (sigmoid від raw tree-sum для V8;
sigmoid від ranker score для V9/V10), використовується лише для ранжування.

## Запуск

Просто відкрийте `index.html` у браузері або запустіть локальний сервер:

```bash
python3 -m http.server 8080
# або
npx serve .
```

## Технології

- Vanilla JavaScript (ES6+) — інференс GBM/LightGBM моделей у браузері
- Python (`research/`) — тренування, бектест, експорт моделей
- HTML5 + CSS3 (CSS Grid, Flexbox)
- OpenDota API + STRATZ GraphQL
