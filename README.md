# Dota 2 Draft Advisor

Web-додаток для прогнозування найкращого вибору героїв на стадії драфту у Dota 2.

## Можливості

- **Вибір позиції (1-5)**: Carry, Mid, Offlane, Soft Support, Hard Support
- **Дві моделі рекомендацій** (вибір у заголовку):
  - **M1 — Curated** (default): курована мапа `HERO_POSITIONS` + OpenDota matchup-дані. Стабільна, працює офлайн.
  - **V7e — GBM** (beta): Gradient Boosted дерева, навчені на 1381 Divine+ матчах. **2× краще top-10 точність** проти M1 ([детальний звіт](research/FINDINGS.md)).
- **Драфт-панель**: додавайте союзних та ворожих героїв
- **Рекомендації в реальному часі**: алгоритм аналізує та рекомендує найкращих героїв
- **Counter-pick аналіз**: враховує матчапи проти ворожих героїв
- **Фільтри**: пошук за назвою, фільтрація за атрибутом (STR/AGI/INT/UNI)
- **Гарячі клавіші**: `1-5` позиція, `Q` перемикання режиму, `R` скидання

## Бектест (1381 Divine+ ranked match)

| Модель | Pick top-10 | Pick top-5 | Pick top-1 | Mean rank | Win-pred acc |
|---|---|---|---|---|---|
| M0 (role-baseline) | 12.4% | 4.4% | 0.4% | 40.3 | 50.7% |
| **M1** (production) | 24.1% | 6.1% | 0.2% | 28.5 | 50.0% |
| M5* (logistic, trained) | 8.8% | 4.3% | 0.8% | 31.0 | 52.2% |
| **V7e** (GBM, pick-rec) | **48.2%** | **32.8%** | **9.3%** | **16.8** | **57.1%** |

## Джерело даних

Використовує [OpenDota API](https://docs.opendota.com/) (безкоштовний, без API ключа):
- `/api/heroStats` — статистика героїв (winrate по рангах)
- `/api/heroes/{id}/matchups` — матчапи (counter-pick дані)

## Алгоритм скорингу

Для кожного кандидата обчислюється зважений скор:

| Компонент | Вага (без ворогів) | Вага (з ворогами) |
|---|---|---|
| Base Winrate | 45% | 25% |
| Position Fit | 40% | 20% |
| Counter Score | 0% | 40% |
| Synergy | 15% | 15% |

- **Base Winrate**: вінрейт героя у рангах Archon-Divine
- **Position Fit**: наскільки ролі героя відповідають обраній позиції
- **Counter Score**: середня перевага проти ворожих героїв (з matchup даних)
- **Synergy**: різноманітність ролей у команді

## Запуск

Просто відкрийте `index.html` у браузері або запустіть локальний сервер:

```bash
python3 -m http.server 8080
# або
npx serve .
```

## Технології

- Vanilla JavaScript (ES6+)
- HTML5 + CSS3 (CSS Grid, Flexbox)
- OpenDota API (CORS-enabled)
- Кешування у пам'яті з TTL 30 хв
- Retry з exponential backoff для API запитів
