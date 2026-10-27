I can't create files from here, but below is the content, ready to save as e.g. `docs/OPERATIONS.md`. I only included commands we've used or seen in this session. Two gaps: I don't know the exact command that builds the v2 feature/dataset files, and I haven't confirmed whether `update_historical_data.py` runs on its own, so check both before relying on them. The prediction tracking, the Pinnacle/xG experiment and the new frontend are still being built in the main conversation. Add their steps once they're done.

````markdown
# MatchHubML – operations guide

All commands run from the project root with the virtual environment:
`.venv\Scripts\python.exe` (Windows). API keys live in `.env`
(`FOOTBALL_DATA_API_KEY`, the 5DollarFootball key, optional `API_FOOTBALL_KEY`).

## 1. Run the frontend

```bash
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/ (the API serves `matchhubml_frontend_live.html` at `/`).

- Model selector: V3 is the default (market-residual ensemble); V1/V2 remain available.
- Useful endpoints: `/api/health`, `/api/matches?model_version=v3&league=...&limit=50`.
- The first request of the day builds the history state (~1 s); later requests use the cache.

## 2. Update data

### Upcoming fixtures + Bet365 opening odds (SQLite cache)
Use the **Sync** button in the frontend (`POST /api/sync`), or:

```bash
python -m src.sync_football_data --days 30 --match-refresh-hours 0 --odds-refresh-hours 0 --max-odds-calls 5
```

Options: `--date-from/--date-to YYYY-MM-DD`, `--leagues`, `--skip-odds`, `--dry-run`.
Note: the window starts today, so results of played matches are only stored if
the window includes past dates (e.g. `--date-from` 14 days back with `--skip-odds`).

### Historical results (football-data.co.uk → data/fixtures.csv)
`update_historical_data.ensure_historical_data_current()` runs automatically when
the prediction CLI starts (weekly refresh; prediction continues on failure).
The feature cache picks up a rewritten `fixtures.csv` automatically (keyed on
file modification time).

### Understat xG (free, no key)
```bash
python src/collect_understat_xg.py
```
Caches raw JSON in `data/raw/understat/`, writes `data/understat_matches.csv`.
Finished seasons are downloaded once; the current season refreshes daily.

## 3. Update / retrain the model

Training data: `data/odds_model_dataset_asof_day_v2.csv`
(27 as-of-day sports features + Bet365 probabilities + result).

1. Refresh historical data (section 2) and rebuild the feature/dataset CSVs.
   The builders (`build_features_asof_day_v1.py`, `build_elo_features_asof_day_v1.py`,
   `prepare_odds_model_dataset_asof_day_v2.py`) **refuse to overwrite** existing
   outputs – move or rename the old files first.
2. Train v3 (refuses to overwrite `models/nn_market_residual_v3/`):
   ```bash
   cd src
   python train_final_nn_market_residual_v3.py
   ```
   To keep the old model, train into a new folder/version instead and register it:
   - `src/api.py`: add it to `MODEL_DIRS` and the `model_version` pattern
   - `matchhubml_frontend_live.html`: add an `<option>` in the version select
3. Optional backtest before deploying (writes `data/nn_market_residual_summary.csv`,
   refuses to overwrite):
   ```bash
   cd src
   python experiment_nn_market_residual.py
   ```
   Compare against the "odds_only" row – the model should not be worse than the market.

## 4. Predict from the command line

```bash
python src/predict_upcoming_match.py --league "La Liga" --home-team "Levante UD" \
  --away-team "Athletic Club" --match-date 2026-11-01 --use-stored-opening-odds
```
Use `--model-dir models/nn_odds_asof_day_v1` to pick another model, or pass
`--odds-prob-home/--odds-prob-draw/--odds-prob-away` instead of stored odds.

## 5. Tests

```bash
python -m unittest tests.test_data_store tests.test_prediction_startup_update \
  tests.test_sync_football_data tests.test_update_historical_data tests.test_model_v3
python -m unittest src.upcoming_match_features src.predict_upcoming_match
python verify_feature_reference.py
```

## Caveats
- Historical odds have no snapshot timestamps; backtests are not proof of pre-kickoff performance.
- Training uses football-data.co.uk B365 prices, while live prediction uses
  5DollarFootball "opening" prices – different snapshots.
- The repo has no commits yet – commit before retraining or bulk data updates.
````

Sync now updates the history (Data tab → "Synkronisér nu"):
- It downloads new results from football-data.co.uk when the last download is over 12 hours old. If the download fails, it keeps the old data and still syncs matches.
- The download URL now follows the season. It was hard-coded to 2026-27 and would have stopped working next July.
- The workbook is now saved to data/raw/, because it holds the Bet365 odds a retrain needs.

Retraining: python src/retrain_v3.py (or --dry-run to test without saving):
1. Rebuilds the training data from fixtures.csv and the workbooks, and checks that all 43,174 existing rows come out the same. The largest difference was 6.8e-13; if they ever don't match, it stops.
2. Holdout test: trains without the last full season, then tests on it.
3. Trains on everything and saves a dated foal_v3_YYYYMMDD/. "v3" in the API and the CLIalways uses the newest such folder; delete it to roll back.

Today's run

- Data: the history update added 49 matches up to 26 October. The new v3 is trained on 43,518 matches, 344 more than
  the old one.
- Holdout on 2025-26: 53.38% accuracy against 53.26% for the odds, and log loss −0.0009 ± 0.0010. That's in line with the market, not a clear improvement.
- Modeller tab: it now shows this holdout re
- Leak guard: the backfill script now skips any match a model was trained on. Otherwise a re-run would have let the new v3 "predict" games it had already seen the results of.

Retraining once a season is enough; the new matches go into the features straight away without retraining. When the Resultater tab has 200–300 scored matches, I can add the calibration chart described above. That shows whether the model really has systematic errors worth fix