"""Retrain the v3 market-residual model on all current data in one command.

Steps:
1. Rebuild the training data from data/fixtures.csv (as-of-day form and Elo
   features) and the Bet365 odds in the football-data.co.uk workbooks
   (data/raw/historical/*.xls[x] and data/raw/all-euro-data-*.xlsx, which
   the history update keeps current).
   Understat xG form (data/understat_matches.csv) is added as of each date.
2. Check that rows already in the current training dataset are reproduced
   exactly (same features and odds), so the pipeline cannot drift silently.
3. Holdout test: train on every season before the last complete one, test on
   that season, and compare with the Bet365 odds on the same matches.
4. Train on everything and save a dated artifact
   (models/nn_market_residual_v3_<YYYYMMDD>/). The API's "v3" always uses the
   newest v3 artifact; delete that folder to roll back.

Retraining once per season is plenty: a season adds ~4% more matches.

Usage:
    python src/retrain_v3.py            # rebuild, test, train and save
    python src/retrain_v3.py --dry-run  # rebuild and test only
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

# Keep PyTorch's import ahead of pandas/NumPy on this Windows setup.
import torch
import numpy as np
import pandas as pd

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from build_elo_features_asof_day_v1 import build_elo_features_asof_day  # noqa: E402
from build_features_asof_day_v1 import build_features_asof_day_optimized  # noqa: E402
from experiment_nn_market_residual import per_match_log_loss  # noqa: E402
from experiment_nn_odds_asof_day_v1 import (  # noqa: E402
    COMMON_FEATURES,
    DEVICE,
    ELO_COLUMNS,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    MODEL_FEATURES,
    ODDS_FEATURES,
    ODDS_MATCH_KEYS,
    calculate_metrics,
)
import train_final_nn_market_residual_v3 as v3  # noqa: E402
import train_final_nn_odds_asof_day_v1 as trainer  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES_PATH = DATA / "fixtures.csv"
RAW_DIR = DATA / "raw"
SHEETS = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga", "I1": "Serie A", "F1": "Ligue 1"}
REPRODUCTION_TOLERANCE = 1e-9
COMPLETE_SEASON_MIN_MATCHES = 1500  # five leagues play ~1,750 matches a season


def load_bet365_odds() -> pd.DataFrame:
    """Bet365 1X2 probabilities (inverse odds, normalised) from every workbook."""
    paths = sorted((RAW_DIR / "historical").glob("all-euro-data-*")) + sorted(RAW_DIR.glob("all-euro-data-*.xlsx"))
    frames = []
    for path in paths:
        excel = pd.ExcelFile(path)
        for sheet, league in SHEETS.items():
            if sheet not in excel.sheet_names:
                continue
            raw = pd.read_excel(excel, sheet_name=sheet)
            if not {"Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"} <= set(raw.columns):
                continue
            raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
            frames.append(pd.DataFrame({
                "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.normalize(),
                "league": league,
                "home_team": raw["HomeTeam"],
                "away_team": raw["AwayTeam"],
                **{name: pd.to_numeric(raw[column], errors="coerce")
                   for name, column in (("odds_home", "B365H"), ("odds_draw", "B365D"), ("odds_away", "B365A"))},
            }))
    odds = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    # A season can exist both in historical/ and as the current workbook: keep the newest file's row.
    odds = odds.drop_duplicates(ODDS_MATCH_KEYS, keep="last")
    decimal = odds[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=np.float64)
    valid = np.isfinite(decimal).all(axis=1) & (decimal > 0).all(axis=1)
    odds = odds[valid].copy()
    implied = 1.0 / odds[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=np.float64)
    odds[ODDS_FEATURES] = implied / implied.sum(axis=1, keepdims=True)
    return odds[ODDS_MATCH_KEYS + ODDS_FEATURES]


def build_dataset() -> pd.DataFrame:
    """Same filters and joins as prepare_odds_model_dataset_asof_day_v1/v2."""
    fixtures = pd.read_csv(FIXTURES_PATH, low_memory=False)
    form = build_features_asof_day_optimized(fixtures)
    elo = build_elo_features_asof_day(fixtures)
    keys = KEY_COLUMNS
    for frame in (form, elo):
        frame["date"] = pd.to_datetime(frame["date"])
    combined = form.merge(
        elo[keys + [c for c in elo.columns if c in MODEL_FEATURES and c not in form.columns]],
        on=keys, how="left", validate="one_to_one",
    )
    combined = combined.dropna(subset=COMMON_FEATURES + ELO_COLUMNS)
    combined = combined[combined["result"].isin(LABEL_TO_INDEX)].copy()
    combined["date"] = combined["date"].dt.normalize()

    dataset = combined.merge(load_bet365_odds(), on=ODDS_MATCH_KEYS, how="inner", validate="one_to_one")
    dataset = v3.add_xg_features(dataset)
    dataset = dataset[KEY_COLUMNS + ["result", "season"] + v3.V3_FEATURES]
    return dataset.sort_values(["date", "time"], kind="stable").reset_index(drop=True)


def check_reproduction(dataset: pd.DataFrame) -> None:
    """Rows in the current training dataset must come out identical.

    Only the 30 v1/v2 features are compared: older datasets have no xG.
    """
    current = pd.read_csv(v3.DATASET_PATH, parse_dates=["date"], low_memory=False)
    merged = current.merge(dataset, on=ODDS_MATCH_KEYS, how="left", suffixes=("_current", "_new"), indicator=True)
    missing = int((merged["_merge"] != "both").sum())
    both = merged[merged["_merge"] == "both"]
    worst = max(
        float(np.nanmax(np.abs(both[f"{c}_current"].to_numpy(float) - both[f"{c}_new"].to_numpy(float))))
        for c in MODEL_FEATURES
    )
    print(
        f"Reproduktion: {len(both):,}/{len(current):,} kampe fra nuværende træningsdata genfundet; "
        f"største afvigelse i en feature: {worst:.2e}"
    )
    if missing or worst > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"Pipeline reproducerer ikke de nuværende træningsdata ({missing} mangler, "
            f"afvigelse {worst:.2e}); intet er trænet."
        )


def predict(ensemble, preprocessor, frame) -> np.ndarray:
    x = torch.as_tensor(preprocessor.transform(frame), dtype=torch.float32, device=DEVICE)
    market = torch.as_tensor(frame[ODDS_FEATURES].to_numpy(np.float64), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return ensemble.to(DEVICE)(x, market).cpu().numpy()


def holdout_test(dataset: pd.DataFrame) -> dict:
    """Train on seasons before the last complete season; test on that season."""
    counts = dataset.groupby("season").size()
    complete = counts[counts >= COMPLETE_SEASON_MIN_MATCHES].index
    season = sorted(complete)[-1]
    train = dataset[dataset["season"] < season]
    test = dataset[dataset["season"] == season]
    print(f"\nHoldout-test: træner på {len(train):,} kampe før {season}, tester på {len(test):,} kampe i {season}")
    selected, _ = v3.select_epochs(train)
    ensemble, preprocessor = v3.train_final(train, {seed: epoch for seed, (epoch, _) in selected.items()})

    actual = test["result"].to_numpy()
    model_probs = predict(ensemble, preprocessor, test)
    odds_probs = test[ODDS_FEATURES].to_numpy(np.float64)
    model, odds = calculate_metrics(actual, model_probs), calculate_metrics(actual, odds_probs)
    diff = per_match_log_loss(actual, model_probs) - per_match_log_loss(actual, odds_probs)
    result = {
        "source": f"retrain_v3.py holdout: trained before {season}, tested on {season}",
        "season": season,
        "matches": int(len(test)),
        "accuracy": model["accuracy"],
        "log_loss": model["log_loss"],
        "odds_only_accuracy": odds["accuracy"],
        "odds_only_log_loss": odds["log_loss"],
        "delta_log_loss_vs_odds": float(diff.mean()),
        "delta_log_loss_vs_odds_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
    }
    print(
        f"  Model: træfsikkerhed {model['accuracy']:.2%}, log loss {model['log_loss']:.5f}\n"
        f"  Odds:  træfsikkerhed {odds['accuracy']:.2%}, log loss {odds['log_loss']:.5f}\n"
        f"  Δ log loss mod odds: {result['delta_log_loss_vs_odds']:+.5f} ± {result['delta_log_loss_vs_odds_se']:.5f}"
    )
    if result["delta_log_loss_vs_odds"] > 2 * result["delta_log_loss_vs_odds_se"]:
        print("  ADVARSEL: modellen var tydeligt dårligere end odds i holdout-sæsonen.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gentræn v3 på alle nuværende data.")
    parser.add_argument("--dry-run", action="store_true", help="Byg data og kør holdout-test, men gem intet.")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    # A second retrain on the same day gets _2, _3, ...: it sorts after the
    # earlier one, so latest_v3_dir() picks it, and nothing is overwritten.
    for attempt in range(1, 100):
        suffix = stamp if attempt == 1 else f"{stamp}_{attempt}"
        artifact_dir = v3.ARTIFACT_DIR.with_name(f"{v3.ARTIFACT_DIR.name}_{suffix}")
        dataset_path = DATA / f"odds_model_dataset_retrain_{suffix}.csv"
        if not (artifact_dir.exists() or dataset_path.exists()):
            break
    else:
        raise FileExistsError(f"For mange gentræninger i dag: {artifact_dir}")

    dataset = build_dataset()
    print(
        f"Træningsdata: {len(dataset):,} kampe, {dataset['date'].min().date()} → {dataset['date'].max().date()} "
        f"(nuværende v3: {len(pd.read_csv(v3.DATASET_PATH, usecols=['date'])):,})"
    )
    check_reproduction(dataset)
    backtest = holdout_test(dataset)
    if args.dry_run:
        print(f"\nTør kørsel: intet gemt. Tid: {time.perf_counter() - started:.0f} sek.")
        return 0

    with dataset_path.open("x", encoding="utf-8", newline="") as handle:
        dataset.drop(columns="season").to_csv(handle, index=False)
    frame = trainer.validate_dataset(dataset_path)
    print("\nEndelig træning på alle kampe:")
    selected, split_info = v3.select_epochs(frame)
    ensemble, preprocessor = v3.train_final(frame, {seed: epoch for seed, (epoch, _) in selected.items()})
    v3.save_artifact(
        ensemble, preprocessor, frame, selected, split_info,
        artifact_dir=artifact_dir, artifact_name=artifact_dir.name,
        dataset_path=dataset_path, backtest=backtest,
    )
    print(
        f"\nGemt: {artifact_dir}\nTræningsdata: {dataset_path}\n"
        f"API'ets V3 bruger nu {artifact_dir.name}. Slet mappen for at rulle tilbage.\n"
        f"Tid: {time.perf_counter() - started:.0f} sek."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
