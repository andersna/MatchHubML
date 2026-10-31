"""Experiment: which training window should the v3 model use?

Strategies (v3 market-residual network on V3_FEATURES, 5-seed ensembles):
  A  all          - every earlier match (current production behaviour)
  B  last3/last5  - only the N most recent seasons before the test period
  C  decay_1y/2y/4y - all history, sample weights 0.5 ** (age / half-life)

Walk-forward folds: for each test season 2022-23 .. 2025-26, train on all
earlier matches and test on the season. Early stopping uses the last 15% of
dates, or with --validation last_season the last complete season (plus any
partial season after it).
A final "this week" fold trains on everything before HOLDOUT_START and tests
on the matches played since, i.e. data no production model has seen.

The data is exactly what src/retrain_v3.py trains on (retrain_v3.build_dataset).
All variants are scored on identical matches, paired against `all` and the
Bet365 odds. Historical odds have no snapshot timestamps; backtest only.
"""

import argparse
from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch  # noqa: F401
import numpy as np
import pandas as pd

import experiment_nn_market_residual as market_residual
from experiment_nn_market_residual import per_match_log_loss, train_residual
from experiment_nn_odds_asof_day_v1 import ODDS_FEATURES, calculate_metrics
from retrain_v3 import COMPLETE_SEASON_MIN_MATCHES, build_dataset
from train_final_nn_market_residual_v3 import V3_FEATURES


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUMMARY_PATH = DATA / "nn_v3_training_window_summary.csv"
PER_MATCH_PATH = DATA / "nn_v3_training_window_per_match.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
TEST_SEASONS = range(2022, 2026)
# Friday of the latest matchweek: the final fold tests on everything from here.
HOLDOUT_START = pd.Timestamp("2026-10-23")
# name -> (seasons kept or None for all, recency half-life in days or None)
VARIANTS = {
    "all": (None, None),
    "last3": (3, None),
    "last5": (5, None),
    "decay_1y": (None, 1 * 365.25),
    "decay_2y": (None, 2 * 365.25),
    "decay_4y": (None, 4 * 365.25),
}
BASELINE = "all"


def folds(data: pd.DataFrame):
    """Yield (name, history, test); history ends where the test period starts."""
    for season in TEST_SEASONS:
        yield (f"{season}-{(season + 1) % 100:02d}",
               data[data["season_start"] < season], data[data["season_start"] == season])
    yield ("this_week", data[data["date"] < HOLDOUT_START], data[data["date"] >= HOLDOUT_START])


def split_validation(history: pd.DataFrame, validation_mode: str):
    """Return (subtrain, validation) for early stopping.

    fraction    - the last 15% of unique dates
    last_season - the last complete season, plus any partial season after it
    """
    if validation_mode == "last_season":
        counts = history.groupby("season_start").size()
        last_complete = counts[counts >= COMPLETE_SEASON_MIN_MATCHES].index.max()
        recent = history["season_start"] >= last_complete
        return history[~recent], history[recent]
    dates = sorted(history["date"].dt.date.unique())
    split = int(len(dates) * 0.85)
    return (history[history["date"].dt.date.isin(dates[:split])],
            history[history["date"].dt.date.isin(dates[split:])])


def fit_variant(history: pd.DataFrame, test: pd.DataFrame, seasons, half_life,
                validation_mode: str = "fraction") -> np.ndarray:
    if seasons is not None:
        recent = sorted(history["season_start"].unique())[-seasons:]
        history = history[history["season_start"].isin(recent)]
    subtrain, validation = split_validation(history, validation_mode)
    if half_life is not None:
        market_residual.RECENCY_HALF_LIFE_DAYS = half_life  # read by recency_weights()
    seed_probs = [
        train_residual(subtrain, validation, test, list(V3_FEATURES), False, half_life is not None, 0.0, 0, seed)[0]
        for seed in SEEDS
    ]
    return np.mean(seed_probs, axis=0)


def summarise(per_match: pd.DataFrame) -> pd.DataFrame:
    # Walk-forward totals reuse each season's row numbers, so pair on the source fold too.
    log_loss = per_match.set_index(["fold", "source", "row", "variant"])["log_loss"].unstack("variant")
    rows = []
    for (fold, variant), group in per_match.groupby(["fold", "variant"], sort=False):
        probs = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        row = {"fold": fold, "variant": variant, "matches": len(group),
               **calculate_metrics(group["actual"].to_numpy(), probs)}
        for reference in (BASELINE, "odds_only"):
            diff = log_loss.loc[fold, variant] - log_loss.loc[fold, reference]
            row[f"delta_log_loss_vs_{reference}"] = diff.mean()
            row[f"delta_log_loss_vs_{reference}_se"] = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sammenlign træningsvinduer for v3.")
    parser.add_argument("--validation", choices=("fraction", "last_season"), default="fraction",
                        help="Early stopping på sidste 15%% af datoerne eller sidste hele sæson.")
    args = parser.parse_args()
    summary_path, per_match_path = SUMMARY_PATH, PER_MATCH_PATH
    if args.validation != "fraction":
        summary_path = SUMMARY_PATH.with_name(f"{SUMMARY_PATH.stem}_{args.validation}.csv")
        per_match_path = PER_MATCH_PATH.with_name(f"{PER_MATCH_PATH.stem}_{args.validation}.csv")
    for path in (summary_path, per_match_path):
        if path.exists():
            raise FileExistsError(f"Output findes allerede; intet overskrives: {path}")
    started = time.perf_counter()
    data = build_dataset()
    data["season_start"] = data["season"].str[:4].astype(int)
    print(f"Datasæt: {len(data):,} kampe, {data['date'].min().date()} → {data['date'].max().date()}")

    rows = []
    for fold, history, test in folds(data):
        if test.empty:
            print(f"{fold}: ingen testkampe; springes over")
            continue
        actual = test["result"].to_numpy()
        probs = {"odds_only": test[ODDS_FEATURES].to_numpy(np.float64)}
        for variant, (seasons, half_life) in VARIANTS.items():
            probs[variant] = fit_variant(history, test, seasons, half_life, args.validation)
        for variant, p in probs.items():
            rows.append(pd.DataFrame({
                "fold": fold, "row": np.arange(len(test)), "variant": variant,
                "date": test["date"].dt.date.to_numpy(), "home_team": test["home_team"].to_numpy(),
                "away_team": test["away_team"].to_numpy(), "actual": actual,
                "log_loss": per_match_log_loss(actual, p),
                "prob_H": p[:, 0], "prob_D": p[:, 1], "prob_A": p[:, 2],
            }))
        ll = {variant: per_match_log_loss(actual, p).mean() for variant, p in probs.items()}
        print(f"{fold} ({len(test):,} kampe, {time.perf_counter() - started:.0f} sek.): "
              + "  ".join(f"{name}={value:.5f}" for name, value in ll.items()), flush=True)

    per_match = pd.concat(rows, ignore_index=True)
    per_match["source"] = per_match["fold"]
    walk_forward = per_match[per_match["fold"] != "this_week"].copy()
    walk_forward["fold"] = "walk_forward_total"
    table = summarise(pd.concat([per_match, walk_forward], ignore_index=True))
    table.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")
    with summary_path.open("x", encoding="utf-8-sig", newline="") as handle:
        table.to_csv(handle, index=False)
    with per_match_path.open("x", encoding="utf-8-sig", newline="") as handle:
        per_match.to_csv(handle, index=False)

    columns = ["matches", "accuracy", "log_loss", "delta_log_loss_vs_all", "delta_log_loss_vs_all_se",
               "delta_log_loss_vs_odds_only", "delta_log_loss_vs_odds_only_se"]
    for fold in ("walk_forward_total", "this_week"):
        print(f"\n=== {fold} ===")
        print(table[table["fold"] == fold].set_index("variant")[columns].to_string(float_format=lambda v: f"{v:.5f}"))
    print("\nNegativ delta = bedre. Pr. sæson: se summary-filen.")
    print(f"Summary: {summary_path}\nPr. kamp: {per_match_path}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
