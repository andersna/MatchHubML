"""Experiment: Understat xG form in the production v3 setup.

Walk-forward by season: for each test season 2016-17 .. 2025-26, train on all
earlier seasons (last 15% of dates for early stopping) and test on the season.
The data is exactly what src/retrain_v3.py trains on (retrain_v3.build_dataset).

Variants (v3 market-residual network, 5-seed ensembles, no class weights):
  current      - the 30 v1/v2 inputs (v3 before xG)
  xg_all       - V3_FEATURES (30 + 10 xG) on all history; xG is NaN before
                 2014-15 and is median-imputed, as in production
  xg_from_2014 - V3_FEATURES, training only on 2014-15 onwards (xG era)

All variants are scored on identical matches, paired against `current`.
Historical odds have no snapshot timestamps; this is a backtest only.
"""

from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch  # noqa: F401
import numpy as np
import pandas as pd

from experiment_nn_market_residual import per_match_log_loss, train_residual
from experiment_nn_odds_asof_day_v1 import MODEL_FEATURES, ODDS_FEATURES, calculate_metrics
from retrain_v3 import build_dataset
from train_final_nn_market_residual_v3 import V3_FEATURES


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUMMARY_PATH = DATA / "nn_v3_xg_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
TEST_SEASONS = range(2016, 2026)
XG_FIRST_SEASON = 2014
# name -> (columns, first training season or None for all history)
VARIANTS = {
    "current": (MODEL_FEATURES, None),
    "xg_all": (V3_FEATURES, None),
    "xg_from_2014": (V3_FEATURES, XG_FIRST_SEASON),
}
BASELINE = "current"


def main() -> None:
    if SUMMARY_PATH.exists():
        raise FileExistsError(f"Output findes allerede; intet overskrives: {SUMMARY_PATH}")
    started = time.perf_counter()
    data = build_dataset()
    data["season_start"] = data["season"].str[:4].astype(int)
    print(f"Datasæt: {len(data):,} kampe; xG-dækning: {data['home_xg_for_last5'].notna().mean():.1%}")

    rows = []
    for season in TEST_SEASONS:
        test = data[data["season_start"] == season]
        if test.empty:
            continue
        actual = test["result"].to_numpy()
        season_probs = {"odds_only": test[ODDS_FEATURES].to_numpy(np.float64)}
        for variant, (columns, first_season) in VARIANTS.items():
            history = data[data["season_start"] < season]
            if first_season is not None:
                history = history[history["season_start"] >= first_season]
            dates = sorted(history["date"].dt.date.unique())
            split = int(len(dates) * 0.85)
            subtrain = history[history["date"].dt.date.isin(dates[:split])]
            validation = history[history["date"].dt.date.isin(dates[split:])]
            seed_probs = [
                train_residual(subtrain, validation, test, list(columns), False, False, 0.0, 0, seed)[0]
                for seed in SEEDS
            ]
            season_probs[variant] = np.mean(seed_probs, axis=0)
        for variant, probs in season_probs.items():
            rows.append(pd.DataFrame({
                "season": season, "variant": variant, "actual": actual,
                "log_loss": per_match_log_loss(actual, probs),
                "prob_H": probs[:, 0], "prob_D": probs[:, 1], "prob_A": probs[:, 2],
            }))
        ll = {variant: per_match_log_loss(actual, probs).mean() for variant, probs in season_probs.items()}
        print(f"{season}-{(season + 1) % 100:02d} ({len(test):,} kampe): "
              + "  ".join(f"{name}={value:.5f}" for name, value in ll.items()), flush=True)

    per_match = pd.concat(rows, ignore_index=True)
    baseline = per_match[per_match["variant"] == BASELINE].reset_index(drop=True)
    summary = []
    for variant, group in per_match.groupby("variant", sort=False):
        group = group.reset_index(drop=True)
        probs = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        diff = group["log_loss"] - baseline["log_loss"]
        summary.append({
            "variant": variant, "matches": len(group),
            **calculate_metrics(group["actual"].to_numpy(), probs),
            "delta_log_loss_vs_current": diff.mean(),
            "delta_log_loss_vs_current_se": diff.std(ddof=1) / np.sqrt(len(diff)),
        })
    table = pd.DataFrame(summary)
    table.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")
    with SUMMARY_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        table.to_csv(handle, index=False)

    columns = ["matches", "accuracy", "log_loss", "brier_score",
               "delta_log_loss_vs_current", "delta_log_loss_vs_current_se"]
    print("\n=== Walk-forward 2016-17 .. 2025-26, 5-seed ensembler ===")
    print(table.set_index("variant")[columns].to_string(float_format=lambda v: f"{v:.5f}"))
    print("Negativ delta = bedre end nuværende v3.")
    print(f"\nSummary: {SUMMARY_PATH}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
