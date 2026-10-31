"""Experiment: do pre-match player absences (injuries, suspensions) add an edge?

Absence features come from player_absence_features.py (API-Football injury
lists weighted by Understat previous-season importance). The free API plan
only covers 2022-23 .. 2024-25, so the folds are 2023-24 and 2024-25, each
trained on all earlier seasons (absence features NaN and median-imputed
before 2022-23). Early stopping as in experiment_nn_v3_training_window.py.

Variants, each with the pre-match and the closing market (closing odds also
in training, as experiment_nn_v3_closing_odds.py):
  v3        - V3_FEATURES (baseline)
  v3_abs    - V3_FEATURES + ABSENCE_FEATURES
  abs_only  - ABSENCE_FEATURES only: a low-capacity correction to the market

A diagnostic first shows, by home-minus-away missing regulars, the actual
home-win rate against both markets on all covered matches: if the market
already prices absences, the rates agree in every bin.
API-Football's historical lists may include information confirmed only at
lineup time; closing odds (which include lineups) are the conservative test.
Historical odds have no snapshot timestamps; backtest only.
"""

from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch  # noqa: F401
import numpy as np
import pandas as pd

from experiment_nn_market_residual import per_match_log_loss, train_residual
from experiment_nn_odds_asof_day_v1 import ODDS_FEATURES, calculate_metrics
from experiment_nn_v3_closing_odds import with_closing
from experiment_nn_v3_training_window import SEEDS, folds, split_validation
from player_absence_features import ABSENCE_FEATURES, add_absence_features
from retrain_v3 import build_dataset
from train_final_nn_market_residual_v3 import V3_FEATURES


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUMMARY_PATH = DATA / "nn_v3_absences_summary.csv"
PER_MATCH_PATH = DATA / "nn_v3_absences_per_match.csv"
TEST_FOLDS = ("2023-24", "2024-25")
VARIANTS = {
    "v3": list(V3_FEATURES),
    "v3_abs": list(V3_FEATURES) + ABSENCE_FEATURES,
    "abs_only": list(ABSENCE_FEATURES),
}
MARKETS = ("pre", "close")
DIAGNOSTIC_BINS = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]


def diagnostic(pre: pd.DataFrame, close: pd.DataFrame) -> None:
    covered = pre["diff_miss_regulars"].notna() & pre["has_closing"]
    frame = pd.DataFrame({
        "bin": pd.cut(pre.loc[covered, "diff_miss_regulars"], DIAGNOSTIC_BINS),
        "home_win": pre.loc[covered, "result"] == "H",
        "market_pre": pre.loc[covered, "odds_prob_home"],
        "market_close": close.loc[covered, "odds_prob_home"],
    })
    table = frame.groupby("bin", observed=True).agg(
        matches=("home_win", "size"), home_win=("home_win", "mean"),
        market_pre=("market_pre", "mean"), market_close=("market_close", "mean"))
    print("\nHjemmesejr efter (hjemme − ude) manglende faste spillere, 2022-23 .. 2024-25:")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    for path in (SUMMARY_PATH, PER_MATCH_PATH):
        if path.exists():
            raise FileExistsError(f"Output findes allerede; intet overskrives: {path}")
    started = time.perf_counter()
    data = build_dataset()
    data["season_start"] = data["season"].str[:4].astype(int)
    data = add_absence_features(data)
    pre, close = with_closing(data)
    print(f"Datasæt: {len(pre):,} kampe; med fraværsdata: {pre['diff_miss_regulars'].notna().sum():,}")
    diagnostic(pre, close)

    rows = []
    for (fold, history_pre, test_pre), (_, history_close, _) in zip(folds(pre), folds(close)):
        if fold not in TEST_FOLDS:
            continue
        keep = test_pre["has_closing"].to_numpy()
        tests = {"pre": test_pre[keep], "close": close.loc[test_pre.index[keep]]}
        histories = {"pre": history_pre, "close": history_close}
        actual = tests["pre"]["result"].to_numpy()
        probs = {f"market_{market}": tests[market][ODDS_FEATURES].to_numpy(np.float64) for market in MARKETS}
        for market in MARKETS:
            subtrain, validation = split_validation(histories[market], "fraction")
            for variant, columns in VARIANTS.items():
                probs[f"{market}_{variant}"] = np.mean([
                    train_residual(subtrain, validation, tests[market], columns, False, False, 0.0, 0, seed)[0]
                    for seed in SEEDS
                ], axis=0)
        test = tests["pre"]
        for variant, p in probs.items():
            rows.append(pd.DataFrame({
                "fold": fold, "row": np.arange(len(test)), "variant": variant,
                "date": test["date"].dt.date.to_numpy(), "home_team": test["home_team"].to_numpy(),
                "away_team": test["away_team"].to_numpy(), "actual": actual,
                "log_loss": per_match_log_loss(actual, p),
                "prob_H": p[:, 0], "prob_D": p[:, 1], "prob_A": p[:, 2],
            }))
        ll = {variant: per_match_log_loss(actual, p).mean() for variant, p in probs.items()}
        print(f"\n{fold} ({len(test):,} kampe, {time.perf_counter() - started:.0f} sek.): "
              + "  ".join(f"{name}={value:.5f}" for name, value in ll.items()), flush=True)

    per_match = pd.concat(rows, ignore_index=True)
    per_match["source"] = per_match["fold"]
    total = per_match.copy()
    total["fold"] = "walk_forward_total"
    scored = pd.concat([per_match, total], ignore_index=True)
    log_loss = scored.set_index(["fold", "source", "row", "variant"])["log_loss"].unstack("variant")
    summary = []
    for (fold, variant), group in scored.groupby(["fold", "variant"], sort=False):
        market = variant.split("_")[0] if not variant.startswith("market") else variant.split("_")[1]
        row = {"fold": fold, "variant": variant, "matches": len(group),
               **calculate_metrics(group["actual"].to_numpy(), group[["prob_H", "prob_D", "prob_A"]].to_numpy())}
        for name, reference in (("v3", f"{market}_v3"), ("market", f"market_{market}")):
            diff = log_loss.loc[fold, variant] - log_loss.loc[fold, reference]
            row[f"delta_log_loss_vs_{name}"] = diff.mean()
            row[f"delta_log_loss_vs_{name}_se"] = diff.std(ddof=1) / np.sqrt(len(diff))
        summary.append(row)
    table = pd.DataFrame(summary)
    table.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")
    with SUMMARY_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        table.to_csv(handle, index=False)
    with PER_MATCH_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        per_match.to_csv(handle, index=False)

    columns = ["matches", "log_loss", "delta_log_loss_vs_v3", "delta_log_loss_vs_v3_se",
               "delta_log_loss_vs_market", "delta_log_loss_vs_market_se"]
    print("\n=== 2023-24 + 2024-25 (deltas mod v3 og markedet med samme odds) ===")
    print(table[table["fold"] == "walk_forward_total"].set_index("variant")[columns]
          .to_string(float_format=lambda v: f"{v:.5f}"))
    print("\nNegativ delta = bedre.")
    print(f"Summary: {SUMMARY_PATH}\nPr. kamp: {PER_MATCH_PATH}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
