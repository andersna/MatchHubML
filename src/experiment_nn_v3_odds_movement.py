"""Experiment: does the odds movement from pre-match to closing add information?

Every variant uses closing odds as the market (production predicts with
Bet365's price at lookup time, near kickoff) and the closing-odds training
of experiment_nn_v3_closing_odds.py. Movement features are log ratios of
margin-free probabilities, closing / pre-match, per outcome:

  close             - V3_FEATURES only (baseline: close_on_close)
  move_b365         - + Bet365 movement (B365C vs B365); both prices exist live
                      as 5DollarFootball's "closing" and "opening" stages
  move_b365_avg     - + market-average movement (AvgC vs Avg); research only,
                      production has no average odds feed

Movement exists from 2019-20 (closing odds); earlier rows are NaN and
median-imputed (~no movement). Walk-forward folds and early stopping as in
experiment_nn_v3_training_window.py. Historical odds have no snapshot
timestamps, and football-data's pre-match collection time need not match
5DollarFootball's opening stage; backtest only.
"""

from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch  # noqa: F401
import numpy as np
import pandas as pd

from experiment_betting_edge import JOIN_KEYS, load_raw_odds
from experiment_nn_market_residual import per_match_log_loss, train_residual
from experiment_nn_odds_asof_day_v1 import ODDS_FEATURES, calculate_metrics
from experiment_nn_v3_closing_odds import with_closing
from experiment_nn_v3_training_window import SEEDS, folds, split_validation
from retrain_v3 import build_dataset
from train_final_nn_market_residual_v3 import V3_FEATURES


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUMMARY_PATH = DATA / "nn_v3_odds_movement_summary.csv"
PER_MATCH_PATH = DATA / "nn_v3_odds_movement_per_match.csv"
SIDES = ("H", "D", "A")
MOVE_B365 = [f"move_b365_{side}" for side in SIDES]
MOVE_AVG = [f"move_avg_{side}" for side in SIDES]
VARIANTS = {
    "close": list(V3_FEATURES),
    "move_b365": list(V3_FEATURES) + MOVE_B365,
    "move_b365_avg": list(V3_FEATURES) + MOVE_B365 + MOVE_AVG,
}
BASELINE = "close"


def _fair(decimal: np.ndarray) -> np.ndarray:
    implied = 1.0 / decimal
    return implied / implied.sum(axis=1, keepdims=True)


def add_movement(closing: pd.DataFrame) -> pd.DataFrame:
    """Add MOVE_B365 and MOVE_AVG; NaN where either price is missing."""
    odds = load_raw_odds()
    merged = closing.merge(odds[JOIN_KEYS + [f"{book}_{side}" for book in ("B365", "B365C", "Avg", "AvgC")
                                             for side in SIDES]],
                           on=JOIN_KEYS, how="left", validate="one_to_one")
    for names, opening, closing_book in ((MOVE_B365, "B365", "B365C"), (MOVE_AVG, "Avg", "AvgC")):
        before = merged[[f"{opening}_{side}" for side in SIDES]].to_numpy(np.float64)
        after = merged[[f"{closing_book}_{side}" for side in SIDES]].to_numpy(np.float64)
        valid = (np.isfinite(before) & np.isfinite(after) & (before > 1.0) & (after > 1.0)).all(axis=1)
        movement = np.full(before.shape, np.nan)
        movement[valid] = np.log(_fair(after[valid]) / _fair(before[valid]))
        merged[names] = movement
    return merged.drop(columns=[c for c in merged.columns if c not in closing.columns and c not in MOVE_B365 + MOVE_AVG])


def main() -> None:
    for path in (SUMMARY_PATH, PER_MATCH_PATH):
        if path.exists():
            raise FileExistsError(f"Output findes allerede; intet overskrives: {path}")
    started = time.perf_counter()
    data = build_dataset()
    data["season_start"] = data["season"].str[:4].astype(int)
    data, closing = with_closing(data)
    closing = add_movement(closing)
    recent = closing[closing["season_start"] >= 2019]
    print(f"Datasæt: {len(closing):,} kampe; bevægelse fra 2019-20: Bet365 {recent[MOVE_B365[0]].notna().mean():.1%}, "
          f"gennemsnit {recent[MOVE_AVG[0]].notna().mean():.1%}; "
          f"gns. |bevægelse| hjemme {recent[MOVE_B365[0]].abs().mean():.4f}")

    rows = []
    for fold, history, test in folds(closing):
        test = test[test["has_closing"]]
        if test.empty:
            continue
        actual = test["result"].to_numpy()
        subtrain, validation = split_validation(history, "fraction")
        probs = {"market_close": test[ODDS_FEATURES].to_numpy(np.float64)}
        for variant, columns in VARIANTS.items():
            probs[variant] = np.mean([
                train_residual(subtrain, validation, test, columns, False, False, 0.0, 0, seed)[0]
                for seed in SEEDS
            ], axis=0)
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
    scored = pd.concat([per_match, walk_forward], ignore_index=True)
    log_loss = scored.set_index(["fold", "source", "row", "variant"])["log_loss"].unstack("variant")
    summary = []
    for (fold, variant), group in scored.groupby(["fold", "variant"], sort=False):
        row = {"fold": fold, "variant": variant, "matches": len(group),
               **calculate_metrics(group["actual"].to_numpy(), group[["prob_H", "prob_D", "prob_A"]].to_numpy())}
        for reference in (BASELINE, "market_close"):
            diff = log_loss.loc[fold, variant] - log_loss.loc[fold, reference]
            row[f"delta_log_loss_vs_{reference}"] = diff.mean()
            row[f"delta_log_loss_vs_{reference}_se"] = diff.std(ddof=1) / np.sqrt(len(diff))
        summary.append(row)
    table = pd.DataFrame(summary)
    table.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")
    with SUMMARY_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        table.to_csv(handle, index=False)
    with PER_MATCH_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        per_match.to_csv(handle, index=False)

    columns = ["matches", "log_loss", f"delta_log_loss_vs_{BASELINE}", f"delta_log_loss_vs_{BASELINE}_se",
               "delta_log_loss_vs_market_close", "delta_log_loss_vs_market_close_se"]
    for fold in ("walk_forward_total", "this_week"):
        print(f"\n=== {fold} ===")
        print(table[table["fold"] == fold].set_index("variant")[columns].to_string(float_format=lambda v: f"{v:.5f}"))
    print("\nNegativ delta = bedre.")
    print(f"Summary: {SUMMARY_PATH}\nPr. kamp: {PER_MATCH_PATH}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
