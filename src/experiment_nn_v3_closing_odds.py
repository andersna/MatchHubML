"""Experiment: should v3 train on Bet365 closing odds instead of pre-match odds?

Training uses football-data's B365H/D/A, collected a day or two before
kickoff. Production predicts with Bet365's price at lookup time, usually close
to kickoff, so the model learns corrections to earlier odds than it receives.
Closing odds (B365CH/CD/CA) exist from 2019-20; earlier rows keep pre-match odds.

Scenarios (v3 market-residual network, 5-seed ensembles, all history):
  pre_on_pre     - train on pre-match, predict with pre-match (current backtests)
  pre_on_close   - train on pre-match, predict with closing (~ production today)
  close_on_close - train on closing where available, predict with closing
Market baselines: market_pre and market_close.

Walk-forward folds and early stopping as in experiment_nn_v3_training_window.py
(last 15% of dates). The question is close_on_close vs pre_on_close, paired on
identical matches. Historical odds have no snapshot timestamps; backtest only.
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
from experiment_nn_v3_training_window import SEEDS, folds, split_validation
from retrain_v3 import build_dataset
from train_final_nn_market_residual_v3 import V3_FEATURES


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUMMARY_PATH = DATA / "nn_v3_closing_odds_summary.csv"
PER_MATCH_PATH = DATA / "nn_v3_closing_odds_per_match.csv"
CLOSING_COLUMNS = ["B365C_H", "B365C_D", "B365C_A"]
BASELINE = "pre_on_close"


def with_closing(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (data, closing): closing has the odds columns replaced by B365C
    probabilities where they exist; `has_closing` marks those rows."""
    odds = load_raw_odds()[JOIN_KEYS + CLOSING_COLUMNS]
    merged = data.merge(odds, on=JOIN_KEYS, how="left", validate="one_to_one")
    decimal = merged[CLOSING_COLUMNS].to_numpy(np.float64)
    valid = np.isfinite(decimal).all(axis=1) & (decimal > 1.0).all(axis=1)
    data = merged.drop(columns=CLOSING_COLUMNS).assign(has_closing=valid)
    closing = data.copy()
    implied = 1.0 / decimal[valid]
    closing.loc[valid, ODDS_FEATURES] = implied / implied.sum(axis=1, keepdims=True)
    return data, closing


def fit(history: pd.DataFrame, tests: list[pd.DataFrame]) -> list[np.ndarray]:
    """5-seed ensemble trained on `history`; one probability array per test frame."""
    subtrain, validation = split_validation(history, "fraction")
    combined = pd.concat(tests, ignore_index=True)
    probs = np.mean([
        train_residual(subtrain, validation, combined, list(V3_FEATURES), False, False, 0.0, 0, seed)[0]
        for seed in SEEDS
    ], axis=0)
    bounds = np.cumsum([0] + [len(test) for test in tests])
    return [probs[start:end] for start, end in zip(bounds[:-1], bounds[1:])]


def main() -> None:
    for path in (SUMMARY_PATH, PER_MATCH_PATH):
        if path.exists():
            raise FileExistsError(f"Output findes allerede; intet overskrives: {path}")
    started = time.perf_counter()
    data = build_dataset()
    data["season_start"] = data["season"].str[:4].astype(int)
    data, closing = with_closing(data)
    coverage = data.groupby("season_start")["has_closing"].mean()
    print(f"Datasæt: {len(data):,} kampe; lukkeodds-dækning fra {coverage[coverage > 0.5].index.min()}-sæsonen: "
          f"{data.loc[data['season_start'] >= 2019, 'has_closing'].mean():.1%}")

    rows = []
    for (fold, history, test), (_, history_close, _) in zip(folds(data), folds(closing)):
        # Score only matches with closing odds, so every scenario sees the same matches.
        keep = test["has_closing"].to_numpy()
        test_pre = test[keep]
        test_close = closing.loc[test.index[keep]]
        if test_pre.empty:
            continue
        actual = test_pre["result"].to_numpy()
        probs = {
            "market_pre": test_pre[ODDS_FEATURES].to_numpy(np.float64),
            "market_close": test_close[ODDS_FEATURES].to_numpy(np.float64),
        }
        probs["pre_on_pre"], probs["pre_on_close"] = fit(history, [test_pre, test_close])
        (probs["close_on_close"],) = fit(history_close, [test_close])
        for variant, p in probs.items():
            rows.append(pd.DataFrame({
                "fold": fold, "row": np.arange(len(test_pre)), "variant": variant,
                "date": test_pre["date"].dt.date.to_numpy(), "home_team": test_pre["home_team"].to_numpy(),
                "away_team": test_pre["away_team"].to_numpy(), "actual": actual,
                "log_loss": per_match_log_loss(actual, p),
                "prob_H": p[:, 0], "prob_D": p[:, 1], "prob_A": p[:, 2],
            }))
        ll = {variant: per_match_log_loss(actual, p).mean() for variant, p in probs.items()}
        print(f"{fold} ({len(test_pre):,} kampe, {time.perf_counter() - started:.0f} sek.): "
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
