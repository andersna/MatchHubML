"""Experiment: would betting on the v3 model's out-of-sample edges make money?

Uses the walk-forward predictions written by experiment_nn_v3_training_window.py
(variant `all`, i.e. production v3: each season predicted by a model trained
only on earlier seasons) and joins the raw decimal odds from the
football-data.co.uk workbooks:

  B365   - Bet365 pre-match prices, the odds the model is trained on and the
           price a bet is placed at
  Max    - best price across bookmakers at the same time (line shopping;
           optimistic, as the best price may not have been available to you)
  B365C  - Bet365 closing odds
  PSC    - Pinnacle closing odds (missing for the current season)

Rule: for each match, take the outcome with the largest edge
p_model * odds - 1; bet 1 unit if that edge exceeds the threshold.

Reported per threshold and price: bets, hit rate, profit, ROI with a 95%
bootstrap interval, and closing-line value (CLV): the expected return of the
bet under Pinnacle's margin-free closing probabilities, which is far less
noisy than ROI and is the standard test of a real edge.
Historical odds have no snapshot timestamps; backtest only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retrain_v3 import RAW_DIR, SHEETS


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEFAULT_PREDICTIONS = DATA / "nn_v3_training_window_per_match.csv"
VARIANT = "all"
THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10)
OUTCOMES = ("H", "D", "A")
PRICES = ("B365", "Max")
BOOTSTRAP_SAMPLES = 2000
RNG_SEED = 42
JOIN_KEYS = ["date", "home_team", "away_team"]


def load_raw_odds() -> pd.DataFrame:
    """Decimal 1X2 odds (opening, best, closing) from every recent workbook."""
    paths = sorted((RAW_DIR / "historical").glob("all-euro-data-*")) + sorted(RAW_DIR.glob("all-euro-data-*.xlsx"))
    books = ("B365", "Max", "Avg", "B365C", "AvgC", "PSC")
    columns = {f"{book}{side}": f"{book}_{side}" for book in books for side in OUTCOMES}
    frames = []
    for path in paths:
        excel = pd.ExcelFile(path)
        for sheet in SHEETS:
            if sheet not in excel.sheet_names:
                continue
            raw = pd.read_excel(excel, sheet_name=sheet)
            if "B365CH" not in raw.columns:  # closing odds start in 2019-20
                continue
            raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
            frame = pd.DataFrame({
                "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.normalize(),
                "home_team": raw["HomeTeam"], "away_team": raw["AwayTeam"],
            })
            for source, target in columns.items():
                frame[target] = pd.to_numeric(raw[source], errors="coerce") if source in raw.columns else np.nan
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates(JOIN_KEYS, keep="last")


def load_bets(predictions_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    predictions = predictions[predictions["variant"] == VARIANT]
    merged = predictions.merge(load_raw_odds(), on=JOIN_KEYS, how="left", validate="one_to_one")
    missing = merged["B365_H"].isna().sum()
    if missing:
        print(f"Advarsel: {missing} kampe uden rå odds udelades")
    return merged.dropna(subset=[f"B365_{side}" for side in OUTCOMES]).reset_index(drop=True)


def choose_bets(frame: pd.DataFrame, price: str, threshold: float) -> pd.DataFrame:
    probs = frame[[f"prob_{side}" for side in OUTCOMES]].to_numpy()
    odds = frame[[f"{price}_{side}" for side in OUTCOMES]].to_numpy()
    edge = probs * odds - 1.0
    pick = np.nanargmax(edge, axis=1)
    rows = np.arange(len(frame))
    chosen = edge[rows, pick] > threshold
    picked = np.array(OUTCOMES)[pick]

    closing = frame[[f"PSC_{side}" for side in OUTCOMES]].to_numpy()
    fair_close = (1.0 / closing) / (1.0 / closing).sum(axis=1, keepdims=True)
    b365_close = frame[[f"B365C_{side}" for side in OUTCOMES]].to_numpy()
    bets = pd.DataFrame({
        "fold": frame["fold"], "pick": picked, "odds": odds[rows, pick], "edge": edge[rows, pick],
        "won": picked == frame["actual"].to_numpy(),
        "clv_pinnacle": fair_close[rows, pick] * odds[rows, pick] - 1.0,
        "beat_b365_close": odds[rows, pick] > b365_close[rows, pick],
    })[chosen]
    bets["profit"] = np.where(bets["won"], bets["odds"] - 1.0, -1.0)
    return bets


def roi_interval(profit: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(profit) < 2:
        return np.nan, np.nan
    samples = rng.choice(profit, size=(BOOTSTRAP_SAMPLES, len(profit))).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def summarise(bets: pd.DataFrame, rng: np.random.Generator) -> dict:
    low, high = roi_interval(bets["profit"].to_numpy(), rng)
    return {
        "bets": len(bets),
        "hit_rate": bets["won"].mean(),
        "avg_odds": bets["odds"].mean(),
        "profit": bets["profit"].sum(),
        "roi": bets["profit"].mean(),
        "roi_ci_low": low, "roi_ci_high": high,
        "clv_pinnacle": bets["clv_pinnacle"].mean(),
        "clv_coverage": bets["clv_pinnacle"].notna().mean(),
        "beat_b365_close": bets["beat_b365_close"].mean(),
        "share_H": (bets["pick"] == "H").mean(),
        "share_D": (bets["pick"] == "D").mean(),
        "share_A": (bets["pick"] == "A").mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest af væddemål på v3's out-of-sample-sandsynligheder.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args()
    output = DATA / f"betting_edge_{args.predictions.stem}.csv"
    if output.exists():
        raise FileExistsError(f"Output findes allerede; intet overskrives: {output}")

    frame = load_bets(args.predictions)
    walk_forward = frame[frame["fold"] != "this_week"]
    margin = (1.0 / walk_forward[[f"B365_{side}" for side in OUTCOMES]]).sum(axis=1).mean() - 1.0
    print(f"Kampe: {len(frame):,} ({len(walk_forward):,} walk-forward); gns. Bet365-margin {margin:.2%}")

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for price in PRICES:
        for threshold in THRESHOLDS:
            bets = choose_bets(frame, price, threshold)
            for scope, subset in (("walk_forward", bets[bets["fold"] != "this_week"]),
                                  ("this_week", bets[bets["fold"] == "this_week"]),
                                  *((f"season_{f}", bets[bets["fold"] == f]) for f in sorted(walk_forward["fold"].unique()))):
                rows.append({"price": price, "threshold": threshold, "scope": scope, **summarise(subset, rng)})
    table = pd.DataFrame(rows)
    table.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        table.to_csv(handle, index=False)

    columns = ["bets", "hit_rate", "avg_odds", "profit", "roi", "roi_ci_low", "roi_ci_high",
               "clv_pinnacle", "beat_b365_close"]
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    for price in PRICES:
        print(f"\n=== Walk-forward, pris: {price} ===")
        view = table[(table["price"] == price) & (table["scope"] == "walk_forward")]
        print(view.set_index("threshold")[columns].to_string(float_format=fmt))
    print("\n=== Denne uge, pris: B365 ===")
    view = table[(table["price"] == "B365") & (table["scope"] == "this_week")]
    print(view.set_index("threshold")[["bets", "hit_rate", "profit", "roi"]].to_string(float_format=fmt))
    print(f"\nROI = gevinst pr. indsat enhed. CLV > 0 = bedre pris end Pinnacles fair lukkeodds.\nOutput: {output}")


if __name__ == "__main__":
    main()
