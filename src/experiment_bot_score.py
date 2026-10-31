"""Experiment: where should the game bot's exact score get its goal level from?

The bot picks V3's most likely outcome and, as its score, the most likely
scoreline with that outcome under Poisson goal rates (game.goal_rates):

  v3_only        - both rates fitted to V3's 1X2 probabilities (current)
  league_avg     - total goals = the league's average; V3 sets the split
  expected_goals - total = home_expected_goals + away_expected_goals
  avg_goals      - total from each team's all-history goals for/against
  last5_goals    - total from goals for/against in the last 5 matches
  xg_last10      - total from Understat xG for/against in the last 10 matches

Every feature total is calibrated (actual = a + b * feature) on 2022-23 and
scored on 2023-24 .. 2025-26 plus this week, using the out-of-sample V3
probabilities from experiment_nn_v3_training_window.py (variant `all`).

Metrics: exact-score hits and game points (1 for the 1X2, +2 for the exact
score) as the game scores them, the mean log-probability of the actual score
under each method's rates (less noisy than hits), and total-goals error.
"""

from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch  # noqa: F401
import numpy as np
import pandas as pd

from game import POINTS_EXACT_SCORE, POINTS_PER_HIT, _poisson_grid, bot_pick, bot_score, goal_rates
from retrain_v3 import build_dataset


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PREDICTIONS_PATH = DATA / "nn_v3_training_window_per_match.csv"
SUMMARY_PATH = DATA / "bot_score_summary.csv"
CALIBRATION_FOLD = "2022-23"
TOTAL_RANGE = (1.0, 5.0)
KEYS = ["date", "home_team", "away_team"]


def pairwise_total(frame: pd.DataFrame, home_for: str, home_against: str, away_for: str, away_against: str) -> pd.Series:
    """Expected goals in the match: each side's scoring averaged with the opponent's conceding."""
    return 0.5 * (frame[home_for] + frame[away_against]) + 0.5 * (frame[away_for] + frame[home_against])


FEATURE_TOTALS = {
    "expected_goals": lambda f: f["home_expected_goals"] + f["away_expected_goals"],
    "avg_goals": lambda f: pairwise_total(f, "home_avg_goals_for", "home_avg_goals_against",
                                          "away_avg_goals_for", "away_avg_goals_against"),
    "last5_goals": lambda f: pairwise_total(f, "home_last5_goals_for", "home_last5_goals_against",
                                            "away_last5_goals_for", "away_last5_goals_against"),
    "xg_last10": lambda f: pairwise_total(f, "home_xg_for_last10", "home_xg_against_last10",
                                          "away_xg_for_last10", "away_xg_against_last10"),
}


def load() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    predictions = predictions[predictions["variant"] == "all"][["fold", *KEYS, "actual", "prob_H", "prob_D", "prob_A"]]
    features = build_dataset()
    fixtures = pd.read_csv(DATA / "fixtures.csv", usecols=[*KEYS, "league", "home_goals", "away_goals"], low_memory=False)
    fixtures["date"] = pd.to_datetime(fixtures["date"]).dt.normalize()
    frame = (predictions
             .merge(features.drop(columns=["result"]), on=KEYS, how="inner", validate="one_to_one")
             .merge(fixtures.drop(columns="league"), on=KEYS, how="inner", validate="one_to_one"))
    frame["total"] = frame["home_goals"] + frame["away_goals"]
    return frame.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)


def totals_by_method(frame: pd.DataFrame) -> dict[str, np.ndarray | None]:
    """Calibrated expected total goals per match for each method (None = V3 only)."""
    calibration = frame["fold"] == CALIBRATION_FOLD
    methods: dict[str, np.ndarray | None] = {"v3_only": None}
    league_mean = frame[calibration].groupby("league")["total"].mean()
    methods["league_avg"] = frame["league"].map(league_mean).to_numpy()
    for name, raw_total in FEATURE_TOTALS.items():
        raw = raw_total(frame)
        usable = calibration & raw.notna()
        slope, intercept = np.polyfit(raw[usable], frame.loc[usable, "total"], 1)
        fitted = (intercept + slope * raw).fillna(frame["league"].map(league_mean))
        methods[name] = np.clip(fitted.to_numpy(), *TOTAL_RANGE)
        print(f"  {name}: kalibrering total = {intercept:.2f} + {slope:.2f} × feature")
    return methods


def score_method(frame: pd.DataFrame, totals: np.ndarray | None) -> pd.DataFrame:
    rows = []
    for i, row in enumerate(frame.itertuples(index=False)):
        probs = (row.prob_H, row.prob_D, row.prob_A)
        total = None if totals is None else round(float(totals[i]), 3)
        home_rate, away_rate = goal_rates(probs, total)
        grid = _poisson_grid(home_rate, away_rate)
        home_goals, away_goals = int(row.home_goals), int(row.away_goals)
        actual_prob = grid[home_goals, away_goals] if max(home_goals, away_goals) < grid.shape[0] else 0.0
        predicted = bot_score(probs, total)
        hit = bot_pick(probs) == row.actual
        exact = predicted == (home_goals, away_goals)
        rows.append({
            "hit": hit, "exact": exact,
            "points": POINTS_PER_HIT * hit + POINTS_EXACT_SCORE * exact,
            "score_log_prob": np.log(max(actual_prob, 1e-12)),
            "total_error": abs(home_rate + away_rate - row.total),
            "predicted": f"{predicted[0]}-{predicted[1]}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    if SUMMARY_PATH.exists():
        raise FileExistsError(f"Output findes allerede; intet overskrives: {SUMMARY_PATH}")
    started = time.perf_counter()
    frame = load()
    print(f"Kampe med V3-sandsynligheder, features og resultat: {len(frame):,}")
    methods = totals_by_method(frame)
    test = (frame["fold"] != CALIBRATION_FOLD).to_numpy()
    print(f"Testkampe (efter kalibreringssæsonen {CALIBRATION_FOLD}): {test.sum():,}")

    scored = {name: score_method(frame[test].reset_index(drop=True), None if totals is None else totals[test])
              for name, totals in methods.items()}
    base = scored["v3_only"]
    summary = []
    for name, result in scored.items():
        exact_diff = result["exact"].astype(float) - base["exact"].astype(float)
        log_diff = result["score_log_prob"] - base["score_log_prob"]
        summary.append({
            "method": name, "matches": len(result),
            "hit_rate": result["hit"].mean(), "exact_rate": result["exact"].mean(),
            "points_per_match": result["points"].mean(),
            "exact_vs_v3_only": exact_diff.mean(), "exact_vs_v3_only_se": exact_diff.std(ddof=1) / np.sqrt(len(result)),
            "score_log_prob": result["score_log_prob"].mean(),
            "log_prob_vs_v3_only": log_diff.mean(), "log_prob_vs_v3_only_se": log_diff.std(ddof=1) / np.sqrt(len(result)),
            "total_goals_mae": result["total_error"].mean(),
            "top_scores": ", ".join(f"{s} {n / len(result):.0%}" for s, n in result["predicted"].value_counts().head(5).items()),
        })
    table = pd.DataFrame(summary)
    with SUMMARY_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        table.to_csv(handle, index=False)

    actual_scores = frame.loc[test, "home_goals"].astype(int).astype(str) + "-" + frame.loc[test, "away_goals"].astype(int).astype(str)
    print("\nFaktiske resultater (top 5): " + ", ".join(f"{s} {n / test.sum():.0%}" for s, n in actual_scores.value_counts().head(5).items()))
    columns = ["exact_rate", "exact_vs_v3_only", "exact_vs_v3_only_se", "points_per_match",
               "score_log_prob", "log_prob_vs_v3_only", "log_prob_vs_v3_only_se", "total_goals_mae"]
    print("\n" + table.set_index("method")[columns].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nForudsagte resultater (top 5):")
    for row in table.itertuples():
        print(f"  {row.method:15s} {row.top_scores}")
    print(f"\nSummary: {SUMMARY_PATH}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
