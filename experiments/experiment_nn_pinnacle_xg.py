"""Experiment: Pinnacle instead of Bet365, and Understat xG form, in the v3 setup.

All models are the v3 market-residual network (5-seed ensembles, no class
weights). The comparison is season-by-season walk-forward on matches where
Bet365, Pinnacle and xG history all exist: train on 2014-15 up to the season
before the test season (last 15% of dates for early stopping), test on each
season 2016-17 .. 2025-26. Every variant is scored on identical matches.

Markets (from the football-data.co.uk workbooks, inverse odds normalised):
  b365     - Bet365 (as v1-v3)
  pinnacle - Pinnacle (PSH/PSD/PSA), a sharper bookmaker
  average  - market average (BbAv* to 2018-19, Avg* afterwards)
  pinnacle_closing - closing line; NOT available before kickoff, reference only

xG features (Understat): mean xG for/against over each team's previous 5 and
10 league matches, strictly before the match date, plus balance differences.
Historical odds have no snapshot timestamps; this is a backtest only.
"""

from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch  # noqa: F401
import numpy as np
import pandas as pd

from experiment_nn_market_residual import per_match_log_loss, train_residual
from experiment_nn_odds_asof_day_v1 import (
    LABEL_TO_INDEX,
    NN_FEATURES,
    ODDS_FEATURES,
    ODDS_MATCH_KEYS,
    calculate_metrics,
    normalize_probabilities,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATASET_PATH = DATA / "odds_model_dataset_asof_day_v2.csv"
UNDERSTAT_PATH = DATA / "understat_matches.csv"
HISTORICAL_DIR = DATA / "raw" / "historical"
OOF_PATH = DATA / "oof_predictions_nn_pinnacle_xg.csv"
SUMMARY_PATH = DATA / "nn_pinnacle_xg_summary.csv"

SHEETS = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga", "I1": "Serie A", "F1": "Ligue 1"}
FIRST_TRAIN_SEASON = 2014
TEST_SEASONS = range(2016, 2026)
SEEDS = (42, 123, 2026, 3407, 9876)
MARKETS = {
    "b365": ("B365H", "B365D", "B365A"),
    "pinnacle": ("PSH", "PSD", "PSA"),
    "average": ("AvgH", "AvgD", "AvgA"),
    "pinnacle_closing": ("PSCH", "PSCD", "PSCA"),
}
XG_FEATURES = [
    f"{side}_xg_{kind}_last{n}"
    for side in ("home", "away") for n in (5, 10) for kind in ("for", "against")
] + ["xg_balance_diff_last5", "xg_balance_diff_last10"]
# name -> (market, correction inputs)
VARIANTS = {
    "res_b365": ("b365", NN_FEATURES),
    "res_pinnacle": ("pinnacle", NN_FEATURES),
    "res_b365_xg": ("b365", NN_FEATURES + XG_FEATURES),
    "res_pinnacle_xg": ("pinnacle", NN_FEATURES + XG_FEATURES),
}


def market_columns(market: str) -> list[str]:
    return [f"{market}_prob_{outcome}" for outcome in ("home", "draw", "away")]


def load_workbook_odds() -> pd.DataFrame:
    """Odds from every workbook season that has Pinnacle prices."""
    frames = []
    for path in sorted(HISTORICAL_DIR.glob("all-euro-data-*")):
        season = int(path.name.split("-")[3])
        if season < FIRST_TRAIN_SEASON:
            continue
        excel = pd.ExcelFile(path)
        for sheet, league in SHEETS.items():
            raw = pd.read_excel(excel, sheet_name=sheet)
            if "AvgH" not in raw and "BbAvH" in raw:
                raw = raw.rename(columns={"BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA"})
            needed = [column for prices in MARKETS.values() for column in prices]
            if any(column not in raw for column in needed):
                continue
            part = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
            part = pd.DataFrame({
                "date": pd.to_datetime(part["Date"], dayfirst=True).dt.normalize(),
                "league": league,
                "home_team": part["HomeTeam"].astype(str).str.strip(),
                "away_team": part["AwayTeam"].astype(str).str.strip(),
                "season": season,
                **{column: pd.to_numeric(part[column], errors="coerce") for column in needed},
            })
            frames.append(part)
    odds = pd.concat(frames, ignore_index=True)
    for market, prices in MARKETS.items():
        decimal = odds[list(prices)].to_numpy(dtype=np.float64)
        valid = np.isfinite(decimal).all(axis=1) & (decimal > 1).all(axis=1)
        implied = np.where(valid[:, None], 1.0 / np.where(valid[:, None], decimal, 2.0), np.nan)
        probs = implied / implied.sum(axis=1, keepdims=True)
        for index, column in enumerate(market_columns(market)):
            odds[column] = probs[:, index]
    columns = ODDS_MATCH_KEYS + ["season"] + [c for m in MARKETS for c in market_columns(m)]
    odds = odds[columns]
    if odds.duplicated(ODDS_MATCH_KEYS).any():
        raise ValueError("Dublerede kampnøgler i workbook-odds.")
    return odds


def xg_features(rows: pd.DataFrame) -> pd.DataFrame:
    """As-of xG form for each row's teams, using only matches before its date."""
    understat = pd.read_csv(UNDERSTAT_PATH, parse_dates=["date"])
    understat = understat[understat["is_result"]].copy()

    # Use the fixtures.csv date for each Understat match (UTC dates can differ
    # by a day), so a match can never count as history for itself.
    fixtures = pd.read_csv(DATA / "fixtures.csv", low_memory=False, parse_dates=["date"])
    aligned = understat.merge(
        fixtures[["league", "home_team", "away_team", "date"]].rename(columns={"date": "fixture_date"}),
        on=["league", "home_team", "away_team"], how="left",
    )
    aligned = aligned[(aligned["fixture_date"] - aligned["date"]).abs() <= pd.Timedelta(days=3)]
    understat = understat.merge(
        aligned[["understat_id", "fixture_date"]].drop_duplicates("understat_id"),
        on="understat_id", how="left",
    )
    understat["date"] = understat["fixture_date"].fillna(understat["date"])

    long = pd.concat([
        understat.assign(team=understat["home_team"], xg_for=understat["home_xg"], xg_against=understat["away_xg"]),
        understat.assign(team=understat["away_team"], xg_for=understat["away_xg"], xg_against=understat["home_xg"]),
    ], ignore_index=True)[["league", "team", "date", "xg_for", "xg_against"]].sort_values("date", kind="stable")
    grouped = long.groupby(["league", "team"], sort=False)
    for n in (5, 10):
        for kind in ("for", "against"):
            # Rolling over a team's matches up to and including this one; the
            # strict-date as-of join below then picks the latest earlier match.
            long[f"xg_{kind}_last{n}"] = grouped[f"xg_{kind}"].transform(
                lambda values, n=n: values.rolling(n, min_periods=3).mean()
            )
    state_columns = [f"xg_{kind}_last{n}" for n in (5, 10) for kind in ("for", "against")]

    result = rows[ODDS_MATCH_KEYS].copy()
    result["row"] = np.arange(len(rows))
    for side in ("home", "away"):
        left = result[["row", "date", "league", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        merged = pd.merge_asof(
            left.sort_values("date"), long[["league", "team", "date", *state_columns]].sort_values("date"),
            on="date", by=["league", "team"], allow_exact_matches=False,
        ).set_index("row").reindex(result["row"])
        for column in state_columns:
            result[f"{side}_{column}"] = merged[column].to_numpy()
    for n in (5, 10):
        result[f"xg_balance_diff_last{n}"] = (
            result[f"home_xg_for_last{n}"] - result[f"home_xg_against_last{n}"]
            - result[f"away_xg_for_last{n}"] + result[f"away_xg_against_last{n}"]
        )
    return result[XG_FEATURES]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH, parse_dates=["date"], low_memory=False)
    data["date"] = data["date"].dt.normalize()
    odds = load_workbook_odds()
    merged = data.merge(odds, on=ODDS_MATCH_KEYS, how="inner", validate="one_to_one")

    # Sanity check: the workbook's Bet365 must equal the dataset's Bet365.
    difference = np.abs(merged[market_columns("b365")].to_numpy() - merged[ODDS_FEATURES].to_numpy())
    agree = (difference.max(axis=1) < 1e-6).mean()
    print(f"Bet365 fra workbooks = datasættets Bet365 i {agree:.2%} af kampene")
    if agree < 0.99:
        raise RuntimeError("Workbook-odds matcher ikke datasættet; join er forkert.")

    merged[XG_FEATURES] = xg_features(merged).to_numpy()
    complete = merged.dropna(subset=[c for m in MARKETS for c in market_columns(m)])
    print(
        f"Kampe fra {FIRST_TRAIN_SEASON}-{FIRST_TRAIN_SEASON + 1}: {len(merged):,}; med alle fire markeder: "
        f"{len(complete):,}; xG-dækning (last5): {complete['home_xg_for_last5'].notna().mean():.1%}"
    )
    return complete.sort_values(["date", "time"], kind="stable").reset_index(drop=True)


def main() -> None:
    for path in (OOF_PATH, SUMMARY_PATH):
        if path.exists():
            raise FileExistsError(f"Output findes allerede; intet overskrives: {path}")
    started = time.perf_counter()
    data = load_data()

    predictions = []
    for season in TEST_SEASONS:
        history = data[data["season"] < season]
        dates = sorted(history["date"].dt.date.unique())
        split = int(len(dates) * 0.85)
        subtrain = history[history["date"].dt.date.isin(dates[:split])]
        validation = history[history["date"].dt.date.isin(dates[split:])]
        test = data[data["season"] == season]
        print(f"\nSæson {season}-{season + 1}: subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}")
        keys = test[ODDS_MATCH_KEYS + ["result"]].reset_index(drop=True)

        for market in MARKETS:
            probs = normalize_probabilities(test[market_columns(market)].to_numpy())
            predictions.append(keys.assign(variant=f"odds_{market}", season=season,
                                           prob_H=probs[:, 0], prob_D=probs[:, 1], prob_A=probs[:, 2]))
        for variant, (market, features) in VARIANTS.items():
            columns = features + market_columns(market)
            seed_probs = []
            for seed in SEEDS:
                probs, _, _ = train_residual(
                    subtrain, validation, test, columns, use_embedding=False, use_recency=False,
                    label_smoothing=0.0, fold=season, seed=seed, market_columns=market_columns(market),
                )
                seed_probs.append(probs)
            probs = np.mean(seed_probs, axis=0)
            metrics = calculate_metrics(test["result"].to_numpy(), probs)
            print(f"  {variant:<16} acc={metrics['accuracy']:.4f} ll={metrics['log_loss']:.5f}", flush=True)
            predictions.append(keys.assign(variant=variant, season=season,
                                           prob_H=probs[:, 0], prob_D=probs[:, 1], prob_A=probs[:, 2]))

    oof = pd.concat(predictions, ignore_index=True)
    baseline = oof[oof["variant"] == "odds_b365"].reset_index(drop=True)
    baseline_ll = per_match_log_loss(baseline["result"].to_numpy(), baseline[["prob_H", "prob_D", "prob_A"]].to_numpy())
    baseline_hit = baseline[["prob_H", "prob_D", "prob_A"]].to_numpy().argmax(1) == baseline["result"].map(LABEL_TO_INDEX).to_numpy()
    rows = []
    for variant, group in oof.groupby("variant", sort=False):
        group = group.reset_index(drop=True)
        if not group[ODDS_MATCH_KEYS].equals(baseline[ODDS_MATCH_KEYS]):
            raise RuntimeError(f"Testkampe afviger for {variant}.")
        probs = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        actual = group["result"].to_numpy()
        diff = per_match_log_loss(actual, probs) - baseline_ll
        hit = (probs.argmax(1) == group["result"].map(LABEL_TO_INDEX).to_numpy()).astype(float) - baseline_hit
        rows.append({
            "variant": variant, "matches": len(group), **calculate_metrics(actual, probs),
            "delta_log_loss_vs_b365": diff.mean(),
            "delta_log_loss_vs_b365_se": diff.std(ddof=1) / np.sqrt(len(diff)),
            "delta_accuracy_vs_b365": hit.mean(),
            "delta_accuracy_vs_b365_se": hit.std(ddof=1) / np.sqrt(len(hit)),
        })
    summary = pd.DataFrame(rows)
    summary.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; closing odds are a reference only")
    with OOF_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        oof.to_csv(handle, index=False)
    with SUMMARY_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        summary.to_csv(handle, index=False)

    print(f"\n=== Testsæsoner {TEST_SEASONS.start}-{TEST_SEASONS.stop} samlet, identiske kampe ===")
    print(summary.drop(columns="odds_timestamp_caveat").set_index("variant")[[
        "matches", "accuracy", "log_loss", "brier_score", "delta_accuracy_vs_b365",
        "delta_accuracy_vs_b365_se", "delta_log_loss_vs_b365", "delta_log_loss_vs_b365_se",
    ]].to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\nOOF: {OOF_PATH}\nSummary: {SUMMARY_PATH}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
