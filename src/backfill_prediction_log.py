"""Reconstruct pre-kickoff predictions for finished cached matches.

Odds, in order of preference:
1. a Bet365 quote from 5DollarFootball observed BEFORE kickoff (the lookup-time
   price, else the opening price - see predict_upcoming_match.ODDS_MARKETS);
2. Bet365 pre-match odds from the football-data.co.uk season workbook
   (data/raw/all-euro-data-*.xlsx, saved by the history update) - the same
   kind of odds the models were trained on.

Sports features are computed as of the match date (history strictly before
it), and a model never predicts a match on or before its last training date,
so a reconstruction uses nothing a live prediction would not have had. Rows
are logged with source='backfill', and a match that already has a prediction
for a model (live or reconstructed) is never predicted again for that model.

Usage:
    python src/backfill_prediction_log.py
"""

from __future__ import annotations

import contextlib
from contextlib import closing
import io
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: F401,E402  (PyTorch before NumPy/Pandas on this setup)
import pandas as pd  # noqa: E402

from src.predict_match import predict_match  # noqa: E402
from src.predict_upcoming_match import ODDS_MARKETS, ODDS_PREFERENCE_SQL, odds_probabilities  # noqa: E402
from src.prediction_log import LOG_PATH, log_predictions, logged_matches, match_identity  # noqa: E402
from src.train_final_nn_odds_asof_day_v1 import ODDS_FEATURES  # noqa: E402
from src.upcoming_match_features import build_upcoming_match_features_partial, history_team_name  # noqa: E402
from src.xg_features import load_xg_history  # noqa: E402

RAW_DIR = _PROJECT_ROOT / "data" / "raw"
OUTCOMES = ("H", "D", "A")
WORKBOOK_ODDS_TAG = "football-data.co.uk:B365"
SHEETS = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga", "I1": "Serie A", "F1": "Ligue 1"}


def finished_matches(database_path: Path) -> list[dict]:
    """Finished cached matches with their preferred pre-kickoff Bet365 quote, if any."""
    uri = "file:" + quote(Path(database_path).resolve().as_posix(), safe="/:\\") + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT m.league, m.match_date, m.kickoff_at, m.home_team, m.away_team,
                   o.odds_home, o.odds_draw, o.odds_away, o.observed_at_utc, o.market
            FROM matches AS m
            LEFT JOIN odds_observations AS o ON o.odds_observation_id = (
                SELECT o2.odds_observation_id FROM odds_observations AS o2
                WHERE o2.match_id = m.match_id
                  AND lower(replace(o2.bookmaker, ' ', '')) = 'bet365'
                  AND o2.market IN {ODDS_MARKETS}
                  AND o2.observed_at_utc < m.kickoff_at
                ORDER BY {ODDS_PREFERENCE_SQL.format(alias="o2")}
                LIMIT 1
            )
            WHERE m.data_source = 'football-data.org'
              AND upper(m.match_status) = 'FINISHED' AND m.result IN ('H', 'D', 'A')
            ORDER BY m.kickoff_at
            """
        ).fetchall()
    return [dict(row) for row in rows]


def workbook_odds(raw_dir: Path = RAW_DIR) -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    """(league, date, home, away) -> Bet365 decimal odds from the saved season workbooks."""
    odds = {}
    for path in sorted(raw_dir.glob("all-euro-data-*.xlsx")):
        excel = pd.ExcelFile(path)
        for sheet, league in SHEETS.items():
            if sheet not in excel.sheet_names:
                continue
            frame = pd.read_excel(excel, sheet_name=sheet)
            if not {"Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"} <= set(frame.columns):
                continue
            dates = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
            for date, row in zip(dates, frame.itertuples(index=False)):
                prices = tuple(float(p) for p in pd.to_numeric([row.B365H, row.B365D, row.B365A], errors="coerce"))
                if pd.isna(date) or any(pd.isna(price) or price <= 1 for price in prices):
                    continue
                odds[(league, date.strftime("%Y-%m-%d"), str(row.HomeTeam).strip(), str(row.AwayTeam).strip())] = prices
    return odds


def backfill(bundles: dict, database_path: Path, *, log_path: Path = LOG_PATH, use_workbook_odds: bool = True) -> dict:
    """Log reconstructed predictions; returns counts for the dashboard."""
    matches = finished_matches(database_path)
    already = logged_matches(path=log_path)
    book = workbook_odds() if use_workbook_odds else {}
    records, skipped = [], []
    counts = {"finished": len(matches), "from_5dollar": 0, "from_workbook": 0, "no_odds": 0}

    candidates = []
    for match in matches:
        key = (match["league"], match["match_date"], match["home_team"], match["away_team"])
        # Only models without any prediction for this match yet (under any earlier date),
        # and never one trained on it.
        versions = [
            version for version, bundle in bundles.items()
            if (version, *match_identity(*key)) not in already
            and match["match_date"] > bundle.metadata["training_data"]["end_date"]
        ]
        if not versions:
            continue
        if match["observed_at_utc"]:
            odds = odds_probabilities(match["odds_home"], match["odds_draw"], match["odds_away"])
            source, market = match["observed_at_utc"], match["market"]
            counts["from_5dollar"] += 1
        else:
            prices = book.get((
                match["league"], match["match_date"],
                history_team_name(match["league"], match["home_team"]),
                history_team_name(match["league"], match["away_team"]),
            ))
            if prices is None:
                counts["no_odds"] += 1
                continue
            odds = odds_probabilities(*prices)
            source = market = WORKBOOK_ODDS_TAG
            counts["from_workbook"] += 1
        candidates.append((match, versions, odds, source, market))

    for match_date in sorted({match["match_date"] for match, *_ in candidates}):
        group = [item for item in candidates if item[0]["match_date"] == match_date]
        requests = [{key: match[key] for key in ("league", "home_team", "away_team", "match_date")} for match, *_ in group]
        with contextlib.redirect_stdout(io.StringIO()):
            features, errors = build_upcoming_match_features_partial(requests, as_of_date=match_date)
        skipped += [(requests[error["request_id"]], error["reason"]) for error in errors]
        for position, (match, versions, odds, source, market) in enumerate(group):
            if position not in features.index:
                continue
            model_input = {**features.loc[position].to_dict(), **dict(zip(ODDS_FEATURES, odds))}
            # xG strictly before the match date, so the match's own xG is never used.
            model_input.update(load_xg_history().match_features(
                match["league"], history_team_name(match["league"], match["home_team"]),
                history_team_name(match["league"], match["away_team"]), match_date,
            ))
            for version in versions:
                result = predict_match(model_input, bundle=bundles[version])
                records.append({
                    "source": "backfill", "model_version": version,
                    "model_artifact": result["model_version"],
                    "logged_at_utc": match["observed_at_utc"] or match["kickoff_at"],
                    "league": match["league"], "match_date": match["match_date"],
                    "kickoff_at": match["kickoff_at"], "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "odds_observed_at_utc": source, "odds_market": market, "odds": odds,
                    "probs": [result["probabilities"][label]["decimal"] for label in OUTCOMES],
                    "expected_goals": (model_input["home_expected_goals"], model_input["away_expected_goals"]),
                })
    counts["logged"] = log_predictions(records, path=log_path)
    counts["matches_added"] = len({(r["league"], r["match_date"], r["home_team"], r["away_team"]) for r in records})
    counts["skipped"] = [f"{r['home_team']} - {r['away_team']}: {reason}" for r, reason in skipped]
    return counts


def main() -> None:
    from src.api import DATABASE_PATH, available_versions, get_bundle

    counts = backfill({version: get_bundle(version) for version in available_versions()}, DATABASE_PATH)
    print(
        f"Afsluttede kampe: {counts['finished']}; nye logrækker: {counts['logged']} "
        f"for {counts['matches_added']} kampe (odds: {counts['from_5dollar']} fra 5DollarFootball, "
        f"{counts['from_workbook']} fra football-data.co.uk; {counts['no_odds']} uden odds)."
    )
    for line in counts["skipped"]:
        print(f"  sprunget over: {line}")


if __name__ == "__main__":
    main()
