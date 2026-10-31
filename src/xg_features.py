"""As-of Understat xG form features, shared by training and live prediction.

For each team: mean xG for/against over its previous 5 and 10 league matches,
strictly before the match date (at least 3 matches, otherwise NaN), plus the
home-minus-away xG balance. Same definition as
experiments/experiment_nn_pinnacle_xg.py. NaN is left for the model's saved
median imputer: Understat starts in 2014-15, and promoted teams have no
top-flight xG history for their first matches.

data/understat_matches.csv is written by src/collect_understat_xg.py, with
team names already mapped to the fixtures.csv vocabulary.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
UNDERSTAT_PATH = ROOT / "data" / "understat_matches.csv"
FIXTURES_PATH = ROOT / "data" / "fixtures.csv"

XG_WINDOWS = (5, 10)
XG_MIN_MATCHES = 3
XG_STATE_COLUMNS = [f"xg_{kind}_last{n}" for n in XG_WINDOWS for kind in ("for", "against")]
XG_FEATURES = [
    f"{side}_xg_{kind}_last{n}"
    for side in ("home", "away") for n in XG_WINDOWS for kind in ("for", "against")
] + [f"xg_balance_diff_last{n}" for n in XG_WINDOWS]
MATCH_KEYS = ["date", "league", "home_team", "away_team"]


class XgHistory:
    """Per-team rolling xG states, looked up strictly before a date."""

    def __init__(self, understat: pd.DataFrame, fixtures: pd.DataFrame | None = None):
        played = understat[understat["is_result"].astype(str).str.lower() == "true"].copy()
        played["date"] = pd.to_datetime(played["date"]).dt.normalize()
        if fixtures is not None and len(played):
            # Use the fixtures.csv date (Understat's UTC date can differ by a
            # day), so a match can never count as history for itself.
            known = fixtures[["league", "home_team", "away_team", "date"]].rename(
                columns={"date": "fixture_date"}
            )
            known["fixture_date"] = pd.to_datetime(known["fixture_date"]).dt.normalize()
            aligned = played[["understat_id", "league", "home_team", "away_team", "date"]].merge(
                known, on=["league", "home_team", "away_team"], how="inner"
            )
            aligned = aligned[(aligned["fixture_date"] - aligned["date"]).abs() <= pd.Timedelta(days=3)]
            fixture_dates = aligned.drop_duplicates("understat_id").set_index("understat_id")["fixture_date"]
            played["date"] = played["understat_id"].map(fixture_dates).fillna(played["date"])

        long = pd.concat([
            played.assign(team=played["home_team"], xg_for=played["home_xg"], xg_against=played["away_xg"]),
            played.assign(team=played["away_team"], xg_for=played["away_xg"], xg_against=played["home_xg"]),
        ], ignore_index=True)[["league", "team", "date", "xg_for", "xg_against"]]
        long = long.sort_values(["league", "team", "date"], kind="stable")
        grouped = long.groupby(["league", "team"], sort=False)
        for n in XG_WINDOWS:
            for kind in ("for", "against"):
                # Includes the match itself; lookups take the latest earlier match.
                long[f"xg_{kind}_last{n}"] = grouped[f"xg_{kind}"].transform(
                    lambda values, n=n: values.rolling(n, min_periods=XG_MIN_MATCHES).mean()
                )

        self._teams: dict[tuple[str, str], tuple[list, np.ndarray]] = {
            key: (group["date"].tolist(), group[XG_STATE_COLUMNS].to_numpy(np.float64))
            for key, group in long.groupby(["league", "team"], sort=False)
        }
        self.last_date = long["date"].max() if len(long) else None

    def team_state(self, league: str, team: str, date) -> np.ndarray:
        """XG_STATE_COLUMNS for the team's latest match strictly before ``date``."""
        entry = self._teams.get((league, team))
        if entry is None:
            return np.full(len(XG_STATE_COLUMNS), np.nan)
        dates, states = entry
        index = bisect_left(dates, pd.Timestamp(date).normalize()) - 1
        return states[index] if index >= 0 else np.full(len(XG_STATE_COLUMNS), np.nan)

    def match_features(self, league: str, home_team: str, away_team: str, date) -> dict[str, float]:
        home = dict(zip(XG_STATE_COLUMNS, self.team_state(league, home_team, date)))
        away = dict(zip(XG_STATE_COLUMNS, self.team_state(league, away_team, date)))
        values = {f"home_{name}": float(value) for name, value in home.items()}
        values.update((f"away_{name}", float(value)) for name, value in away.items())
        for n in XG_WINDOWS:
            values[f"xg_balance_diff_last{n}"] = (
                home[f"xg_for_last{n}"] - home[f"xg_against_last{n}"]
                - away[f"xg_for_last{n}"] + away[f"xg_against_last{n}"]
            )
        return {name: float(values[name]) for name in XG_FEATURES}

    def features_for(self, rows: pd.DataFrame) -> pd.DataFrame:
        """XG_FEATURES for every row of a frame with MATCH_KEYS columns."""
        values = [
            self.match_features(league, home, away, date)
            for date, league, home, away in zip(
                pd.to_datetime(rows["date"]), rows["league"], rows["home_team"], rows["away_team"]
            )
        ]
        return pd.DataFrame(values, columns=XG_FEATURES, index=rows.index, dtype=np.float64)


_cache: dict[tuple, XgHistory] = {}


def load_xg_history(
    understat_path: Path = UNDERSTAT_PATH, fixtures_path: Path = FIXTURES_PATH
) -> XgHistory:
    """XgHistory for the current files, rebuilt only when either file changes.

    A missing Understat file gives an empty history (all NaN features), so
    predictions keep working and the imputer fills the medians.
    """
    understat_path, fixtures_path = Path(understat_path), Path(fixtures_path)
    if not understat_path.is_file():
        return XgHistory(pd.DataFrame(columns=[
            "is_result", "date", "league", "home_team", "away_team", "home_xg", "away_xg", "understat_id",
        ]))
    key = tuple(
        (str(path), path.stat().st_mtime_ns if path.is_file() else None)
        for path in (understat_path, fixtures_path)
    )
    if key not in _cache:
        fixtures = pd.read_csv(fixtures_path, low_memory=False) if fixtures_path.is_file() else None
        _cache.clear()
        _cache[key] = XgHistory(pd.read_csv(understat_path, low_memory=False), fixtures)
    return _cache[key]
