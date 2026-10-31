"""Pre-match player-absence features from API-Football injuries + Understat.

Each listed absence (data/api_football_injuries.csv, collect_api_football_injuries.py)
is weighted by the player's importance in the PREVIOUS season, from Understat's
season player tables (data/raw/understat/players_<league>_<season>.json), so
nothing from the match's own season leaks in:

  regular - min(previous-season minutes / 2700, 1): 1 = an every-week starter
  attack  - previous-season npxG + xA: attacking output lost

Per team and match: *_miss_regulars and *_miss_attack sum "Missing Fixture"
absences, *_doubt_regulars sums "Questionable" ones. Players without an
Understat match or previous season count 0. Rows from league seasons without
injury data are NaN (median-imputed by the model); covered matches with no
listed absence are 0.

Team names are mapped to the fixtures.csv vocabulary by date overlap
(API-Football) and via data/understat_matches.csv (Understat).
"""

from __future__ import annotations

import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from collect_understat_xg import LEAGUES as UNDERSTAT_CODES
from collect_understat_xg import RAW_DIR as UNDERSTAT_RAW_DIR


ROOT = Path(__file__).resolve().parent.parent
INJURIES_PATH = ROOT / "data" / "api_football_injuries.csv"
FIXTURES_PATH = ROOT / "data" / "fixtures.csv"
UNDERSTAT_MATCHES_PATH = ROOT / "data" / "understat_matches.csv"
FULL_SEASON_MINUTES = 2700
MIN_TEAM_DATE_OVERLAP = 0.8
SIDES = ("home", "away")
ABSENCE_FEATURES = (
    [f"{side}_{name}" for side in SIDES for name in ("miss_regulars", "miss_attack", "doubt_regulars")]
    + ["diff_miss_regulars", "diff_miss_attack"]
)


def _normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(text.lower().replace("-", " ").replace("'", "").split())


def _name_matches(api_name: str, understat_name: str) -> bool:
    """'M. Salah' ~ 'Mohamed Salah'; 'V. van Dijk' ~ 'Virgil van Dijk'; 'Rodri' ~ 'Rodri'."""
    api, full = _normalise(api_name), _normalise(understat_name)
    first, _, rest = api.partition(" ")
    if first.endswith(".") and rest:
        return full.startswith(first[0]) and (full == rest or full.endswith(" " + rest))
    return full == api or full.endswith(" " + api) or full.startswith(api + " ")


def _season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def _map_api_teams(injuries: pd.DataFrame, fixtures: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(league, API team) -> fixtures team, by share of the API team's dates it also played on."""
    played = pd.concat([
        fixtures[["league", "date", "home_team"]].rename(columns={"home_team": "team"}),
        fixtures[["league", "date", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    dates_by_team = played.groupby(["league", "team"])["date"].agg(set)
    mapping = {}
    for (league, team), dates in injuries.groupby(["league", "team"])["date"].agg(set).items():
        shifted = dates | {d + pd.Timedelta(days=1) for d in dates} | {d - pd.Timedelta(days=1) for d in dates}
        candidates = dates_by_team.loc[league]
        scores = candidates.map(lambda own: len(own & shifted) / len(dates))
        if scores.max() >= MIN_TEAM_DATE_OVERLAP:
            mapping[(league, team)] = scores.idxmax()
    return mapping


def _understat_players() -> pd.DataFrame:
    """One row per player and season, with the team in fixtures.csv vocabulary."""
    teams = pd.read_csv(UNDERSTAT_MATCHES_PATH)
    team_names = dict(zip(zip(teams["league"], teams["understat_home"]), teams["home_team"]))
    team_names.update(zip(zip(teams["league"], teams["understat_away"]), teams["away_team"]))
    rows = []
    for league, code in UNDERSTAT_CODES.items():
        for path in sorted(UNDERSTAT_RAW_DIR.glob(f"players_{code}_*.json")):
            season = int(path.stem.rsplit("_", 1)[1])
            for player in json.loads(path.read_text(encoding="utf-8")):
                # A player transferred mid-season has "Team A,Team B"; keep each team.
                for team in player["team_title"].split(","):
                    rows.append({
                        "league": league, "season": season, "player_id": int(player["id"]),
                        "player_name": player["player_name"],
                        "team": team_names.get((league, team), team),
                        "minutes": float(player["time"]),
                        "attack": float(player["npxG"]) + float(player["xA"]),
                    })
    return pd.DataFrame(rows)


def _importance(injuries: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Add `regular` and `attack` from each absentee's previous Understat season."""
    by_team_season = {key: group for key, group in players.groupby(["team", "season"])}
    by_season = {season: group for season, group in players.groupby("season")}
    # A player can appear twice in a season after a transfer; sum both stints.
    previous = players.groupby(["player_id", "season"])[["minutes", "attack"]].sum()
    cache: dict[tuple[str, str, int], int | None] = {}

    def player_id(name: str, team: str, season: int) -> int | None:
        key = (name, team, season)
        if key not in cache:
            found = None
            # Prefer this season's squad; else anyone from last season (newly signed).
            for pool in (by_team_season.get((team, season)), by_season.get(season - 1)):
                if pool is None:
                    continue
                hits = pool[pool["player_name"].map(lambda full: _name_matches(name, full))]
                if not hits.empty:
                    found = int(hits.sort_values("minutes").iloc[-1]["player_id"])
                    break
            cache[key] = found
        return cache[key]

    ids = [player_id(row.player, row.fixture_team, row.season) for row in injuries.itertuples()]
    result = injuries.assign(understat_id=ids)
    lookup = [previous.loc[(pid, season - 1)] if pid is not None and (pid, season - 1) in previous.index else None
              for pid, season in zip(ids, result["season"])]
    result["regular"] = [min(s["minutes"] / FULL_SEASON_MINUTES, 1.0) if s is not None else 0.0 for s in lookup]
    result["attack"] = [s["attack"] if s is not None else 0.0 for s in lookup]
    return result


def build_absence_features(verbose: bool = True) -> pd.DataFrame:
    """One row per (league, date, team) with that team's absence sums."""
    injuries = pd.read_csv(INJURIES_PATH)
    injuries["date"] = pd.to_datetime(injuries["kickoff_utc"]).dt.tz_localize(None).dt.normalize()
    fixtures = pd.read_csv(FIXTURES_PATH, usecols=["date", "league", "home_team", "away_team"], low_memory=False)
    fixtures["date"] = pd.to_datetime(fixtures["date"]).dt.normalize()

    teams = _map_api_teams(injuries, fixtures)
    injuries["fixture_team"] = [teams.get((league, team)) for league, team in zip(injuries["league"], injuries["team"])]
    unmapped = injuries["fixture_team"].isna().mean()
    injuries = injuries.dropna(subset=["fixture_team"])

    # Snap each absence to the team's own fixture date (UTC vs local date may differ by a day).
    played = pd.concat([
        fixtures[["league", "date", "home_team"]].rename(columns={"home_team": "fixture_team"}),
        fixtures[["league", "date", "away_team"]].rename(columns={"away_team": "fixture_team"}),
    ]).rename(columns={"date": "match_date"}).sort_values("match_date")
    injuries = pd.merge_asof(
        injuries.sort_values("date"), played, left_on="date", right_on="match_date",
        by=["league", "fixture_team"], direction="nearest", tolerance=pd.Timedelta(days=1),
    ).dropna(subset=["match_date"])

    injuries = _importance(injuries, _understat_players())
    if verbose:
        print(f"Fravær: {len(injuries):,}; hold ikke mappet {unmapped:.1%}; "
              f"spillere fundet i Understat {injuries['understat_id'].notna().mean():.1%}; "
              f"med forrige sæson {(injuries['regular'] > 0).mean():.1%}")

    missing = injuries["type"] == "Missing Fixture"
    sums = injuries.assign(
        miss_regulars=np.where(missing, injuries["regular"], 0.0),
        miss_attack=np.where(missing, injuries["attack"], 0.0),
        doubt_regulars=np.where(~missing, injuries["regular"], 0.0),
    ).groupby(["league", "match_date", "fixture_team"])[["miss_regulars", "miss_attack", "doubt_regulars"]].sum()
    return sums.reset_index().rename(columns={"match_date": "date", "fixture_team": "team"})


def add_absence_features(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add ABSENCE_FEATURES to a frame with league, date, season, home_team, away_team."""
    absences = build_absence_features(verbose)
    covered_seasons = {
        (league, _season_label(season))
        for league, season in pd.read_csv(INJURIES_PATH, usecols=["league", "season"]).drop_duplicates().itertuples(index=False)
    }
    result = data.copy()
    for side in SIDES:
        side_frame = absences.rename(columns={"team": f"{side}_team", **{
            name: f"{side}_{name}" for name in ("miss_regulars", "miss_attack", "doubt_regulars")}})
        result = result.merge(side_frame, on=["league", "date", f"{side}_team"], how="left", validate="many_to_one")
    covered = np.array([(league, season) in covered_seasons for league, season in zip(result["league"], result["season"])])
    side_columns = [c for c in ABSENCE_FEATURES if not c.startswith("diff_")]
    result.loc[covered, side_columns] = result.loc[covered, side_columns].fillna(0.0)
    result.loc[~covered, side_columns] = np.nan
    result["diff_miss_regulars"] = result["home_miss_regulars"] - result["away_miss_regulars"]
    result["diff_miss_attack"] = result["home_miss_attack"] - result["away_miss_attack"]
    return result
