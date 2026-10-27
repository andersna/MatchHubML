"""Collect per-match expected goals (xG) from Understat for the five leagues.

Understat is free and needs no API key. Its league pages load data from
``/getLeagueData/<league>/<season>`` (season = start year, 2014 onwards).
Raw responses are cached in data/raw/understat/; finished seasons are never
re-downloaded, the current season is refreshed when older than a day.

Team names are mapped to the fixtures.csv vocabulary automatically by
matching league, date (+-1 day) and final score, so no manual alias list is
needed. Understat's own "forecast" field is derived from the match's own xG
and is deliberately not stored (it would leak the result).

Usage:
    python src/collect_understat_xg.py
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "understat"
FIXTURES_PATH = ROOT / "data" / "fixtures.csv"
OUTPUT_PATH = ROOT / "data" / "understat_matches.csv"

BASE_URL = "https://understat.com/getLeagueData"
LEAGUES = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}
FIRST_SEASON = 2014
REQUEST_PAUSE_SECONDS = 1.5
CURRENT_SEASON_MAX_AGE_SECONDS = 24 * 3600
MIN_MAPPING_SHARE = 0.8


def current_season_start(today: datetime | None = None) -> int:
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


def _fetch_season(code: str, season: int, *, force: bool) -> dict:
    path = RAW_DIR / f"{code}_{season}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    response = requests.get(
        f"{BASE_URL}/{code}/{season}",
        headers={"X-Requested-With": "XMLHttpRequest"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "dates" not in payload:
        raise ValueError(f"Uventet Understat-svar for {code} {season}.")
    # Only match data is kept; the player tables are large and unused.
    payload = {"dates": payload["dates"]}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    time.sleep(REQUEST_PAUSE_SECONDS)
    return payload


def collect_raw(seasons: range | None = None) -> pd.DataFrame:
    """Return one row per Understat match with Understat team names."""
    current = current_season_start()
    seasons = seasons or range(FIRST_SEASON, current + 1)
    rows = []
    for league, code in LEAGUES.items():
        for season in seasons:
            path = RAW_DIR / f"{code}_{season}.json"
            stale = (
                season == current and path.is_file()
                and time.time() - path.stat().st_mtime > CURRENT_SEASON_MAX_AGE_SECONDS
            )
            payload = _fetch_season(code, season, force=stale)
            for match in payload.get("dates") or []:
                rows.append({
                    "league": league,
                    "season": season,
                    "understat_id": match["id"],
                    "kickoff_utc": match["datetime"],
                    "understat_home": match["h"]["title"],
                    "understat_away": match["a"]["title"],
                    "is_result": bool(match.get("isResult")),
                    "home_goals": pd.to_numeric(match["goals"]["h"], errors="coerce"),
                    "away_goals": pd.to_numeric(match["goals"]["a"], errors="coerce"),
                    "home_xg": pd.to_numeric(match["xG"]["h"], errors="coerce"),
                    "away_xg": pd.to_numeric(match["xG"]["a"], errors="coerce"),
                })
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["kickoff_utc"]).dt.normalize()
    return frame


def build_team_mapping(understat: pd.DataFrame, fixtures: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map (league, Understat name) -> fixtures.csv name by matching date and score."""
    played = understat[understat["is_result"]]
    history = fixtures.dropna(subset=["home_goals", "away_goals"]).copy()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    keys = ["league", "date", "home_goals", "away_goals"]
    votes: dict[tuple[str, str], Counter] = {}
    for shift in (0, -1, 1):  # kickoff is UTC; allow a day either side
        shifted = played.assign(date=played["date"] + pd.Timedelta(days=shift))
        merged = shifted.merge(history[keys + ["home_team", "away_team"]], on=keys, how="inner")
        for row in merged.itertuples(index=False):
            votes.setdefault((row.league, row.understat_home), Counter())[row.home_team] += 1
            votes.setdefault((row.league, row.understat_away), Counter())[row.away_team] += 1

    mapping, unmapped = {}, []
    appearances = Counter(
        [(r.league, r.understat_home) for r in played.itertuples()]
        + [(r.league, r.understat_away) for r in played.itertuples()]
    )
    for key, total in appearances.items():
        counter = votes.get(key)
        if not counter:
            unmapped.append(key)
            continue
        name, count = counter.most_common(1)[0]
        # Same-date same-score coincidences give a few wrong votes; the
        # correct name must dominate and explain most of the team's matches.
        if count / sum(counter.values()) >= 0.5 and count / total >= MIN_MAPPING_SHARE:
            mapping[key] = name
        else:
            unmapped.append(key)
    if unmapped:
        print(f"Advarsel: {len(unmapped)} Understat-hold kunne ikke mappes: {sorted(unmapped)[:10]}")
    return mapping


def main() -> None:
    understat = collect_raw()
    fixtures = pd.read_csv(FIXTURES_PATH, low_memory=False)
    mapping = build_team_mapping(understat, fixtures)
    understat["home_team"] = [mapping.get((l, n)) for l, n in zip(understat["league"], understat["understat_home"])]
    understat["away_team"] = [mapping.get((l, n)) for l, n in zip(understat["league"], understat["understat_away"])]
    mapped = understat.dropna(subset=["home_team", "away_team"])
    columns = [
        "date", "league", "season", "home_team", "away_team", "is_result",
        "home_goals", "away_goals", "home_xg", "away_xg",
        "understat_home", "understat_away", "understat_id", "kickoff_utc",
    ]
    mapped[columns].sort_values(["date", "league", "home_team"]).to_csv(OUTPUT_PATH, index=False)
    print(
        f"Understat: {len(understat):,} kampe hentet; {len(mapped):,} med mappede holdnavne "
        f"({mapped['is_result'].sum():,} spillede). Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
