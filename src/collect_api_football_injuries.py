"""Collect pre-match player absences (injuries, suspensions) from API-Football.

``/injuries?league=<id>&season=<year>`` returns every listed absence of a
league season in one call: player, team, fixture and type ("Missing Fixture"
or "Questionable") with a reason ("Ankle Injury", "Red Card", ...). The key is
API_FOOTBALL_KEY in .env (never printed). Raw responses are cached in
data/raw/api_football/; finished seasons are never re-downloaded. Seasons the
plan cannot access are reported and skipped.

Usage:
    python src/collect_api_football_injuries.py
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pandas as pd
import requests

from collect_understat_xg import current_season_start
from test_api_football import _read_dotenv_value


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "api_football"
OUTPUT_PATH = ROOT / "data" / "api_football_injuries.csv"

API_URL = "https://v3.football.api-sports.io/injuries"
LEAGUES = {"Premier League": 39, "La Liga": 140, "Bundesliga": 78, "Serie A": 135, "Ligue 1": 61}
FIRST_SEASON = 2019
REQUEST_PAUSE_SECONDS = 7  # free plan: 10 requests per minute
REQUEST_TIMEOUT_SECONDS = 60


def _fetch(key: str, league_id: int, season: int, *, force: bool) -> dict:
    path = RAW_DIR / f"injuries_{league_id}_{season}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    response = requests.get(
        API_URL, params={"league": league_id, "season": season},
        headers={"x-apisports-key": key}, timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    time.sleep(REQUEST_PAUSE_SECONDS)
    if payload.get("errors"):
        return payload  # plan or parameter error: report, do not cache
    if payload.get("paging", {}).get("total", 1) > 1:
        raise ValueError(f"Uventet paginering for liga {league_id} {season}.")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    key = _read_dotenv_value("API_FOOTBALL_KEY")
    if not key:
        raise SystemExit("API_FOOTBALL_KEY mangler i .env.")
    current = current_season_start()
    rows = []
    for league, league_id in LEAGUES.items():
        for season in range(FIRST_SEASON, current + 1):
            payload = _fetch(key, league_id, season, force=season == current)
            if payload.get("errors"):
                print(f"{league} {season}: sprunget over ({payload['errors']})")
                continue
            for item in payload["response"]:
                rows.append({
                    "league": league, "season": season,
                    "fixture_id": item["fixture"]["id"],
                    "kickoff_utc": item["fixture"]["date"],
                    "team_id": item["team"]["id"], "team": item["team"]["name"],
                    "player_id": item["player"]["id"], "player": item["player"]["name"],
                    "type": item["player"]["type"], "reason": item["player"]["reason"],
                })
            print(f"{league} {season}: {len(payload['response']):,} fravær")
    frame = pd.DataFrame(rows).drop_duplicates(["fixture_id", "team_id", "player_id"])
    frame.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n{len(frame):,} fravær gemt: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
