
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.api import open_database_read_only
from src.upcoming_match_features import build_upcoming_match_features

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "feature_reference.json"

now_utc = (
    datetime.now(timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z")
)

with open_database_read_only() as connection:
    row = connection.execute(
        """
        SELECT m.league, m.match_date, m.home_team, m.away_team
        FROM matches AS m
        WHERE m.data_source = 'football-data.org'
          AND m.league = 'La Liga'
          AND m.match_date = '2026-10-21'
          AND m.home_team = 'Levante UD'
          AND m.away_team = 'Athletic Club'
          AND m.kickoff_at > ?
          AND EXISTS (
              SELECT 1
              FROM odds_observations AS o
              WHERE o.match_id = m.match_id
                AND lower(replace(o.bookmaker, ' ', '')) = 'bet365'
                AND o.market = '1X2:opening'
          )
        LIMIT 1
        """,
        (now_utc,),
    ).fetchone()

if row is None:
    raise RuntimeError("Referenc kampen blev ikke fundet i den lokale database.")

request = {
    "league": row["league"],
    "home_team": row["home_team"],
    "away_team": row["away_team"],
    "match_date": row["match_date"],
}

features = build_upcoming_match_features(request)

record = {
    **request,
    "features": {
        column: float(features.iloc[0][column])
        for column in features.columns
    },
}

OUTPUT.write_text(
    json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
    encoding="utf-8",
)

print(f"Reference gemt: {OUTPUT}")
print(f"Kamp: {request['home_team']} - {request['away_team']}")
print(f"Antal features: {len(record['features'])}")
