
from datetime import datetime, timezone

from src.api import open_database_read_only

now_utc = (
    datetime.now(timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z")
)

with open_database_read_only() as connection:
    rows = connection.execute(
        """
        SELECT
            m.league,
            m.match_date,
            m.kickoff_at,
            m.home_team,
            m.away_team
        FROM matches AS m
        WHERE m.data_source = 'football-data.org'
          AND m.league IN (
              'Premier League',
              'La Liga',
              'Bundesliga',
              'Serie A',
              'Ligue 1'
          )
          AND upper(m.match_status) NOT IN (
              'FINISHED', 'CANCELED', 'CANCELLED'
          )
          AND m.kickoff_at IS NOT NULL
          AND m.kickoff_at > ?
          AND EXISTS (
              SELECT 1
              FROM odds_observations AS o
              WHERE o.match_id = m.match_id
                AND lower(replace(o.bookmaker, ' ', '')) = 'bet365'
                AND o.market = '1X2:opening'
          )
        ORDER BY m.kickoff_at, m.league, m.home_team
        LIMIT 8
        """,
        (now_utc,),
    ).fetchall()

for row in rows:
    print(dict(row))

print(f"\nAntal kommende kampe: {len(rows)}")
