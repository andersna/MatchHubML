"""Started matches stay in /api/matches for IN_PLAY_WINDOW and are never logged."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from src import api
from src.data_store import DataStore


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


class InPlayWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows: sqlite file may still be open
        self.db_path = Path(self.tmp.name) / "cache.sqlite"
        with DataStore(self.db_path) as store:
            store.initialize_schema()
        now = datetime.now(timezone.utc)
        self.fixtures = {
            # name: (kickoff offset, status)
            "Future FC": (timedelta(hours=1), "TIMED"),
            "Started FC": (timedelta(minutes=-60), "IN_PLAY"),
            "Old FC": (timedelta(hours=-3), "IN_PLAY"),        # outside the 2 h window
            "Finished FC": (timedelta(minutes=-30), "FINISHED"),  # ended early: gone
        }
        connection = sqlite3.connect(self.db_path)
        for index, (home, (offset, status)) in enumerate(self.fixtures.items(), start=1):
            kickoff = now + offset
            connection.execute(
                """INSERT INTO matches (match_id, data_source, source_match_key, league, match_date,
                       kickoff_at, home_team, away_team, match_status, source_payload_json,
                       imported_at_utc, updated_at_utc)
                   VALUES (?, 'football-data.org', ?, 'Premier League', ?, ?, ?, 'Away FC', ?, '{}', ?, ?)""",
                (index, f"k{index}", kickoff.date().isoformat(), _iso(kickoff), home, status,
                 _iso(now), _iso(now)),
            )
            connection.execute(
                """INSERT INTO odds_observations (match_id, bookmaker, market, odds_home, odds_draw,
                       odds_away, observed_at_utc, data_source, observation_key, created_at_utc)
                   VALUES (?, 'Bet365', '1X2:closing', 2.0, 3.4, 3.8, ?, 'test', ?, ?)""",
                (index, _iso(now - timedelta(hours=4)), f"o{index}", _iso(now)),
            )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_window_keeps_started_matches_for_two_hours(self):
        with patch.object(api, "DATABASE_PATH", self.db_path):
            rows = api._upcoming_with_odds(None, 50)
        by_home = {row["home_team"]: row for row in rows}
        self.assertEqual(set(by_home), {"Future FC", "Started FC"})
        self.assertFalse(by_home["Future FC"]["in_play"])
        self.assertTrue(by_home["Started FC"]["in_play"])
        # Started matches come first (kickoff order).
        self.assertEqual(rows[0]["home_team"], "Started FC")

    def test_in_play_predictions_are_not_logged(self):
        fixture = {
            "league": "Ligue 1", "match_date": datetime.now(timezone.utc).date().isoformat(),
            "kickoff_at": _iso(datetime.now(timezone.utc) - timedelta(minutes=30)),
            "home_team": "Olympique de Marseille", "away_team": "Toulouse FC",
            "odds_home": 1.6, "odds_draw": 4.0, "odds_away": 5.5,
            "observed_at_utc": _iso(datetime.now(timezone.utc) - timedelta(hours=3)),
            "market": "1X2:closing",
        }
        for in_play in (True, False):
            with self.subTest(in_play=in_play), \
                    patch.object(api, "_upcoming_with_odds", return_value=[fixture | {"in_play": in_play}]), \
                    patch.object(api.prediction_log, "log_predictions", return_value=0) as log:
                payload = api.predict_upcoming("v3", None, 10)
            self.assertEqual(payload["count"], 1, payload["skipped"])
            self.assertIs(payload["matches"][0]["in_play"], in_play)
            logged = list(log.call_args.args[0])
            if in_play:
                self.assertEqual(logged, [])
            else:
                self.assertTrue(logged)


if __name__ == "__main__":
    unittest.main()
