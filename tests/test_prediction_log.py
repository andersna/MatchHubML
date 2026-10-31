from __future__ import annotations

import math
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src import prediction_log


def _record(version, probs, *, source="live", odds=(0.5, 0.3, 0.2), home="Home FC", logged="2026-10-20T10:00:00Z",
            market="1X2:closing"):
    return {
        "source": source, "model_version": version, "model_artifact": f"artifact_{version}",
        "logged_at_utc": logged, "league": "Premier League", "match_date": "2026-10-25",
        "kickoff_at": "2026-10-25T14:00:00Z", "home_team": home, "away_team": "Away FC",
        "odds_observed_at_utc": "2026-10-19T08:00:00Z", "odds_market": market, "odds": odds, "probs": probs,
    }


class PredictionLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.log = root / "log.sqlite"
        self.cache = root / "cache.sqlite"
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute(
                """CREATE TABLE matches (data_source TEXT, league TEXT, match_date TEXT,
                home_team TEXT, away_team TEXT, match_status TEXT, home_score INT,
                away_score INT, result TEXT)"""
            )
            connection.execute(
                "INSERT INTO matches VALUES ('football-data.org', 'Premier League', '2026-10-25', "
                "'Home FC', 'Away FC', 'FINISHED', 2, 1, 'H')"
            )
        self.addCleanup(self.temp.cleanup)

    def test_same_model_match_and_odds_snapshot_is_logged_once(self):
        first = prediction_log.log_predictions([_record("v3", (0.6, 0.25, 0.15))], path=self.log)
        again = prediction_log.log_predictions([_record("v3", (0.1, 0.1, 0.8))], path=self.log)
        self.assertEqual((first, again), (1, 0))

    def test_scores_models_against_result_and_odds(self):
        prediction_log.log_predictions([
            _record("v3", (0.6, 0.25, 0.15)),
            _record("v1", (0.4, 0.3, 0.3)),
            _record("v3", (0.5, 0.3, 0.2), home="Other FC"),  # not finished yet
        ], path=self.log)
        report = prediction_log.evaluate(self.cache, path=self.log)

        self.assertEqual(report["pending"], 1)
        self.assertEqual(len(report["matches"]), 1)
        models = {row["model"]: row for row in report["models"]}
        self.assertAlmostEqual(models["v3"]["log_loss"], -math.log(0.6))
        self.assertAlmostEqual(models["v3"]["delta_log_loss_vs_odds"], -math.log(0.6) + math.log(0.5))
        self.assertEqual(models["v3"]["accuracy"], 1.0)
        self.assertAlmostEqual(report["odds"]["log_loss"], -math.log(0.5))
        self.assertEqual(models["v3"]["cumulative_delta_log_loss"], [round(-math.log(0.6) + math.log(0.5), 6)])

    def test_same_lookup_is_logged_once_per_price_type(self):
        # One 5DollarFootball lookup stores opening and lookup price with the same timestamp.
        opening = prediction_log.log_predictions([_record("v3", (0.5, 0.3, 0.2), market="1X2:opening")], path=self.log)
        lookup = prediction_log.log_predictions(
            [_record("v3", (0.6, 0.25, 0.15), market="1X2:closing", logged="2026-10-20T11:00:00Z")], path=self.log
        )
        self.assertEqual((opening, lookup), (1, 1))
        report = prediction_log.evaluate(self.cache, path=self.log)
        self.assertEqual(report["matches"][0]["v3"]["probs"], [0.6, 0.25, 0.15])  # latest prediction is scored

    def test_old_log_without_odds_market_is_migrated_as_opening(self):
        old_schema = prediction_log.SCHEMA.replace(
            "    odds_market TEXT NOT NULL DEFAULT '1X2:opening',\n", ""
        ).replace(", odds_market)", ")")
        with closing(sqlite3.connect(self.log)) as connection, connection:
            connection.executescript(old_schema)
            connection.execute(
                """INSERT INTO predictions (logged_at_utc, source, model_version, model_artifact, league,
                match_date, kickoff_at, home_team, away_team, odds_observed_at_utc, odds_prob_home,
                odds_prob_draw, odds_prob_away, prob_home, prob_draw, prob_away)
                VALUES ('2026-10-20T10:00:00Z', 'live', 'v3', 'a', 'Premier League', '2026-10-25', NULL,
                'Home FC', 'Away FC', '2026-10-19T08:00:00Z', 0.5, 0.3, 0.2, 0.6, 0.25, 0.15)"""
            )
        self.assertEqual(prediction_log.logged_matches(path=self.log), {("v3", "Premier League", 2026, "Home FC", "Away FC")})
        with closing(sqlite3.connect(self.log)) as connection:
            self.assertEqual(connection.execute("SELECT odds_market FROM predictions").fetchall(), [("1X2:opening",)])

    def test_old_log_without_expected_goals_gets_the_columns(self):
        old_schema = prediction_log.SCHEMA.replace("    home_expected_goals REAL,\n    away_expected_goals REAL,\n", "")
        with closing(sqlite3.connect(self.log)) as connection, connection:
            connection.executescript(old_schema)
        record = {**_record("v3", (0.6, 0.25, 0.15)), "expected_goals": (1.8, 1.1)}
        self.assertEqual(prediction_log.log_predictions([record], path=self.log), 1)
        with closing(sqlite3.connect(self.log)) as connection:
            self.assertEqual(connection.execute("SELECT home_expected_goals, away_expected_goals FROM predictions").fetchall(),
                             [(1.8, 1.1)])

    def test_repeat_fills_in_missing_expected_goals_without_counting_as_new(self):
        prediction_log.log_predictions([_record("v3", (0.6, 0.25, 0.15))], path=self.log)
        again = prediction_log.log_predictions([{**_record("v3", (0.6, 0.25, 0.15)), "expected_goals": (1.8, 1.1)}],
                                               path=self.log)
        self.assertEqual(again, 0)
        with closing(sqlite3.connect(self.log)) as connection:
            self.assertEqual(connection.execute("SELECT home_expected_goals, away_expected_goals FROM predictions").fetchall(),
                             [(1.8, 1.1)])

    def test_logged_matches_are_keyed_by_season_not_date(self):
        prediction_log.log_predictions([_record("v3", (0.6, 0.25, 0.15))], path=self.log)
        logged = prediction_log.logged_matches(path=self.log)
        moved = prediction_log.match_identity("Premier League", "2026-11-04", "Home FC", "Away FC")
        self.assertIn(("v3", *moved), logged)

    def test_season_starts_in_july(self):
        self.assertEqual(prediction_log.season_start("2026-07-01"), 2026)
        self.assertEqual(prediction_log.season_start("2027-05-20"), 2026)

    def _rescheduled_cache(self):
        """The match was logged for 25 Oct but moved to 24 Oct 18:00 and played then."""
        cache = Path(self.temp.name) / "moved.sqlite"
        with closing(sqlite3.connect(cache)) as connection, connection:
            connection.execute(
                """CREATE TABLE matches (data_source TEXT, league TEXT, match_date TEXT, kickoff_at TEXT,
                home_team TEXT, away_team TEXT, match_status TEXT, home_score INT, away_score INT, result TEXT)"""
            )
            connection.execute(
                "INSERT INTO matches VALUES ('football-data.org', 'Premier League', '2026-10-24', "
                "'2026-10-24T18:00:00Z', 'Home FC', 'Away FC', 'FINISHED', 2, 1, 'H')"
            )
        return cache

    def test_rescheduled_match_keeps_its_prediction_and_shows_the_new_date(self):
        prediction_log.log_predictions([_record("v3", (0.6, 0.25, 0.15))], path=self.log)  # logged for 25 Oct
        report = prediction_log.evaluate(self._rescheduled_cache(), path=self.log)
        self.assertEqual(report["pending"], 0)
        self.assertEqual(len(report["matches"]), 1)
        self.assertEqual(report["matches"][0]["match_date"], "2026-10-24")
        self.assertEqual(report["matches"][0]["result"], "H")

    def test_live_prediction_beats_a_later_reconstruction_of_the_same_match(self):
        prediction_log.log_predictions([
            _record("v3", (0.6, 0.25, 0.15)),  # live, logged for 25 Oct
            {**_record("v3", (0.2, 0.3, 0.5), source="backfill", logged="2026-10-26T10:00:00Z"), "match_date": "2026-10-24"},
        ], path=self.log)
        match = prediction_log.evaluate(self._rescheduled_cache(), path=self.log)["matches"][0]
        self.assertEqual(match["v3"]["source"], "live")
        self.assertEqual(match["v3"]["probs"], [0.6, 0.25, 0.15])

    def test_live_prediction_after_the_actual_kickoff_is_ignored(self):
        # Logged while the cache still said 25 Oct, but the match had kicked off on 24 Oct.
        prediction_log.log_predictions([_record("v3", (0.9, 0.05, 0.05), logged="2026-10-24T19:00:00Z")], path=self.log)
        report = prediction_log.evaluate(self._rescheduled_cache(), path=self.log)
        self.assertEqual(report["matches"], [])

    def test_fallback_supplies_results_the_cache_lacks(self):
        prediction_log.log_predictions([_record("v3", (0.2, 0.3, 0.5), home="Old FC")], path=self.log)
        fallback = lambda keys: {key: {"home_score": 0, "away_score": 1, "result": "A"} for key in keys}
        report = prediction_log.evaluate(self.cache, path=self.log, fallback_results=fallback)
        self.assertEqual(report["pending"], 0)
        scored = {item["home_team"]: item for item in report["matches"]}
        self.assertEqual(scored["Old FC"]["result"], "A")

    def test_missing_log_file_gives_empty_report(self):
        report = prediction_log.evaluate(self.cache, path=Path(self.temp.name) / "missing.sqlite")
        self.assertEqual(report["matches"], [])


if __name__ == "__main__":
    unittest.main()
