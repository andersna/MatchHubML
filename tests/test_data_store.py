from __future__ import annotations

import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.data_store import DataStore
from src.export_sqlite_to_csv import export_matches_csv
from src.import_csv_to_sqlite import import_csv


FIXTURE_COLUMNS = [
    "date", "time", "home_team", "away_team", "home_goals", "away_goals",
    "result", "league", "season",
]


class DataStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "test.sqlite"
        self.csv_path = self.root / "fixtures.csv"
        self.write_fixture_csv([
            ["2025-01-01", "19:45:00", "Home FC", "Away FC", "2", "1", "H", "Test League", "2024-25"],
        ])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_fixture_csv(self, rows: list[list[str]]) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIXTURE_COLUMNS)
            writer.writerows(rows)

    def test_database_schema(self) -> None:
        with DataStore(self.database) as store:
            names = {
                row["name"] for row in store.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertTrue({"matches", "odds_observations", "sync_state", "fixture_match_audit"}.issubset(names))

    def test_import_valid_invalid_and_missing_values(self) -> None:
        self.write_fixture_csv([
            ["2025-01-01", "19:45:00", "Home FC", "Away FC", "2", "1", "H", "Test League", "2024-25"],
            ["not-a-date", "", "Bad FC", "Away FC", "0", "0", "D", "Test League", "2024-25"],
            ["2025-01-03", "", "Missing Score", "Away FC", "", "", "", "Test League", "2024-25"],
            ["2025-01-04", "25:61:00", "Bad Time", "Away FC", "0", "0", "D", "Test League", "2024-25"],
        ])
        report = import_csv(self.csv_path, self.database)
        self.assertEqual(report.rows_read, 4)
        self.assertEqual(report.imported, 2)
        self.assertEqual(report.rejected, 2)
        with DataStore(self.database) as store:
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 2)
            missing_score = store.connection.execute(
                "SELECT match_status, home_score, result FROM matches WHERE home_team='Missing Score'"
            ).fetchone()
            self.assertEqual(tuple(missing_score), ("SCHEDULED", None, None))

    def test_repeated_import_is_idempotent(self) -> None:
        first = import_csv(self.csv_path, self.database)
        second = import_csv(self.csv_path, self.database)
        self.assertEqual(first.imported, 1)
        self.assertEqual(second.imported, 0)
        self.assertEqual(second.duplicates, 1)
        with DataStore(self.database) as store:
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 1)

    def test_export_and_reimport(self) -> None:
        import_csv(self.csv_path, self.database)
        export_path = self.root / "exports" / "roundtrip.csv"
        written = export_matches_csv(self.database, export_path)
        self.assertEqual(written, export_path)
        new_db = self.root / "reimport.sqlite"
        report = import_csv(export_path, new_db)
        self.assertEqual(report.imported, 1)
        with export_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["home_team"], "Home FC")
        self.assertEqual(rows[0]["result"], "H")
        with self.assertRaisesRegex(FileExistsError, "findes allerede"):
            export_matches_csv(self.database, export_path)

    def test_odds_snapshots_are_appended_not_replaced(self) -> None:
        import_csv(self.csv_path, self.database)
        with DataStore(self.database) as store:
            match_id = store.connection.execute("SELECT match_id FROM matches").fetchone()[0]
            args = dict(
                match_id=match_id, bookmaker="Bookmaker", market="1X2",
                odds_home=2.1, odds_draw=3.2, odds_away=3.7,
                observed_at="2025-01-01T10:00:00+01:00", data_source="test",
            )
            with store.transaction():
                self.assertTrue(store.add_odds_observation(**args))
                self.assertFalse(store.add_odds_observation(**args))
                self.assertTrue(store.add_odds_observation(
                    **{**args, "odds_home": 2.0, "observed_at": "2025-01-01T11:00:00+01:00"}
                ))
            rows = store.connection.execute(
                "SELECT odds_home, observed_at_utc FROM odds_observations ORDER BY observed_at_utc"
            ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["observed_at_utc"], "2025-01-01T09:00:00Z")
            self.assertEqual(rows[1]["odds_home"], 2.0)

    def test_transaction_rolls_back_all_writes_on_failure(self) -> None:
        import_csv(self.csv_path, self.database)
        with DataStore(self.database) as store:
            store.connection.execute("DELETE FROM matches")
            with self.assertRaises(sqlite3.IntegrityError):
                with store.transaction():
                    store.connection.execute(
                        "INSERT INTO matches (data_source,source_match_key,league,match_date,home_team,away_team,match_status,source_payload_json,imported_at_utc,updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        ("test", "temporary", "L", "2025-01-01", "A", "B", "SCHEDULED", "{}", "now", "now"),
                    )
                    store.connection.execute(
                        "INSERT INTO odds_observations (match_id,bookmaker,market,odds_home,odds_draw,odds_away,observed_at_utc,data_source,observation_key,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (999999, "B", "1X2", 2.0, 3.0, 4.0, "2025-01-01T00:00:00Z", "test", "bad-fk", "now"),
                    )
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 0)

    def test_naive_odds_timestamp_is_rejected(self) -> None:
        with DataStore(self.database) as store:
            with self.assertRaisesRegex(ValueError, "skal indeholde tidszone"):
                store.add_odds_observation(
                    match_id=1, bookmaker="B", market="1X2", odds_home=2.0,
                    odds_draw=3.0, odds_away=4.0, observed_at="2025-01-01T12:00:00",
                    data_source="test",
                )

    def test_fixture_match_audits_append_by_sync_run(self) -> None:
        import_csv(self.csv_path, self.database)
        with DataStore(self.database) as store:
            match_id = store.connection.execute("SELECT match_id FROM matches").fetchone()[0]
            fixture = {
                "id": 77,
                "league": {"id": 4, "name": "League"},
                "teams": {"home": {"name": "Home"}, "away": {"name": "Away"}},
                "date": "2025-01-02",
                "kickoff_utc": "2025-01-02T19:00:00Z",
            }
            with store.transaction():
                store.add_fixture_match_audit(
                    sync_run_id="run-one", audited_at="2025-01-01T12:00:00+01:00",
                    data_source="5dollarfootballapi", fixture=fixture,
                    candidate_matches=[{"external_match_id": "123", "same_date": True}],
                    selected_match_id=match_id, outcome="matched", reason="matched",
                )
                store.add_fixture_match_audit(
                    sync_run_id="run-two", audited_at="2025-01-01T12:01:00Z",
                    data_source="5dollarfootballapi", fixture=fixture,
                    candidate_matches=[], selected_match_id=None,
                    outcome="unmatched", reason="team_names_mismatch",
                )
            rows = store.connection.execute(
                "SELECT * FROM fixture_match_audit ORDER BY audit_id"
            ).fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["sync_run_id"] for row in rows], ["run-one", "run-two"])
            self.assertEqual(rows[0]["audited_at_utc"], "2025-01-01T11:00:00Z")
            self.assertEqual(rows[0]["external_fixture_id"], "77")
            self.assertEqual(rows[0]["selected_match_id"], match_id)
            self.assertIn('"name":"League"', rows[0]["raw_league_json"])


if __name__ == "__main__":
    unittest.main()
