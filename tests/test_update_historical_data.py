from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import update_historical_data as updater


class FakeDownload:
    status_code = 200

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def _workbook_bytes() -> bytes:
    rows = {
        "E0": [
            {"Date": "10/10/2026", "HomeTeam": "Existing Home", "AwayTeam": "Existing Away",
             "FTHG": 2, "FTAG": 1, "HS": 10, "AS": 4, "B365H": 2.0, "B365D": 3.0, "B365A": 4.0},
            {"Date": "10/10/2026", "HomeTeam": "Existing Home", "AwayTeam": "Existing Away",
             "FTHG": 2, "FTAG": 1, "HS": 11, "AS": 4, "B365H": 2.0, "B365D": 3.0, "B365A": 4.0},
            {"Date": "17/10/2026", "HomeTeam": "New Home", "AwayTeam": "New Away",
             "FTHG": None, "FTAG": None, "HS": None, "AS": None,
             "B365H": None, "B365D": None, "B365A": None},
        ],
        "SP1": [{"Date": "11/10/2026", "HomeTeam": "Spain Home", "AwayTeam": "Spain Away",
                 "FTHG": 1, "FTAG": 1, "B365H": 2.0, "B365D": 3.0, "B365A": 4.0}],
        "D1": [{"Date": "11/10/2026", "HomeTeam": "Germany Home", "AwayTeam": "Germany Away",
                "FTHG": 1, "FTAG": 0, "B365H": 2.0, "B365D": 3.0, "B365A": 4.0}],
        "I1": [{"Date": "11/10/2026", "HomeTeam": "Italy Home", "AwayTeam": "Italy Away",
                "FTHG": 0, "FTAG": 0, "B365H": 2.0, "B365D": 3.0, "B365A": 4.0}],
        "F1": [{"Date": "11/10/2026", "HomeTeam": "France Home", "AwayTeam": "France Away",
                "FTHG": 0, "FTAG": 1, "B365H": 2.0, "B365D": 3.0, "B365A": 4.0}],
    }
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, sheet_rows in rows.items():
            pd.DataFrame(sheet_rows).to_excel(writer, sheet_name=sheet, index=False)
    return buffer.getvalue()


def _existing_csv(path: Path) -> bytes:
    frame = pd.DataFrame([
        {"date": "2026-10-10", "time": "15:00:00", "home_team": "Existing Home",
         "away_team": "Existing Away", "home_goals": None, "away_goals": None,
         "result": None, "league": "Premier League", "season": "2026-27",
         "home_shots": None, "away_shots": None, "B365H": 6.5, "odds_history_marker": "keep-me"},
        {"date": "2020-01-01", "time": None, "home_team": "Old Home", "away_team": "Old Away",
         "home_goals": 1, "away_goals": 0, "result": "H", "league": "Premier League",
         "season": "2019-20", "home_shots": 8, "away_shots": 2,
         "B365H": 3.25, "odds_history_marker": "old-row"},
    ])
    frame.to_csv(path, index=False)
    return path.read_bytes()


class HistoricalUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.fixtures = root / "fixtures.csv"
        self.state = root / "historical_update_state.json"
        self.old_csv = _existing_csv(self.fixtures)
        self.now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
        self.content = _workbook_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, *, now=None, force=False, content=None):
        with patch("src.update_historical_data.requests.get", return_value=FakeDownload(
            self.content if content is None else content
        )) as mocked:
            report = updater.update_historical_data(
                fixtures_path=self.fixtures,
                state_path=self.state,
                now=now or self.now,
                force=force,
            )
        return report, mocked

    def test_weekly_interval_skips_until_seven_days_then_downloads(self):
        self.state.write_text(json.dumps({
            "last_success_at_utc": (self.now - timedelta(days=7) + timedelta(minutes=1))
            .isoformat().replace("+00:00", "Z")
        }), encoding="utf-8")
        report, mocked = self._run()
        self.assertTrue(report.skipped_by_interval)
        mocked.assert_not_called()
        expired, mocked = self._run(now=self.now + timedelta(minutes=1))
        self.assertIsNone(expired.error)
        mocked.assert_called_once()

    def test_merges_new_matches_updates_scores_and_preserves_existing_odds(self):
        report, mocked = self._run()
        self.assertIsNone(report.error)
        mocked.assert_called_once()
        self.assertEqual(report.duplicate_source_rows, 1)
        self.assertEqual(report.new_matches, 5)  # one new fixture in each competition
        self.assertEqual(report.updated_matches, 1)
        self.assertEqual(report.unchanged_matches, 0)
        self.assertEqual(report.missing_results, 1)
        self.assertEqual(report.missing_odds, 1)

        data = pd.read_csv(self.fixtures, low_memory=False)
        self.assertEqual(len(data), 7)
        updated = data.loc[data["home_team"].eq("Existing Home")].iloc[0]
        self.assertEqual((int(updated["home_goals"]), int(updated["away_goals"]), updated["result"]), (2, 1, "H"))
        self.assertEqual(float(updated["B365H"]), 6.5)
        self.assertEqual(updated["odds_history_marker"], "keep-me")
        self.assertEqual(data.loc[data["home_team"].eq("Old Home"), "result"].iloc[0], "H")
        self.assertEqual(json.loads(self.state.read_text(encoding="utf-8"))["last_success_at_utc"], "2026-10-20T12:00:00Z")

    def test_invalid_download_preserves_csv_and_status(self):
        old_state = b'{"last_success_at_utc":"2026-10-01T00:00:00Z"}\n'
        self.state.write_bytes(old_state)
        report, _ = self._run(content=b"not an excel workbook")
        self.assertIsNotNone(report.error)
        self.assertEqual(self.fixtures.read_bytes(), self.old_csv)
        self.assertEqual(self.state.read_bytes(), old_state)

    def test_download_failure_preserves_csv_and_status(self):
        old_state = b'{"last_success_at_utc":"2026-10-01T00:00:00Z"}\n'
        self.state.write_bytes(old_state)
        with patch(
            "src.update_historical_data.requests.get",
            side_effect=OSError("simulated network failure"),
        ):
            report = updater.update_historical_data(
                fixtures_path=self.fixtures, state_path=self.state, now=self.now, force=True
            )
        self.assertIn("simulated network failure", report.error)
        self.assertEqual(self.fixtures.read_bytes(), self.old_csv)
        self.assertEqual(self.state.read_bytes(), old_state)

    def test_failed_state_commit_rolls_back_fixtures_csv(self):
        old_state = b'{"last_success_at_utc":"2026-10-01T00:00:00Z"}\n'
        self.state.write_bytes(old_state)
        real_replace = updater.os.replace

        def fail_only_state(source, destination):
            if Path(destination) == self.state:
                raise OSError("simulated state commit failure")
            return real_replace(source, destination)

        with patch("src.update_historical_data.requests.get", return_value=FakeDownload(self.content)), \
             patch("src.update_historical_data.os.replace", side_effect=fail_only_state):
            report = updater.update_historical_data(
                fixtures_path=self.fixtures, state_path=self.state, now=self.now, force=True
            )
        self.assertIn("simulated state commit failure", report.error)
        self.assertEqual(self.fixtures.read_bytes(), self.old_csv)
        self.assertEqual(self.state.read_bytes(), old_state)

    def test_source_odds_columns_do_not_overwrite_existing_odds_values(self):
        # The workbook has B365 odds, but the canonical fixtures parser omits
        # them; existing values in fixtures.csv must remain untouched.
        report, _ = self._run()
        self.assertIsNone(report.error)
        data = pd.read_csv(self.fixtures, low_memory=False)
        self.assertEqual(float(data.loc[data["home_team"].eq("Existing Home"), "B365H"].iloc[0]), 6.5)

    def test_successful_update_keeps_workbook_next_to_history(self):
        report, _ = self._run(force=True)
        self.assertIsNone(report.error)
        saved = self.fixtures.parent / "raw" / "all-euro-data-2026-2027.xlsx"
        self.assertEqual(saved.read_bytes(), self.content)


class SeasonSourceUrlTests(unittest.TestCase):
    def test_url_follows_the_season_that_starts_in_july(self):
        self.assertTrue(updater.season_source_url(datetime(2026, 10, 1)).endswith("/2627/all-euro-data-2026-2027.xlsx"))
        self.assertTrue(updater.season_source_url(datetime(2027, 6, 30)).endswith("/2627/all-euro-data-2026-2027.xlsx"))
        self.assertTrue(updater.season_source_url(datetime(2027, 7, 1)).endswith("/2728/all-euro-data-2027-2028.xlsx"))
        self.assertTrue(updater.season_source_url(datetime(2099, 8, 1)).endswith("/9900/all-euro-data-2099-2100.xlsx"))


if __name__ == "__main__":
    unittest.main()
