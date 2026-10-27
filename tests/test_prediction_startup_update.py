from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.predict_upcoming_match import check_historical_data_on_startup
from src import update_historical_data as updater


class PredictionStartupUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.fixtures = root / "fixtures.csv"
        self.state = root / "state.json"
        self.fixtures.write_text(
            "date,home_team,away_team,result,league,season\n"
            "2020-01-01,Home,Away,H,Premier League,2019-20\n",
            encoding="utf-8",
        )
        self.original_fixtures = self.fixtures.read_bytes()
        self.now = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def _check(self):
        return updater.ensure_historical_data_current(
            fixtures_path=self.fixtures,
            state_path=self.state,
            now=self.now,
        )

    def test_startup_does_not_download_within_seven_days(self):
        self.state.write_text(json.dumps({
            "last_success_at_utc": (self.now - timedelta(days=6, hours=23))
            .isoformat().replace("+00:00", "Z")
        }), encoding="utf-8")
        with patch("src.update_historical_data.requests.get") as mocked, \
             redirect_stdout(io.StringIO()):
            report = check_historical_data_on_startup(self._check)
        mocked.assert_not_called()
        self.assertTrue(report.skipped_by_interval)
        self.assertEqual(self.fixtures.read_bytes(), self.original_fixtures)

    def test_startup_attempts_download_after_seven_days(self):
        self.state.write_text(json.dumps({
            "last_success_at_utc": (self.now - timedelta(days=7))
            .isoformat().replace("+00:00", "Z")
        }), encoding="utf-8")
        with patch(
            "src.update_historical_data.requests.get",
            side_effect=OSError("simulated network outage"),
        ) as mocked, redirect_stdout(io.StringIO()):
            report = check_historical_data_on_startup(self._check)
        mocked.assert_called_once()
        self.assertTrue(report.attempted)
        self.assertIn("simulated network outage", report.error)
        self.assertEqual(self.fixtures.read_bytes(), self.original_fixtures)

    def test_startup_continues_when_updater_raises(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            report = check_historical_data_on_startup(
                lambda: (_ for _ in ()).throw(RuntimeError("startup test failure"))
            )
        self.assertIsNone(report)
        self.assertIn("fortsætter med eksisterende data", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
