from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import api
from src.sync_football_data import SyncReport
from src.update_historical_data import UpdateReport


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


def _league(code, fetchable, earliest=None, checked_hours_ago=1.0):
    return {"code": code, "league": api.COMPETITIONS[code]["league"], "fetchable": fetchable,
            "earliest_fetchable": earliest, "list_checked_utc": _iso(checked_hours_ago)}


def _coverage(leagues, **extra):
    fetchable = sum(item["fetchable"] for item in leagues)
    return {"total": 10, "has_odds": 10 - fetchable, "fetchable": fetchable, "no_odds_yet": 0,
            "not_listed": 0, "presses_needed": -(-fetchable // 5), "next_press_calls": 0,
            "leagues": leagues, "missing": [], **extra}


ALL_FRESH = [_league(code, 0) for code in api.COMPETITIONS]


def _report(*, calls=(), failures=(), obtained=0):
    report = SyncReport(date(2026, 10, 13), date(2026, 11, 26))
    report.api_calls = [dict(call) for call in calls]
    report.failures = list(failures)
    if obtained:
        report.odds_collection_by_league = {"SA": {"opening_obtained": obtained}}
    return report


class PlanOddsLeaguesTests(unittest.TestCase):
    def test_one_league_with_five_fetchable_is_the_only_fixture_list(self):
        leagues = [item for item in ALL_FRESH if item["code"] != "SA"] + [_league("SA", 13, "2026-10-28T19:45:00Z")]
        self.assertEqual(api.plan_odds_leagues(_coverage(leagues)), ("SA",))

    def test_earliest_kickoffs_first_and_biggest_backlog_last(self):
        leagues = [
            _league("PL", 3, "2026-11-08T14:00:00Z"),
            _league("PD", 0),
            _league("BL1", 9, "2026-11-06T19:30:00Z"),
            _league("SA", 2, "2026-10-28T19:45:00Z"),
            _league("FL1", 11, "2026-11-01T16:15:00Z"),
        ]
        # SA (2) is earliest, then FL1 (11) reaches 5 fetchable; BL1/PL wait.
        self.assertEqual(api.plan_odds_leagues(_coverage(leagues)), ("SA", "FL1"))

    def test_stale_fixture_lists_are_rechecked_for_new_matches(self):
        leagues = [item for item in ALL_FRESH if item["code"] != "PD"] + [_league("PD", 0, checked_hours_ago=7)]
        self.assertEqual(api.plan_odds_leagues(_coverage(leagues)), ("PD",))
        self.assertEqual(api.plan_odds_leagues(_coverage(ALL_FRESH)), ())


class SyncEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        before = _coverage([item for item in ALL_FRESH if item["code"] != "SA"] + [_league("SA", 6, "2026-10-28T19:45:00Z")])
        after = {**before, "has_odds": 9, "fetchable": 1, "presses_needed": 1, "next_press_calls": 2}
        patches = [
            patch.object(api, "LAST_SYNC_PATH", Path(self.temp.name) / "last_sync.json"),
            patch.object(api, "update_historical_data", return_value=UpdateReport(skipped_by_interval=True, updated_at_utc="x")),
            patch.object(api, "odds_coverage", side_effect=[before, after]),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_results_for_all_leagues_then_odds_only_for_planned_leagues(self):
        football = [{"source": "football-data.org", "label": "matches", "status": 200, "rate_remaining": "3"}] * 2
        odds = [{"source": "5dollarfootballapi", "label": "odds", "status": 200, "rate_remaining": "95"}] * 6
        with patch.object(api, "run_sync", side_effect=[_report(calls=football), _report(calls=odds, obtained=5)]) as run:
            result = api.sync_data()
        matches_config, odds_config = (call.args[0] for call in run.call_args_list)
        self.assertTrue(matches_config.skip_odds)
        self.assertEqual(matches_config.leagues, tuple(api.COMPETITIONS))
        self.assertEqual(odds_config.leagues, ("SA",))
        self.assertFalse(odds_config.skip_odds)
        self.assertEqual(odds_config.max_odds_calls, api.ODDS_CALLS_PER_SYNC)
        self.assertEqual(result["odds_fetched"], 5)
        self.assertEqual(result["api_calls"]["football-data.org"]["calls"], 2)
        self.assertEqual(result["api_calls"]["football-data.org"]["quota_remaining"], "3")
        self.assertEqual(result["api_calls"]["5dollarfootballapi"]["calls"], 6)
        self.assertEqual(result["cooldown_seconds"], 60)  # 3 calls left < 5 needed by the next press
        self.assertEqual(json.loads(api.LAST_SYNC_PATH.read_text(encoding="utf-8"))["odds_fetched"], 5)

    def test_no_cooldown_when_matches_were_not_refetched(self):
        with patch.object(api, "run_sync", side_effect=[_report(), _report()]):
            result = api.sync_data()
        self.assertEqual(result["cooldown_seconds"], 0)
        self.assertFalse(result["matches"]["fetched"])

    def test_rate_limit_is_explained_in_plain_language(self):
        failure = "5dollarfootballapi: rate limit (HTTP 429); intet retry udføres."
        with patch.object(api, "run_sync", side_effect=[_report(), _report(failures=[failure])]):
            result = api.sync_data()
        self.assertEqual(result["status"], "partial")
        self.assertIn("5DollarFootball har nået sin grænse", result["failures"][0])

    def test_quota_reset_time_comes_from_retry_after(self):
        limited = {"source": "5dollarfootballapi", "label": "odds", "status": 429,
                   "rate_remaining": "0", "rate_limit": "15", "retry_after": "120"}
        with patch.object(api, "run_sync", side_effect=[_report(), _report(calls=[limited])]):
            result = api.sync_data()
        quota = result["api_calls"]["5dollarfootballapi"]
        self.assertEqual((quota["quota_remaining"], quota["quota_limit"]), ("0", "15"))
        reset = datetime.fromisoformat(quota["quota_reset_utc"].replace("Z", "+00:00"))
        self.assertAlmostEqual((reset - datetime.now(timezone.utc)).total_seconds(), 120, delta=5)


if __name__ == "__main__":
    unittest.main()
