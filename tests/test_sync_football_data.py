from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data_store import DataStore
from src.sync_football_data import (
    FOOTBALL_DATA_SOURCE,
    FIVE_DOLLAR_SOURCE,
    FIVE_DOLLAR_TEAM_ALIASES,
    SyncConfig,
    SyncError,
    SyncReport,
    _has_stored_opening_odds,
    _match_provider_fixture,
    _match_provider_fixture_with_diagnostics,
    _norm_name,
    _provider_team_key,
    _extract_odds,
    _request_json,
    build_dry_run_plan,
    normalize_football_data_match,
    run_sync,
)


class FakeResponse:
    def __init__(self, status: int, payload: dict, headers: dict | None = None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        return self._payload


def _future_kickoff(now: datetime) -> str:
    return (now + timedelta(days=2)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _football_match(kickoff: str, *, status: str = "SCHEDULED", home_score=None, away_score=None):
    return {
        "id": 701,
        "utcDate": kickoff,
        "status": status,
        "homeTeam": {"name": "Test FC"},
        "awayTeam": {"name": "Away SC"},
        "competition": {"code": "PL", "name": "Premier League"},
        "season": {"id": 2026},
        "score": {"fullTime": {"home": home_score, "away": away_score}},
    }


def _five_dollar_fixtures(kickoff: str):
    return {
        "success": 1,
        "data": [{
            "id": 9901,
            "league": {"id": 555, "name": "England Premier League"},
            "teams": {"home": {"name": "Test"}, "away": {"name": "Away"}},
            "kickoff_utc": kickoff,
            "status": "scheduled",
        }],
    }


def _odds_payload():
    return {
        "success": 1,
        "data": {
            "fixture_id": 9901,
            "bookmakers": [{
                "name": "Bet 365",
                "slug": "bet365",
                "odds": {"1x2": {
                    "opening": {"home": 2.1, "draw": 3.3, "away": 3.4},
                    "closing": {"home": 2.0, "draw": 3.4, "away": 3.5},
                    "inplay": None,
                }},
            }],
        },
    }


class SyncFootballDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "cache.sqlite"
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.kickoff = _future_kickoff(self.now)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalizes_football_data_and_preserves_missing_kickoff(self):
        match = normalize_football_data_match(_football_match(self.kickoff), "PL")
        self.assertEqual(match["data_source"], FOOTBALL_DATA_SOURCE)
        self.assertEqual(match["source_match_key"], "external:701")
        self.assertTrue(match["kickoff_at"].endswith("Z"))
        no_time = _football_match("2026-10-22")
        match_no_time = normalize_football_data_match(no_time, "PL")
        self.assertEqual(match_no_time["match_date"], "2026-10-22")
        self.assertIsNone(match_no_time["kickoff_at"])

    def test_invalid_odds_do_not_produce_a_storable_stage(self):
        payload = _odds_payload()
        payload["data"]["bookmakers"][0]["odds"]["1x2"]["opening"]["draw"] = None
        bookmaker, stages = _extract_odds(payload["data"])
        self.assertEqual(bookmaker, "Bet 365")
        self.assertNotIn("opening", stages)
        self.assertIn("closing", stages)

    def test_opening_cache_ignores_closing_only_observation(self):
        match = normalize_football_data_match(_football_match(self.kickoff), "PL")
        with DataStore(self.database) as store:
            with store.transaction():
                match_id, _ = store.upsert_match(match)
                store.add_odds_observation(
                    match_id=match_id, bookmaker="Bet 365", market="1X2:closing",
                    odds_home=2.1, odds_draw=3.2, odds_away=3.4,
                    observed_at=self.now, data_source=FIVE_DOLLAR_SOURCE,
                    fetched_at=self.now,
                )
            self.assertFalse(_has_stored_opening_odds(store, match_id=match_id))
            with store.transaction():
                store.add_odds_observation(
                    match_id=match_id, bookmaker="Bet365", market="1X2:opening",
                    odds_home=2.2, odds_draw=3.1, odds_away=3.3,
                    observed_at=self.now, data_source=FIVE_DOLLAR_SOURCE,
                    fetched_at=self.now,
                )
            self.assertTrue(_has_stored_opening_odds(store, match_id=match_id))
            # Opening observations are immutable historical data. Age alone
            # must never cause them to be fetched again.
            self.assertTrue(_has_stored_opening_odds(store, match_id=match_id))

    def test_global_odds_cap_gives_each_selected_league_one_call(self):
        leagues = ("PL", "PD", "BL1", "SA", "FL1")
        names = {
            "PL": ("Premier League", "England"), "PD": ("La Liga", "Spain"),
            "BL1": ("Bundesliga", "Germany"), "SA": ("Serie A", "Italy"),
            "FL1": ("Ligue 1", "France"),
        }
        responses = []
        for index, code in enumerate(leagues):
            kickoff = self.kickoff
            match = _football_match(kickoff)
            match["id"] = 800 + index
            responses.append(FakeResponse(200, {"matches": [match]}))
            league_id = 600 + index
            league_name, country = names[code]
            responses.append(FakeResponse(200, {"success": 1, "data": [{
                "id": league_id, "name": league_name, "country": {"name": country},
            }]}))
            fixture = _five_dollar_fixtures(kickoff)["data"][0]
            fixture["id"] = 9901 + index
            fixture["league"] = {"id": league_id, "name": league_name}
            responses.append(FakeResponse(200, {"success": 1, "data": [fixture]}))
            odds = _odds_payload()
            odds["data"]["fixture_id"] = fixture["id"]
            responses.append(FakeResponse(200, odds))

        config = SyncConfig(
            database_path=self.database, leagues=leagues,
            date_from=self.now.date(), date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=5,
        )
        with patch("src.sync_football_data.requests.get", side_effect=responses) as mocked:
            report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now,
            )
        self.assertEqual(mocked.call_count, 20)
        self.assertEqual(report.odds_detail_calls_by_league, {code: 1 for code in leagues})
        self.assertEqual(report.odds_added, 10)
        self.assertEqual(report.odds_deferred, 0)
        for code in leagues:
            self.assertEqual(report.odds_collection_by_league[code]["upcoming_fixtures"], 1)
            self.assertEqual(report.odds_collection_by_league[code]["opening_obtained"], 1)
            self.assertEqual(report.odds_collection_by_league[code]["opening_reused"], 0)

    def test_cached_league_releases_odds_call_capacity_to_other_leagues(self):
        leagues = ("PL", "PD", "BL1", "SA", "FL1")
        names = {
            "PL": ("Premier League", "England"), "PD": ("La Liga", "Spain"),
            "BL1": ("Bundesliga", "Germany"), "SA": ("Serie A", "Italy"),
            "FL1": ("Ligue 1", "France"),
        }
        # Preload the PL fixture's opening quote. The other leagues have five
        # eligible details in total, so all five global call slots should be
        # used without repeating the PL detail request.
        with DataStore(self.database) as store:
            with store.transaction():
                preloaded = normalize_football_data_match(
                    {**_football_match(self.kickoff), "id": 800}, "PL"
                )
                preloaded["home_team"] = "PL Home"
                preloaded["away_team"] = "PL Away"
                match_id, _ = store.upsert_match(preloaded)
                store.add_odds_observation(
                    match_id=match_id, bookmaker="Bet 365", market="1X2:opening",
                    odds_home=2.1, odds_draw=3.3, odds_away=3.4,
                    observed_at=self.now - timedelta(days=2),
                    data_source=FIVE_DOLLAR_SOURCE,
                    fetched_at=self.now - timedelta(days=2),
                )

        responses = []
        expected_counts = {"PL": 0, "PD": 2, "BL1": 1, "SA": 1, "FL1": 1}
        fixture_id = 12000
        for index, code in enumerate(leagues):
            league_name, country = names[code]
            count = 2 if code == "PD" else 1
            match_rows = []
            fixture_rows = []
            for item_index in range(count):
                current_id = 800 + index * 10 + item_index
                current_fixture_id = fixture_id + index * 10 + item_index
                home, away = (
                    ("PL Home", "PL Away") if code == "PL"
                    else (f"{code} Home {item_index}", f"{code} Away {item_index}")
                )
                match = _football_match(self.kickoff)
                match["id"] = current_id
                match["homeTeam"]["name"] = home
                match["awayTeam"]["name"] = away
                match["competition"]["code"] = code
                match_rows.append(match)
                fixture_rows.append({
                    "id": current_fixture_id,
                    "league": {"id": 600 + index, "name": league_name},
                    "teams": {"home": {"name": home}, "away": {"name": away}},
                    "kickoff_utc": self.kickoff,
                    "status": "scheduled",
                })
            responses.extend([
                FakeResponse(200, {"matches": match_rows}),
                FakeResponse(200, {"success": 1, "data": [{
                    "id": 600 + index, "name": league_name,
                    "country": {"name": country},
                }]}),
                FakeResponse(200, {"success": 1, "data": fixture_rows}),
            ])
            for selected in fixture_rows[:expected_counts[code]]:
                odds = _odds_payload()
                odds["data"]["fixture_id"] = selected["id"]
                responses.append(FakeResponse(200, odds))

        config = SyncConfig(
            database_path=self.database, leagues=leagues,
            date_from=self.now.date(), date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=5, odds_refresh_hours=1,
        )
        with patch("src.sync_football_data.requests.get", side_effect=responses) as mocked:
            report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now,
            )
        self.assertEqual(report.odds_detail_calls_by_league, {"PD": 2, "BL1": 1, "SA": 1, "FL1": 1})
        self.assertEqual(sum(report.odds_detail_calls_by_league.values()), 5)
        self.assertEqual(mocked.call_count, 20)
        self.assertEqual(report.odds_deferred, 0)
        self.assertEqual(report.odds_duplicate, 0)
        self.assertEqual(report.odds_collection_by_league["PL"]["opening_reused"], 1)
        self.assertEqual(report.odds_collection_by_league["PL"]["upcoming_fixtures"], 1)
        self.assertEqual(report.odds_collection_by_league["PD"]["opening_obtained"], 2)

    def test_started_fixture_is_audited_but_gets_no_odds_request(self):
        config = SyncConfig(
            database_path=self.database, leagues=("PL",),
            date_from=self.now.date(), date_to=self.now.date() + timedelta(days=3),
            max_odds_calls=1,
        )
        fixture = _five_dollar_fixtures(self.now.isoformat())["data"]
        responses = [
            FakeResponse(200, {"matches": [_football_match(self.now.isoformat())]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"},
            }]}),
            FakeResponse(200, {"success": 1, "data": fixture}),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=responses) as mocked:
            report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now,
            )
        self.assertEqual(mocked.call_count, 3)
        self.assertFalse(any(call["label"].startswith("odds:") for call in report.api_calls))
        self.assertEqual(report.odds_collection_by_league["PL"]["upcoming_fixtures"], 0)
        with DataStore(self.database) as store:
            audit = store.connection.execute(
                "SELECT outcome,reason FROM fixture_match_audit WHERE external_fixture_id='9901'"
            ).fetchone()
            self.assertEqual(tuple(audit), ("rejected", "kickoff_not_future"))

    def test_closing_only_is_saved_separately_and_opening_is_reported_missing(self):
        config = SyncConfig(
            database_path=self.database, leagues=("PL",),
            date_from=self.now.date(), date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=1,
        )
        odds = _odds_payload()
        odds["data"]["bookmakers"][0]["odds"]["1x2"]["opening"]["draw"] = None
        responses = [
            FakeResponse(200, {"matches": [_football_match(self.kickoff)]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"},
            }]}),
            FakeResponse(200, _five_dollar_fixtures(self.kickoff)),
            FakeResponse(200, odds),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=responses):
            report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now,
            )
        self.assertEqual(report.odds_added, 1)
        self.assertEqual(report.odds_no_quote, 1)
        self.assertEqual(
            report.odds_collection_by_league["PL"]["opening_missing_after_lookup"], 1
        )
        with DataStore(self.database) as store:
            markets = [row[0] for row in store.connection.execute(
                "SELECT market FROM odds_observations"
            )]
            self.assertEqual(markets, ["1X2:closing"])
            state = store.connection.execute(
                "SELECT details_json FROM sync_state WHERE dataset='odds_fixture:9901'"
            ).fetchone()
            self.assertIn('"opening_status":"missing_or_invalid"', state[0])

        # While both cache layers are fresh, report the missing quote under
        # the fixture-cache category rather than as an API miss or range skip.
        with patch(
            "src.sync_football_data.requests.get",
            side_effect=AssertionError("fresh cache must not call API"),
        ) as mocked:
            cached_report = run_sync(
                config,
                football_data_token="fd-test",
                five_dollar_token="odds-test",
                now=self.now + timedelta(minutes=30),
            )
        mocked.assert_not_called()
        cached_metrics = cached_report.odds_collection_by_league["PL"]
        self.assertEqual(cached_metrics["fixture_cache_missing"], 1)
        self.assertEqual(cached_metrics["opening_missing_after_lookup"], 0)
        self.assertEqual(cached_metrics["range_cache_skipped"], 0)

    def test_missing_opening_can_be_retried_after_refresh_ttl(self):
        config = SyncConfig(
            database_path=self.database, leagues=("PL",),
            date_from=self.now.date(), date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=1, odds_refresh_hours=1,
        )
        first_odds = _odds_payload()
        first_odds["data"]["bookmakers"][0]["odds"]["1x2"]["opening"]["draw"] = None
        first_responses = [
            FakeResponse(200, {"matches": [_football_match(self.kickoff)]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"},
            }]}),
            FakeResponse(200, _five_dollar_fixtures(self.kickoff)),
            FakeResponse(200, first_odds),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=first_responses) as mocked:
            first_report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now,
            )
        self.assertEqual(first_report.odds_no_quote, 1)
        self.assertEqual(
            first_report.odds_collection_by_league["PL"]["opening_missing_after_lookup"], 1
        )
        self.assertEqual(mocked.call_count, 4)

        later = self.now + timedelta(hours=2)
        second_responses = [
            FakeResponse(200, _five_dollar_fixtures(self.kickoff)),
            FakeResponse(200, _odds_payload()),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=second_responses) as mocked:
            second_report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=later,
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(second_report.odds_detail_calls_by_league, {"PL": 1})
        self.assertGreaterEqual(second_report.odds_added, 1)
        self.assertEqual(second_report.odds_no_quote, 0)
        self.assertEqual(second_report.odds_collection_by_league["PL"]["opening_obtained"], 1)
        with DataStore(self.database) as store:
            counts = dict(store.connection.execute(
                "SELECT market, COUNT(*) FROM odds_observations GROUP BY market"
            ))
            self.assertEqual(counts, {"1X2:closing": 1, "1X2:opening": 1})

    def test_fresh_fixture_cache_without_opening_is_reported_separately(self):
        config = SyncConfig(
            database_path=self.database, leagues=("PL",),
            date_from=self.now.date(), date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=1, odds_refresh_hours=1,
        )
        missing = _odds_payload()
        missing["data"]["bookmakers"][0]["odds"]["1x2"]["opening"]["draw"] = None
        initial = [
            FakeResponse(200, {"matches": [_football_match(self.kickoff)]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"},
            }]}),
            FakeResponse(200, _five_dollar_fixtures(self.kickoff)),
            FakeResponse(200, missing),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=initial):
            run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now,
            )

        # Simulate an interrupted/partial range sync: the per-fixture cache is
        # still fresh, while the league-wide odds cache is absent.
        with DataStore(self.database) as store:
            with store.transaction():
                store.connection.execute(
                    "DELETE FROM sync_state WHERE data_source=? AND dataset=?",
                    (FIVE_DOLLAR_SOURCE, "odds:PL"),
                )
        later = self.now + timedelta(minutes=30)
        with patch(
            "src.sync_football_data.requests.get",
            return_value=FakeResponse(200, _five_dollar_fixtures(self.kickoff)),
        ) as mocked:
            report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=later,
            )
        self.assertEqual(mocked.call_count, 1)  # fixture list only; no detail call
        metrics = report.odds_collection_by_league["PL"]
        self.assertEqual(metrics["upcoming_fixtures"], 1)
        self.assertEqual(metrics["fixture_cache_missing"], 1)
        self.assertEqual(metrics["opening_missing_after_lookup"], 0)

    def test_fixture_without_usable_kickoff_is_reported(self):
        config = SyncConfig(
            database_path=self.database, leagues=("PL",),
            date_from=self.now.date(), date_to=self.now.date() + timedelta(days=3),
            max_odds_calls=1,
        )
        fixture = _five_dollar_fixtures(self.kickoff)
        fixture["data"][0]["kickoff_utc"] = None
        responses = [
            FakeResponse(200, {"matches": [_football_match(self.kickoff)]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"},
            }]}),
            FakeResponse(200, fixture),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=responses) as mocked:
            report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now,
            )
        self.assertEqual(mocked.call_count, 3)
        metrics = report.odds_collection_by_league["PL"]
        self.assertEqual(metrics["upcoming_fixtures"], 0)
        self.assertEqual(metrics["no_usable_kickoff"], 1)

    def test_conservative_join_rejects_ambiguous_fixture(self):
        fixture = _five_dollar_fixtures(self.kickoff)["data"][0]
        first = normalize_football_data_match(_football_match(self.kickoff), "PL")
        duplicate = {**first, "source_match_key": "external:702", "external_match_id": "702"}
        candidate, reason = _match_provider_fixture(
            fixture, [first, duplicate], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "ambiguous")

    def test_match_audit_reasons_cover_match_date_team_and_kickoff(self):
        fd = normalize_football_data_match(
            _football_match("2030-01-03T12:00:00Z"), "PL"
        )
        fixture = _five_dollar_fixtures("2030-01-03T12:00:00Z")["data"][0]
        candidate, reason, evidence = _match_provider_fixture_with_diagnostics(
            fixture, [fd], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(reason, "matched")
        self.assertTrue(evidence[0]["normalized_teams_match"])

        date_mismatch = {**fixture, "kickoff_utc": "2030-01-04T12:00:00Z"}
        self.assertEqual(
            _match_provider_fixture_with_diagnostics(
                date_mismatch, [fd], provider_source=FIVE_DOLLAR_SOURCE
            )[1],
            "no_same_date_candidate",
        )
        name_mismatch = {
            **fixture,
            "teams": {"home": {"name": "Other Team"}, "away": {"name": "Away"}},
        }
        self.assertEqual(
            _match_provider_fixture_with_diagnostics(
                name_mismatch, [fd], provider_source=FIVE_DOLLAR_SOURCE
            )[1],
            "team_names_mismatch",
        )
        kickoff_mismatch = {**fixture, "kickoff_utc": "2030-01-03T12:31:00Z"}
        candidate, reason, evidence = _match_provider_fixture_with_diagnostics(
            kickoff_mismatch, [fd], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "kickoff_difference_over_30_minutes")
        self.assertEqual(evidence[0]["kickoff_delta_minutes"], 31.0)

    def test_all_verified_five_dollar_aliases_are_exact_and_source_scoped(self):
        canonical_names = {
            "ipswich": "Ipswich Town", "nottm forest": "Nottingham Forest",
            "man city": "Manchester City", "coventry": "Coventry City",
            "tottenham": "Tottenham Hotspur", "newcastle": "Newcastle United",
            "hull": "Hull City", "brighton": "Brighton & Hove Albion",
            "man utd": "Manchester United", "leeds": "Leeds United",
        }
        self.assertTrue(set(canonical_names).issubset(FIVE_DOLLAR_TEAM_ALIASES))
        for fixture_name, canonical in canonical_names.items():
            with self.subTest(alias=fixture_name):
                kickoff = "2030-01-03T12:00:00Z"
                fd_raw = _football_match(kickoff)
                fd_raw["homeTeam"]["name"] = canonical + " FC"
                fd_match = normalize_football_data_match(fd_raw, "PL")
                fixture = _five_dollar_fixtures(kickoff)["data"][0]
                fixture["teams"]["home"]["name"] = fixture_name
                candidate, reason = _match_provider_fixture(
                    fixture, [fd_match], provider_source=FIVE_DOLLAR_SOURCE
                )
                self.assertIsNotNone(candidate)
                self.assertEqual(reason, "matched")
                # The same name is not rewritten when the provider isn't 5Dollar.
                self.assertNotEqual(
                    _provider_team_key(fixture_name, source=FOOTBALL_DATA_SOURCE),
                    _provider_team_key(fixture_name, source=FIVE_DOLLAR_SOURCE),
                )

    def test_audit_verified_league_names_are_canonicalized_only_when_needed(self):
        # Each pair comes from the persisted audit candidates. A pair whose
        # existing normalization already agrees must not get a redundant alias.
        pairs = {
            "La Liga": [
                ("Levante", "Levante UD"), ("CD Alaves", "Deportivo Alavés"),
                ("Malaga", "Málaga CF"), ("Rayo Vallecano", "Rayo Vallecano de Madrid"),
                ("Racing Santander", "Real Racing Club de Santander"),
                ("Espanyol", "RCD Espanyol de Barcelona"),
                ("Atletico Madrid", "Club Atlético de Madrid"),
                ("Deportivo A Coruna", "RC Deportivo La Coruña"),
                ("Celta Vigo", "RC Celta de Vigo"), ("Real Betis", "Real Betis Balompié"),
                ("Real Sociedad", "Real Sociedad de Fútbol"), ("Sevilla", "Sevilla FC"),
                ("Osasuna", "CA Osasuna"),
            ],
            "Bundesliga": [
                ("Borussia M'gladbach", "Borussia Mönchengladbach"),
                ("Augsburg", "FC Augsburg"), ("Union Berlin", "1. FC Union Berlin"),
                ("Cologne", "1. FC Köln"), ("Schalke", "FC Schalke 04"),
                ("Mainz", "1. FSV Mainz 05"), ("Werder Bremen", "SV Werder Bremen"),
                ("Paderborn", "SC Paderborn 07"), ("Hamburg", "Hamburger SV"),
                ("Elversberg", "SV 07 Elversberg"),
                ("TSG Hoffenheim", "TSG 1899 Hoffenheim"),
                ("Bayer Leverkusen", "Bayer 04 Leverkusen"),
                ("Bayern Munich", "FC Bayern München"), ("SC Freiburg", "SC Freiburg"),
            ],
            "Serie A": [
                ("Torino", "Torino FC"), ("Monza", "AC Monza"),
                ("Cagliari", "Cagliari Calcio"), ("Bologna", "Bologna FC 1909"),
                ("Como", "Como 1907"), ("Sassuolo", "US Sassuolo Calcio"),
                ("Napoli", "SSC Napoli"), ("Roma", "AS Roma"), ("Lazio", "SS Lazio"),
                ("Parma", "Parma Calcio 1913"),
                ("Inter Milan", "FC Internazionale Milano"),
                ("Fiorentina", "ACF Fiorentina"), ("Atalanta", "Atalanta BC"),
                ("Frosinone", "Frosinone Calcio"), ("Genoa", "Genoa CFC"),
                ("Venezia", "Venezia FC"), ("Lecce", "US Lecce"),
                ("Juventus", "Juventus FC"), ("Udinese", "Udinese Calcio"),
                ("AC Milan", "AC Milan"),
            ],
            "Ligue 1": [
                ("Brest", "Stade Brestois 29"), ("Nice", "OGC Nice"),
                ("Toulouse", "Toulouse FC"), ("Troyes", "ES Troyes AC"),
                ("Lens", "Racing Club de Lens"), ("Paris FC", "Paris FC"),
                ("Monaco", "AS Monaco FC"), ("Lille", "Lille OSC"),
                ("Angers", "Angers SCO"), ("Lorient", "FC Lorient"),
                ("Auxerre", "AJ Auxerre"), ("Le Mans", "Le Mans FC"),
                ("Marseille", "Olympique de Marseille"), ("Le Havre", "Le Havre AC"),
                ("Rennes", "Stade Rennais FC 1901"),
                ("Strasbourg", "RC Strasbourg Alsace"),
                ("PSG", "Paris Saint-Germain FC"), ("Lyon", "Olympique Lyonnais"),
            ],
        }
        self.assertEqual(set(pairs), {"La Liga", "Bundesliga", "Serie A", "Ligue 1"})
        for league, team_pairs in pairs.items():
            with self.subTest(league=league):
                for source_name, target_name in team_pairs:
                    normalized_source = _norm_name(source_name)
                    normalized_target = _norm_name(target_name)
                    key = _provider_team_key(source_name, source=FIVE_DOLLAR_SOURCE)
                    self.assertEqual(key, normalized_target, (league, source_name))
                    if normalized_source == normalized_target:
                        self.assertNotIn(normalized_source, FIVE_DOLLAR_TEAM_ALIASES)
                    else:
                        self.assertEqual(FIVE_DOLLAR_TEAM_ALIASES[normalized_source], normalized_target)
                    # The same raw string is never rewritten on the other source.
                    self.assertEqual(
                        _provider_team_key(source_name, source=FOOTBALL_DATA_SOURCE),
                        normalized_source,
                    )

    def test_alias_matching_keeps_home_away_order_and_date_kickoff_guards(self):
        kickoff = "2030-01-03T12:00:00Z"
        fixture = _five_dollar_fixtures(kickoff)["data"][0]
        fixture["teams"]["home"]["name"] = "Brest"
        fixture["teams"]["away"]["name"] = "Nice"
        fd_raw = _football_match(kickoff)
        fd_raw["homeTeam"]["name"] = "Stade Brestois 29"
        fd_raw["awayTeam"]["name"] = "OGC Nice"
        fd = normalize_football_data_match(fd_raw, "FL1")
        candidate, reason = _match_provider_fixture(
            fixture, [fd], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(reason, "matched")

        reversed_fd = {**fd, "home_team": fd["away_team"], "away_team": fd["home_team"]}
        candidate, reason = _match_provider_fixture(
            fixture, [reversed_fd], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "unmatched")

        wrong_date = {**fd, "match_date": "2030-01-04"}
        candidate, reason = _match_provider_fixture(
            fixture, [wrong_date], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "unmatched")

        late_kickoff = {**fd, "kickoff_at": "2030-01-03T12:31:00Z"}
        candidate, reason = _match_provider_fixture(
            fixture, [late_kickoff], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "unmatched")

    def test_existing_normalizer_still_handles_accents_punctuation_and_club_suffixes(self):
        self.assertEqual(_norm_name("AFC-Bournemouth"), _norm_name("Bournemouth FC"))
        self.assertEqual(_norm_name("Atlético CF"), "atletico")

    def test_unknown_names_stay_unmatched_and_alias_does_not_resolve_ambiguity(self):
        kickoff = "2030-01-03T12:00:00Z"
        fixture = _five_dollar_fixtures(kickoff)["data"][0]
        fixture["teams"]["home"]["name"] = "Unknown Club"
        fd = normalize_football_data_match(_football_match(kickoff), "PL")
        candidate, reason = _match_provider_fixture(
            fixture, [fd], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "unmatched")

        fixture["teams"]["home"]["name"] = "Ipswich"
        fd_raw = _football_match(kickoff)
        fd_raw["homeTeam"]["name"] = "Ipswich Town FC"
        canonical = normalize_football_data_match(fd_raw, "PL")
        second = {**canonical, "source_match_key": "external:702", "external_match_id": "702"}
        candidate, reason = _match_provider_fixture(
            fixture, [canonical, second], provider_source=FIVE_DOLLAR_SOURCE
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "ambiguous")

    def test_persisted_audit_records_matched_unmatched_and_ambiguous(self):
        config = SyncConfig(
            database_path=self.database,
            leagues=("PL",),
            date_from=self.now.date(),
            date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=1,
        )
        fixtures = _five_dollar_fixtures(self.kickoff)["data"]
        fixtures[0]["teams"]["home"]["name"] = "Ipswich"
        fixtures.extend([
            {
                "id": 9902,
                "league": {"id": 555, "name": "England Premier League"},
                "teams": {"home": {"name": "Unknown FC"}, "away": {"name": "Away"}},
                "kickoff_utc": self.kickoff,
                "status": "scheduled",
            },
            {
                "id": 9903,
                "league": {"id": 555, "name": "England Premier League"},
                "teams": {"home": {"name": "No Time"}, "away": {"name": "Away"}},
                "kickoff_utc": None,
                "status": "scheduled",
            },
        ])
        responses = [
            FakeResponse(200, {"matches": [{
                **_football_match(self.kickoff),
                "homeTeam": {"name": "Ipswich Town FC"},
            }]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"}
            }]}),
            FakeResponse(200, {"success": 1, "data": fixtures}),
            FakeResponse(200, _odds_payload()),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=responses):
            report = run_sync(
                config, football_data_token="fd-test", five_dollar_token="odds-test", now=self.now
            )
        with DataStore(self.database) as store:
            rows = store.connection.execute(
                "SELECT * FROM fixture_match_audit ORDER BY external_fixture_id"
            ).fetchall()
            self.assertEqual(len(rows), 3)
            by_id = {row["external_fixture_id"]: row for row in rows}
            self.assertEqual(by_id["9901"]["outcome"], "matched")
            self.assertEqual(by_id["9901"]["reason"], "matched")
            self.assertIsNotNone(by_id["9901"]["selected_match_id"])
            self.assertEqual(by_id["9902"]["outcome"], "unmatched")
            self.assertEqual(by_id["9902"]["reason"], "team_names_mismatch")
            self.assertEqual(by_id["9903"]["outcome"], "rejected")
            self.assertEqual(by_id["9903"]["reason"], "invalid_or_missing_kickoff")
            self.assertEqual(by_id["9901"]["sync_run_id"], report.sync_run_id)
            self.assertEqual(by_id["9901"]["data_source"], FIVE_DOLLAR_SOURCE)
            self.assertIn('"name":"Ipswich"', by_id["9901"]["raw_home_team_json"])
            self.assertIn('"provider_home_match_key":"ipswich town"', by_id["9901"]["candidate_matches_json"])
            self.assertIn('"normalized_teams_match":true', by_id["9901"]["candidate_matches_json"])

        ambiguous_db = Path(self.temp_dir.name) / "ambiguous.sqlite"
        ambiguous_config = SyncConfig(
            database_path=ambiguous_db,
            leagues=("PL",),
            date_from=self.now.date(),
            date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=1,
        )
        second_match = {**_football_match(self.kickoff), "id": 702}
        ambiguous_responses = [
            FakeResponse(200, {"matches": [_football_match(self.kickoff), second_match]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"}
            }]}),
            FakeResponse(200, _five_dollar_fixtures(self.kickoff)),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=ambiguous_responses):
            run_sync(
                ambiguous_config,
                football_data_token="fd-test",
                five_dollar_token="odds-test",
                now=self.now,
            )
        with DataStore(ambiguous_db) as store:
            audit = store.connection.execute(
                "SELECT outcome,reason,selected_match_id,candidate_matches_json FROM fixture_match_audit"
            ).fetchone()
            self.assertEqual(audit["outcome"], "ambiguous")
            self.assertEqual(audit["reason"], "multiple_candidates")
            self.assertIsNone(audit["selected_match_id"])
            self.assertEqual(audit["candidate_matches_json"].count('"same_date":true'), 2)

    def test_dry_run_is_offline_and_does_not_create_database(self):
        config = SyncConfig(
            database_path=self.database,
            leagues=("PL",),
            date_from=self.now.date(),
            date_to=self.now.date() + timedelta(days=30),
            dry_run=True,
        )
        with patch("src.sync_football_data.requests.get", side_effect=AssertionError("network called")):
            report = run_sync(config, now=self.now)
        self.assertEqual(report.api_calls, [])
        self.assertFalse(self.database.exists())

    def test_dry_run_reads_cache_read_only_and_checks_covered_range(self):
        start = self.now.date()
        end = start + timedelta(days=30)
        with DataStore(self.database) as store:
            with store.transaction():
                store.set_sync_state(
                    FOOTBALL_DATA_SOURCE,
                    "matches:PL",
                    last_success_at=self.now,
                    cursor=f"{start.isoformat()}..{end.isoformat()}",
                )
                store.set_sync_state(
                    FIVE_DOLLAR_SOURCE,
                    "odds:PL",
                    last_success_at=self.now,
                    cursor=f"{start.isoformat()}..{end.isoformat()}",
                )
        before = self.database.read_bytes()
        fresh_config = SyncConfig(
            database_path=self.database,
            leagues=("PL",),
            date_from=start,
            date_to=end,
            dry_run=True,
        )
        plan = build_dry_run_plan(fresh_config, now=self.now + timedelta(minutes=1))
        self.assertEqual(plan["total_calls_upper_bound"], 0)
        extended_config = SyncConfig(
            database_path=self.database,
            leagues=("PL",),
            date_from=start,
            date_to=end + timedelta(days=1),
            dry_run=True,
        )
        extended = build_dry_run_plan(extended_config, now=self.now + timedelta(minutes=1))
        self.assertEqual(extended["football_data_calls"], 1)
        self.assertEqual(extended["five_dollar_fixture_lists"], 1)
        self.assertEqual(self.database.read_bytes(), before)

    def test_fixture_upsert_odds_and_cache_avoid_repeat_calls(self):
        config = SyncConfig(
            database_path=self.database,
            leagues=("PL",),
            date_from=self.now.date(),
            date_to=self.now.date() + timedelta(days=30),
            max_odds_calls=1,
        )
        responses = [
            FakeResponse(200, {"matches": [_football_match(self.kickoff)]}),
            FakeResponse(200, {"success": 1, "data": [{
                "id": 555, "name": "England Premier League", "country": {"name": "England"}
            }]}),
            FakeResponse(200, _five_dollar_fixtures(self.kickoff)),
            FakeResponse(200, _odds_payload()),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=responses) as mocked:
            report = run_sync(
                config,
                football_data_token="fd-secret-test",
                five_dollar_token="odds-secret-test",
                now=self.now,
            )
            self.assertEqual(mocked.call_count, 4)
            self.assertNotIn("fd-secret-test", repr(report))
            self.assertNotIn("odds-secret-test", repr(report))
        self.assertEqual(report.matches_inserted, 1)
        self.assertEqual(report.odds_added, 2)
        self.assertEqual(len(report.api_calls), 4)
        with DataStore(self.database) as store:
            match = store.connection.execute("SELECT * FROM matches").fetchone()
            self.assertEqual(match["external_match_id"], "701")
            self.assertEqual(match["data_source"], FOOTBALL_DATA_SOURCE)
            markets = [row[0] for row in store.connection.execute(
                "SELECT market FROM odds_observations ORDER BY market"
            )]
            self.assertEqual(markets, ["1X2:closing", "1X2:opening"])
            self.assertTrue(all(row[0] == FIVE_DOLLAR_SOURCE for row in store.connection.execute(
                "SELECT data_source FROM odds_observations"
            )))

        # Both source-level TTLs are fresh: rerun performs no HTTP calls.
        with patch("src.sync_football_data.requests.get", side_effect=AssertionError("unexpected call")) as mocked:
            second = run_sync(
                config,
                football_data_token="fd-secret-test",
                five_dollar_token="odds-secret-test",
                now=self.now + timedelta(minutes=5),
            )
            mocked.assert_not_called()
        self.assertEqual(second.api_calls, [])
        self.assertEqual(second.odds_collection_by_league["PL"]["upcoming_fixtures"], 1)
        self.assertEqual(second.odds_collection_by_league["PL"]["opening_reused"], 1)

    def test_finished_match_metadata_updates_in_place(self):
        scheduled = normalize_football_data_match(_football_match(self.kickoff), "PL")
        finished = normalize_football_data_match(
            _football_match(self.kickoff, status="FINISHED", home_score=2, away_score=1), "PL"
        )
        with DataStore(self.database) as store:
            with store.transaction():
                match_id, status1 = store.upsert_match(scheduled)
            with store.transaction():
                updated_id, status2 = store.upsert_match(finished)
            self.assertEqual(match_id, updated_id)
            self.assertEqual(status1, "inserted")
            self.assertEqual(status2, "updated")
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 1)
            row = store.connection.execute("SELECT match_status,result,home_score FROM matches").fetchone()
            self.assertEqual(tuple(row), ("FINISHED", "H", 2))

    def test_reimport_of_same_external_match_id_is_idempotent(self):
        match = normalize_football_data_match(_football_match(self.kickoff), "PL")
        with DataStore(self.database) as store:
            with store.transaction():
                first_id, first_status = store.upsert_match(match)
            with store.transaction():
                second_id, second_status = store.upsert_match(match)
            self.assertEqual(first_id, second_id)
            self.assertEqual(first_status, "inserted")
            self.assertEqual(second_status, "unchanged")
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 1)

    def test_failure_for_one_competition_does_not_block_another(self):
        config = SyncConfig(
            database_path=self.database,
            leagues=("PD", "PL"),
            date_from=self.now.date(),
            date_to=self.now.date() + timedelta(days=30),
            skip_odds=True,
        )
        responses = [
            FakeResponse(500, {"message": "temporary outage"}),
            FakeResponse(200, {"matches": [_football_match(self.kickoff)]}),
        ]
        with patch("src.sync_football_data.requests.get", side_effect=responses):
            report = run_sync(config, football_data_token="fd-token", now=self.now)
        self.assertEqual(len(report.failures), 1)
        with DataStore(self.database) as store:
            saved = store.connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            self.assertEqual(saved, 1)
            pl_state = store.connection.execute(
                "SELECT 1 FROM sync_state WHERE data_source=? AND dataset='matches:PL'",
                (FOOTBALL_DATA_SOURCE,),
            ).fetchone()
            failed_state = store.connection.execute(
                "SELECT 1 FROM sync_state WHERE data_source=? AND dataset='matches:PD'",
                (FOOTBALL_DATA_SOURCE,),
            ).fetchone()
        self.assertIsNotNone(pl_state)
        self.assertIsNone(failed_state)

    def test_rate_limit_is_reported_without_retry_or_success_state(self):
        config = SyncConfig(
            database_path=self.database,
            leagues=("PL",),
            date_from=self.now.date(),
            date_to=self.now.date() + timedelta(days=30),
            skip_odds=True,
        )
        with patch("src.sync_football_data.requests.get", return_value=FakeResponse(
            429, {"message": "slow down"}, {"Retry-After": "60"}
        )) as mocked:
            report = run_sync(config, football_data_token="fd-token", now=self.now)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(len(report.api_calls), 1)
        self.assertIn("HTTP 429", report.failures[0])
        with DataStore(self.database) as store:
            state = store.connection.execute("SELECT COUNT(*) FROM sync_state").fetchone()[0]
        self.assertEqual(state, 0)

    def test_api_error_redacts_token_and_does_not_retry(self):
        secret = "unit-test-token"
        report = SyncReport(self.now.date(), self.now.date() + timedelta(days=1))
        response = FakeResponse(401, {"error": {"message": f"bad key {secret}"}})
        with patch("src.sync_football_data.requests.get", return_value=response) as mocked:
            with self.assertRaises(SyncError) as caught:
                _request_json(
                    url="https://example.invalid/endpoint",
                    source="test",
                    token=secret,
                    headers={"Authorization": f"Bearer {secret}"},
                    params={},
                    report=report,
                    label="auth-test",
                )
        self.assertEqual(mocked.call_count, 1)
        self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(len(report.api_calls), 1)


if __name__ == "__main__":
    unittest.main()
