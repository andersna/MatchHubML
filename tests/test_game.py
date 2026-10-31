from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src import game, prediction_log
from src.prediction_log import match_identity


KICKOFF = "2026-11-07T15:00:00Z"
BEFORE = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
AFTER = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
WEEK = "2026-W45"  # Monday 2 Nov – Sunday 8 Nov 2026
MATCH = {"league": "Premier League", "match_date": "2026-11-07", "home_team": "Home FC", "away_team": "Away FC"}
OTHER = {"league": "Premier League", "match_date": "2026-11-07", "home_team": "Other FC", "away_team": "Rival FC"}
KEY = tuple(MATCH.values())
OTHER_KEY = tuple(OTHER.values())


def _bot_record(probs, *, logged, match=MATCH, source="live"):
    return {
        "source": source, "model_version": "v3", "model_artifact": "v3_artifact", "logged_at_utc": logged,
        **match, "kickoff_at": KICKOFF, "odds_observed_at_utc": "2026-11-06T08:00:00Z",
        "odds_market": "1X2:closing", "odds": (0.5, 0.3, 0.2), "probs": probs,
    }


class GameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.path, self.cache, self.log = root / "game.sqlite", root / "cache.sqlite", root / "log.sqlite"
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute(
                """CREATE TABLE matches (data_source TEXT, league TEXT, match_date TEXT, kickoff_at TEXT,
                home_team TEXT, away_team TEXT, match_status TEXT, home_score INT, away_score INT, result TEXT)"""
            )
            for match in (MATCH, OTHER):
                connection.execute(
                    "INSERT INTO matches VALUES ('football-data.org', ?, ?, ?, ?, ?, 'TIMED', NULL, NULL, NULL)",
                    (match["league"], match["match_date"], KICKOFF, match["home_team"], match["away_team"]),
                )
        self.results = {KEY: {"result": "H", "home_score": 2, "away_score": 0},
                        OTHER_KEY: {"result": "D", "home_score": 1, "away_score": 1}}

    def results_fn(self, keys):
        # Like api.game_results: matched on match_identity, so a moved match finds its result.
        by_identity = {match_identity(*key): result for key, result in self.results.items()}
        return {key: by_identity[match_identity(*key)] for key in keys if match_identity(*key) in by_identity}

    def move(self, match, match_date, kickoff):
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute(
                "UPDATE matches SET match_date = ?, kickoff_at = ? WHERE home_team = ? AND away_team = ?",
                (match_date, kickoff, match["home_team"], match["away_team"]),
            )

    def player(self, name):
        token = game.register(name, "secret123", path=self.path, now=BEFORE)
        return game.user_for_token(token, path=self.path, now=BEFORE)["user_id"]

    def enter(self, user_id, pick, match=MATCH, now=BEFORE, score=None):
        entry = {**match, "pick": pick}
        if score is not None:
            entry.update(home_goals=score[0], away_goals=score[1])
        return game.save_entries(user_id, [entry], cache_path=self.cache, path=self.path, now=now)

    def round(self, user_id, now):
        return game.round_view(user_id, WEEK, cache_path=self.cache, log_path=self.log,
                               results=self.results_fn, path=self.path, now=now)

    def table(self, code=game.GLOBAL_LEAGUE, now=AFTER):
        return game.league_table(code, cache_path=self.cache, log_path=self.log, results=self.results_fn,
                                 week=WEEK, path=self.path, now=now)

    # Scoring
    def test_points_one_for_the_result_three_with_the_exact_score(self):
        final = {"result": "H", "home_score": 2, "away_score": 0}
        self.assertEqual(game.points({"pick": "H", "home_goals": None, "away_goals": None}, final), 1)
        self.assertEqual(game.points({"pick": "H", "home_goals": 2, "away_goals": 0}, final), 3)
        self.assertEqual(game.points({"pick": "H", "home_goals": 1, "away_goals": 0}, final), 1)
        self.assertEqual(game.points({"pick": "D", "home_goals": 0, "away_goals": 0}, final), 0)

    def test_bot_score_agrees_with_its_pick_and_its_confidence(self):
        for probs in [(0.5, 0.27, 0.23), (0.2, 0.25, 0.55), (0.34, 0.35, 0.31), (0.8, 0.13, 0.07)]:
            prediction = game.bot_prediction(probs)
            self.assertEqual(game.outcome_of(prediction["home_goals"], prediction["away_goals"]), prediction["pick"])
        strong, narrow = game.bot_score((0.8, 0.13, 0.07)), game.bot_score((0.45, 0.28, 0.27))
        self.assertGreaterEqual(strong[0] - strong[1], narrow[0] - narrow[1])

    def test_bot_score_follows_the_matchs_goal_level(self):
        probs = (0.55, 0.25, 0.2)
        tight = game.bot_score(probs, game.match_total(0.6, 0.5))
        open_ = game.bot_score(probs, game.match_total(2.6, 1.9))
        self.assertLess(sum(tight), sum(open_))
        for score in (tight, open_):
            self.assertEqual(game.outcome_of(*score), "H")  # the score always agrees with the pick
        self.assertIsNone(game.match_total(None, 1.2))

    def test_bot_uses_logged_expected_goals(self):
        record = {**_bot_record((0.6, 0.25, 0.15), logged="2026-11-07T10:00:00Z"), "expected_goals": (2.4, 1.6)}
        prediction_log.log_predictions([record], path=self.log)
        self.assertEqual(game.bot_entries(self.log, {KEY: KICKOFF}), {KEY: ((0.6, 0.25, 0.15), game.match_total(2.4, 1.6))})

    def test_bot_picks_its_most_likely_outcome(self):
        self.assertEqual(game.bot_pick((0.3, 0.25, 0.45)), "A")
        self.assertEqual(game.bot_pick((0.4, 0.2, 0.4)), "H")  # tie: earlier outcome

    # Accounts
    def test_register_login_and_bad_password(self):
        self.player("Anna")
        token = game.login("anna", "secret123", path=self.path, now=BEFORE)  # case-insensitive
        self.assertEqual(game.user_for_token(token, path=self.path, now=BEFORE)["username"], "Anna")
        with self.assertRaises(game.GameError):
            game.login("Anna", "wrong", path=self.path)
        with self.assertRaises(game.GameError):
            game.register("ANNA", "secret123", path=self.path)

    def test_bot_name_is_reserved(self):
        with self.assertRaises(game.GameError):
            game.register("MatchHub V3", "secret123", path=self.path)

    def test_logout_ends_the_session(self):
        token = game.register("Anna", "secret123", path=self.path, now=BEFORE)
        game.logout(token, path=self.path)
        self.assertIsNone(game.user_for_token(token, path=self.path, now=BEFORE))

    # Entries
    def test_entries_lock_at_kickoff(self):
        anna = self.player("Anna")
        self.assertEqual(self.enter(anna, "H")["saved"], 1)
        late = self.enter(anna, "A", now=AFTER)
        self.assertEqual(late["saved"], 0)
        self.assertIn("låst", late["rejected"][0]["reason"])
        self.assertEqual(self.round(anna, AFTER)["matches"][0]["my_entry"]["pick"], "H")

    def test_resubmitting_before_kickoff_replaces_the_entry(self):
        anna = self.player("Anna")
        self.enter(anna, "H")
        self.enter(anna, "d", score=(1, 1))
        self.assertEqual(self.round(anna, BEFORE)["matches"][0]["my_entry"],
                         {"pick": "D", "home_goals": 1, "away_goals": 1})

    def test_pick_must_be_1_x_or_2_and_match_must_exist(self):
        anna = self.player("Anna")
        self.assertIn("1, X eller 2", self.enter(anna, "Z")["rejected"][0]["reason"])
        unknown = {**MATCH, "home_team": "Nobody FC"}
        self.assertIn("findes ikke", self.enter(anna, "H", match=unknown)["rejected"][0]["reason"])

    def test_score_must_match_the_pick_and_be_complete(self):
        anna = self.player("Anna")
        self.assertIn("passer ikke", self.enter(anna, "H", score=(0, 1))["rejected"][0]["reason"])
        self.assertIn("begge hold", self.enter(anna, "H", score=(2, None))["rejected"][0]["reason"])
        self.assertIn("mellem 0", self.enter(anna, "H", score=(40, 0))["rejected"][0]["reason"])
        self.assertEqual(self.enter(anna, "H", score=(2, 0))["saved"], 1)

    # The bot
    def test_bot_is_hidden_before_kickoff_and_shown_after(self):
        anna = self.player("Anna")
        prediction_log.log_predictions([_bot_record((0.6, 0.25, 0.15), logged="2026-11-07T10:00:00Z")], path=self.log)
        before = self.round(anna, BEFORE)["matches"][0]
        self.assertFalse(before["locked"])
        self.assertNotIn("bot_entry", before)
        after = self.round(anna, AFTER)["matches"][0]
        bot = after["bot_entry"]
        self.assertEqual((bot["pick"], bot["probability"]), ("H", 0.6))
        self.assertEqual((bot["home_goals"], bot["away_goals"]), game.bot_score((0.6, 0.25, 0.15)))
        self.assertEqual(after["bot_points"], game.points(bot, self.results[KEY]))

    def test_bot_uses_only_live_predictions_logged_before_kickoff(self):
        prediction_log.log_predictions([
            _bot_record((0.4, 0.3, 0.3), logged="2026-11-07T10:00:00Z"),
            _bot_record((0.9, 0.05, 0.05), logged="2026-11-07T16:00:00Z"),  # after kickoff
            _bot_record((0.8, 0.1, 0.1), logged="2026-11-07T11:00:00Z", source="backfill"),
        ], path=self.log)
        self.assertEqual(game.bot_entries(self.log, {KEY: KICKOFF}), {KEY: ((0.4, 0.3, 0.3), None)})

    # Standings
    def test_table_uses_the_league_slate_and_missing_entries_score_zero(self):
        anna, ben = self.player("Anna"), self.player("Ben")
        self.enter(anna, "H", score=(2, 0))  # exact: 3 points
        self.enter(anna, "D", match=OTHER)  # right result: 1 point
        self.enter(ben, "A")  # wrong; Ben skips OTHER
        prediction_log.log_predictions([_bot_record((0.5, 0.3, 0.2), logged="2026-11-07T10:00:00Z")], path=self.log)
        table = self.table()
        rows = {row["player"]: row for row in table["rows"]}
        self.assertEqual(table["matches"], 2)
        self.assertEqual(rows["Anna"]["total_points"], 4)
        self.assertEqual(rows["Anna"]["hit_rate"], 1.0)
        self.assertEqual(rows["Anna"]["exact"], 1)
        self.assertEqual(rows["Ben"]["total_points"], 0)
        self.assertEqual(rows["Ben"]["predicted"], 1)
        bot = game.bot_prediction((0.5, 0.3, 0.2))  # H on MATCH, no entry for OTHER
        self.assertEqual(rows[game.BOT_NAME]["total_points"], game.points(bot, self.results[KEY]))
        self.assertTrue(rows[game.BOT_NAME]["is_bot"])
        self.assertEqual([row["rank"] for row in table["rows"]], [1, 2, 3])
        self.assertEqual(rows["Anna"]["week_points"], rows["Anna"]["total_points"])

    def test_unfinished_matches_are_not_scored(self):
        anna = self.player("Anna")
        self.enter(anna, "H")
        self.results = {}
        self.assertEqual(self.table()["matches"], 0)
        self.assertEqual(self.table(now=BEFORE)["matches"], 0)  # not kicked off

    def test_private_league_contains_only_its_members(self):
        anna, ben = self.player("Anna"), self.player("Ben")
        league = game.create_league(anna, "Kontoret", path=self.path)
        self.assertEqual(len(league["code"]), game.INVITE_LENGTH)
        self.assertEqual({r["player"] for r in self.table(league["code"])["rows"]}, {"Anna", game.BOT_NAME})
        game.join_league(ben, league["code"].lower(), path=self.path)
        self.assertEqual({r["player"] for r in self.table(league["code"])["rows"]}, {"Anna", "Ben", game.BOT_NAME})
        self.assertEqual(game.user_leagues(ben, path=self.path)[0]["members"], 2)
        with self.assertRaises(game.GameError):
            game.join_league(ben, "NOPE00", path=self.path)

    # Rescheduled matches
    def test_pick_survives_when_the_match_is_moved(self):
        anna = self.player("Anna")
        self.enter(anna, "H", score=(2, 0))
        self.move(MATCH, "2026-11-08", "2026-11-08T15:00:00Z")
        view = self.round(anna, BEFORE)
        moved = next(m for m in view["matches"] if m["home_team"] == MATCH["home_team"])
        self.assertEqual(moved["match_date"], "2026-11-08")
        self.assertEqual(moved["my_entry"], {"pick": "H", "home_goals": 2, "away_goals": 0})
        rows = {r["player"]: r for r in self.table(now=datetime(2026, 11, 8, 18, 0, tzinfo=timezone.utc))["rows"]}
        self.assertEqual(rows["Anna"]["total_points"], 3)

    def test_new_pick_after_a_move_replaces_the_old_one(self):
        anna = self.player("Anna")
        self.enter(anna, "H")
        self.move(MATCH, "2026-11-08", "2026-11-08T15:00:00Z")
        self.enter(anna, "A", match={**MATCH, "match_date": "2026-11-08"})
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT match_date, pick FROM entries").fetchall(), [("2026-11-08", "A")])

    def test_bot_ignores_predictions_after_a_kickoff_moved_earlier(self):
        anna = self.player("Anna")
        self.enter(anna, "H")
        # Logged for 7 Nov 15:00, but the match was moved to 6 Nov 19:00 and V3 predicted it again at 20:00.
        prediction_log.log_predictions([
            _bot_record((0.3, 0.3, 0.4), logged="2026-11-06T10:00:00Z"),
            _bot_record((0.9, 0.05, 0.05), logged="2026-11-06T20:00:00Z"),
        ], path=self.log)
        self.move(MATCH, "2026-11-06", "2026-11-06T19:00:00Z")
        moved_key = (MATCH["league"], "2026-11-06", MATCH["home_team"], MATCH["away_team"])
        self.assertEqual(game.bot_entries(self.log, {moved_key: "2026-11-06T19:00:00Z"}), {moved_key: ((0.3, 0.3, 0.4), None)})

    # Older databases
    def _old_entries_table(self, columns):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(f"""CREATE TABLE entries (user_id INTEGER, league TEXT, match_date TEXT,
                home_team TEXT, away_team TEXT, kickoff_at TEXT, {columns}, submitted_at_utc TEXT,
                PRIMARY KEY (user_id, league, match_date, home_team, away_team))""")

    def test_slider_entries_become_the_outcome_rated_highest(self):
        self._old_entries_table("prob_home REAL, prob_draw REAL, prob_away REAL")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("INSERT INTO entries VALUES (1, ?, ?, ?, ?, ?, 0.2, 0.5, 0.3, 'x')", (*KEY, KICKOFF))
        anna = self.player("Anna")
        self.assertEqual(self.round(anna, BEFORE)["matches"][0]["my_entry"],
                         {"pick": "D", "home_goals": None, "away_goals": None})
        self.assertEqual(self.enter(anna, "H", score=(1, 0))["saved"], 1)

    def test_pick_only_entries_get_score_columns(self):
        self._old_entries_table("pick TEXT")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("INSERT INTO entries VALUES (1, ?, ?, ?, ?, ?, 'A', 'x')", (*KEY, KICKOFF))
        anna = self.player("Anna")
        self.assertEqual(self.round(anna, BEFORE)["matches"][0]["my_entry"]["pick"], "A")
        self.assertEqual(self.enter(anna, "H", score=(2, 1))["saved"], 1)

    # Weeks
    def test_week_bounds_and_navigation(self):
        start, end = game.week_bounds(WEEK)
        self.assertEqual(start, datetime(2026, 11, 2, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 11, 9, tzinfo=timezone.utc))
        self.assertEqual(game.iso_week(AFTER), WEEK)
        view = self.round(self.player("Anna"), BEFORE)
        self.assertEqual((view["previous_week"], view["next_week"]), ("2026-W44", "2026-W46"))
        self.assertEqual(len(view["matches"]), 2)
        with self.assertRaises(game.GameError):
            game.week_bounds("week 45")


if __name__ == "__main__":
    unittest.main()
