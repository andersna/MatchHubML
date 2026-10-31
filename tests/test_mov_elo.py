from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from build_elo_features_asof_day_v1 import (  # noqa: E402
    INITIAL_ELO,
    MOV_ELO_COLUMNS,
    MOV_SEASON_REGRESSION,
    build_elo_features_asof_day,
    build_mov_elo_features,
    margin_multiplier,
    replay_form_ratings,
)


def _fixtures(rows):
    return pd.DataFrame(
        [
            {"date": date, "time": None, "league": "L", "home_team": home, "away_team": away,
             "home_goals": hg, "away_goals": ag, "season": season}
            for date, home, away, hg, ag, season in rows
        ]
    )


def _home_elo_after(first_match, column="home_elo_mov_k32"):
    """A's MoV Elo going into a second match, after ``first_match`` (A at home vs B)."""
    hg, ag = first_match
    features = build_mov_elo_features(_fixtures([
        ("2020-01-01", "A", "B", hg, ag, "2019-20"),
        ("2020-01-08", "A", "C", 0, 0, "2019-20"),
    ]))
    return features.loc[1, column]


class MarginMultiplierTests(unittest.TestCase):
    def test_draw_and_one_goal_win_between_equals_are_neutral(self):
        self.assertEqual(margin_multiplier(0, 200), 1.0)
        self.assertAlmostEqual(margin_multiplier(1, 0), 1.0)

    def test_bigger_margins_count_more(self):
        self.assertGreater(margin_multiplier(3, 0), margin_multiplier(2, 0))
        self.assertGreater(margin_multiplier(2, 0), margin_multiplier(1, 0))

    def test_underdog_margins_count_more_than_favourite_margins(self):
        self.assertGreater(margin_multiplier(2, -200), margin_multiplier(2, 200))


class MovEloTests(unittest.TestCase):
    def test_bigger_win_earns_more_than_narrow_win(self):
        self.assertGreater(_home_elo_after((4, 0)), _home_elo_after((1, 0)))
        self.assertGreater(_home_elo_after((1, 0)), INITIAL_ELO)

    def test_win_bonus_is_zero_sum(self):
        features = build_mov_elo_features(_fixtures([
            ("2020-01-01", "A", "B", 3, 0, "2019-20"),
            ("2020-01-08", "A", "B", 0, 0, "2019-20"),
        ]))
        row = features.loc[1]
        self.assertAlmostEqual(row["home_elo_mov_k32"] + row["away_elo_mov_k32"], 2 * INITIAL_ELO)

    def test_same_day_result_does_not_leak_into_same_day_features(self):
        rows = [
            ("2020-01-01", "A", "B", 1, 0, "2019-20"),
            ("2020-01-01", "C", "D", 0, 1, "2019-20"),
        ]
        changed = [("2020-01-01", "A", "B", 5, 0, "2019-20"), rows[1]]
        pd.testing.assert_frame_equal(
            build_mov_elo_features(_fixtures(rows))[MOV_ELO_COLUMNS],
            build_mov_elo_features(_fixtures(changed))[MOV_ELO_COLUMNS],
        )

    def test_new_season_regresses_toward_initial_rating_only_in_movr(self):
        features = build_mov_elo_features(_fixtures([
            ("2020-05-01", "A", "B", 3, 0, "2019-20"),
            ("2020-08-20", "A", "B", 0, 0, "2020-21"),
        ]))
        row = features.loc[1]
        gained = row["home_elo_mov_k32"] - INITIAL_ELO
        self.assertAlmostEqual(
            row["home_elo_movr_k32"] - INITIAL_ELO, gained * (1 - MOV_SEASON_REGRESSION)
        )

    def test_form_rating_weights_the_current_season_more_than_elo(self):
        # A was much better last season; this season B beats A.
        fixtures = _fixtures([
            ("2020-01-01", "A", "B", 4, 0, "2019-20"),
            ("2020-01-08", "B", "A", 0, 4, "2019-20"),
            ("2020-01-15", "A", "B", 4, 0, "2019-20"),
            ("2020-08-20", "A", "B", 0, 2, "2020-21"),
        ])
        ratings = replay_form_ratings(fixtures)
        self.assertGreater(ratings[("L", "B")], ratings[("L", "A")])
        self.assertAlmostEqual(ratings[("L", "A")] + ratings[("L", "B")], 2 * INITIAL_ELO)
        elo = build_elo_features_asof_day(pd.concat([fixtures, fixtures.tail(1).assign(date="2020-08-27")]))
        self.assertGreater(elo.iloc[-1]["home_elo_k48"], elo.iloc[-1]["away_elo_k48"])

    def test_existing_elo_columns_are_unchanged(self):
        sample = _fixtures([
            ("2020-01-01", "A", "B", 1, 0, "2019-20"),
            ("2020-01-08", "A", "B", 4, 0, "2019-20"),
            ("2020-01-15", "B", "A", 0, 0, "2019-20"),
        ])
        features = build_elo_features_asof_day(sample)
        # Plain Elo ignores the margin: 1-0 and 4-0 move A by the same formula.
        self.assertNotIn("home_elo_mov_k32", features.columns)
        self.assertGreater(features.loc[2, "away_elo_k32"], INITIAL_ELO)


if __name__ == "__main__":
    unittest.main()
