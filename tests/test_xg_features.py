from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nn_preprocessing import MedianStandardPreprocessor  # noqa: E402
from predict_match import ModelBundle, predict_match  # noqa: E402
from train_final_nn_market_residual_v3 import (  # noqa: E402
    MODEL_FEATURES,
    ODDS_FEATURES,
    V3_FEATURES,
    MarketResidualEnsemble,
)
from xg_features import XG_FEATURES, XgHistory, load_xg_history  # noqa: E402


def _understat(rows):
    """rows: (date, home, away, home_xg, away_xg); all in league L and played."""
    return pd.DataFrame([
        {"date": date, "league": "L", "home_team": home, "away_team": away, "is_result": True,
         "home_xg": hxg, "away_xg": axg, "understat_id": index}
        for index, (date, home, away, hxg, axg) in enumerate(rows)
    ])


FOUR_A_MATCHES = [
    ("2020-01-01", "A", "B", 1.0, 0.5),
    ("2020-01-08", "C", "A", 0.0, 2.0),
    ("2020-01-15", "A", "D", 3.0, 1.0),
    ("2020-01-22", "B", "A", 1.0, 1.0),
]


class XgHistoryTests(unittest.TestCase):
    def test_uses_only_matches_strictly_before_the_date(self):
        history = XgHistory(_understat(FOUR_A_MATCHES))
        # Before the fourth match: A's xG for 1, 2, 3 -> mean 2.0; against 0.5, 0, 1 -> 0.5.
        state = history.team_state("L", "A", "2020-01-22")
        self.assertAlmostEqual(state[0], 2.0)
        self.assertAlmostEqual(state[1], 0.5)
        # On the day after, the fourth match counts too.
        self.assertAlmostEqual(history.team_state("L", "A", "2020-01-23")[0], 7.0 / 4)

    def test_fewer_than_three_matches_or_unknown_team_is_nan(self):
        history = XgHistory(_understat(FOUR_A_MATCHES))
        self.assertTrue(np.isnan(history.team_state("L", "A", "2020-01-15")).all())
        self.assertTrue(np.isnan(history.team_state("L", "Unknown", "2020-02-01")).all())

    def test_match_features_order_and_balance(self):
        history = XgHistory(_understat(FOUR_A_MATCHES + [
            ("2020-01-29", "B", "C", 1.0, 1.0), ("2020-02-05", "D", "B", 1.0, 1.0),
        ]))
        features = history.match_features("L", "A", "B", "2020-03-01")
        self.assertEqual(list(features), XG_FEATURES)
        expected = (features["home_xg_for_last5"] - features["home_xg_against_last5"]
                    - features["away_xg_for_last5"] + features["away_xg_against_last5"])
        self.assertAlmostEqual(features["xg_balance_diff_last5"], expected)

    def test_fixture_date_replaces_understat_date(self):
        # Understat (UTC) dates the match a day later than fixtures.csv; the
        # match must not count as history on its own fixture date + 1 lookup.
        understat = _understat(FOUR_A_MATCHES[:3] + [("2020-01-23", "B", "A", 1.0, 1.0)])
        fixtures = understat[["league", "home_team", "away_team"]].assign(
            date=["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]
        )
        history = XgHistory(understat, fixtures)
        self.assertAlmostEqual(history.team_state("L", "A", "2020-01-23")[0], 7.0 / 4)

    def test_missing_understat_file_gives_nan_features(self):
        with tempfile.TemporaryDirectory() as temp:
            history = load_xg_history(Path(temp) / "missing.csv", Path(temp) / "fixtures.csv")
        features = history.match_features("L", "A", "B", "2020-01-01")
        self.assertTrue(all(np.isnan(value) for value in features.values()))


def _bundle(feature_columns):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(50, len(feature_columns))), columns=list(feature_columns))
    preprocessor = MedianStandardPreprocessor(list(feature_columns)).fit(frame)
    model = MarketResidualEnsemble(len(feature_columns), member_count=2).eval()
    metadata = {"artifact_name": "test", "index_to_class": ["H", "D", "A"]}
    return ModelBundle(model, preprocessor, metadata, tuple(feature_columns), torch.device("cpu"))


class FortyFeaturePredictionTests(unittest.TestCase):
    def setUp(self):
        self.row = {name: 0.1 for name in MODEL_FEATURES}
        self.row.update(dict(zip(ODDS_FEATURES, (0.5, 0.3, 0.2))))
        self.row.update({name: np.nan for name in XG_FEATURES})

    def test_xg_model_accepts_missing_xg_and_old_models_ignore_xg(self):
        for columns in (V3_FEATURES, MODEL_FEATURES):
            result = predict_match(self.row, bundle=_bundle(columns))
            self.assertAlmostEqual(result["sum"], 1.0, places=9)

    def test_xg_model_requires_xg_columns(self):
        row = {name: self.row[name] for name in MODEL_FEATURES}
        with self.assertRaises(ValueError):
            predict_match(row, bundle=_bundle(V3_FEATURES))


if __name__ == "__main__":
    unittest.main()
