from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# predict_match.py uses script-local imports (as in api.py); import the same
# module names so isinstance checks see one MarketResidualEnsemble class.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from predict_match import load_model_bundle, predict_match  # noqa: E402
from train_final_nn_market_residual_v3 import (  # noqa: E402
    ARTIFACT_DIR,
    MODEL_FEATURES,
    ODDS_FEATURES,
    MarketResidualEnsemble,
)


class MarketResidualEnsembleTests(unittest.TestCase):
    def test_untrained_ensemble_returns_market_probabilities(self):
        # Zero-initialised corrections mean an untrained model is exactly the market.
        ensemble = MarketResidualEnsemble(len(MODEL_FEATURES), member_count=3).eval()
        market = torch.tensor([[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]])
        with torch.no_grad():
            probs = ensemble(torch.randn(2, len(MODEL_FEATURES)), market)
        self.assertTrue(torch.allclose(probs, market.double(), atol=1e-6))


class LatestV3DirTests(unittest.TestCase):
    def test_newest_complete_retrain_wins_and_incomplete_folders_are_ignored(self):
        import tempfile
        from predict_upcoming_match import MODEL_FILES, latest_v3_dir

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "nn_market_residual_v3"
            self.assertEqual(latest_v3_dir(base), base)
            for name, complete in (("nn_market_residual_v3_20270801", True),
                                   ("nn_market_residual_v3_20280801", True),
                                   ("nn_market_residual_v3_20290801", False)):
                folder = Path(temp) / name
                folder.mkdir()
                for file in MODEL_FILES[: None if complete else 1]:
                    (folder / file).write_text("{}")
            self.assertEqual(latest_v3_dir(base).name, "nn_market_residual_v3_20280801")


TRAINING_DATASET = ROOT / "data" / "odds_model_dataset_asof_day_v2.csv"


@unittest.skipUnless(
    (ARTIFACT_DIR / "metadata.json").is_file() and TRAINING_DATASET.is_file(),
    "v3-artefakt eller træningsdatasæt mangler (genereres lokalt, ikke i git)",
)
class SavedV3ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle(artifact_dir=ARTIFACT_DIR)
        dataset = pd.read_csv(TRAINING_DATASET, low_memory=False)
        cls.rows = dataset.tail(200).reset_index(drop=True)

    def test_loads_as_five_member_ensemble(self):
        self.assertIsInstance(self.bundle.model, MarketResidualEnsemble)
        self.assertEqual(len(self.bundle.model.members), 5)
        self.assertEqual(self.bundle.metadata["artifact_name"], "nn_market_residual_v3")

    def test_predictions_are_valid_and_stay_close_to_market(self):
        for _, row in self.rows.iterrows():
            result = predict_match(row[MODEL_FEATURES].to_dict(), bundle=self.bundle)
            probs = np.array([result["probabilities"][label]["decimal"] for label in ("H", "D", "A")])
            self.assertAlmostEqual(probs.sum(), 1.0, places=9)
            # A market correction, not a replacement: shifts are small.
            market = row[ODDS_FEATURES].to_numpy(dtype=np.float64)
            self.assertLess(np.abs(probs - market).max(), 0.10)

    def test_model_input_changes_the_prediction(self):
        row = self.rows.iloc[0][MODEL_FEATURES].to_dict()
        base = predict_match(row, bundle=self.bundle)["probabilities"]["H"]["decimal"]
        row["elo_difference_k32"] += 300
        stronger_home = predict_match(row, bundle=self.bundle)["probabilities"]["H"]["decimal"]
        self.assertNotEqual(base, stronger_home)


if __name__ == "__main__":
    unittest.main()
