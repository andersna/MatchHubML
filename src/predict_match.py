"""Load the saved MatchHubML model and predict H/D/A probabilities for a match.

Example technical smoke test (does not score the known result):
    python src/predict_match.py --row-index -1

The CSV loader deliberately does not load the actual result column. Historical
odds have no documented snapshot timestamps, and the model was trained on the
entire historical dataset, so this smoke test is not a generalization estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

# Preserve the project's PyTorch-before-NumPy import order.
import torch
import numpy as np
import pandas as pd

# Reuse the training module's model class, ordered features, labels and paths.
# Its main guard prevents training when imported here.
from train_final_nn_odds_asof_day_v1 import (
    ARTIFACT_DIR,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    MODEL_FEATURES,
    ODDS_DATA_PATH,
    ODDS_FEATURES,
    MatchNet,
)
from experiment_nn_odds_asof_day_v1 import INDEX_TO_LABEL
from nn_preprocessing import MedianStandardPreprocessor
from train_final_nn_market_residual_v3 import MODEL_CLASS as RESIDUAL_MODEL_CLASS
from train_final_nn_market_residual_v3 import MarketResidualEnsemble, V3_FEATURES


CLASS_ORDER = tuple(INDEX_TO_LABEL.tolist())
# v1, v2 and v3 artifacts trained before xG use MODEL_FEATURES; v3 with xG uses V3_FEATURES.
SUPPORTED_FEATURE_SETS = (tuple(MODEL_FEATURES), tuple(V3_FEATURES))


@dataclass
class ModelBundle:
    model: MatchNet | MarketResidualEnsemble
    preprocessor: MedianStandardPreprocessor
    metadata: dict[str, Any]
    feature_columns: tuple[str, ...]
    device: torch.device


def load_model_bundle(
    artifact_dir: Path = ARTIFACT_DIR,
    device: torch.device | str = "cpu",
) -> ModelBundle:
    """Load weights and saved preprocessing without fitting or mutating them."""
    artifact_dir = Path(artifact_dir)
    weights_path = artifact_dir / "model_state_dict.pt"
    preprocessing_path = artifact_dir / "preprocessing.json"
    metadata_path = artifact_dir / "metadata.json"
    for path in (weights_path, preprocessing_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Modelartefakt mangler: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = tuple(metadata.get("feature_columns_in_order", ()))
    if feature_columns not in SUPPORTED_FEATURE_SETS:
        raise ValueError("Metadataens featureliste/-rækkefølge afviger fra træningskoden.")
    if metadata.get("feature_count") != len(feature_columns):
        raise ValueError("Metadataens featureantal stemmer ikke med træningskoden.")
    if metadata.get("class_to_index") != LABEL_TO_INDEX:
        raise ValueError("Metadataens H/D/A-mapping afviger fra træningskoden.")
    if metadata.get("index_to_class") != list(CLASS_ORDER):
        raise ValueError("Metadataens klasseorden er ugyldig.")
    model_class = metadata.get("model_class")
    if model_class not in ("MatchNet", RESIDUAL_MODEL_CLASS):
        raise ValueError("Ukendt modelklasse i metadata.")

    preprocessing_data = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    preprocessor = MedianStandardPreprocessor.from_dict(preprocessing_data)
    if tuple(preprocessor.feature_columns) != feature_columns:
        raise ValueError("Preprocessingens featureorden matcher ikke modellen.")

    selected_device = torch.device(device)
    if model_class == RESIDUAL_MODEL_CLASS:
        model = MarketResidualEnsemble(
            len(feature_columns), int(metadata["member_count"])
        ).to(selected_device)
    else:
        model = MatchNet(len(feature_columns)).to(selected_device)
    state = torch.load(weights_path, map_location=selected_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return ModelBundle(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        feature_columns=feature_columns,
        device=selected_device,
    )


def _as_one_row(
    match: Mapping[str, Any] | pd.Series | pd.DataFrame, feature_columns: tuple[str, ...]
) -> pd.DataFrame:
    if isinstance(match, pd.DataFrame):
        if len(match) != 1:
            raise ValueError("predict_match forventer præcis én kamprække.")
        row = match.copy()
    elif isinstance(match, pd.Series):
        row = match.to_frame().T
    elif isinstance(match, Mapping):
        row = pd.DataFrame([dict(match)])
    else:
        raise TypeError("Kampen skal være en mapping, pandas Series eller én-rækkes DataFrame.")

    missing = [name for name in feature_columns if name not in row.columns]
    if missing:
        raise ValueError(f"Kampen mangler modelfeatures: {missing}")
    return row.loc[:, list(feature_columns)].reset_index(drop=True)


def predict_match(
    match: Mapping[str, Any] | pd.Series | pd.DataFrame,
    bundle: ModelBundle | None = None,
) -> dict[str, Any]:
    """Return H/D/A probabilities as decimals and percentages for one match.

    Missing non-odds numeric values are passed to the saved median imputer.
    Odds probabilities must be present, numeric, in (0, 1], and sum to 1.
    """
    bundle = bundle or load_model_bundle()
    row = _as_one_row(match, bundle.feature_columns)
    numeric_row = pd.DataFrame(index=[0])
    for column in bundle.feature_columns:
        raw_value = row.at[0, column]
        if pd.isna(raw_value):
            if column in ODDS_FEATURES:
                raise ValueError(f"Oddsfeature {column} må ikke mangle.")
            numeric_row[column] = [np.nan]
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {column} skal være numerisk.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Feature {column} skal være en endelig numerisk værdi.")
        numeric_row[column] = [value]

    odds = numeric_row.loc[:, ODDS_FEATURES].to_numpy(dtype=np.float64)[0]
    if (odds <= 0).any() or (odds > 1).any():
        raise ValueError("Odds-sandsynligheder skal ligge i intervallet (0, 1].")
    if not np.isclose(odds.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Odds-sandsynlighederne skal summere til cirka 1.")

    scaled = bundle.preprocessor.transform(numeric_row)
    input_tensor = torch.as_tensor(
        scaled.copy(), dtype=torch.float32, device=bundle.device
    )
    with torch.inference_mode():
        if isinstance(bundle.model, MarketResidualEnsemble):
            # v3 corrects the market: it needs the raw (unscaled) odds probabilities.
            market = torch.as_tensor(
                odds.reshape(1, 3), dtype=torch.float32, device=bundle.device
            )
            probabilities = bundle.model(input_tensor, market)
        else:
            # Softmax in float64: float32 sums can miss 1 by more than the 1e-7 check below.
            probabilities = torch.softmax(bundle.model(input_tensor).double(), dim=1)
    probs = probabilities.detach().cpu().numpy()[0].astype(np.float64)
    if probs.shape != (3,) or not np.isfinite(probs).all() or (probs < 0).any():
        raise RuntimeError("Modellen returnerede ugyldige sandsynligheder.")
    total = float(probs.sum())
    if total <= 0 or not np.isclose(total, 1.0, atol=1e-7, rtol=0):
        raise RuntimeError("Modellens sandsynligheder summerer ikke til 1.")
    probs /= total

    by_class = {
        label: {"decimal": float(probs[index]), "percent": float(probs[index] * 100.0)}
        for index, label in enumerate(bundle.metadata["index_to_class"])
    }
    return {
        "model_version": bundle.metadata["artifact_name"],
        "probabilities": by_class,
        "sum": float(probs.sum()),
    }


def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Forudsig H/D/A-sandsynligheder med den gemte MatchHubML-model."
    )
    parser.add_argument(
        "--csv", type=Path, default=ODDS_DATA_PATH,
        help="CSV med eksisterende kampfeatures (default: as-of-day odds-datasæt).",
    )
    parser.add_argument(
        "--row-index", type=int, default=-1,
        help="Rækkeindeks i CSV; -1 vælger sidste række.",
    )
    args = parser.parse_args()

    use_columns = list(KEY_COLUMNS) + list(MODEL_FEATURES)
    missing_columns = [name for name in use_columns if name not in pd.read_csv(args.csv, nrows=0).columns]
    if missing_columns:
        raise ValueError(f"CSV mangler påkrævede kolonner: {missing_columns}")
    # `result` is intentionally not in use_columns and is never passed to predict_match.
    matches = pd.read_csv(
        args.csv, usecols=use_columns, parse_dates=["date"], low_memory=False
    )
    if matches.empty:
        raise ValueError("CSV indeholder ingen kampe.")
    if not -len(matches) <= args.row_index < len(matches):
        raise IndexError(f"row-index {args.row_index} er uden for CSV-rækkerne.")
    match = matches.iloc[[args.row_index]].copy()
    bundle = load_model_bundle()
    result = predict_match(match, bundle)
    selected = match.iloc[0]

    print("Teknisk smoke-test: modellen er trænet på hele datasættet; dette måler ikke generaliseringsevne.")
    print("Oddsbegrænsning: snapshot-tidspunkter er ikke dokumenteret; før-kickoff-tilgængelighed er ukendt.")
    print(f"Modelversion: {result['model_version']}")
    print(
        f"Kamp: {selected['home_team']} vs. {selected['away_team']} "
        f"({selected['date'].date()}, {selected['league']})"
    )
    for label in CLASS_ORDER:
        values = result["probabilities"][label]
        print(f"{label}: {values['decimal']:.8f} ({values['percent']:.4f}%)")
    print(f"Sum: {result['sum']:.10f}")


if __name__ == "__main__":
    _cli()
