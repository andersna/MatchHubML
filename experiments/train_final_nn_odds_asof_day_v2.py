"""Train and save the separate MatchHubML v2 NN + odds artifact.

Epoch selection uses a chronological 85/15 split of unique dates and
unweighted validation CrossEntropyLoss. The selected epoch count is then used
to refit on all eligible v2 rows. Historical bookmaker odds lack verified
publication timestamps; their availability before kickoff is unproven.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch import nn

from experiment_nn_odds_asof_day_v1 import (
    ALPHA,
    DEVICE,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    MODEL_FEATURES,
    NN_FEATURES,
    ODDS_FEATURES,
    ODDS_MATCH_KEYS,
    MatchNet,
    alpha_quarter_weights,
)
from nn_preprocessing import MedianStandardPreprocessor
import train_final_nn_odds_asof_day_v1 as trainer


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "odds_model_dataset_asof_day_v2.csv"
V1_DATASET_PATH = ROOT / "data" / "odds_model_dataset_asof_day_v1.csv"
ARTIFACT_DIR = ROOT / "models" / "nn_odds_asof_day_v2"
WEIGHTS_PATH = ARTIFACT_DIR / "model_state_dict.pt"
PREPROCESSING_PATH = ARTIFACT_DIR / "preprocessing.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
OUTPUT_PATHS = (WEIGHTS_PATH, PREPROCESSING_PATH, METADATA_PATH)

FINAL_SEED = 42
MAX_EPOCHS = 100
BATCH_SIZE = 256
PATIENCE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EXPECTED_ROWS = 43174
EXPECTED_V2_ADDITIONS = 48


def check_configuration() -> None:
    trainer.check_configuration()
    if len(NN_FEATURES) != 27 or len(ODDS_FEATURES) != 3 or len(MODEL_FEATURES) != 30:
        raise AssertionError("V2 kræver fortsat 27 sportsfeatures og 3 oddsfeatures.")
    if ALPHA != 0.25:
        raise AssertionError("V2 kræver alpha=0.25.")


def check_outputs_absent() -> None:
    existing = [path for path in (ARTIFACT_DIR, *OUTPUT_PATHS) if path.exists()]
    if existing:
        raise FileExistsError(
            "V2-modelmål findes allerede; intet overskrives: "
            + ", ".join(map(str, existing))
        )


def validate_v2_dataset() -> pd.DataFrame:
    frame = trainer.validate_dataset(DATASET_PATH)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Forventede {EXPECTED_ROWS} rækker i v2-datasættet, fandt {len(frame)}.")

    old = pd.read_csv(V1_DATASET_PATH, low_memory=False)
    old_keys = set(map(tuple, old[ODDS_MATCH_KEYS].astype(str).to_numpy()))
    new_keys = set(map(tuple, frame[ODDS_MATCH_KEYS].astype(str).to_numpy()))
    if not old_keys.issubset(new_keys):
        raise ValueError("V2-datasættet mangler kampnøgler fra v1-oddsdatasættet.")
    additions = new_keys - old_keys
    if len(additions) != EXPECTED_V2_ADDITIONS:
        raise ValueError(
            f"Forventede {EXPECTED_V2_ADDITIONS} nye v2-kampnøgler, fandt {len(additions)}."
        )
    return frame


def _sha256(path: Path) -> str:
    return trainer.sha256_file(path)


def _metadata(frame: pd.DataFrame, epochs: int, validation_loss: float, split_info: dict) -> dict:
    y = frame["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    return {
        "artifact_name": "nn_odds_asof_day_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": "MatchNet",
        "architecture": [30, 32, "ReLU", "Dropout(0.15)", 16, "ReLU", 3],
        "feature_columns_in_order": list(MODEL_FEATURES),
        "feature_count": 30,
        "class_to_index": dict(LABEL_TO_INDEX),
        "index_to_class": ["H", "D", "A"],
        "alpha": ALPHA,
        "class_weights_h_d_a": alpha_quarter_weights(y).astype(float).tolist(),
        "final_seed": FINAL_SEED,
        "optimizer": {
            "name": "AdamW", "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "epochs": epochs, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "training_loss": "weighted CrossEntropyLoss",
        },
        "epoch_selection": {
            "method": "chronological 85/15 unique-date holdout; best checkpoint by unweighted validation CrossEntropyLoss; refit all rows for selected epochs",
            "best_validation_cross_entropy": validation_loss,
            **split_info,
        },
        "training_data": {
            "path": str(DATASET_PATH.relative_to(ROOT)),
            "sha256": _sha256(DATASET_PATH),
            "rows": int(len(frame)),
            "start_date": str(frame.date.min().date()),
            "end_date": str(frame.date.max().date()),
            "unique_odds_match_keys": int(len(frame[ODDS_MATCH_KEYS].drop_duplicates())),
            "v1_rows_retained": len(frame) - EXPECTED_V2_ADDITIONS,
            "new_rows_since_v1": EXPECTED_V2_ADDITIONS,
            "v1_dataset_sha256": _sha256(V1_DATASET_PATH),
        },
        "preprocessing": "Median imputation and StandardScaler; fit on chronological subtrain for epoch selection, then refit on all v2 rows for final artifact.",
        "odds_timestamp_caveat": "Historical odds have no documented snapshot timestamps; availability before kickoff is unverified.",
        "runtime": {
            "python": platform.python_version(), "pytorch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "device_used_for_training": str(DEVICE),
        },
        "files": {"weights": WEIGHTS_PATH.name, "preprocessing": PREPROCESSING_PATH.name},
    }


def save_artifact(model, preprocessor, frame, epochs: int, validation_loss: float, split_info: dict) -> None:
    check_outputs_absent()
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payloads = {
        PREPROCESSING_PATH: preprocessor.to_dict(),
        METADATA_PATH: _metadata(frame, epochs, validation_loss, split_info),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []
    try:
        with WEIGHTS_PATH.open("xb") as stream:
            created.append(WEIGHTS_PATH)
            torch.save(state, stream)
        for path, payload in payloads.items():
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        try:
            ARTIFACT_DIR.rmdir()
        except OSError:
            pass
        raise


def smoke_test(frame: pd.DataFrame) -> np.ndarray:
    from predict_match import load_model_bundle

    bundle = load_model_bundle(ARTIFACT_DIR, device=DEVICE)
    row = frame.loc[:, MODEL_FEATURES].iloc[[0]]
    scaled = bundle.preprocessor.transform(row)
    inputs = torch.as_tensor(scaled.copy(), dtype=torch.float32, device=bundle.device)
    with torch.inference_mode():
        probs = torch.softmax(bundle.model(inputs), dim=1).cpu().numpy()[0]
    if probs.shape != (3,) or not np.isfinite(probs).all() or (probs < 0).any():
        raise RuntimeError("Genindlæst v2-model gav ugyldige sandsynligheder.")
    if not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise RuntimeError("Genindlæst v2-models sandsynligheder summerer ikke til 1.")
    return probs


def main() -> None:
    check_configuration()
    check_outputs_absent()
    print("Oddsbegrænsning: historiske odds mangler dokumenterede snapshot-tidspunkter.")
    frame = validate_v2_dataset()
    print(
        f"Validerede v2-træningsrækker: {len(frame):,}; "
        f"periode {frame.date.min().date()} til {frame.date.max().date()}"
    )
    best_epoch, best_val_loss, split_info = trainer.chronological_epoch_selection(frame)
    print(
        f"Valgt epoketal: {best_epoch}; uvægtet validation CE={best_val_loss:.6f}; "
        f"subtrain={split_info['subtrain_rows']:,}, validation={split_info['validation_rows']:,}"
    )
    model, preprocessor = trainer.train_final(frame, best_epoch)
    save_artifact(model, preprocessor, frame, best_epoch, best_val_loss, split_info)
    probabilities = smoke_test(frame)
    print(f"Smoke-test H/D/A: {probabilities[0]:.6f}, {probabilities[1]:.6f}, {probabilities[2]:.6f}")
    print(f"Weights: {WEIGHTS_PATH}\nPreprocessing: {PREPROCESSING_PATH}\nMetadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
