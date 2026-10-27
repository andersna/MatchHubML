"""Train and save the first final MatchHubML NN + odds artifact.

This script is intentionally separate from the OOF experiments. It selects a
fixed epoch count using a chronological 85/15 date holdout and unweighted
validation CrossEntropyLoss, then refits preprocessing and the same model on
all eligible historical rows for exactly that many epochs. Test-fold metrics
are not used for epoch selection. Historical odds have no documented snapshot
timestamps, so the resulting model is not evidence of pre-kickoff odds access.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path

# Keep PyTorch's import ahead of pandas/NumPy on this Windows setup.
import torch
import numpy as np
import pandas as pd
from torch import nn

# Reuse the tested model, exact feature order, H/D/A mapping, alpha and weight
# calculation. The imported experiment has a main guard and does not train on
# import. Its inline fold preprocessing is centralized here for final fitting.
from experiment_nn_odds_asof_day_v1 import (
    ALPHA,
    DEVICE,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    MODEL_FEATURES,
    NN_FEATURES,
    ODDS_DATA_PATH,
    ODDS_FEATURES,
    ODDS_MATCH_KEYS,
    MatchNet,
    _baseline_dates,
    alpha_quarter_weights,
)
from nn_preprocessing import MedianStandardPreprocessor


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "models" / "nn_odds_asof_day_v1"
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
VALIDATION_FRACTION = 0.15


def check_configuration() -> None:
    if len(NN_FEATURES) != 27 or len(ODDS_FEATURES) != 3:
        raise AssertionError("Forventede 27 sportsfeatures og 3 oddsfeatures.")
    if len(MODEL_FEATURES) != 30 or len(set(MODEL_FEATURES)) != 30:
        raise AssertionError("Modelens input skal være 30 unikke features.")
    if list(MODEL_FEATURES) != list(NN_FEATURES) + list(ODDS_FEATURES):
        raise AssertionError("Modelens featureorden afviger fra den godkendte orden.")
    if ALPHA != 0.25:
        raise AssertionError(f"Forventede alpha=0.25, fik {ALPHA}.")


def check_outputs_absent() -> None:
    existing = [path for path in OUTPUT_PATHS if path.exists()]
    if existing:
        raise FileExistsError(
            "Modelartefakt findes allerede; intet overskrives: "
            + ", ".join(str(path) for path in existing)
        )


def validate_dataset(path: Path = ODDS_DATA_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Træningsdatasættet findes ikke: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = KEY_COLUMNS + ["result"] + MODEL_FEATURES
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Datasættet mangler kolonner: {missing_columns}")
    if frame.empty:
        raise ValueError("Træningsdatasættet er tomt.")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("Datasættet indeholder manglende eller ugyldige kampdatoer.")
    if frame[ODDS_MATCH_KEYS].isna().any().any():
        raise ValueError("Kampnøgler uden dato/liga/hold kan ikke bruges sikkert.")
    if frame.duplicated(ODDS_MATCH_KEYS).any():
        raise ValueError("Datasættet har dubletter på odds-kampnøgler.")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Datasættet har dubletter på fulde kampnøgler.")
    if not frame["result"].isin(LABEL_TO_INDEX).all():
        raise ValueError("Ugyldigt resultat; forventes udelukkende H, D eller A.")

    numeric = frame[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        cols = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Modelinput mangler værdier eller er ikke numerisk: {cols}")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Modelinput indeholder uendelige eller ugyldige værdier.")

    odds = numeric[ODDS_FEATURES].to_numpy(dtype=np.float64)
    if (odds <= 0).any() or (odds > 1).any():
        raise ValueError("Odds-sandsynligheder skal være i intervallet (0, 1].")
    if not np.allclose(odds.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Odds-sandsynlighederne skal summere til 1 pr. kamp.")

    frame[MODEL_FEATURES] = numeric
    return frame.sort_values(["date", "time"], kind="stable").reset_index(drop=True)


def validate_asof_day_provenance(frame: pd.DataFrame) -> dict:
    """Check that dataset dates sit inside the existing full-baseline blocks."""
    old_full, old_blocks = _baseline_dates(
        ROOT / "data" / "features.csv", ROOT / "data" / "elo_features.csv"
    )
    asof_full, asof_blocks = _baseline_dates(
        ROOT / "data" / "features_asof_day_v1.csv",
        ROOT / "data" / "elo_features_asof_day_v1.csv",
    )
    if old_blocks != asof_blocks:
        raise RuntimeError(
            "As-of-day-dataets kalenderblokke afviger fra moderne baseline."
        )
    allowed_dates = {day for block in asof_blocks for day in block}
    input_dates = set(frame["date"].dt.date.unique())
    if not input_dates.issubset(allowed_dates):
        raise ValueError("Odds-datasættet indeholder datoer uden for as-of-day-baselinen.")
    return {
        "modern_baseline_rows": int(len(old_full)),
        "asof_day_baseline_rows": int(len(asof_full)),
        "calendar_blocks_match": True,
        "asof_day_blocks": [
            {"start": str(block[0]), "end": str(block[-1]), "dates": len(block)}
            for block in asof_blocks
        ],
    }


def chronological_epoch_selection(frame: pd.DataFrame) -> tuple[int, float, dict]:
    """Choose best epoch on the latest 15% of dates, with early stopping."""
    dates = sorted(frame["date"].dt.date.unique())
    split = int(len(dates) * (1.0 - VALIDATION_FRACTION))
    if split < 1 or split >= len(dates):
        raise ValueError("For få unikke datoer til kronologisk 85/15-validering.")
    train_dates, validation_dates = set(dates[:split]), set(dates[split:])
    subtrain = frame[frame["date"].dt.date.isin(train_dates)].copy()
    validation = frame[frame["date"].dt.date.isin(validation_dates)].copy()
    if subtrain.empty or validation.empty:
        raise ValueError("Kronologisk epokevalg gav tomt subtrain eller validation.")

    preprocessor = MedianStandardPreprocessor(list(MODEL_FEATURES))
    x_train = preprocessor.fit_transform(subtrain)
    x_validation = preprocessor.transform(validation)
    y_train = subtrain["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    y_validation = validation["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Subtrain har for få klasser til at træne netværket.")

    np.random.seed(FINAL_SEED)
    torch.manual_seed(FINAL_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(FINAL_SEED)
    model = MatchNet(len(MODEL_FEATURES)).to(DEVICE)
    train_loss = nn.CrossEntropyLoss(
        weight=torch.as_tensor(alpha_quarter_weights(y_train), device=DEVICE)
    )
    validation_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=DEVICE)
    x_validation_t = torch.as_tensor(
        x_validation, dtype=torch.float32, device=DEVICE
    )
    y_validation_t = torch.as_tensor(
        y_validation, dtype=torch.long, device=DEVICE
    )

    best_epoch, best_loss, best_state, stale_epochs = 0, float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_train_t), device=DEVICE)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = train_loss(model(x_train_t[batch]), y_train_t[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = validation_loss(model(x_validation_t), y_validation_t).item()
        if value < best_loss:
            best_epoch, best_loss = epoch, value
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Kronologisk validering fandt intet gyldigt checkpoint.")

    return best_epoch, best_loss, {
        "subtrain_rows": int(len(subtrain)),
        "validation_rows": int(len(validation)),
        "subtrain_start": str(min(train_dates)),
        "subtrain_end": str(max(train_dates)),
        "validation_start": str(min(validation_dates)),
        "validation_end": str(max(validation_dates)),
        "epochs_run": int(epoch),
    }


def train_final(frame: pd.DataFrame, epochs: int):
    """Refit preprocessing and the model on all rows for the selected epochs."""
    preprocessor = MedianStandardPreprocessor(list(MODEL_FEATURES))
    x_all = preprocessor.fit_transform(frame)
    y_all = frame["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    np.random.seed(FINAL_SEED)
    torch.manual_seed(FINAL_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(FINAL_SEED)
    model = MatchNet(len(MODEL_FEATURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(alpha_quarter_weights(y_all), device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    x_all_t = torch.as_tensor(x_all, dtype=torch.float32, device=DEVICE)
    y_all_t = torch.as_tensor(y_all, dtype=torch.long, device=DEVICE)
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_all_t), device=DEVICE)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(x_all_t[batch]), y_all_t[batch])
            loss.backward()
            optimizer.step()
    return model, preprocessor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(model, preprocessor, frame, epochs, validation_loss, split_info, provenance):
    check_outputs_absent()
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    preprocessing = preprocessor.to_dict()
    final_class_weights = alpha_quarter_weights(
        frame["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    )
    metadata = {
        "artifact_name": "nn_odds_asof_day_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": "MatchNet",
        "architecture": [30, 32, "ReLU", "Dropout(0.15)", 16, "ReLU", 3],
        "feature_columns_in_order": list(MODEL_FEATURES),
        "feature_count": len(MODEL_FEATURES),
        "class_to_index": dict(LABEL_TO_INDEX),
        "index_to_class": ["H", "D", "A"],
        "alpha": ALPHA,
        "class_weights_h_d_a": final_class_weights.astype(float).tolist(),
        "final_seed": FINAL_SEED,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": epochs,
            "training_loss": "weighted CrossEntropyLoss",
            "class_weights": "inverse class frequencies on the final fit data raised to alpha, mean-normalized over present classes",
        },
        "epoch_selection": {
            "method": "chronological 85/15 unique-date holdout; best epoch by unweighted validation CrossEntropyLoss; patience 8; max 100 epochs; refit all rows for selected epoch count",
            "best_validation_cross_entropy": validation_loss,
            **split_info,
        },
        "training_data": {
            "path": str(ODDS_DATA_PATH.relative_to(ROOT)),
            "sha256": sha256_file(ODDS_DATA_PATH),
            "rows": int(len(frame)),
            "start_date": str(frame.date.min().date()),
            "end_date": str(frame.date.max().date()),
            "unique_odds_match_keys": int(len(frame[ODDS_MATCH_KEYS].drop_duplicates())),
            "asof_day_provenance_check": provenance,
        },
        "odds_timestamp_caveat": "Historical odds have no documented snapshot timestamps; availability before kickoff is unverified.",
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device_used_for_training": str(DEVICE),
        },
        "files": {
            "weights": WEIGHTS_PATH.name,
            "preprocessing": PREPROCESSING_PATH.name,
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    check_outputs_absent()
    created: list[Path] = []
    try:
        with WEIGHTS_PATH.open("xb") as stream:
            created.append(WEIGHTS_PATH)
            torch.save(state, stream)
        for path, payload in (
            (PREPROCESSING_PATH, preprocessing),
            (METADATA_PATH, metadata),
        ):
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    except Exception:
        for path in created:
            if path.exists():
                path.unlink()
        raise


def main() -> None:
    check_configuration()
    check_outputs_absent()
    print(
        "Oddsbegrænsning: historiske odds mangler dokumenterede snapshot-tidspunkter; "
        "før-kickoff-tilgængelighed er ikke verificeret."
    )
    frame = validate_dataset()
    provenance = validate_asof_day_provenance(frame)
    print(f"Validerede træningskampe: {len(frame):,}; features: {len(MODEL_FEATURES)}")
    best_epoch, best_val_loss, split_info = chronological_epoch_selection(frame)
    print(
        f"Valgt epoketal: {best_epoch} (uvægtet validation CE={best_val_loss:.6f}); "
        "foldenes testresultater indgår ikke i valget."
    )
    model, preprocessor = train_final(frame, best_epoch)
    save_artifact(
        model, preprocessor, frame, best_epoch, best_val_loss, split_info, provenance
    )
    print(f"Modelvægte: {WEIGHTS_PATH}")
    print(f"Preprocessing: {PREPROCESSING_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
