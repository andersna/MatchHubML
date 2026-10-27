"""Train and save the MatchHubML v3 market-residual NN ensemble.

Each member predicts logits = log(market probabilities) + correction(inputs),
with the correction's last layer initialised to zero, so an untrained member
is exactly the Bet365 market. Inputs are the same 30 columns as v1/v2 (27
sports features + 3 odds probabilities). There are no class weights.

For every seed, the epoch count is chosen on a chronological 85/15 holdout of
unique dates by unweighted CrossEntropyLoss, with the untouched market as
epoch 0; the member is then refit on all rows for exactly that many epochs.
The saved model averages the members' probabilities.

Backtest (experiment_nn_market_residual.py, variant residual_27, 5-seed
ensemble, folds 2-3): accuracy 53.64% and log loss 0.96876, versus 53.41% /
0.97122 for the v1 architecture and 53.72% / 0.96909 for odds alone.
Historical odds lack snapshot timestamps; pre-kickoff availability is unverified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from pathlib import Path

# Keep PyTorch's import ahead of pandas/NumPy on this Windows setup.
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F

from experiment_nn_market_residual import MarketResidualNet
from experiment_nn_odds_asof_day_v1 import (
    DEVICE,
    LABEL_TO_INDEX,
    MODEL_FEATURES,
    ODDS_FEATURES,
    ODDS_MATCH_KEYS,
)
from nn_preprocessing import MedianStandardPreprocessor
import train_final_nn_odds_asof_day_v1 as trainer


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "odds_model_dataset_asof_day_v2.csv"
ARTIFACT_DIR = ROOT / "models" / "nn_market_residual_v3"
WEIGHTS_PATH = ARTIFACT_DIR / "model_state_dict.pt"
PREPROCESSING_PATH = ARTIFACT_DIR / "preprocessing.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
OUTPUT_PATHS = (WEIGHTS_PATH, PREPROCESSING_PATH, METADATA_PATH)

ARTIFACT_NAME = "nn_market_residual_v3"
MODEL_CLASS = "MarketResidualEnsemble"
SEEDS = (42, 123, 2026, 3407, 9876)
MAX_EPOCHS = 100
BATCH_SIZE = 256
PATIENCE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
VALIDATION_FRACTION = 0.15
ODDS_INDICES = [MODEL_FEATURES.index(name) for name in ODDS_FEATURES]


class MarketResidualEnsemble(nn.Module):
    """Average of MarketResidualNet members' probabilities."""

    def __init__(self, input_size: int, member_count: int):
        super().__init__()
        self.members = nn.ModuleList(
            MarketResidualNet(input_size, n_leagues=1, embedding_dim=0)
            for _ in range(member_count)
        )

    def forward(self, x: torch.Tensor, market_probs: torch.Tensor) -> torch.Tensor:
        """Return probabilities (not logits) for scaled inputs and raw market probabilities."""
        market_log = torch.log(market_probs)
        league = torch.zeros(len(x), dtype=torch.long, device=x.device)  # unused: no embedding
        probs = [torch.softmax(member(x, league, market_log), dim=1) for member in self.members]
        # Average in float64: float32 member averages can miss a sum of 1 by ~1e-7.
        mean = torch.stack(probs).double().mean(dim=0)
        return mean / mean.sum(dim=1, keepdim=True)


def _output_paths(artifact_dir: Path) -> tuple[Path, Path, Path]:
    return (
        artifact_dir / WEIGHTS_PATH.name,
        artifact_dir / PREPROCESSING_PATH.name,
        artifact_dir / METADATA_PATH.name,
    )


def check_outputs_absent(artifact_dir: Path = ARTIFACT_DIR) -> None:
    existing = [path for path in _output_paths(artifact_dir) if path.exists()]
    if existing:
        raise FileExistsError(
            "V3-modelartefakt findes allerede; intet overskrives: "
            + ", ".join(map(str, existing))
        )


def _tensors(preprocessor, frame):
    x = torch.as_tensor(preprocessor.transform(frame), dtype=torch.float32, device=DEVICE)
    market = torch.as_tensor(
        frame[ODDS_FEATURES].to_numpy(dtype=np.float64), dtype=torch.float32, device=DEVICE
    )
    y = torch.as_tensor(frame["result"].map(LABEL_TO_INDEX).to_numpy(np.int64), device=DEVICE)
    return x, torch.log(market), y


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _train_epoch(model, optimizer, x, market_log, y, league):
    model.train()
    order = torch.randperm(len(x), device=DEVICE)
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x[batch], league[batch], market_log[batch]), y[batch])
        loss.backward()
        optimizer.step()


def _new_member(seed: int):
    _seed_everything(seed)
    model = MarketResidualNet(len(MODEL_FEATURES), n_leagues=1, embedding_dim=0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def select_epochs(frame: pd.DataFrame) -> tuple[dict[int, tuple[int, float]], dict]:
    """Best epoch per seed on the latest 15% of dates; epoch 0 is the market."""
    dates = sorted(frame["date"].dt.date.unique())
    split = int(len(dates) * (1.0 - VALIDATION_FRACTION))
    subtrain = frame[frame["date"].dt.date.isin(dates[:split])]
    validation = frame[frame["date"].dt.date.isin(dates[split:])]
    preprocessor = MedianStandardPreprocessor(list(MODEL_FEATURES))
    preprocessor.fit(subtrain)
    x_train, market_train, y_train = _tensors(preprocessor, subtrain)
    x_val, market_val, y_val = _tensors(preprocessor, validation)
    league_train = torch.zeros(len(x_train), dtype=torch.long, device=DEVICE)
    league_val = torch.zeros(len(x_val), dtype=torch.long, device=DEVICE)

    market_ce = F.cross_entropy(market_val, y_val).item()
    selected = {}
    for seed in SEEDS:
        model, optimizer = _new_member(seed)

        def validation_ce():
            model.eval()
            with torch.no_grad():
                return F.cross_entropy(model(x_val, league_val, market_val), y_val).item()

        best_epoch, best_ce, stale = 0, validation_ce(), 0
        for epoch in range(1, MAX_EPOCHS + 1):
            _train_epoch(model, optimizer, x_train, market_train, y_train, league_train)
            value = validation_ce()
            if value < best_ce:
                best_epoch, best_ce, stale = epoch, value, 0
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        selected[seed] = (best_epoch, best_ce)
        print(f"  seed {seed}: epoke {best_epoch}, validation CE {best_ce:.6f} (marked: {market_ce:.6f})")

    return selected, {
        "subtrain_rows": int(len(subtrain)),
        "validation_rows": int(len(validation)),
        "subtrain_start": str(dates[0]),
        "subtrain_end": str(dates[split - 1]),
        "validation_start": str(dates[split]),
        "validation_end": str(dates[-1]),
        "market_only_validation_cross_entropy": market_ce,
    }


def train_final(frame: pd.DataFrame, epochs_by_seed: dict[int, int]):
    preprocessor = MedianStandardPreprocessor(list(MODEL_FEATURES))
    preprocessor.fit(frame)
    x, market_log, y = _tensors(preprocessor, frame)
    league = torch.zeros(len(x), dtype=torch.long, device=DEVICE)
    ensemble = MarketResidualEnsemble(len(MODEL_FEATURES), len(SEEDS))
    for index, seed in enumerate(SEEDS):
        model, optimizer = _new_member(seed)
        for _epoch in range(epochs_by_seed[seed]):
            _train_epoch(model, optimizer, x, market_log, y, league)
        ensemble.members[index].load_state_dict(model.state_dict())
    ensemble.eval()
    return ensemble, preprocessor


def save_artifact(
    ensemble, preprocessor, frame, selected, split_info, *,
    artifact_dir: Path = ARTIFACT_DIR,
    artifact_name: str = ARTIFACT_NAME,
    dataset_path: Path = DATASET_PATH,
    backtest: dict | None = None,
) -> None:
    check_outputs_absent(artifact_dir)
    weights_path, preprocessing_path, metadata_path = _output_paths(artifact_dir)
    state = {key: value.detach().cpu() for key, value in ensemble.state_dict().items()}
    metadata = {
        "artifact_name": artifact_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_class": MODEL_CLASS,
        "member_count": len(SEEDS),
        "architecture": {
            "member": "logits = log(odds probabilities) + correction(x)",
            "correction": [len(MODEL_FEATURES), 32, "ReLU", "Dropout(0.15)", 16, "ReLU", 3],
            "correction_last_layer_init": "zeros",
            "ensemble": "mean of member softmax probabilities",
        },
        "feature_columns_in_order": list(MODEL_FEATURES),
        "feature_count": len(MODEL_FEATURES),
        "market_columns": list(ODDS_FEATURES),
        "class_to_index": dict(LABEL_TO_INDEX),
        "index_to_class": ["H", "D", "A"],
        "seeds": list(SEEDS),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs_by_seed": {str(seed): epoch for seed, (epoch, _) in selected.items()},
            "training_loss": "unweighted CrossEntropyLoss (no class weights)",
        },
        "epoch_selection": {
            "method": "per seed: chronological 85/15 unique-date holdout; best epoch by unweighted validation CrossEntropyLoss, epoch 0 = market; patience 8; max 100; refit all rows",
            "validation_cross_entropy_by_seed": {str(seed): ce for seed, (_, ce) in selected.items()},
            **split_info,
        },
        "backtest": backtest or {
            "source": "experiment_nn_market_residual.py, variant residual_27_ensemble, folds 2-3",
            "accuracy": 0.53636,
            "log_loss": 0.96876,
            "odds_only_accuracy": 0.53716,
            "odds_only_log_loss": 0.96909,
        },
        "training_data": {
            "path": str(dataset_path.relative_to(ROOT)),
            "sha256": trainer.sha256_file(dataset_path),
            "rows": int(len(frame)),
            "start_date": str(frame.date.min().date()),
            "end_date": str(frame.date.max().date()),
            "unique_odds_match_keys": int(len(frame[ODDS_MATCH_KEYS].drop_duplicates())),
        },
        "odds_timestamp_caveat": "Historical odds have no documented snapshot timestamps; availability before kickoff is unverified.",
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device_used_for_training": str(DEVICE),
        },
        "files": {"weights": weights_path.name, "preprocessing": preprocessing_path.name},
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with weights_path.open("xb") as stream:
            created.append(weights_path)
            torch.save(state, stream)
        for path, payload in ((preprocessing_path, preprocessor.to_dict()), (metadata_path, metadata)):
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
    trainer.check_configuration()
    check_outputs_absent()
    print(
        "Oddsbegrænsning: historiske odds mangler dokumenterede snapshot-tidspunkter; "
        "før-kickoff-tilgængelighed er ikke verificeret."
    )
    frame = trainer.validate_dataset(DATASET_PATH)
    print(f"Validerede træningskampe: {len(frame):,}; features: {len(MODEL_FEATURES)}")
    print("Epokevalg pr. seed (kronologisk 85/15):")
    selected, split_info = select_epochs(frame)
    ensemble, preprocessor = train_final(frame, {seed: epoch for seed, (epoch, _) in selected.items()})
    save_artifact(ensemble, preprocessor, frame, selected, split_info)
    print(f"Modelvægte: {WEIGHTS_PATH}\nPreprocessing: {PREPROCESSING_PATH}\nMetadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
