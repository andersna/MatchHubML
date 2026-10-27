from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_repeated_seeds.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_repeated_seeds_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHAS = (0.0, 0.25)

BASE_FEATURES = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
    "home_last5_goals_for",
    "home_last5_goals_against",
    "home_last5_points",
    "away_last5_goals_for",
    "away_last5_goals_against",
    "away_last5_points",
]

ATTACK_DEFENSE_FEATURES = [
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "home_expected_goals",
    "away_expected_goals",
]

KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]

COMMON_FEATURES = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
    "home_last5_goals_for",
    "home_last5_goals_against",
    "home_last5_points",
    "away_last5_goals_for",
    "away_last5_goals_against",
    "away_last5_points",
    "home_form_rating",
    "away_form_rating",
    "home_venue_form_rating",
    "away_venue_form_rating",
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "home_expected_goals",
    "away_expected_goals",
]

ELO_COLUMNS = [
    f"{prefix}_elo_k{k}"
    for k in [16, 32, 48]
    for prefix in ["home", "away"]
] + [f"elo_difference_k{k}" for k in [16, 32, 48]]

NN_FEATURES = BASE_FEATURES + ATTACK_DEFENSE_FEATURES + ELO_COLUMNS
LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
INDEX_TO_LABEL = np.array(["H", "D", "A"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchNet(nn.Module):
    """Baseline-netværket: 27 → 32 → 16 → 3."""

    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.layers(x)


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        raise ValueError("Ugyldige sandsynligheder blev produceret.")
    return probabilities / row_sums


def calculate_metrics(y_true, probabilities):
    probabilities = normalize_probabilities(probabilities)
    actual_indices = np.array(
        [LABEL_TO_INDEX[value] for value in y_true], dtype=np.int64
    )
    predicted_indices = probabilities.argmax(axis=1)
    predictions = INDEX_TO_LABEL[predicted_indices]
    one_hot = np.eye(3, dtype=np.float64)[actual_indices]
    brier_score = np.mean(
        np.sum((one_hot - probabilities) ** 2, axis=1)
    )
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            actual_indices,
            probabilities,
            labels=[0, 1, 2],
        ),
        "brier_score": brier_score,
    }
    return metrics, predictions, probabilities


def make_class_weights(y_subtrain, alpha):
    """Compute normalized alpha weights using this fold's subtrain only."""
    if alpha not in ALPHAS:
        raise ValueError(f"Ukendt alpha: {alpha}")

    labels = np.asarray(y_subtrain, dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    present = counts > 0
    baseline_weights = np.zeros(3, dtype=np.float64)
    baseline_weights[present] = (
        len(labels) / (present.sum() * counts[present])
    )

    # Alpha zero is exactly unweighted CrossEntropyLoss.
    if alpha == 0.0:
        return None

    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline_weights[present] ** alpha
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def train_one_alpha(subtrain, validation, test, fold, seed, alpha):
    # Fit imputation and scaling only on subtrain in this fold.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_subtrain = scaler.fit_transform(
        imputer.fit_transform(subtrain[NN_FEATURES])
    )
    x_validation = scaler.transform(
        imputer.transform(validation[NN_FEATURES])
    )
    x_test = scaler.transform(imputer.transform(test[NN_FEATURES]))

    y_subtrain = subtrain["result"].map(
        LABEL_TO_INDEX
    ).to_numpy(dtype=np.int64)
    y_validation = validation["result"].map(
        LABEL_TO_INDEX
    ).to_numpy(dtype=np.int64)

    x_subtrain_tensor = torch.tensor(
        x_subtrain, dtype=torch.float32, device=DEVICE
    )
    y_subtrain_tensor = torch.tensor(
        y_subtrain, dtype=torch.long, device=DEVICE
    )
    x_validation_tensor = torch.tensor(
        x_validation, dtype=torch.float32, device=DEVICE
    )
    y_validation_tensor = torch.tensor(
        y_validation, dtype=torch.long, device=DEVICE
    )
    x_test_tensor = torch.tensor(
        x_test, dtype=torch.float32, device=DEVICE
    )

    # Both alpha variants for a seed/fold reset to this same effective seed.
    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)

    network = MatchNet(len(NN_FEATURES)).to(DEVICE)
    class_weights = make_class_weights(y_subtrain, alpha)
    if class_weights is None:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=DEVICE)
        )

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )
    max_epochs = 100
    batch_size = 256
    patience = 8
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        network.train()
        order = torch.randperm(len(x_subtrain_tensor), device=DEVICE)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_x = x_subtrain_tensor[indices]
            batch_y = y_subtrain_tensor[indices]
            optimizer.zero_grad()
            loss = loss_function(network(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            validation_logits = network(x_validation_tensor)
            validation_loss = loss_function(
                validation_logits, y_validation_tensor
            ).item()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in network.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Ingen gyldig model blev gemt.")

    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        probabilities = torch.softmax(
            network(x_test_tensor), dim=1
        ).cpu().numpy()

    print(
        f"    seed={seed}, alpha={alpha:g}: bedste epoke={best_epoch}, "
        f"valideringstab={best_validation_loss:.4f}"
    )
    return normalize_probabilities(probabilities)


def metric_summary_row(scope, alpha, seed, fold, metrics, matches,
                       metric_stds=None):
    row = {
        "scope": scope,
        "alpha": alpha,
        "seed": seed,
        "fold": fold,
        "kampe": matches,
    }
    metric_stds = metric_stds or {}
    for name in ("accuracy", "macro_f1", "log_loss", "brier_score"):
        row[name] = metrics.get(name, np.nan)
        row[f"{name}_std"] = metric_stds.get(name, np.nan)
    return row


def main():
    # Refuse to overwrite any existing outputs before doing any training.
    existing_outputs = [
        path for path in (OOF_PATH, SUMMARY_PATH) if path.exists()
    ]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Resultatfil findes allerede; intet blev overskrevet: {paths}"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"])
    df = df.merge(
        elo[
            KEY_COLUMNS
            + ELO_COLUMNS
            + ATTACK_DEFENSE_FEATURES
        ],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    df = df.sort_values(["date", "time"], kind="stable").reset_index(
        drop=True
    )
    # Same row selection as the class-weight-strength experiment; no season filter.
    df = df.dropna(
        subset=COMMON_FEATURES + ELO_COLUMNS
    ).reset_index(drop=True)
    df = df[df["result"].isin(LABEL_TO_INDEX)].reset_index(drop=True)
    if not df["result"].isin(LABEL_TO_INDEX).all():
        raise ValueError("Resultatkolonnen indeholder ukendte klasser.")

    dates = sorted(df["date"].dt.date.unique())
    date_blocks = [
        list(block)
        for block in np.array_split(dates, 4)
        if len(block) > 0
    ]
    print(f"PyTorch: {torch.__version__}")
    print(f"Enhed: {DEVICE}")
    print(f"Kampe i backtesten: {len(df)}")
    print(f"Features: {len(NN_FEATURES)}")

    oof_predictions = []
    summary_rows = []

    for fold in range(1, len(date_blocks)):
        train_dates = [
            date
            for block in date_blocks[:fold]
            for date in block
        ]
        test_dates = date_blocks[fold]
        train = df[df["date"].dt.date.isin(train_dates)]
        test = df[df["date"].dt.date.isin(test_dates)]

        unique_train_dates = sorted(train["date"].dt.date.unique())
        split_index = int(len(unique_train_dates) * 0.85)
        subtrain_dates = unique_train_dates[:split_index]
        validation_dates = unique_train_dates[split_index:]
        subtrain = train[
            train["date"].dt.date.isin(subtrain_dates)
        ].copy()
        validation = train[
            train["date"].dt.date.isin(validation_dates)
        ].copy()

        if train.empty or test.empty:
            continue
        if subtrain.empty or validation.empty:
            raise RuntimeError(f"Fold {fold} har tomt subtrain/validering.")

        print(
            f"\nFold {fold}: træning={len(subtrain) + len(validation)}, "
            f"subtrain={len(subtrain)}, validering={len(validation)}, "
            f"test={len(test)}"
        )
        fold_keys = test[KEY_COLUMNS].reset_index(drop=True)

        for seed in SEEDS:
            for alpha in ALPHAS:
                probabilities = train_one_alpha(
                    subtrain, validation, test, fold, seed, alpha
                )
                metrics, predictions, probabilities = calculate_metrics(
                    test["result"].to_numpy(), probabilities
                )
                summary_rows.append(metric_summary_row(
                    "seed_fold", alpha, seed, fold, metrics, len(test)
                ))

                for i, (_, match) in enumerate(test.iterrows()):
                    oof_predictions.append({
                        "seed": seed,
                        "alpha": alpha,
                        "fold": fold,
                        "date": match["date"],
                        "time": match["time"],
                        "league": match["league"],
                        "home_team": match["home_team"],
                        "away_team": match["away_team"],
                        "actual": match["result"],
                        "predicted": predictions[i],
                        "prob_H": float(probabilities[i, 0]),
                        "prob_D": float(probabilities[i, 1]),
                        "prob_A": float(probabilities[i, 2]),
                    })

                candidate_keys = pd.DataFrame(
                    [
                        row for row in oof_predictions
                        if row["seed"] == seed
                        and row["alpha"] == alpha
                        and row["fold"] == fold
                    ]
                )[KEY_COLUMNS].reset_index(drop=True)
                # Normalize nullable key dtypes while preserving values/order.
                if not candidate_keys.astype("string").equals(
                    fold_keys.astype("string")
                ):
                    raise RuntimeError(
                        f"Testkampene afviger for seed={seed}, alpha={alpha}, "
                        f"fold={fold}."
                    )

    oof_df = pd.DataFrame(oof_predictions)
    if oof_df.empty:
        raise RuntimeError("Backtesten producerede ingen OOF-forudsigelser.")

    # Pooled OOF metrics for each seed/alpha, combining the chronological folds.
    seed_overall_metrics = []
    for (seed, alpha), group in oof_df.groupby(
        ["seed", "alpha"], sort=True
    ):
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        metrics, _, _ = calculate_metrics(
            group["actual"].to_numpy(), probabilities
        )
        seed_overall_metrics.append({
            "seed": seed,
            "alpha": alpha,
            "metrics": metrics,
            "kampe": len(group),
        })
        summary_rows.append(metric_summary_row(
            "seed_overall", alpha, seed, "all", metrics, len(group)
        ))

    # Fold-wise mean and sample standard deviation across the five seeds.
    fold_summary_df = pd.DataFrame(
        [row for row in summary_rows if row["scope"] == "seed_fold"]
    )
    for (alpha, fold), group in fold_summary_df.groupby(
        ["alpha", "fold"], sort=True
    ):
        means = {
            name: group[name].mean()
            for name in ("accuracy", "macro_f1", "log_loss", "brier_score")
        }
        stds = {
            name: group[name].std(ddof=1)
            for name in ("accuracy", "macro_f1", "log_loss", "brier_score")
        }
        summary_rows.append(metric_summary_row(
            "fold_seed_summary", alpha, "all", fold, means,
            group["kampe"].mean(), stds
        ))

    # Overall mean and sample standard deviation across seed-level OOF results.
    seed_overall_df = pd.DataFrame([
        {
            "seed": item["seed"],
            "alpha": item["alpha"],
            **item["metrics"],
            "kampe": item["kampe"],
        }
        for item in seed_overall_metrics
    ])
    for alpha, group in seed_overall_df.groupby("alpha", sort=True):
        means = {
            name: group[name].mean()
            for name in ("accuracy", "macro_f1", "log_loss", "brier_score")
        }
        stds = {
            name: group[name].std(ddof=1)
            for name in ("accuracy", "macro_f1", "log_loss", "brier_score")
        }
        summary_rows.append(metric_summary_row(
            "overall_seed_summary", alpha, "all", "all", means,
            group["kampe"].mean(), stds
        ))

    summary_columns = [
        "scope", "alpha", "seed", "fold", "kampe",
        "accuracy", "accuracy_std", "macro_f1", "macro_f1_std",
        "log_loss", "log_loss_std", "brier_score", "brier_score_std",
    ]
    summary_df = pd.DataFrame(summary_rows)[summary_columns]

    # Recheck immediately before writing; never overwrite an existing result.
    existing_outputs = [
        path for path in (OOF_PATH, SUMMARY_PATH) if path.exists()
    ]
    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Resultatfil findes allerede; intet blev overskrevet: {paths}"
        )

    oof_df.to_csv(OOF_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"\nOOF-forudsigelser gemt i: {OOF_PATH}")
    print(f"Seed-, fold- og samlet sammenligning gemt i: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
