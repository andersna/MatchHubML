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
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_learning_rate.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_learning_rate_summary.csv"

LEARNING_RATES = (0.0005, 0.001, 0.002)

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

torch.manual_seed(42)
np.random.seed(42)
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

    return {
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
    }, predictions, probabilities


def train_one_candidate(
    subtrain, validation, test, fold, learning_rate
):
    # Tilpas kun preprocessing på subtrain i dette fold.
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

    # Samme fold-seed giver kandidaterne samme initialisering og shuffle-seed.
    torch.manual_seed(42 + fold)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42 + fold)

    network = MatchNet(len(NN_FEATURES)).to(DEVICE)

    counts = np.bincount(y_subtrain, minlength=3)
    present = counts > 0
    class_weights = np.zeros(3, dtype=np.float32)
    class_weights[present] = (
        len(y_subtrain) / (present.sum() * counts[present])
    )
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=learning_rate,
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
        f"    lr={learning_rate:g}: bedste epoke={best_epoch}, "
        f"valideringstab={best_validation_loss:.4f}"
    )
    return normalize_probabilities(probabilities)


def main():
    # Refuse to overwrite any existing result, including files created after
    # this script was reviewed.
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

    # Matcher baselineens datagrundlag: intet sæsonfilter.
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
    fold_summaries = []

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

        # Én fælles testmængde pr. fold anvendes af samtlige kandidater.
        fold_keys = test[KEY_COLUMNS].reset_index(drop=True)
        for learning_rate in LEARNING_RATES:
            probabilities = train_one_candidate(
                subtrain, validation, test, fold, learning_rate
            )
            scores, predictions, probabilities = calculate_metrics(
                test["result"].to_numpy(), probabilities
            )
            fold_summaries.append({
                "scope": "fold",
                "fold": fold,
                "learning_rate": learning_rate,
                **scores,
                "kampe": len(test),
            })

            for i, (_, match) in enumerate(test.iterrows()):
                oof_predictions.append({
                    "learning_rate": learning_rate,
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

            # Guard against accidental changes to candidate-specific test rows.
            candidate_keys = pd.DataFrame(
                [row for row in oof_predictions
                 if row["fold"] == fold
                 and row["learning_rate"] == learning_rate]
            )[KEY_COLUMNS].reset_index(drop=True)
            # Pandas infers an all-missing `time` column as float in the
            # candidate frame but keeps it as object in `test`; compare
            # normalized values so dtype inference does not cause a false alarm.
            if not candidate_keys.astype("string").equals(
                fold_keys.astype("string")
            ):
                raise RuntimeError(
                    f"Testkampene afviger for learning rate {learning_rate}, "
                    f"fold {fold}."
                )

    oof_df = pd.DataFrame(oof_predictions)
    if oof_df.empty:
        raise RuntimeError("Backtesten producerede ingen OOF-forudsigelser.")

    # Pooled metrics over all OOF matches, in addition to per-fold rows.
    for learning_rate, group in oof_df.groupby("learning_rate", sort=True):
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        scores, _, _ = calculate_metrics(group["actual"].to_numpy(), probabilities)
        fold_summaries.append({
            "scope": "overall",
            "fold": "all",
            "learning_rate": learning_rate,
            **scores,
            "kampe": len(group),
        })

    summary_df = pd.DataFrame(fold_summaries)[
        ["scope", "fold", "learning_rate", "accuracy", "macro_f1",
         "log_loss", "brier_score", "kampe"]
    ]

    # Check once more just before writing so a newly created file is preserved.
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
    print(f"Fold- og samlet sammenligning gemt i: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
