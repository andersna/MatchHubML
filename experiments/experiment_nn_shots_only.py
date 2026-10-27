
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Stier og indstillinger
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "oof_predictions_nn_shots_only.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "nn_shots_only_summary.csv"
)

LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
INDEX_TO_LABEL = np.array(["H", "D", "A"])

ODDS_FEATURES = [
    "odds_prob_home",
    "odds_prob_draw",
    "odds_prob_away",
]

KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]

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
] + [
    f"elo_difference_k{k}" for k in [16, 32, 48]
]

FORM_FEATURES = [
    "home_form_rating",
    "away_form_rating",
    "home_venue_form_rating",
    "away_venue_form_rating",
]

SHOT_FEATURES = [
    "home_last5_shots_for",
    "home_last5_shots_against",
    "home_last5_shots_on_target_for",
    "home_last5_shots_on_target_against",
    "away_last5_shots_for",
    "away_last5_shots_against",
    "away_last5_shots_on_target_for",
    "away_last5_shots_on_target_against",
]

NN_FEATURES = BASE_FEATURES + ATTACK_DEFENSE_FEATURES + ELO_COLUMNS
NN_FEATURES_WITH_SHOTS = NN_FEATURES + SHOT_FEATURES

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch: {torch.__version__}")
print(f"Enhed: {DEVICE}")


# --------------------------------------------------
# Samme netværksarkitektur som baseline
# --------------------------------------------------

class MatchNet(nn.Module):
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


# --------------------------------------------------
# Evaluering
# --------------------------------------------------


def calculate_metrics(y_true, probabilities):
    # Sikr, at sandsynlighederne er gyldige og summerer til 1.
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    actual_indices = np.array(
        [LABEL_TO_INDEX[value] for value in y_true]
    )
    predicted_indices = probabilities.argmax(axis=1)
    predictions = INDEX_TO_LABEL[predicted_indices]

    one_hot = np.eye(3)[actual_indices]
    brier = np.mean(
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
        "brier_score": brier,
    }, predictions


# --------------------------------------------------
# Træn én model i ét fold
# --------------------------------------------------

def train_neural_network(subtrain, validation, test, feature_columns, fold):
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    scaler = StandardScaler()

    # Imputer og scaler tilpasses kun på subtrain.
    x_subtrain = scaler.fit_transform(
        imputer.fit_transform(subtrain[feature_columns])
    )
    x_validation = scaler.transform(
        imputer.transform(validation[feature_columns])
    )
    x_test = scaler.transform(
        imputer.transform(test[feature_columns])
    )

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

    # Samme seed for begge modeller i det samme fold.
    torch.manual_seed(42 + fold)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42 + fold)

    network = MatchNet(len(feature_columns)).to(DEVICE)

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
        order = torch.randperm(
            len(x_subtrain_tensor), device=DEVICE
        )

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]

            batch_x = x_subtrain_tensor[indices]
            batch_y = y_subtrain_tensor[indices]

            optimizer.zero_grad()
            logits = network(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            validation_logits = network(x_validation_tensor)
            validation_loss = loss_function(
                validation_logits,
                y_validation_tensor,
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
        f"    Bedste epoke: {best_epoch}, "
        f"valideringstab: {best_validation_loss:.4f}"
    )

    return probabilities


# --------------------------------------------------
# Indlæs og klargør data
# --------------------------------------------------

df = pd.read_csv(
    "data/features.csv",
    parse_dates=["date"],
)
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

df = df.sort_values(
    ["date", "time"], kind="stable"
).reset_index(drop=True)

# Begræns eksperimentet til sæsoner med næsten komplet skuddata.
df = df[df["season"] >= "2006-07"].reset_index(drop=True)

df = df.dropna(
    subset=COMMON_FEATURES + ELO_COLUMNS
).reset_index(drop=True)

df = df[df["result"].isin(LABEL_TO_INDEX)].reset_index(drop=True)

if df.duplicated(KEY_COLUMNS).any():
    raise ValueError("Der findes dubletter på kampnøglen.")

print(f"\nKampe i fælles datasæt: {len(df)}")
print(f"Features i Model A: {len(NN_FEATURES)}")
print(f"Features i Model B: {len(NN_FEATURES_WITH_SHOTS)}")


# --------------------------------------------------
# Kronologisk backtest: samme datoer for alle modeller
# --------------------------------------------------

dates = sorted(df["date"].dt.date.unique())

date_blocks = [
    list(block)
    for block in np.array_split(dates, 4)
    if len(block) > 0
]

oof_predictions = []

for fold in range(1, len(date_blocks)):
    train_dates = [
        date
        for block in date_blocks[:fold]
        for date in block
    ]
    test_dates = date_blocks[fold]

    train = df[df["date"].dt.date.isin(train_dates)].copy()
    test = df[df["date"].dt.date.isin(test_dates)].copy()

    unique_train_dates = sorted(
        train["date"].dt.date.unique()
    )
    split_index = int(len(unique_train_dates) * 0.85)

    subtrain_dates = unique_train_dates[:split_index]
    validation_dates = unique_train_dates[split_index:]

    subtrain = train[
        train["date"].dt.date.isin(subtrain_dates)
    ].copy()
    validation = train[
        train["date"].dt.date.isin(validation_dates)
    ].copy()

    if subtrain.empty or validation.empty or test.empty:
        raise RuntimeError(f"Fold {fold} har et tomt datasæt.")

    print(
        f"\nFold {fold}: "
        f"træning={len(subtrain)}, "
        f"validering={len(validation)}, "
        f"test={len(test)}"
    )

    
    # Model A: eksisterende 27 features.
    print("  Model A: neuralt netværk med 27 features")
    probabilities_a = train_neural_network(
        subtrain, validation, test, NN_FEATURES, fold
    )

    # Model B: 27 eksisterende features + 12 form- og skud-features.
    print("  Model B: neuralt netværk med skud")
    probabilities_b = train_neural_network(
        subtrain,
        validation,
        test,
        NN_FEATURES_WITH_SHOTS,
        fold,
    )

    # Bookmakerens sandsynligheder bruges direkte som reference.
    models = [
        ("Model A - 27 features", probabilities_a),
        ("Model B - form og skud", probabilities_b),
    ]


    y_test = test["result"].to_numpy()

    for model_name, probabilities in models:
        scores, predictions = calculate_metrics(
            y_test, probabilities
        )

        print(
            f"  {model_name}: "
            f"accuracy={scores['accuracy']:.4f}, "
            f"macro F1={scores['macro_f1']:.4f}, "
            f"log loss={scores['log_loss']:.4f}, "
            f"Brier={scores['brier_score']:.4f}"
        )

        for i, (_, match) in enumerate(test.iterrows()):
            oof_predictions.append({
                "fold": fold,
                "date": match["date"],
                "league": match["league"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "model": model_name,
                "actual": y_test[i],
                "predicted": predictions[i],
                "prob_H": float(probabilities[i, 0]),
                "prob_D": float(probabilities[i, 1]),
                "prob_A": float(probabilities[i, 2]),
                **scores,
            })


# --------------------------------------------------
# Samlet resultat og nye outputfiler
# --------------------------------------------------

oof_df = pd.DataFrame(oof_predictions)

summary = (
    oof_df.groupby("model")
    .agg(
        accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        log_loss=("log_loss", "mean"),
        brier_score=("brier_score", "mean"),
        kampe=("actual", "size"),
    )
)

# Beregn også målinger samlet på tværs af alle OOF-kampe.
print("\n=== SAMLET OOF-RESULTAT ===")

overall_rows = []

for model_name, group in oof_df.groupby("model", sort=False):
    probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
    scores, _ = calculate_metrics(
        group["actual"].to_numpy(), probabilities
    )

    overall_rows.append({
        "model": model_name,
        **scores,
        "kampe": len(group),
    })

overall = pd.DataFrame(overall_rows).set_index("model")
print(overall.round(4).to_string())

oof_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
overall.to_csv(SUMMARY_PATH, encoding="utf-8-sig")

print(f"\nOOF-forudsigelser gemt i: {OUTPUT_PATH}")
print(f"Samlet sammenligning gemt i: {SUMMARY_PATH}")