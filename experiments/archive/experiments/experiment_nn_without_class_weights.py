
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"


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

KEY_COLUMNS = [
    "date", "time", "league", "home_team", "away_team"
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

NN_FEATURES = (
    BASE_FEATURES
    + ATTACK_DEFENSE_FEATURES
    + ELO_COLUMNS
)

LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
INDEX_TO_LABEL = np.array(["H", "D", "A"])

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch: {torch.__version__}")
print(f"Enhed: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class MatchNet(nn.Module):
    """Lille neuralt netværk til klassifikation af kampresultater."""

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


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0,
    )


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


# Indlæs de samme data og anvend samme udvælgelse
# af kampe som i den eksisterende backtest.
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
elo = pd.read_csv(ELO_PATH, parse_dates=["date"])

df = df.merge(
    elo[KEY_COLUMNS + ELO_COLUMNS + [
        "home_attack_strength",
        "away_attack_strength",
        "home_defense_strength",
        "away_defense_strength",
        "home_expected_goals",
        "away_expected_goals",
    ]],
    on=KEY_COLUMNS,
    how="left",
    validate="one_to_one",
)

df = df.sort_values(
    ["date", "time"], kind="stable"
).reset_index(drop=True)


# Behold den eksisterende udvælgelse af kampe.
# Manglende værdier i modelinputtene håndteres senere
# ved imputering, der kun tilpasses på træningsdata.
df = df.dropna(
    subset=COMMON_FEATURES + ELO_COLUMNS
).reset_index(drop=True)

# Fjern eventuelle rækker uden et gyldigt kampresultat.
df = df[df["result"].isin(LABEL_TO_INDEX)].reset_index(drop=True)

if not df["result"].isin(LABEL_TO_INDEX).all():
    raise ValueError("Resultatkolonnen indeholder ukendte klasser.")

dates = sorted(df["date"].dt.date.unique())
date_blocks = [
    list(block)
    for block in np.array_split(dates, 4)
    if len(block) > 0
]

print(f"Kampe i backtesten: {len(df)}")


results = []
oof_predictions = []

for fold in range(1, len(date_blocks)):
    train_dates = [
        date
        for block in date_blocks[:fold]
        for date in block
    ]
    test_dates = date_blocks[fold]

    train = df[df["date"].dt.date.isin(train_dates)]
    test = df[df["date"].dt.date.isin(test_dates)]

    # Brug de seneste 15 % af træningsperiodens datoer
    # som kronologisk valideringssæt til early stopping.
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

    y_train = train["result"].map(
        LABEL_TO_INDEX
    ).to_numpy(dtype=np.int64)

    y_test = test["result"].to_numpy()

    print(
        f"\nFold {fold}: "
        f"træning={len(train)} kampe, "
        f"test={len(test)} kampe"
    )

    # Reference: den eksisterende logistiske regression med C=10.
    logistic = make_pipeline(
        SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        ),
        StandardScaler(),
        LogisticRegression(
            C=10.0,
            max_iter=1000,
            class_weight="balanced",
        ),
    )

    logistic.fit(train[BASE_FEATURES], train["result"])
    logistic_predictions = logistic.predict(test[BASE_FEATURES])

    for name, predictions in [
        ("Logistisk regression C=10", logistic_predictions)
    ]:
        scores = evaluate(y_test, predictions)

        results.append({
            "fold": fold,
            "model": name,
            **scores,
        })

        for actual, predicted in zip(y_test, predictions):
            oof_predictions.append({
                "fold": fold,
                "model": name,
                "actual": actual,
                "predicted": predicted,
            })

        print(
            f"  {name}: "
            f"accuracy={scores['accuracy']:.3f}, "
            f"macro F1={scores['macro_f1']:.3f}"
        )


    # Skalering læres kun fra subtrain i det aktuelle fold.
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    scaler = StandardScaler()

    x_subtrain = scaler.fit_transform(
        imputer.fit_transform(subtrain[NN_FEATURES])
    )
    x_validation = scaler.transform(
        imputer.transform(validation[NN_FEATURES])
    )
    x_test = scaler.transform(
        imputer.transform(test[NN_FEATURES])
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

    torch.manual_seed(42 + fold)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42 + fold)

    network = MatchNet(len(NN_FEATURES)).to(DEVICE)

    # Eksperiment: ingen klassevægte.
    loss_function = nn.CrossEntropyLoss()

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
            batch_indices = order[start:start + batch_size]

            batch_x = x_subtrain_tensor[batch_indices]
            batch_y = y_subtrain_tensor[batch_indices]

            optimizer.zero_grad()
            logits = network(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validering bruges til at vælge epoke,
        # aldrig til at opdatere vægtene direkte.
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

    print(
        f"  Early stopping: bedste epoke={best_epoch}, "
        f"valideringstab={best_validation_loss:.4f}, "
        f"trænede epoker={epoch}"
    )

    network.eval()
    with torch.no_grad():
        logits = network(x_test_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predicted_indices = probabilities.argmax(axis=1)

    neural_predictions = INDEX_TO_LABEL[predicted_indices]
    scores = evaluate(y_test, neural_predictions)

    results.append({
        "fold": fold,
        "model": "Neuralt netværk",
        **scores,
    })

    for actual, predicted, probs in zip(
        y_test, neural_predictions, probabilities
    ):
        oof_predictions.append({
            "fold": fold,
            "model": "Neuralt netværk",
            "actual": actual,
            "predicted": predicted,
            "prob_H": float(probs[0]),
            "prob_D": float(probs[1]),
            "prob_A": float(probs[2]),
        })

    print(
        f"  Neuralt netværk: "
        f"accuracy={scores['accuracy']:.3f}, "
        f"macro F1={scores['macro_f1']:.3f}"
    )


# Gennemsnit af hver folds målinger.
results_df = pd.DataFrame(results)

print("\n=== GENNEMSNIT PÅ TVÆRS AF FOLDS ===")
summary = results_df.groupby("model").agg(
    mean_accuracy=("accuracy", "mean"),
    mean_macro_f1=("macro_f1", "mean"),
    folds=("fold", "nunique"),
).sort_values("mean_macro_f1", ascending=False)

print(summary.round(3).to_string())


# Faktisk samlet evaluering af alle out-of-fold-forudsigelser.
oof_df = pd.DataFrame(oof_predictions)

# Gem out-of-fold-forudsigelser til senere sammenligning.
predictions_path = (
    PROJECT_ROOT / "data" / "oof_predictions_nn_without_class_weights.csv"
)
oof_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

print(f"\nGemte out-of-fold-forudsigelser i: {predictions_path}")


print("\n=== SAMLET OUT-OF-FOLD RESULTAT ===")
for name, group in oof_df.groupby("model", sort=False):
    accuracy = accuracy_score(group["actual"], group["predicted"])
    f1 = macro_f1(group["actual"], group["predicted"])

    print(
        f"{name}: accuracy={accuracy:.3f}, "
        f"macro F1={f1:.3f}, kampe={len(group)}"
    )

print("\n=== FORVEKSLINGSMATRICER ===")

for name, group in oof_df.groupby("model", sort=False):
    matrix = pd.crosstab(
        group["actual"],
        group["predicted"],
        rownames=["Faktisk resultat"],
        colnames=["Forudsagt resultat"],
    ).reindex(
        index=["H", "D", "A"],
        columns=["H", "D", "A"],
        fill_value=0,
    )

    print(f"\n{name}")
    print("Rækker = faktisk resultat, kolonner = forudsigelse")
    print(matrix.to_string())