
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
]

ELO_COLUMNS = [
    f"{prefix}_elo_k{k}"
    for k in [16, 32, 48]
    for prefix in ["home", "away"]
] + [
    f"elo_difference_k{k}" for k in [16, 32, 48]
]

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
    elo[KEY_COLUMNS + ELO_COLUMNS],
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

    # Skalering læres kun fra træningsdata i det aktuelle fold.
    imputer = SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )
    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        imputer.fit_transform(train[BASE_FEATURES])
    )
    x_test = scaler.transform(
        imputer.transform(test[BASE_FEATURES])
    )

    torch.manual_seed(42 + fold)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42 + fold)

    x_train_tensor = torch.tensor(
        x_train, dtype=torch.float32, device=DEVICE
    )
    y_train_tensor = torch.tensor(
        y_train, dtype=torch.long, device=DEVICE
    )
    x_test_tensor = torch.tensor(
        x_test, dtype=torch.float32, device=DEVICE
    )

    network = MatchNet(len(BASE_FEATURES)).to(DEVICE)

    # Balancer klasserne i træningsdata, når de er til stede.
    counts = np.bincount(y_train, minlength=3)
    present = counts > 0
    class_weights = np.zeros(3, dtype=np.float32)
    class_weights[present] = (
        len(y_train) / (present.sum() * counts[present])
    )

    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=DEVICE)
    )

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )

    # Første baseline-eksperiment med minibatches.
    epochs = 50
    batch_size = 256

    network.train()

    for epoch in range(epochs):
        order = torch.randperm(len(x_train_tensor), device=DEVICE)

        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]

            batch_x = x_train_tensor[batch_indices]
            batch_y = y_train_tensor[batch_indices]

            optimizer.zero_grad()
            logits = network(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()

    network.eval()
    with torch.no_grad():
        logits = network(x_test_tensor)
        predicted_indices = logits.argmax(dim=1).cpu().numpy()

    neural_predictions = INDEX_TO_LABEL[predicted_indices]
    scores = evaluate(y_test, neural_predictions)

    results.append({
        "fold": fold,
        "model": "Neuralt netværk",
        **scores,
    })

    for actual, predicted in zip(y_test, neural_predictions):
        oof_predictions.append({
            "fold": fold,
            "model": "Neuralt netværk",
            "actual": actual,
            "predicted": predicted,
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
predictions_path = PROJECT_ROOT / "data" / "oof_predictions.csv"
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