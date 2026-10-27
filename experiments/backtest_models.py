from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"

BASIC_FEATURES = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
]

FORM_FEATURES = [
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

BASE_FEATURES = BASIC_FEATURES + FORM_FEATURES

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
elo = pd.read_csv(ELO_PATH, parse_dates=["date"])

elo_columns = []
for k in [16, 32, 48]:
    elo_columns.extend([
        f"home_elo_k{k}",
        f"away_elo_k{k}",
        f"elo_difference_k{k}",
    ])

df = df.merge(
    elo[KEY_COLUMNS + elo_columns],
    on=KEY_COLUMNS,
    how="left",
    validate="one_to_one",
)

df = df.sort_values(
    ["date", "time"], kind="stable"
).reset_index(drop=True)

# Samme kampe for grundmodellen og alle Elo-varianter.
# Vi beholder det hidtidige fælles datagrundlag.
COMMON_FEATURES = BASE_FEATURES + [
    "home_form_rating",
    "away_form_rating",
    "home_venue_form_rating",
    "away_venue_form_rating",
]
df = df.dropna(subset=COMMON_FEATURES + elo_columns).reset_index(drop=True)

dates = sorted(df["date"].dt.date.unique())
date_blocks = [
    list(block)
    for block in np.array_split(dates, 4)
    if len(block) > 0
]



def make_logistic_model(c=1.0):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=1000,
            class_weight="balanced",
        ),
    )


models = {
    "Baseline": (
        lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            DummyClassifier(strategy="most_frequent"),
        ),
        BASIC_FEATURES,
    ),
}

for c in [0.01, 0.1, 1.0, 10.0]:
    models[f"Eksisterende features + C={c}"] = (
        lambda c=c: make_logistic_model(c=c),
        BASE_FEATURES,
    )


for k in [16, 32, 48]:
    features = BASE_FEATURES + [
        f"home_elo_k{k}",
        f"away_elo_k{k}",
        f"elo_difference_k{k}",
    ]
    models[f"Eksisterende features + Elo K={k}"] = (
        make_logistic_model,
        features,
    )

results = []

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

    print(
        f"\nFold {fold}: træning "
        f"{train['date'].min().date()} til "
        f"{train['date'].max().date()} | test "
        f"{test['date'].min().date()} til "
        f"{test['date'].max().date()}"
    )

    for name, (make_model, columns) in models.items():
        model = make_model()
        model.fit(train[columns], train["result"])
        predictions = model.predict(test[columns])

        accuracy = accuracy_score(test["result"], predictions)
        macro_f1 = f1_score(
            test["result"],
            predictions,
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0,
        )

        results.append({
            "fold": fold,
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "test_matches": len(test),
        })

        print(
            f"  {name}: accuracy={accuracy:.3f}, "
            f"macro F1={macro_f1:.3f}, kampe={len(test)}"
        )

results_df = pd.DataFrame(results)

print("\n=== GENNEMSNIT PÅ TVÆRS AF FOLDS ===")
summary = results_df.groupby("model").agg(
    mean_accuracy=("accuracy", "mean"),
    mean_macro_f1=("macro_f1", "mean"),
    folds=("fold", "nunique"),
).sort_values("mean_macro_f1", ascending=False)

print(summary.round(3).to_string())

print("\n=== SAMLET OUT-OF-FOLD RESULTAT ===")
for name in models:
    subset = results_df[results_df["model"] == name]
    print(
        f"{name}: gennemsnitlig accuracy="
        f"{subset['accuracy'].mean():.3f}, "
        f"gennemsnitlig macro F1="
        f"{subset['macro_f1'].mean():.3f}"
    )