
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, f1_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"

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

POISSON_COLUMNS = [
    "home_expected_goals",
    "away_expected_goals",
]

LABELS = ["H", "D", "A"]
MAX_GOALS = 15


def predict_probabilities(home_lambda, away_lambda):
    """Beregn sandsynligheder for hjemmesejr, uafgjort og udesejr."""

    goals = np.arange(MAX_GOALS + 1)

    home_probs = poisson.pmf(goals, home_lambda)
    away_probs = poisson.pmf(goals, away_lambda)

    score_matrix = np.outer(home_probs, away_probs)

    p_home = np.tril(score_matrix, k=-1).sum()
    p_draw = np.trace(score_matrix)
    p_away = np.triu(score_matrix, k=1).sum()

    probabilities = np.array([p_home, p_draw, p_away])

    # Normalisér for den meget lille sandsynlighedsmasse
    # uden for intervallet 0-15 mål.
    probabilities /= probabilities.sum()

    return probabilities


def evaluate(y_true, probabilities):
    predictions = np.array(LABELS)[
        np.argmax(probabilities, axis=1)
    ]

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=LABELS,
        ),
    }


# Indlæs data.
df = pd.read_csv(DATA_PATH, parse_dates=["date"], low_memory=False)
elo = pd.read_csv(ELO_PATH, parse_dates=["date"], low_memory=False)

# Tilføj både Elo- og Poisson-features.
df = df.merge(
    elo[KEY_COLUMNS + ELO_COLUMNS + POISSON_COLUMNS],
    on=KEY_COLUMNS,
    how="left",
    validate="one_to_one",
)

df = df.sort_values(
    ["date", "time"], kind="stable"
).reset_index(drop=True)

# Samme udvælgelse af kampe som i den eksisterende backtest.
df = df.dropna(
    subset=COMMON_FEATURES + ELO_COLUMNS
).reset_index(drop=True)

df = df[df["result"].isin(LABELS)].reset_index(drop=True)

if df[POISSON_COLUMNS].isna().any().any():
    raise ValueError("Poisson-features indeholder manglende værdier.")

if (df[POISSON_COLUMNS] <= 0).any().any():
    raise ValueError("Forventede mål skal være positive.")

dates = sorted(df["date"].dt.date.unique())

date_blocks = [
    list(block)
    for block in np.array_split(dates, 4)
    if len(block) > 0
]

print(f"Kampe i backtesten: {len(df)}")
print("Model: Poisson med dynamiske angrebs- og forsvarsratinger")
print(f"Maksimalt antal mål i beregningen: {MAX_GOALS}")

all_actual = []
all_probabilities = []
fold_results = []

for fold in range(1, len(date_blocks)):
    test_dates = date_blocks[fold]

    test = df[
        df["date"].dt.date.isin(test_dates)
    ].copy()

    if test.empty:
        continue

    y_test = test["result"].to_numpy()

    probabilities = np.vstack([
        predict_probabilities(home_lambda, away_lambda)
        for home_lambda, away_lambda in zip(
            test["home_expected_goals"],
            test["away_expected_goals"],
        )
    ])

    scores = evaluate(y_test, probabilities)

    fold_results.append({
        "fold": fold,
        "matches": len(test),
        **scores,
    })

    all_actual.extend(y_test)
    all_probabilities.extend(probabilities)

    print(
        f"\nFold {fold}: {len(test)} kampe"
        f"\n  Accuracy: {scores['accuracy']:.4f}"
        f"\n  Macro F1: {scores['macro_f1']:.4f}"
        f"\n  Log loss: {scores['log_loss']:.4f}"
    )

overall = evaluate(
    np.array(all_actual),
    np.array(all_probabilities),
)

print("\nSamlet resultat på alle out-of-fold-testkampe:")
print(f"Accuracy: {overall['accuracy']:.4f}")
print(f"Macro F1: {overall['macro_f1']:.4f}")
print(f"Log loss: {overall['log_loss']:.4f}")

print("\nFoldoversigt:")
print(pd.DataFrame(fold_results).round(4).to_string(index=False))
predicted_labels = np.array(LABELS)[
    np.argmax(np.array(all_probabilities), axis=1)
]

print("\nForvekslingsmatrix (rækker=faktisk, kolonner=forudsagt):")
print(
    pd.DataFrame(
        confusion_matrix(
            all_actual,
            predicted_labels,
            labels=LABELS,
        ),
        index=[f"Faktisk {label}" for label in LABELS],
        columns=[f"Forudsagt {label}" for label in LABELS],
    )
)

print("\nKlasserapport:")
print(
    classification_report(
        all_actual,
        predicted_labels,
        labels=LABELS,
        zero_division=0,
    )
)

print("\nFordeling af faktiske resultater:")
print(pd.Series(all_actual).value_counts(normalize=True).reindex(LABELS).round(4))

print("\nFordeling af forudsagte resultater:")
print(pd.Series(predicted_labels).value_counts(normalize=True).reindex(LABELS).round(4))
