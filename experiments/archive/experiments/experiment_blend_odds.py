
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "oof_predictions_nn_odds_comparison.csv"
OUTPUT = ROOT / "data" / "blend_odds_weight_results.csv"

NN_NAME = "Model B - med odds"
BOOKMAKER_NAME = "Bet365 - odds alene"

KEYS = [
    "fold", "date", "league", "home_team", "away_team", "actual"
]
PROB_COLS = ["prob_H", "prob_D", "prob_A"]


def metrics(actual, probabilities):
    labels = {"H": 0, "D": 1, "A": 2}
    y = np.array([labels[value] for value in actual])
    probabilities = np.asarray(probabilities, dtype=float)

    # Undgå små afrundingsforskelle i sandsynlighedssummen.
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    predicted = np.array(["H", "D", "A"])[probabilities.argmax(axis=1)]
    one_hot = np.eye(3)[y]

    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(y, probabilities, labels=[0, 1, 2]),
        "brier_score": np.mean(
            np.sum((one_hot - probabilities) ** 2, axis=1)
        ),
    }


# Indlæs de eksisterende out-of-fold-forudsigelser.
df = pd.read_csv(INPUT, parse_dates=["date"])

required_models = {NN_NAME, BOOKMAKER_NAME}
available_models = set(df["model"].unique())

if not required_models.issubset(available_models):
    raise ValueError(
        f"Mangler modeller. Fundet: {sorted(available_models)}"
    )

nn = df[df["model"] == NN_NAME][KEYS + PROB_COLS].copy()
bookmaker = df[
    df["model"] == BOOKMAKER_NAME
][KEYS + PROB_COLS].copy()

nn = nn.rename(columns={col: f"{col}_nn" for col in PROB_COLS})
bookmaker = bookmaker.rename(
    columns={col: f"{col}_bookmaker" for col in PROB_COLS}
)

data = nn.merge(
    bookmaker,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

if data.empty:
    raise ValueError("Ingen fælles kampe mellem modellerne.")

print(f"Fælles kampe: {len(data)}")
print("Kampe pr. fold:")
print(data.groupby("fold").size().to_string())

nn_probs = data[[f"{c}_nn" for c in PROB_COLS]].to_numpy()
bookmaker_probs = data[
    [f"{c}_bookmaker" for c in PROB_COLS]
].to_numpy()

# Normaliser begge kilders sandsynligheder.
nn_probs = np.clip(nn_probs, 1e-12, 1.0)
nn_probs /= nn_probs.sum(axis=1, keepdims=True)

bookmaker_probs = np.clip(bookmaker_probs, 1e-12, 1.0)
bookmaker_probs /= bookmaker_probs.sum(axis=1, keepdims=True)

actual = data["actual"].to_numpy()

# Brug fold 1 og 2 til at vælge vægt.
tuning_mask = data["fold"].isin([1, 2]).to_numpy()
test_mask = (data["fold"] == 3).to_numpy()

if not tuning_mask.any() or not test_mask.any():
    raise ValueError("Forventede at finde både fold 1+2 og fold 3.")

weights = np.round(np.arange(0.0, 1.01, 0.1), 2)
results = []

for weight in weights:
    blended = (
        weight * nn_probs
        + (1.0 - weight) * bookmaker_probs
    )
    blended /= blended.sum(axis=1, keepdims=True)

    tuning_scores = metrics(
        actual[tuning_mask],
        blended[tuning_mask],
    )
    test_scores = metrics(
        actual[test_mask],
        blended[test_mask],
    )

    results.append({
        "nn_weight": weight,
        "bookmaker_weight": round(1.0 - weight, 2),
        **{
            f"tuning_{key}": value
            for key, value in tuning_scores.items()
        },
        **{
            f"fold3_{key}": value
            for key, value in test_scores.items()
        },
    })

results_df = pd.DataFrame(results)

# Log loss er det primære kriterium for valg af vægt.
best_row = results_df.loc[
    results_df["tuning_log_loss"].idxmin()
]

best_weight = float(best_row["nn_weight"])
best_bookmaker_weight = float(best_row["bookmaker_weight"])

print("\n=== VÆGTTEST: FOLD 1+2 ===")
print(
    results_df[
        [
            "nn_weight",
            "bookmaker_weight",
            "tuning_accuracy",
            "tuning_macro_f1",
            "tuning_log_loss",
            "tuning_brier_score",
        ]
    ].round(4).to_string(index=False)
)

print("\n=== VALGT VÆGT ===")
print(f"Neuralt netværk: {best_weight:.0%}")
print(f"Bet365:          {best_bookmaker_weight:.0%}")
print("Valgt ud fra laveste log loss på fold 1+2.")

print("\n=== EVALUERING PÅ SENERE FOLD 3 ===")

# Sammenlign den valgte kombination med begge kilder alene.
selected_probs = (
    best_weight * nn_probs[test_mask]
    + best_bookmaker_weight * bookmaker_probs[test_mask]
)
selected_probs /= selected_probs.sum(axis=1, keepdims=True)

comparisons = [
    ("Neuralt netværk med odds", nn_probs[test_mask]),
    ("Bet365 alene", bookmaker_probs[test_mask]),
    ("Kombineret model", selected_probs),
]

for name, probabilities in comparisons:
    scores = metrics(actual[test_mask], probabilities)
    print(f"\n{name}:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")

results_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nAlle vægtresultater gemt i: {OUTPUT}")
