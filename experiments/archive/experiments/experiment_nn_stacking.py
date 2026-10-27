
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss


ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "oof_predictions_nn_odds_comparison.csv"
OUTPUT = ROOT / "data" / "nn_stacking_results.csv"

NN_NAME = "Model B - med odds"
BOOKMAKER_NAME = "Bet365 - odds alene"

KEYS = [
    "fold", "date", "league", "home_team", "away_team", "actual"
]
PROB_COLS = ["prob_H", "prob_D", "prob_A"]
LABELS = ["H", "D", "A"]



def get_metrics(actual, probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities = probabilities / probabilities.sum(
        axis=1, keepdims=True
    )

    predicted = np.array(LABELS)[probabilities.argmax(axis=1)]
    one_hot = np.eye(3)[
        np.array([LABELS.index(value) for value in actual])
    ]

    # Vores kolonner er H, D, A.
    # sklearn skal have både labels og sandsynligheder i samme rækkefølge.
    probabilities_for_log_loss = probabilities[:, [2, 1, 0]]

    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            actual,
            probabilities_for_log_loss,
            labels=["A", "D", "H"],
        ),
        "brier_score": np.mean(
            np.sum((one_hot - probabilities) ** 2, axis=1)
        ),
    }

# Hent de gemte out-of-fold-forudsigelser.
df = pd.read_csv(INPUT, parse_dates=["date"])

nn = df[df["model"] == NN_NAME][KEYS + PROB_COLS].copy()
bookmaker = df[
    df["model"] == BOOKMAKER_NAME
][KEYS + PROB_COLS].copy()

nn = nn.rename(
    columns={c: f"{c}_nn" for c in PROB_COLS}
)
bookmaker = bookmaker.rename(
    columns={c: f"{c}_bookmaker" for c in PROB_COLS}
)

data = nn.merge(
    bookmaker,
    on=KEYS,
    how="inner",
    validate="one_to_one",
)

if data.empty:
    raise ValueError("Ingen fælles kampe fundet.")

nn_columns = [f"{c}_nn" for c in PROB_COLS]
bookmaker_columns = [f"{c}_bookmaker" for c in PROB_COLS]

nn_probs = data[nn_columns].to_numpy(dtype=float)
bookmaker_probs = data[bookmaker_columns].to_numpy(dtype=float)

# Normalisér sandsynlighederne før brug.
nn_probs = np.clip(nn_probs, 1e-12, 1.0)
nn_probs /= nn_probs.sum(axis=1, keepdims=True)

bookmaker_probs = np.clip(bookmaker_probs, 1e-12, 1.0)
bookmaker_probs /= bookmaker_probs.sum(axis=1, keepdims=True)

# Stacking-modellen får seks inputs:
# tre fra netværket og tre fra Bet365.
X = np.column_stack([nn_probs, bookmaker_probs])
y = data["actual"].to_numpy()

train_mask = data["fold"].isin([1, 2]).to_numpy()
test_mask = (data["fold"] == 3).to_numpy()

if not train_mask.any() or not test_mask.any():
    raise ValueError("Fold 1+2 til træning og fold 3 til test kræves.")

print(f"Fælles kampe: {len(data)}")
print(f"Træningskampe (fold 1+2): {train_mask.sum()}")
print(f"Testkampe (fold 3): {test_mask.sum()}")

# Meta-modellen trænes kun på tidligere OOF-forudsigelser.
stacker = LogisticRegression(
    C=1.0,
    max_iter=2000,
)
stacker.fit(X[train_mask], y[train_mask])

# Sørg for, at sandsynlighederne står i rækkefølgen H, D, A.
raw_probabilities = stacker.predict_proba(X[test_mask])
class_positions = [
    list(stacker.classes_).index(label) for label in LABELS
]
stacking_probs = raw_probabilities[:, class_positions]

actual_test = y[test_mask]

# Baselines og stacking evalueres på præcis samme testkampe.
comparisons = [
    ("Bet365 alene", bookmaker_probs[test_mask]),
    ("Neuralt netværk med odds", nn_probs[test_mask]),
    ("Stacking - logistisk regression", stacking_probs),
]

results = []

print("\n=== RESULTATER PÅ FOLD 3 ===")

for name, probabilities in comparisons:
    scores = get_metrics(actual_test, probabilities)

    results.append({
        "model": name,
        **scores,
        "kampe": len(actual_test),
    })

    print(f"\n{name}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print(f"\nResultater gemt i: {OUTPUT}")