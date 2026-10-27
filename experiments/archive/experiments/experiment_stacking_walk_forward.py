
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "oof_predictions_nn_odds_comparison.csv"
OUTPUT = ROOT / "data" / "stacking_walk_forward_results.csv"

NN_NAME = "Model B - med odds"
BOOKMAKER_NAME = "Bet365 - odds alene"

KEYS = ["fold", "date", "league", "home_team", "away_team", "actual"]
PROB_COLS = ["prob_H", "prob_D", "prob_A"]
LABELS = ["H", "D", "A"]


def metrics(actual, probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    predicted = np.array(LABELS)[probabilities.argmax(axis=1)]
    one_hot = np.eye(3)[
        np.array([LABELS.index(value) for value in actual])
    ]

    # log_loss kræver, at sandsynlighedernes kolonner følger
    # samme rækkefølge som labels.
    probabilities_for_loss = probabilities[:, [2, 1, 0]]

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
            probabilities_for_loss,
            labels=["A", "D", "H"],
        ),
        "brier_score": np.mean(
            np.sum((one_hot - probabilities) ** 2, axis=1)
        ),
    }


# Indlæs tidligere out-of-fold-forudsigelser.
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
).sort_values(["fold", "date"]).reset_index(drop=True)

if data.empty:
    raise ValueError("Ingen fælles forudsigelser fundet.")

nn_probs = data[[f"{c}_nn" for c in PROB_COLS]].to_numpy(float)
book_probs = data[
    [f"{c}_bookmaker" for c in PROB_COLS]
].to_numpy(float)

nn_probs = np.clip(nn_probs, 1e-12, 1.0)
nn_probs /= nn_probs.sum(axis=1, keepdims=True)

book_probs = np.clip(book_probs, 1e-12, 1.0)
book_probs /= book_probs.sum(axis=1, keepdims=True)

# Seks inputværdier: H/D/A fra netværket og H/D/A fra Bet365.
X = np.column_stack([nn_probs, book_probs])
y = data["actual"].to_numpy()
folds = data["fold"].to_numpy()

results = []

# Walk-forward: hver testfold bruger kun tidligere folds til træning.
for test_fold in sorted(np.unique(folds))[1:]:
    train_mask = folds < test_fold
    test_mask = folds == test_fold

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    stacker = LogisticRegression(
        C=1.0,
        max_iter=2000,
    )
    stacker.fit(X_train, y_train)

    raw_probs = stacker.predict_proba(X_test)
    class_positions = [
        list(stacker.classes_).index(label)
        for label in LABELS
    ]
    stacking_probs = raw_probs[:, class_positions]

    test_nn = nn_probs[test_mask]
    test_book = book_probs[test_mask]

    models = [
        ("Bet365 alene", test_book),
        ("Neuralt netværk med odds", test_nn),
        ("Stacking", stacking_probs),
    ]

    print(
        f"\n=== Fold {test_fold}: "
        f"træning={train_mask.sum()}, test={test_mask.sum()} ==="
    )

    for name, probabilities in models:
        scores = metrics(y_test, probabilities)
        print(
            f"{name}: "
            f"accuracy={scores['accuracy']:.4f}, "
            f"macro_f1={scores['macro_f1']:.4f}, "
            f"log_loss={scores['log_loss']:.4f}, "
            f"brier={scores['brier_score']:.4f}"
        )

        results.append({
            "test_fold": int(test_fold),
            "model": name,
            "train_matches": int(train_mask.sum()),
            "test_matches": int(test_mask.sum()),
            **scores,
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n=== RESULTATER GEMT ===")
print(OUTPUT)