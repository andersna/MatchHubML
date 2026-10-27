from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "oof_predictions_early_stopping.csv"

LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
LABELS = [0, 1, 2]
PROB_COLUMNS = ["prob_H", "prob_D", "prob_A"]

df = pd.read_csv(DATA_PATH)
df = df[df["model"] == "Neuralt netværk"].copy()
df["y"] = df["actual"].map(LABEL_TO_INDEX)

train = df[df["fold"].isin([1, 2])].copy()
test = df[df["fold"] == 3].copy()

def get_log_probabilities(data):
    probabilities = data[PROB_COLUMNS].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    return np.log(probabilities)

X_train = get_log_probabilities(train)
y_train = train["y"].to_numpy()
X_test = get_log_probabilities(test)
y_test = test["y"].to_numpy()

# Baseline: det neurale netværks oprindelige sandsynligheder.
baseline_probabilities = test[PROB_COLUMNS].to_numpy(dtype=float, copy=True)
baseline_probabilities /= baseline_probabilities.sum(axis=1, keepdims=True)

# Kalibratoren lærer kun fra fold 1 og 2.
calibrator = LogisticRegression(max_iter=2000, C=1.0)
calibrator.fit(X_train, y_train)

calibrated_probabilities = calibrator.predict_proba(X_test)

# Sørg for, at kolonnerne følger rækkefølgen H, D, A.
class_positions = {label: i for i, label in enumerate(calibrator.classes_)}
calibrated_probabilities = calibrated_probabilities[
    :, [class_positions[label] for label in LABELS]
]

def evaluate(name, probabilities):
    predictions = probabilities.argmax(axis=1)
    print(f"\n{name}")
    print(f"Log loss:  {log_loss(y_test, probabilities, labels=LABELS):.5f}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Macro F1:  "
        f"{f1_score(y_test, predictions, labels=LABELS, average='macro'):.4f}"
    )
    print(
        "Gennemsnitlige sandsynligheder H/D/A:",
        np.round(probabilities.mean(axis=0), 4),
    )

print("Neuralt netværk – logistisk kalibrering")
print(f"Træningskampe til kalibrering: {len(train)}")
print(f"Testkampe i fold 3: {len(test)}")

evaluate("FØR kalibrering", baseline_probabilities)
evaluate("EFTER kalibrering", calibrated_probabilities)

print("\nEksperiment færdigt. Ingen eksisterende filer er ændret.")

