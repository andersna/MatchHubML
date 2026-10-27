from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "oof_predictions_early_stopping.csv"

PROB_COLUMNS = ["prob_H", "prob_D", "prob_A"]
LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
LABELS = [0, 1, 2]

df = pd.read_csv(DATA_PATH)
df = df[df["model"] == "Neuralt netværk"].copy()
df["y"] = df["actual"].map(LABEL_TO_INDEX)

train = df[df["fold"].isin([1, 2])]
test = df[df["fold"] == 3]

def normalize(probabilities):
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)

def power_calibrate(probabilities, temperature):
    adjusted = np.clip(probabilities, 1e-8, 1.0) ** (1.0 / temperature)
    return normalize(adjusted)

p_train = normalize(train[PROB_COLUMNS].to_numpy(dtype=float))
y_train = train["y"].to_numpy()

p_test = normalize(test[PROB_COLUMNS].to_numpy(dtype=float))
y_test = test["y"].to_numpy()

def objective(temperature):
    calibrated = power_calibrate(p_train, temperature)
    return log_loss(y_train, calibrated, labels=LABELS)

result = minimize_scalar(
    objective,
    bounds=(0.25, 3.0),
    method="bounded",
)

best_temperature = float(result.x)
p_calibrated = power_calibrate(p_test, best_temperature)

def report(name, probabilities):
    predictions = probabilities.argmax(axis=1)
    print(f"\n{name}")
    print(f"Log loss: {log_loss(y_test, probabilities, labels=LABELS):.5f}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Macro F1: {f1_score(y_test, predictions, labels=LABELS, average='macro'):.4f}"
    )
    print("Gennemsnitlige sandsynligheder H/D/A:", np.round(probabilities.mean(axis=0), 4))
    print("Forudsagte klasser H/D/A:", np.round(
        np.bincount(predictions, minlength=3) / len(predictions), 4
    ))

print("Neuralt netværk – power calibration")
print(f"Kalibreringskampe: {len(train)}")
print(f"Testkampe: {len(test)}")
print(f"Valgt temperatur fra fold 1+2: {best_temperature:.4f}")

report("FØR kalibrering – fold 3", p_test)
report("EFTER kalibrering – fold 3", p_calibrated)

print("\nEksperiment færdigt. Ingen eksisterende filer er ændret.")
