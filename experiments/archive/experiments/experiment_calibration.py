import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

df = pd.read_csv("data/oof_predictions_early_stopping.csv")
df = df[df["model"] == "Neuralt netværk"].copy()

label_to_index = {"H": 0, "D": 1, "A": 2}
prob_cols = ["prob_H", "prob_D", "prob_A"]

def prepare(group):
    probabilities = group[prob_cols].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    # Log-sandsynligheder bruges som input til kalibreringen.
    X = np.log(probabilities)
    y = group["actual"].map(label_to_index).to_numpy()
    return X, y, probabilities

def evaluate(y, probabilities):
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, predicted),
        "macro_f1": f1_score(
            y, predicted, labels=[0, 1, 2], average="macro"
        ),
        "log_loss": log_loss(y, probabilities, labels=[0, 1, 2]),
        "andel_D": np.mean(predicted == 1),
    }


def temperature_scale(probabilities, temperature):
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    logits = np.log(probabilities)
    logits = logits / temperature

    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def find_best_temperature(y, probabilities):
    candidates = np.arange(0.70, 2.01, 0.05)
    results = []

    for temperature in candidates:
        scaled = temperature_scale(probabilities, temperature)
        loss = log_loss(y, scaled, labels=[0, 1, 2])
        results.append((loss, temperature))

    best_loss, best_temperature = min(results)
    return best_temperature, best_loss


for test_fold in [2, 3]:
    train = df[df["fold"] < test_fold]
    test = df[df["fold"] == test_fold]

    _, y_train, train_probs = prepare(train)
    _, y_test, test_probs = prepare(test)

    temperature, train_loss = find_best_temperature(
        y_train, train_probs
    )

    scaled_probs = temperature_scale(test_probs, temperature)

    original = evaluate(y_test, test_probs)
    calibrated = evaluate(y_test, scaled_probs)

    print(f"\n=== TEST PÅ FOLD {test_fold} ===")
    print(f"Træningsdata til kalibrering: {len(train)} kampe")
    print(f"Testkampe: {len(test)}")
    print(f"Valgt temperatur: {temperature:.2f}")
    print(f"Log loss på kalibreringsdata: {train_loss:.4f}")

    for name, result in [
        ("Original model", original),
        ("Temperature scaling", calibrated),
    ]:
        print(
            f"{name:22s} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro F1: {result['macro_f1']:.4f} | "
            f"Log loss: {result['log_loss']:.4f} | "
            f"Andel D: {result['andel_D']:.3f}"
        )

