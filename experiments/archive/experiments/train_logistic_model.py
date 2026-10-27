
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"

FEATURE_COLUMNS = [
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

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["date", "time"]).reset_index(drop=True)
df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

# Træn på de ældste 80 %, og test på de nyeste 20 %.
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[FEATURE_COLUMNS]
y_train = train["result"]
X_test = test[FEATURE_COLUMNS]
y_test = test["result"]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("=== LOGISTISK REGRESSION ===")
print(f"Træningskampe: {len(train)}")
print(f"Testkampe: {len(test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

print("\n=== KLASSIFIKATIONSRAPPORT ===")
print(classification_report(
    y_test,
    predictions,
    labels=["H", "D", "A"],
    zero_division=0,
))

print("=== EKSEMPLER PÅ FORUDSIGELSER ===")
results = test[["date", "league", "home_team", "away_team", "result"]].copy()
results["prediction"] = predictions
print(results.tail(10).to_string(index=False))
