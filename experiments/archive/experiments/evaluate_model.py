
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"

FEATURES = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
]
LABELS = ["H", "D", "A"]

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["date", "time"]).reset_index(drop=True)
df = df.dropna(subset=FEATURES).reset_index(drop=True)

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)

model.fit(train[FEATURES], train["result"])
predictions = model.predict(test[FEATURES])

matrix = confusion_matrix(
    test["result"],
    predictions,
    labels=LABELS,
)

result = pd.DataFrame(
    matrix,
    index=[f"Faktisk {label}" for label in LABELS],
    columns=[f"Forudsagt {label}" for label in LABELS],
)

print("Rækker = faktiske resultater")
print("Kolonner = modellens forudsigelser\n")
print(result.to_string())
print(f"\nAntal testkampe: {len(test)}")
