
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"

FEATURE_COLUMNS = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
]

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values(["date", "time"]).reset_index(drop=True)

# Fjern kampe, hvor et af holdene endnu ikke har historik.
df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

# De ældste 80 % bruges til træning, de nyeste 20 % til test.
split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[FEATURE_COLUMNS]
y_train = train["result"]
X_test = test[FEATURE_COLUMNS]
y_test = test["result"]

# Baseline: forudsiger altid den mest almindelige klasse i træningsdata.
model = make_pipeline(
    SimpleImputer(strategy="median"),
    DummyClassifier(strategy="most_frequent"),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("=== DATA ===")
print(f"Kampe efter fjernelse af manglende historik: {len(df)}")
print(f"Træningskampe: {len(train)}")
print(f"Testkampe: {len(test)}")

print("\n=== BASELINE-RESULTAT ===")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

print("\n=== KLASSIFIKATIONSRAPPORT ===")
print(classification_report(
    y_test,
    predictions,
    labels=["H", "D", "A"],
    zero_division=0,
))

print("Bemærk: Modellen forudsiger altid den mest almindelige klasse.")