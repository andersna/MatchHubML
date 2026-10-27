from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features.csv"

BASIC_FEATURES = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
]

RECENT_FORM_FEATURES = [
    "home_last5_goals_for",
    "home_last5_goals_against",
    "home_last5_points",
    "away_last5_goals_for",
    "away_last5_goals_against",
    "away_last5_points",
]

FORM_RATING_FEATURES = [
    "home_form_rating",
    "away_form_rating",
]

VENUE_FORM_FEATURES = [
    "home_venue_form_rating",
    "away_venue_form_rating",
]

BASE_FEATURES = BASIC_FEATURES + RECENT_FORM_FEATURES
WITH_FORM = BASE_FEATURES + FORM_RATING_FEATURES
WITH_VENUE = BASE_FEATURES + VENUE_FORM_FEATURES
WITH_BOTH = BASE_FEATURES + FORM_RATING_FEATURES + VENUE_FORM_FEATURES

df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
df = df.sort_values(
    ["date", "time"], kind="stable"
).reset_index(drop=True)

# Samme kampe for alle modeller i denne sammenligning.
df = df.dropna(subset=WITH_BOTH).reset_index(drop=True)

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

y_train = train["result"]
y_test = test["result"]


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    )


models = {
    "Eksisterende features": BASE_FEATURES,
    "Eksisterende + samlet form-rating": WITH_FORM,
    "Eksisterende + hjemme-/udeform": WITH_VENUE,
    "Eksisterende + begge form-ratings": WITH_BOTH,
}

print("=== FÆLLES TESTGRUNDLAG ===")
print(f"Kampe i alt: {len(df)}")
print(f"Træningskampe: {len(train)}")
print(f"Testkampe: {len(test)}")
print(
    f"Testperiode: {test['date'].min().date()} "
    f"til {test['date'].max().date()}"
)

print("\n=== MODELSAMMENLIGNING ===")

for name, feature_columns in models.items():
    model = make_model()
    model.fit(train[feature_columns], y_train)
    predictions = model.predict(test[feature_columns])

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0,
    )

    print(f"\n{name}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Macro F1: {macro_f1:.3f}")