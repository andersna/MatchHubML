
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/features_with_odds.csv")
OUTPUT_FILE = Path("data/odds_model_dataset.csv")

df = pd.read_csv(INPUT_FILE, low_memory=False)

odds_columns = ["odds_home", "odds_draw", "odds_away"]

for column in odds_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Behold kun kampe med tre gyldige, positive odds.
valid_odds = df[odds_columns].gt(0).all(axis=1)
df = df.loc[valid_odds].copy()

# Omdan decimalodds til implicitte sandsynligheder.
implied = 1 / df[odds_columns]
implied_sum = implied.sum(axis=1)

# Normalisér, så sandsynlighederne summerer til 1.
probabilities = implied.div(implied_sum, axis=0)

df["odds_prob_home"] = probabilities["odds_home"]
df["odds_prob_draw"] = probabilities["odds_draw"]
df["odds_prob_away"] = probabilities["odds_away"]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Kampe i oddsmodellen:", len(df))
print("Antal inputfeatures fra odds: 3")

print("\nEksempel på odds og normaliserede sandsynligheder:")
print(
    df[
        [
            "home_team",
            "away_team",
            "odds_home",
            "odds_draw",
            "odds_away",
            "odds_prob_home",
            "odds_prob_draw",
            "odds_prob_away",
        ]
    ].head().round(4).to_string(index=False)
)

probability_sum = probabilities.sum(axis=1)

print(
    "\nStørste afvigelse fra sandsynlighedssum 1:",
    round(float((probability_sum - 1).abs().max()), 10),
)

print(f"\nGemt i: {OUTPUT_FILE}")
