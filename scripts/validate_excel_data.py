
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"

df = pd.read_csv(DATA_PATH)

print("=== DATASET ===")
print(f"Antal rækker: {len(df)}")
print(f"Antal kolonner: {len(df.columns)}")

print("\n=== LIGAER ===")
print(df["league"].value_counts().to_string())

print("\n=== DATOER ===")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print(f"Tidligste dato: {df['date'].min()}")
print(f"Seneste dato:   {df['date'].max()}")
print(f"Manglende datoer: {df['date'].isna().sum()}")

print("\n=== DUBLETTER ===")
duplicates = df.duplicated(
    subset=["league", "date", "home_team", "away_team"]
)
print(f"Mulige dubletter: {duplicates.sum()}")

print("\n=== RESULTATER ===")
print("Slutresultater fordelt på udfald:")
print(df["result"].value_counts(dropna=False).to_string())

print("\n=== MANGLENDE VÆRDIER ===")
print(df.isna().sum().to_string())

print("\n=== EKSEMPLER ===")
print(
    df[
        ["date", "league", "home_team", "away_team",
         "home_goals", "away_goals", "result"]
    ].head(10).to_string(index=False)
)
