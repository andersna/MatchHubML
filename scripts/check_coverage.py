
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"

df = pd.read_csv(DATA_PATH)

# Hver kamp tæller som én kamp for begge hold.
home = df[["league", "home_team"]].rename(
    columns={"home_team": "team"}
)
away = df[["league", "away_team"]].rename(
    columns={"away_team": "team"}
)

appearances = pd.concat([home, away], ignore_index=True)

counts = (
    appearances.groupby(["league", "team"])
    .size()
    .rename("matches")
    .reset_index()
)

print("=== HOLD OG KAMPDÆKNING ===")
for league, group in counts.groupby("league"):
    print(f"\n{league}: {group['team'].nunique()} hold")
    print(
        group.sort_values("matches").to_string(index=False)
    )
    print(
        f"Min. kampe pr. hold: {group['matches'].min()}"
        f" | Maks.: {group['matches'].max()}"
    )