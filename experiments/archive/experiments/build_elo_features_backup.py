from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "elo_features.csv"

INITIAL_ELO = 1500
K_FACTORS = [16, 32, 48]
HOME_ADVANTAGE = 60


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def calculate_elo(data: pd.DataFrame, k_factor: int) -> pd.DataFrame:
    ratings = {}
    home_elo_before = []
    away_elo_before = []

    for row in data.itertuples(index=False):
        league = row.league
        home_team = row.home_team
        away_team = row.away_team

        home_key = (league, home_team)
        away_key = (league, away_team)

        home_rating = ratings.get(home_key, INITIAL_ELO)
        away_rating = ratings.get(away_key, INITIAL_ELO)

        # Gem ratings før kampens resultat opdaterer dem.
        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)

        expected_home = expected_score(
            home_rating + HOME_ADVANTAGE,
            away_rating,
        )

        home_actual = actual_score(row.home_goals, row.away_goals)
        away_actual = 1 - home_actual

        ratings[home_key] = home_rating + k_factor * (
            home_actual - expected_home
        )
        ratings[away_key] = away_rating + k_factor * (
            away_actual - (1 - expected_home)
        )

    return pd.DataFrame({
        f"home_elo_k{k_factor}": home_elo_before,
        f"away_elo_k{k_factor}": away_elo_before,
        f"elo_difference_k{k_factor}": [
            home - away
            for home, away in zip(home_elo_before, away_elo_before)
        ],
    })


df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["match_id"] = range(len(df))
df = df.sort_values(
    ["date", "time", "match_id"]
).reset_index(drop=True)

elo_columns = []

for k_factor in K_FACTORS:
    elo_features = calculate_elo(df, k_factor)
    df = pd.concat([df, elo_features], axis=1)
    elo_columns.extend(elo_features.columns)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Antal kampe: {len(df)}")
print(f"K-faktorer: {K_FACTORS}")
print("\nEksempel på Elo-ratinger:")
print(
    df[
        [
            "date",
            "league",
            "home_team",
            "away_team",
            "elo_difference_k16",
            "elo_difference_k32",
            "elo_difference_k48",
        ]
    ]
    .head(10)
    .to_string(index=False)
)
print(f"\nGemte Elo-features i: {OUTPUT_PATH}")