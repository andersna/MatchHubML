
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df["match_id"] = range(len(df))
df = df.sort_values(
    ["date", "time", "match_id"]
).reset_index(drop=True)

# Historisk hjemmebanefordel pr. liga.
# Hjemme- og udepoint beregnes kun ud fra tidligere afsluttede kampe.
df["home_points"] = (
    (df["home_goals"] > df["away_goals"]) * 3
    + (df["home_goals"] == df["away_goals"]) * 1
)

df["away_points"] = (
    (df["away_goals"] > df["home_goals"]) * 3
    + (df["home_goals"] == df["away_goals"]) * 1
)

df["league_home_advantage"] = (
    df.groupby("league", sort=False)["home_points"]
    .transform(lambda s: s.shift(1).expanding().mean())
    - df.groupby("league", sort=False)["away_points"]
    .transform(lambda s: s.shift(1).expanding().mean())
)


# Opret én observation pr. hold pr. kamp.
# Medtag skudstatistik, så vi senere kan beregne historisk
# angrebs- og forsvarsform uden at bruge aktuelle kampdata.

home = df[
    [
        "match_id", "date", "time", "league",
        "home_team", "home_goals", "away_goals",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
    ]
].copy()

home.columns = [
    "match_id", "date", "time", "league",
    "team", "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_on_target_for", "shots_on_target_against",
]
home["is_home"] = 1

away = df[
    [
        "match_id", "date", "time", "league",
        "away_team", "home_goals", "away_goals",
        "away_shots", "home_shots",
        "away_shots_on_target", "home_shots_on_target",
    ]
].copy()

away.columns = [
    "match_id", "date", "time", "league",
    "team", "goals_against", "goals_for",
    "shots_for", "shots_against",
    "shots_on_target_for", "shots_on_target_against",
]
away["is_home"] = 0

team_matches = pd.concat([home, away], ignore_index=True)

team_matches = team_matches.sort_values(
    ["league", "team", "date", "time", "match_id"]
).reset_index(drop=True)

team_matches["points"] = (
    (team_matches["goals_for"] > team_matches["goals_against"]) * 3
    + (team_matches["goals_for"] == team_matches["goals_against"]) * 1
)


# Målforskel for hver kamp.
team_matches["goal_difference"] = (
    team_matches["goals_for"] - team_matches["goals_against"]
)

# Samlet form-rating fra de fem seneste tidligere kampe.
# Point vægtes højere end målforskel.
grouped = team_matches.groupby(["league", "team"], sort=False)

team_matches["form_rating"] = (
    grouped["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ) * 2
    + grouped["goal_difference"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
)


grouped = team_matches.groupby(["league", "team"], sort=False)


# Form baseret på tidligere kampe i samme rolle:
# hjemmeholdets tidligere hjemmekampe og udeholdets tidligere udekampe.
venue_grouped = team_matches.groupby(
    ["league", "team", "is_home"],
    sort=False,
)

team_matches["venue_form_rating"] = (
    venue_grouped["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    ) * 2
    + venue_grouped["goal_difference"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
)


# Historik fra alle tidligere kampe.
for column, output_name in [
    ("goals_for", "avg_goals_for"),
    ("goals_against", "avg_goals_against"),
    ("points", "avg_points"),
]:
    team_matches[output_name] = grouped[column].transform(
        lambda s: s.shift(1).expanding().mean()
    )


# Historik fra de seneste fem tidligere kampe.
for column, output_name in [
    ("goals_for", "last5_goals_for"),
    ("goals_against", "last5_goals_against"),
    ("points", "last5_points"),
    ("shots_for", "last5_shots_for"),
    ("shots_against", "last5_shots_against"),
    ("shots_on_target_for", "last5_shots_on_target_for"),
    ("shots_on_target_against", "last5_shots_on_target_against"),
]:
    team_matches[output_name] = grouped[column].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )



feature_columns = [
    "avg_goals_for",
    "avg_goals_against",
    "avg_points",
    "last5_goals_for",
    "last5_goals_against",
    "last5_points",
    "last5_shots_for",
    "last5_shots_against",
    "last5_shots_on_target_for",
    "last5_shots_on_target_against",
    "form_rating",
    "venue_form_rating",
]


# Adskil hjemmeholdets og udeholdets features.
home_features = team_matches[team_matches["is_home"] == 1][
    ["match_id"] + feature_columns
].rename(columns={
    column: f"home_{column}" for column in feature_columns
})

away_features = team_matches[team_matches["is_home"] == 0][
    ["match_id"] + feature_columns
].rename(columns={
    column: f"away_{column}" for column in feature_columns
})

features = df.merge(
    home_features, on="match_id", how="left", validate="one_to_one"
).merge(
    away_features, on="match_id", how="left", validate="one_to_one"
)

features = features.drop(columns=["match_id"])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Antal kampe: {len(features)}")
print(f"Antal kolonner: {len(features.columns)}")

print("\nEksempel på de nye form-features:")
print(features[
    ["date", "league", "home_team", "away_team",
     "home_last5_points", "away_last5_points",
     "home_last5_goals_for", "away_last5_goals_for"]
].head(12).to_string(index=False))

print("\nManglende værdier i de nye features:")
new_columns = [
    "home_last5_goals_for",
    "home_last5_goals_against",
    "home_last5_points",
    "away_last5_goals_for",
    "away_last5_goals_against",
    "away_last5_points",
]
print(features[new_columns].isna().sum().to_string())
print(f"\nGemte features i: {OUTPUT_PATH}")