from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "elo_features.csv"

INITIAL_ELO = 1500
K_FACTORS = [16, 32, 48]
HOME_ADVANTAGE = 60
# Angrebs- og forsvarsratinger til en senere Poisson-model
INITIAL_ATTACK = 1.0
INITIAL_DEFENSE = 1.0

ATTACK_K = 0.05
DEFENSE_K = 0.05

MIN_EXPECTED_GOALS = 0.05


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



def calculate_attack_defense(data: pd.DataFrame) -> pd.DataFrame:
    ratings = {}
    league_stats = {}

    results = []

    # Udgangspunkt for ligaens målgennemsnit.
    INITIAL_HOME_GOALS = 1.45
    INITIAL_AWAY_GOALS = 1.15
    PRIOR_MATCHES = 20

    for _, day in data.groupby("date", sort=True):
        day_results = []
        day_updates = []

        # Først beregnes ALLE forudsigelser for datoen.
        # Ingen resultater fra denne dato er endnu indarbejdet.
        for row in day.itertuples(index=False):
            league = row.league
            home_team = row.home_team
            away_team = row.away_team

            home_key = (league, home_team)
            away_key = (league, away_team)

            home_ratings = ratings.setdefault(
                home_key,
                {"attack": INITIAL_ATTACK, "defense": INITIAL_DEFENSE},
            )
            away_ratings = ratings.setdefault(
                away_key,
                {"attack": INITIAL_ATTACK, "defense": INITIAL_DEFENSE},
            )

            stats = league_stats.setdefault(
                league,
                {
                    "home_goals": INITIAL_HOME_GOALS * PRIOR_MATCHES,
                    "away_goals": INITIAL_AWAY_GOALS * PRIOR_MATCHES,
                    "matches": PRIOR_MATCHES,
                },
            )

            avg_home_goals = stats["home_goals"] / stats["matches"]
            avg_away_goals = stats["away_goals"] / stats["matches"]

            home_attack = home_ratings["attack"]
            away_attack = away_ratings["attack"]
            home_defense = home_ratings["defense"]
            away_defense = away_ratings["defense"]

            expected_home = max(
                MIN_EXPECTED_GOALS,
                avg_home_goals * home_attack / away_defense,
            )
            expected_away = max(
                MIN_EXPECTED_GOALS,
                avg_away_goals * away_attack / home_defense,
            )

            day_results.append({
                "home_attack_strength": home_attack,
                "away_attack_strength": away_attack,
                "home_defense_strength": home_defense,
                "away_defense_strength": away_defense,
                "home_expected_goals": expected_home,
                "away_expected_goals": expected_away,
            })

            day_updates.append({
                "league": league,
                "home_key": home_key,
                "away_key": away_key,
                "actual_home": row.home_goals,
                "actual_away": row.away_goals,
                "expected_home": expected_home,
                "expected_away": expected_away,
            })

        # Først efter alle dagens forudsigelser opdateres ratings.
        for update in day_updates:
            league = update["league"]
            home_key = update["home_key"]
            away_key = update["away_key"]

            actual_home = float(update["actual_home"])
            actual_away = float(update["actual_away"])
            expected_home = update["expected_home"]
            expected_away = update["expected_away"]

            home_ratings = ratings[home_key]
            away_ratings = ratings[away_key]

            # Opdater angreb ud fra mål minus forventede mål.
            home_ratings["attack"] *= __import__("math").exp(
                ATTACK_K
                * (actual_home - expected_home)
                / (expected_home + 1.0)
            )

            away_ratings["attack"] *= __import__("math").exp(
                ATTACK_K
                * (actual_away - expected_away)
                / (expected_away + 1.0)
            )

            # Et stærkt forsvar tillader færre mål end forventet.
            home_ratings["defense"] *= __import__("math").exp(
                DEFENSE_K
                * (expected_away - actual_away)
                / (expected_away + 1.0)
            )

            away_ratings["defense"] *= __import__("math").exp(
                DEFENSE_K
                * (expected_home - actual_home)
                / (expected_home + 1.0)
            )

            # Begræns ekstreme ratings.
            for team_ratings in (home_ratings, away_ratings):
                team_ratings["attack"] = min(
                    3.0, max(0.3, team_ratings["attack"])
                )
                team_ratings["defense"] = min(
                    3.0, max(0.3, team_ratings["defense"])
                )

            # Opdater ligagennemsnittet efter kampdagen.
            stats = league_stats[league]
            stats["home_goals"] += actual_home
            stats["away_goals"] += actual_away
            stats["matches"] += 1

        results.extend(day_results)

    return pd.DataFrame(results)



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

# Beregn angrebs- og forsvarsratinger til Poisson-modellen
attack_defense_features = calculate_attack_defense(df)

df = pd.concat(
    [df, attack_defense_features],
    axis=1,
)

new_columns = attack_defense_features.columns.tolist()

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