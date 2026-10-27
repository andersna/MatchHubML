"""Build Elo and attack/defense features from start-of-date state.

This version preserves the existing feature names and update formulas. Elo
inputs are snapshotted at the start of each calendar date, then all results
from the date are applied. The existing attack/defense implementation already
uses a two-pass daily calculation and is retained here.
"""

from collections import defaultdict
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "elo_features.csv"
DAY_FEATURES_PATH = PROJECT_ROOT / "data" / "features_asof_day_v1.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "elo_features_asof_day_v1.csv"

MATCH_KEYS = ["date", "league", "home_team", "away_team"]
INITIAL_ELO = 1500
K_FACTORS = [16, 32, 48]
HOME_ADVANTAGE = 60
INITIAL_ATTACK = 1.0
INITIAL_DEFENSE = 1.0
ATTACK_K = 0.05
DEFENSE_K = 0.05
MIN_EXPECTED_GOALS = 0.05
INITIAL_HOME_GOALS = 1.45
INITIAL_AWAY_GOALS = 1.15
PRIOR_MATCHES = 20

ELO_COLUMNS = [
    name
    for k in K_FACTORS
    for name in (f"home_elo_k{k}", f"away_elo_k{k}", f"elo_difference_k{k}")
]
ATTACK_DEFENSE_COLUMNS = [
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "home_expected_goals",
    "away_expected_goals",
]
FEATURE_COLUMNS = ELO_COLUMNS + ATTACK_DEFENSE_COLUMNS


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0

def calculate_elo_asof_day(data: pd.DataFrame, k_factor: int) -> pd.DataFrame:
    """Calculate pre-match Elo using one immutable rating snapshot per date."""
    ratings = {}
    results = {}

    for _, day in data.groupby(data["date"].dt.normalize(), sort=True, dropna=False):
        day_rows = []
        day_features = {}

        # Read only the state that existed at the beginning of this date.
        for row in day.itertuples(index=False):
            league = row.league
            home_key = (league, row.home_team)
            away_key = (league, row.away_team)
            home_rating = ratings.get(home_key, INITIAL_ELO)
            away_rating = ratings.get(away_key, INITIAL_ELO)
            expected_home = expected_score(home_rating + HOME_ADVANTAGE, away_rating)
            home_actual = actual_score(row.home_goals, row.away_goals)
            away_actual = 1 - home_actual
            day_features[row.match_id] = {
                f"home_elo_k{k_factor}": home_rating,
                f"away_elo_k{k_factor}": away_rating,
                f"elo_difference_k{k_factor}": home_rating - away_rating,
            }
            day_rows.append(
                (
                    home_key,
                    away_key,
                    k_factor * (home_actual - expected_home),
                    k_factor * (away_actual - (1 - expected_home)),
                )
            )

        # Aggregate updates after every feature for the date has been computed.
        # Summing per-match deltas also handles a team's multiple same-day
        # appearances without allowing one result to affect another prediction.
        daily_deltas = defaultdict(float)
        for home_key, away_key, home_delta, away_delta in day_rows:
            daily_deltas[home_key] += home_delta
            daily_deltas[away_key] += away_delta
        for team_key, delta in daily_deltas.items():
            ratings[team_key] = ratings.get(team_key, INITIAL_ELO) + delta

        results.update(day_features)

    return pd.DataFrame(
        [{column: results[match_id][column] for column in ELO_COLUMNS if column.endswith(f"k{k_factor}") or column == f"elo_difference_k{k_factor}"} for match_id in data["match_id"]]
    )


def _day_bounds(dates: pd.Series) -> list[tuple[int, int]]:
    """(start, end) row positions of each distinct date in a date-sorted frame."""
    days = dates.to_numpy()
    if not len(days):
        return []
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]]).tolist()
    return list(zip(starts, starts[1:] + [len(days)]))


def _replay_elo_all_k(leagues, home_teams, away_teams, home_goals, away_goals,
                      bounds, feature_arrays=None) -> dict[int, dict]:
    """Replay all Elo variants day by day; optionally record pre-match features.

    Plain lists are used instead of per-day pandas groups, which dominated the
    runtime. Formulas and update order are unchanged.
    """
    ratings = {k: {} for k in K_FACTORS}

    for start, end in bounds:
        daily_deltas = {k: {} for k in K_FACTORS}

        # Read all match values before applying any updates for this date.
        for i in range(start, end):
            league = leagues[i]
            home_key = (league, home_teams[i])
            away_key = (league, away_teams[i])
            home_actual = actual_score(home_goals[i], away_goals[i])
            away_actual = 1 - home_actual

            for k_factor in K_FACTORS:
                current_ratings = ratings[k_factor]
                home_rating = current_ratings.get(home_key, INITIAL_ELO)
                away_rating = current_ratings.get(away_key, INITIAL_ELO)
                expected_home = expected_score(home_rating + HOME_ADVANTAGE, away_rating)

                if feature_arrays is not None:
                    feature_arrays[f"home_elo_k{k_factor}"][i] = home_rating
                    feature_arrays[f"away_elo_k{k_factor}"][i] = away_rating
                    feature_arrays[f"elo_difference_k{k_factor}"][i] = home_rating - away_rating

                # Summing per-match deltas handles multiple same-day appearances
                # without letting one result affect another prediction.
                deltas = daily_deltas[k_factor]
                deltas[home_key] = deltas.get(home_key, 0.0) + k_factor * (home_actual - expected_home)
                deltas[away_key] = deltas.get(away_key, 0.0) + k_factor * (away_actual - (1 - expected_home))

        for k_factor in K_FACTORS:
            current_ratings = ratings[k_factor]
            for team_key, delta in daily_deltas[k_factor].items():
                current_ratings[team_key] = current_ratings.get(team_key, INITIAL_ELO) + delta

    return ratings


def calculate_elo_all_k_asof_day(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all pre-match Elo variants in one historical pass."""
    feature_arrays = {
        column: np.empty(len(data), dtype=np.float64) for column in ELO_COLUMNS
    }
    _replay_elo_all_k(
        data["league"].tolist(), data["home_team"].tolist(), data["away_team"].tolist(),
        data["home_goals"].tolist(), data["away_goals"].tolist(),
        _day_bounds(data["date"].dt.normalize()), feature_arrays,
    )
    return pd.DataFrame(feature_arrays, columns=ELO_COLUMNS)


_PRIOR_LEAGUE_STATS = (
    INITIAL_HOME_GOALS * PRIOR_MATCHES,
    INITIAL_AWAY_GOALS * PRIOR_MATCHES,
    PRIOR_MATCHES,
)
_INITIAL_TEAM_RATING = (INITIAL_ATTACK, INITIAL_DEFENSE)


def _attack_defense_read(ratings, league_stats, league, home_key, away_key):
    """Start-of-day attack/defense features for one match; does not mutate state."""
    home_attack, home_defense = ratings.get(home_key, _INITIAL_TEAM_RATING)
    away_attack, away_defense = ratings.get(away_key, _INITIAL_TEAM_RATING)
    stats = league_stats.get(league, _PRIOR_LEAGUE_STATS)

    avg_home_goals = stats[0] / stats[2]
    avg_away_goals = stats[1] / stats[2]
    expected_home = max(MIN_EXPECTED_GOALS, avg_home_goals * home_attack / away_defense)
    expected_away = max(MIN_EXPECTED_GOALS, avg_away_goals * away_attack / home_defense)
    return (home_attack, away_attack, home_defense, away_defense, expected_home, expected_away)


def _replay_attack_defense(leagues, home_teams, away_teams, home_goals, away_goals,
                           bounds, out=None) -> tuple[dict, dict]:
    """Replay attack/defense ratings day by day; optionally record features in ``out``."""
    ratings = {}       # (league, team) -> [attack, defense]
    league_stats = {}  # league -> [home_goals, away_goals, matches]

    for start, end in bounds:
        day_updates = []

        # All matches on this date use the same start-of-day state.
        for i in range(start, end):
            league = leagues[i]
            home_key = (league, home_teams[i])
            away_key = (league, away_teams[i])
            features = _attack_defense_read(ratings, league_stats, league, home_key, away_key)
            if out is not None:
                out[i] = features
            day_updates.append((
                league, home_key, away_key, home_goals[i], away_goals[i],
                features[4], features[5],
            ))

        # Apply updates only after all predictions for this date are computed.
        for (
            league,
            home_key,
            away_key,
            actual_home_value,
            actual_away_value,
            expected_home,
            expected_away,
        ) in day_updates:
            actual_home = float(actual_home_value)
            actual_away = float(actual_away_value)

            home_ratings = ratings.setdefault(home_key, list(_INITIAL_TEAM_RATING))
            away_ratings = ratings.setdefault(away_key, list(_INITIAL_TEAM_RATING))

            home_ratings[0] *= exp(
                ATTACK_K * (actual_home - expected_home) / (expected_home + 1.0)
            )
            away_ratings[0] *= exp(
                ATTACK_K * (actual_away - expected_away) / (expected_away + 1.0)
            )
            home_ratings[1] *= exp(
                DEFENSE_K * (expected_away - actual_away) / (expected_away + 1.0)
            )
            away_ratings[1] *= exp(
                DEFENSE_K * (expected_home - actual_home) / (expected_home + 1.0)
            )

            for team_ratings in (home_ratings, away_ratings):
                team_ratings[0] = min(3.0, max(0.3, team_ratings[0]))
                team_ratings[1] = min(3.0, max(0.3, team_ratings[1]))

            stats = league_stats.setdefault(league, list(_PRIOR_LEAGUE_STATS))
            stats[0] += actual_home
            stats[1] += actual_away
            stats[2] += 1

    return ratings, league_stats


def calculate_attack_defense_optimized(data: pd.DataFrame) -> pd.DataFrame:
    """Optimized attack/defense calculation with unchanged daily semantics."""
    out = np.empty((len(data), len(ATTACK_DEFENSE_COLUMNS)), dtype=np.float64)
    _replay_attack_defense(
        data["league"].tolist(), data["home_team"].tolist(), data["away_team"].tolist(),
        data["home_goals"].tolist(), data["away_goals"].tolist(),
        _day_bounds(data["date"]), out,
    )
    return pd.DataFrame(out, columns=ATTACK_DEFENSE_COLUMNS)


class RatingState:
    """Elo and attack/defense state at the start of the date after a replay."""

    def __init__(self, elo_ratings, attack_defense_ratings, league_stats):
        self.elo_ratings = elo_ratings
        self.attack_defense_ratings = attack_defense_ratings
        self.league_stats = league_stats

    def match_features(self, league: str, home_team: str, away_team: str) -> dict[str, float]:
        """Return FEATURE_COLUMNS for a match played on the next date."""
        home_key = (league, home_team)
        away_key = (league, away_team)
        result = {}
        for k_factor in K_FACTORS:
            current_ratings = self.elo_ratings[k_factor]
            home_rating = current_ratings.get(home_key, INITIAL_ELO)
            away_rating = current_ratings.get(away_key, INITIAL_ELO)
            result[f"home_elo_k{k_factor}"] = home_rating
            result[f"away_elo_k{k_factor}"] = away_rating
            result[f"elo_difference_k{k_factor}"] = home_rating - away_rating
        result.update(zip(ATTACK_DEFENSE_COLUMNS, _attack_defense_read(
            self.attack_defense_ratings, self.league_stats, league, home_key, away_key,
        )))
        return result


def _prepare_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    df = fixtures.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["match_id"] = np.arange(len(df))
    df = df.sort_values(
        ["date", "time", "match_id"], na_position="last"
    ).reset_index(drop=True)

    if "__request_id" in df.columns:
        history_for_check = df.loc[df["__request_id"].isna()]
    else:
        history_for_check = df

    appearances = pd.concat(
        [
            history_for_check[["date", "league", "home_team"]].rename(
                columns={"home_team": "team"}
            ),
            history_for_check[["date", "league", "away_team"]].rename(
                columns={"away_team": "team"}
            ),
        ],
        ignore_index=True,
    )
    repeated = appearances.duplicated(["date", "league", "team"], keep=False)
    if repeated.any():
        examples = appearances.loc[
            repeated, ["date", "league", "team"]
        ].head(10)
        raise ValueError(
            "Tvetydig dagsopdatering: et hold spiller flere kampe i samme liga/dato. "
            "Eksempler:\n" + examples.to_string(index=False)
        )
    return df


def replay_rating_state(fixtures: pd.DataFrame) -> RatingState:
    """Return Elo and attack/defense state after every fixture has been applied."""
    df = _prepare_fixtures(fixtures)
    columns = (
        df["league"].tolist(), df["home_team"].tolist(), df["away_team"].tolist(),
        df["home_goals"].tolist(), df["away_goals"].tolist(),
    )
    elo_ratings = _replay_elo_all_k(*columns, _day_bounds(df["date"].dt.normalize()))
    attack_defense_ratings, league_stats = _replay_attack_defense(
        *columns, _day_bounds(df["date"])
    )
    return RatingState(elo_ratings, attack_defense_ratings, league_stats)


def build_elo_features_asof_day(fixtures: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_fixtures(fixtures)
    elo_features = calculate_elo_all_k_asof_day(df)
    attack_defense_features = calculate_attack_defense_optimized(df)
    return pd.concat(
        [
            df.reset_index(drop=True),
            elo_features.reset_index(drop=True),
            attack_defense_features.reset_index(drop=True),
        ],
        axis=1,
    )



def _temporal_boundary_self_check():
    sample = pd.DataFrame(
        [
            {"date": "2020-01-01", "time": "19:45:00", "league": "L", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
            {"date": "2020-01-01", "time": None, "league": "L", "home_team": "C", "away_team": "D", "home_goals": 0, "away_goals": 1},
            {"date": "2020-01-02", "time": None, "league": "L", "home_team": "A", "away_team": "C", "home_goals": 0, "away_goals": 0},
        ]
    )
    changed = sample.copy()
    changed.loc[0, "home_goals"] = 0
    changed.loc[0, "away_goals"] = 3
    first = build_elo_features_asof_day(sample)
    second = build_elo_features_asof_day(changed)
    first_same_day = first.loc[:1, FEATURE_COLUMNS].reset_index(drop=True)
    second_same_day = second.loc[:1, FEATURE_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_same_day, second_same_day)
    if first.loc[2, "home_elo_k16"] == second.loc[2, "home_elo_k16"]:
        raise AssertionError("Selvtest fejlede: ændret resultat påvirkede ikke senere Elo.")
    if first.loc[2, "home_attack_strength"] == second.loc[2, "home_attack_strength"]:
        raise AssertionError("Selvtest fejlede: ændret resultat påvirkede ikke senere angrebsrating.")


def _comparison_report(old: pd.DataFrame, new: pd.DataFrame):
    old_duplicates = int(old.duplicated(MATCH_KEYS).sum())
    new_duplicates = int(new.duplicated(MATCH_KEYS).sum())
    merged = old[MATCH_KEYS + FEATURE_COLUMNS].merge(
        new[MATCH_KEYS + FEATURE_COLUMNS],
        on=MATCH_KEYS,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    matched = merged[merged["_merge"] == "both"]
    print("\n=== Elo-features: eksisterende vs dagsvis ===")
    print(f"Rækker: gammel={len(old):,}; ny={len(new):,}")
    print(f"Dubletter på kampnøgler: gammel={old_duplicates:,}; ny={new_duplicates:,}")
    print(
        "Kampnøgler uden match: "
        f"gammel={int((merged['_merge'] == 'right_only').sum()):,}; "
        f"ny={int((merged['_merge'] == 'left_only').sum()):,}"
    )
    print(f"Manglende featureværdier: gammel={int(old[FEATURE_COLUMNS].isna().sum().sum()):,}; ny={int(new[FEATURE_COLUMNS].isna().sum().sum()):,}")

    changed_counts = {}
    changed_rows = set()
    for column in FEATURE_COLUMNS:
        old_values = pd.to_numeric(matched[f"{column}_old"], errors="coerce")
        new_values = pd.to_numeric(matched[f"{column}_new"], errors="coerce")
        equal = np.isclose(old_values, new_values, rtol=1e-10, atol=1e-10, equal_nan=True)
        changed = ~equal
        changed_counts[column] = int(changed.sum())
        changed_rows.update(matched.index[changed].tolist())
    print(f"Fælles kampe med mindst én ændret feature: {len(changed_rows):,}")
    print("Ændrede værdier pr. featurekolonne:")
    print(pd.Series(changed_counts, dtype="int64").to_string())


def _check_day_feature_join(new_elo: pd.DataFrame):
    if not DAY_FEATURES_PATH.exists():
        print(f"Sammenkobling sprunget over: mangler {DAY_FEATURES_PATH}")
        return
    day_features = pd.read_csv(DAY_FEATURES_PATH, parse_dates=["date"])
    duplicates = int(day_features.duplicated(MATCH_KEYS).sum())
    if duplicates:
        raise AssertionError(f"Dagsvis featurefil har {duplicates} dublerede kampnøgler.")
    joined = day_features[MATCH_KEYS].merge(
        new_elo[MATCH_KEYS], on=MATCH_KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    left_only = int((joined["_merge"] == "left_only").sum())
    right_only = int((joined["_merge"] == "right_only").sum())
    print("\n=== Kobling med features_asof_day_v1.csv ===")
    print(f"Dagsvis feature-rækker: {len(day_features):,}; dagsvis Elo-rækker: {len(new_elo):,}")
    print(f"Dubletter på kampnøgler i featuresfilen: {duplicates:,}")
    print(f"Kampnøgler uden match: features={left_only:,}; Elo={right_only:,}")
    if left_only or right_only:
        raise AssertionError("Kampnøglerne kan ikke kobles fuldstændigt og entydigt.")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Output findes allerede; intet overskrives: {OUTPUT_PATH}")
    _temporal_boundary_self_check()
    fixtures = pd.read_csv(DATA_PATH, parse_dates=["date"])
    new = build_elo_features_asof_day(fixtures)
    if len(new) != len(fixtures):
        raise AssertionError("Rækketallet ændrede sig under Elo-featureberegningen.")

    try:
        with OUTPUT_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
            new.to_csv(handle, index=False)
    except FileExistsError:
        raise FileExistsError(f"Output blev oprettet under kørslen; intet overskrevet: {OUTPUT_PATH}")

    print("PASS: Elo, angrebs-/forsvarsratings og forventede mål beregnes før dagens opdatering.")
    print("PASS: in-memory test viste, at ændring af dagens resultat ikke ændrer samme dags features, men kan ændre senere datoer.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rækker: {len(new):,}; kolonner: {len(new.columns)}")
    old = pd.read_csv(REFERENCE_PATH, parse_dates=["date"])
    new_for_compare = pd.read_csv(OUTPUT_PATH, parse_dates=["date"])
    _comparison_report(old, new_for_compare)
    _check_day_feature_join(new_for_compare)


if __name__ == "__main__":
    main()
