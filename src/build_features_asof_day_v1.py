"""Build team form features from the state at the start of each match date.

This is a versioned alternative to build_features.py. It intentionally keeps
the existing feature names and formulas, while preventing matches on the same
calendar date from contributing to one another's features.
"""

from collections import defaultdict, deque, namedtuple
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
REFERENCE_PATH = PROJECT_ROOT / "data" / "features.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features_asof_day_v1.csv"

MATCH_KEYS = ["date", "league", "home_team", "away_team"]
FEATURE_COLUMNS = [
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
HISTORY_COLUMNS = [
    "goals_for",
    "goals_against",
    "points",
    "goal_difference",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
]
ROLLING_COLUMNS = [
    "goals_for",
    "goals_against",
    "points",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
]


def _points(goals_for, goals_against):
    """Match the original boolean point calculation, including missing goals."""
    if pd.isna(goals_for) or pd.isna(goals_against):
        return 0
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _mean(values):
    """Pandas expanding/rolling mean semantics: ignore missing observations."""
    valid = [value for value in values if not pd.isna(value)]
    return float(np.mean(valid)) if valid else np.nan

def _missing(value):
    """Cheap scalar replacement for pd.isna on floats, ints and None."""
    return value is None or value != value


def _points_fast(goals_for, goals_against):
    if _missing(goals_for) or _missing(goals_against):
        return 0
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _mean_optimized(values):
    """Calculate mean while ignoring missing values."""
    total = 0.0
    count = 0

    for value in values:
        if not _missing(value):
            total += value
            count += 1

    return total / count if count else np.nan


def _make_observation(row, side):
    if side == "home":
        gf, ga = row["home_goals"], row["away_goals"]
        sf, sa = row["home_shots"], row["away_shots"]
        stf, sta = row["home_shots_on_target"], row["away_shots_on_target"]
        is_home = 1
    else:
        gf, ga = row["away_goals"], row["home_goals"]
        sf, sa = row["away_shots"], row["home_shots"]
        stf, sta = row["away_shots_on_target"], row["home_shots_on_target"]
        is_home = 0

    return {
        "league": row["league"],
        "team": row[f"{side}_team"],
        "is_home": is_home,
        "goals_for": gf,
        "goals_against": ga,
        "points": _points(gf, ga),
        "goal_difference": gf - ga if not (pd.isna(gf) or pd.isna(ga)) else np.nan,
        "shots_for": sf,
        "shots_against": sa,
        "shots_on_target_for": stf,
        "shots_on_target_against": sta,
    }



def _features_from_history(team_history, venue_history):
    result = {}

    for column, output_name in [
        ("goals_for", "avg_goals_for"),
        ("goals_against", "avg_goals_against"),
        ("points", "avg_points"),
    ]:
        result[output_name] = _mean([item[column] for item in team_history])

    recent = team_history[-5:]

    for column in ROLLING_COLUMNS:
        result[f"last5_{column}"] = _mean(
            [item[column] for item in recent]
        )

    recent_points = [item["points"] for item in recent]
    recent_goal_difference = [item["goal_difference"] for item in recent]

    avg_recent_points = _mean(recent_points)
    avg_recent_goal_difference = _mean(recent_goal_difference)

    result["form_rating"] = (
        2 * avg_recent_points + avg_recent_goal_difference
        if not pd.isna(avg_recent_points)
        and not pd.isna(avg_recent_goal_difference)
        else np.nan
    )

    recent_venue = venue_history[-5:]
    venue_points = _mean([item["points"] for item in recent_venue])
    venue_goal_difference = _mean(
        [item["goal_difference"] for item in recent_venue]
    )

    result["venue_form_rating"] = (
        2 * venue_points + venue_goal_difference
        if not pd.isna(venue_points)
        and not pd.isna(venue_goal_difference)
        else np.nan
    )

    return result

# Observation tuple layout used by FormState: one tuple per team appearance,
# in HISTORY_COLUMNS order.
_GF, _GA, _PTS, _GD, _SF, _SA, _STF, _STA = range(len(HISTORY_COLUMNS))
_ROLLING_INDICES = [HISTORY_COLUMNS.index(column) for column in ROLLING_COLUMNS]
_EMPTY_TOTALS = (0.0, 0, 0.0, 0, 0.0, 0)


class FormState:
    """Team and league form state after replaying complete match dates.

    Expanding means are kept as running sums added in match order, which gives
    the same floating-point result as summing the full history on every read.
    Reads never mutate the state, so a finished state can be shared.
    """

    def __init__(self):
        # (league, team) -> [sum_gf, n_gf, sum_ga, n_ga, sum_points, n_points]
        self.totals = {}
        self.recent = {}        # (league, team) -> last 5 observations
        self.venue_recent = {}  # (league, team, is_home) -> last 5 observations
        self.league_points = {}  # league -> [home_points, away_points, matches]

    def home_advantage(self, league):
        stats = self.league_points.get(league)
        if not stats or not stats[2]:
            return np.nan
        return (stats[0] / stats[2]) - (stats[1] / stats[2])

    def team_features(self, league, team, is_home):
        """Return FEATURE_COLUMNS for one team from the current state."""
        totals = self.totals.get((league, team), _EMPTY_TOTALS)
        recent = self.recent.get((league, team), ())
        recent_venue = self.venue_recent.get((league, team, is_home), ())

        result = {
            "avg_goals_for": totals[0] / totals[1] if totals[1] else np.nan,
            "avg_goals_against": totals[2] / totals[3] if totals[3] else np.nan,
            "avg_points": totals[4] / totals[5] if totals[5] else np.nan,
        }
        for column, index in zip(ROLLING_COLUMNS, _ROLLING_INDICES):
            result[f"last5_{column}"] = _mean_optimized([obs[index] for obs in recent])

        avg_recent_points = result["last5_points"]
        avg_recent_goal_difference = _mean_optimized([obs[_GD] for obs in recent])
        result["form_rating"] = (
            2 * avg_recent_points + avg_recent_goal_difference
            if not _missing(avg_recent_points)
            and not _missing(avg_recent_goal_difference)
            else np.nan
        )

        venue_points = _mean_optimized([obs[_PTS] for obs in recent_venue])
        venue_goal_difference = _mean_optimized([obs[_GD] for obs in recent_venue])
        result["venue_form_rating"] = (
            2 * venue_points + venue_goal_difference
            if not _missing(venue_points) and not _missing(venue_goal_difference)
            else np.nan
        )
        return result

    def _add_team(self, league, team, is_home, obs):
        key = (league, team)
        totals = self.totals.get(key)
        if totals is None:
            totals = self.totals[key] = [0.0, 0, 0.0, 0, 0.0, 0]
        for slot, value in ((0, obs[_GF]), (2, obs[_GA]), (4, obs[_PTS])):
            if not _missing(value):
                totals[slot] += value
                totals[slot + 1] += 1

        recent = self.recent.get(key)
        if recent is None:
            recent = self.recent[key] = deque(maxlen=5)
        recent.append(obs)

        venue_key = (league, team, is_home)
        venue = self.venue_recent.get(venue_key)
        if venue is None:
            venue = self.venue_recent[venue_key] = deque(maxlen=5)
        venue.append(obs)

    def add_matches(self, cols, start, end):
        """Append rows [start, end) of a _FixtureColumns to the state, in order."""
        for i in range(start, end):
            league = cols.league[i]
            hg, ag = cols.home_goals[i], cols.away_goals[i]
            hp, ap = cols.home_points[i], cols.away_points[i]
            goals_missing = _missing(hg) or _missing(ag)
            self._add_team(league, cols.home_team[i], 1, (
                hg, ag, hp, np.nan if goals_missing else hg - ag,
                cols.home_shots[i], cols.away_shots[i],
                cols.home_shots_on_target[i], cols.away_shots_on_target[i],
            ))
            self._add_team(league, cols.away_team[i], 0, (
                ag, hg, ap, np.nan if goals_missing else ag - hg,
                cols.away_shots[i], cols.home_shots[i],
                cols.away_shots_on_target[i], cols.home_shots_on_target[i],
            ))
            stats = self.league_points.get(league)
            if stats is None:
                stats = self.league_points[league] = [0.0, 0.0, 0]
            stats[0] += hp
            stats[1] += ap
            stats[2] += 1


_FixtureColumns = namedtuple("_FixtureColumns", [
    "league", "home_team", "away_team", "home_goals", "away_goals",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_points", "away_points",
])


def _fixture_columns(df):
    """Convert the needed columns to plain lists once; pandas per-row access is slow."""
    return _FixtureColumns(*(df[name].tolist() for name in _FixtureColumns._fields))


def _day_bounds(dates):
    """(start, end) row positions of each calendar date in a date-sorted frame."""
    days = dates.dt.normalize().to_numpy()
    if not len(days):
        return []
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]]).tolist()
    return list(zip(starts, starts[1:] + [len(days)]))


def _prepare_fixtures(fixtures):
    """Copy, sort and validate fixtures exactly as the day builders expect."""
    df = fixtures.copy()

    if "date" not in df:
        raise ValueError("fixtures.csv mangler kolonnen 'date'.")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["match_id"] = np.arange(len(df))
    df = df.sort_values(
        ["date", "time", "match_id"], na_position="last"
    ).reset_index(drop=True)

    required = [
        "league", "home_team", "away_team", "home_goals", "away_goals",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"fixtures.csv mangler nødvendige kolonner: {missing}")

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
            "Tvetydig dagsrækkefølge: et hold optræder flere gange i samme liga/dato. "
            "Ingen features er skrevet. Eksempler:\n" + examples.to_string(index=False)
        )

    df["home_points"] = [
        _points_fast(home, away)
        for home, away in zip(df["home_goals"].tolist(), df["away_goals"].tolist())
    ]
    df["away_points"] = [
        _points_fast(away, home)
        for home, away in zip(df["home_goals"].tolist(), df["away_goals"].tolist())
    ]
    return df


def replay_form_state(fixtures):
    """Return the FormState after every fixture, i.e. the start of the next date."""
    df = _prepare_fixtures(fixtures)
    state = FormState()
    # Without reads, appending in sorted row order equals the day-wise replay.
    state.add_matches(_fixture_columns(df), 0, len(df))
    return state


def build_features_asof_day(fixtures, *, only_requested_rows=False):
    """Return existing feature schema, computed from pre-date state only."""
    df = fixtures.copy()
    if "date" not in df:
        raise ValueError("fixtures.csv mangler kolonnen 'date'.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["match_id"] = np.arange(len(df))
    df = df.sort_values(["date", "time", "match_id"], na_position="last").reset_index(drop=True)

    required = [
        "league", "home_team", "away_team", "home_goals", "away_goals",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"fixtures.csv mangler nødvendige kolonner: {missing}")

    # If one team has multiple fixtures in the same league/date, their order is
    # unknowable where kickoff is missing. Reject rather than impose a result-based order.

    # Validate duplicate team appearances in historical matches only.
    # Upcoming requests share one calculation date by design and must not
    # trigger this historical ordering check.
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
            "Tvetydig dagsrækkefølge: et hold optræder flere gange i samme liga/dato. "
            "Ingen features er skrevet. Eksempler:\n" + examples.to_string(index=False)
        )


    df["home_points"] = [
        _points(home, away) for home, away in zip(df["home_goals"], df["away_goals"])
    ]
    df["away_points"] = [
        _points(away, home) for home, away in zip(df["home_goals"], df["away_goals"])
    ]

    histories = defaultdict(list)
    venue_histories = defaultdict(list)
    league_home_points = defaultdict(float)
    league_away_points = defaultdict(float)
    league_match_counts = defaultdict(int)
    feature_by_match = {}
    hfa_by_match = {}

    # Each group is a complete calendar date. All features are computed before
    # any results from that date are appended to team or league state.
    for _, day in df.groupby(df["date"].dt.normalize(), sort=True, dropna=False):
        day_observations = []
        day_features = {}
        day_hfa = {}

        for row in day.to_dict("records"):
            match_id = row["match_id"]
            league = row["league"]
            count = league_match_counts[league]
            day_hfa[match_id] = (
                (league_home_points[league] / count) - (league_away_points[league] / count)
                if count
                else np.nan
            )

            for side in ("home", "away"):
                obs = _make_observation(row, side)
                team_key = (obs["league"], obs["team"])
                venue_key = (obs["league"], obs["team"], obs["is_home"])
                prefix = f"{side}_"
                if not only_requested_rows or pd.notna(row.get("__request_id", np.nan)):
                    for feature, value in _features_from_history(
                        histories[team_key], venue_histories[venue_key]
                    ).items():
                        day_features.setdefault(match_id, {})[prefix + feature] = value
                day_observations.append((team_key, venue_key, obs))

        # State mutation is deliberately a separate pass after today's features.
        for team_key, venue_key, obs in day_observations:
            histories[team_key].append(obs)
            venue_histories[venue_key].append(obs)
        for row in day.to_dict("records"):
            league = row["league"]
            league_home_points[league] += row["home_points"]
            league_away_points[league] += row["away_points"]
            league_match_counts[league] += 1
        feature_by_match.update(day_features)
        hfa_by_match.update(day_hfa)

    df["league_home_advantage"] = df["match_id"].map(hfa_by_match)
    for side in ("home", "away"):
        for feature in FEATURE_COLUMNS:
            df[f"{side}_{feature}"] = df["match_id"].map(
                lambda match_id, key=f"{side}_{feature}": feature_by_match.get(match_id, {}).get(key, np.nan)
            )

    return df.drop(columns="match_id")

def build_features_asof_day_reference(fixtures, *, only_requested_rows=False):
    """Return existing feature schema, computed from pre-date state only."""
    df = fixtures.copy()
    if "date" not in df:
        raise ValueError("fixtures.csv mangler kolonnen 'date'.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["match_id"] = np.arange(len(df))
    df = df.sort_values(["date", "time", "match_id"], na_position="last").reset_index(drop=True)

    required = [
        "league", "home_team", "away_team", "home_goals", "away_goals",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"fixtures.csv mangler nødvendige kolonner: {missing}")

    # If one team has multiple fixtures in the same league/date, their order is
    # unknowable where kickoff is missing. Reject rather than impose a result-based order.

    # Validate duplicate team appearances in historical matches only.
    # Upcoming requests share one calculation date by design and must not
    # trigger this historical ordering check.
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
            "Tvetydig dagsrækkefølge: et hold optræder flere gange i samme liga/dato. "
            "Ingen features er skrevet. Eksempler:\n" + examples.to_string(index=False)
        )


    df["home_points"] = [
        _points(home, away) for home, away in zip(df["home_goals"], df["away_goals"])
    ]
    df["away_points"] = [
        _points(away, home) for home, away in zip(df["home_goals"], df["away_goals"])
    ]

    histories = defaultdict(list)
    venue_histories = defaultdict(list)
    league_home_points = defaultdict(float)
    league_away_points = defaultdict(float)
    league_match_counts = defaultdict(int)
    feature_by_match = {}
    hfa_by_match = {}

    # Each group is a complete calendar date. All features are computed before
    # any results from that date are appended to team or league state.
    for _, day in df.groupby(df["date"].dt.normalize(), sort=True, dropna=False):
        day_observations = []
        day_features = {}
        day_hfa = {}

        for row in day.to_dict("records"):
            match_id = row["match_id"]
            league = row["league"]
            count = league_match_counts[league]
            day_hfa[match_id] = (
                (league_home_points[league] / count) - (league_away_points[league] / count)
                if count
                else np.nan
            )

            for side in ("home", "away"):
                obs = _make_observation(row, side)
                team_key = (obs["league"], obs["team"])
                venue_key = (obs["league"], obs["team"], obs["is_home"])
                prefix = f"{side}_"
                if not only_requested_rows or pd.notna(row.get("__request_id", np.nan)):
                    for feature, value in _features_from_history(
                        histories[team_key], venue_histories[venue_key]
                    ).items():
                        day_features.setdefault(match_id, {})[prefix + feature] = value
                day_observations.append((team_key, venue_key, obs))

        # State mutation is deliberately a separate pass after today's features.
        for team_key, venue_key, obs in day_observations:
            histories[team_key].append(obs)
            venue_histories[venue_key].append(obs)
        for row in day.to_dict("records"):
            league = row["league"]
            league_home_points[league] += row["home_points"]
            league_away_points[league] += row["away_points"]
            league_match_counts[league] += 1
        feature_by_match.update(day_features)
        hfa_by_match.update(day_hfa)

    df["league_home_advantage"] = df["match_id"].map(hfa_by_match)
    for side in ("home", "away"):
        for feature in FEATURE_COLUMNS:
            df[f"{side}_{feature}"] = df["match_id"].map(
                lambda match_id, key=f"{side}_{feature}": feature_by_match.get(match_id, {}).get(key, np.nan)
            )

    return df.drop(columns="match_id")


def build_features_asof_day_optimized(fixtures, *, only_requested_rows=False):
    """Optimized version preserving the existing daily feature semantics."""
    df = _prepare_fixtures(fixtures)
    n = len(df)
    cols = _fixture_columns(df)

    requested = (
        df["__request_id"].notna().tolist()
        if "__request_id" in df.columns
        else [False] * n
    )
    home_advantage = np.full(n, np.nan)
    outputs = {
        f"{side}_{feature}": np.full(n, np.nan)
        for side in ("home", "away")
        for feature in FEATURE_COLUMNS
    }

    state = FormState()
    for start, end in _day_bounds(df["date"]):
        # Every match on this date reads the same start-of-day state.
        for i in range(start, end):
            league = cols.league[i]
            home_advantage[i] = state.home_advantage(league)
            if only_requested_rows and not requested[i]:
                continue
            for side, team, is_home in (
                ("home", cols.home_team[i], 1),
                ("away", cols.away_team[i], 0),
            ):
                for feature, value in state.team_features(league, team, is_home).items():
                    outputs[f"{side}_{feature}"][i] = value

        # Append today's results only after all today's features are computed.
        state.add_matches(cols, start, end)

    df["league_home_advantage"] = home_advantage
    for key, values in outputs.items():
        df[key] = values

    return df.drop(columns="match_id")


def _temporal_boundary_self_check():
    """In-memory proof that day-one outcomes cannot change day-one features."""
    sample = pd.DataFrame(
        [
            {"date": "2020-01-01", "time": "19:45:00", "league": "L", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "home_shots": 5, "away_shots": 3, "home_shots_on_target": 2, "away_shots_on_target": 1},
            {"date": "2020-01-01", "time": None, "league": "L", "home_team": "C", "away_team": "D", "home_goals": 0, "away_goals": 1, "home_shots": 2, "away_shots": 6, "home_shots_on_target": 1, "away_shots_on_target": 3},
            {"date": "2020-01-02", "time": None, "league": "L", "home_team": "A", "away_team": "C", "home_goals": 0, "away_goals": 0, "home_shots": 0, "away_shots": 0, "home_shots_on_target": 0, "away_shots_on_target": 0},
        ]
    )
    changed = sample.copy()
    changed.loc[0, "home_goals"] = 4
    first = build_features_asof_day(sample)
    second = build_features_asof_day(changed)
    check_cols = ["league_home_advantage"] + [f"{side}_{col}" for side in ("home", "away") for col in FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(first.loc[:1, check_cols], second.loc[:1, check_cols])
    if first.loc[2, "home_avg_goals_for"] == second.loc[2, "home_avg_goals_for"]:
        raise AssertionError("Selvtest fejlede: dag 1-resultatet kom ikke ind i senere historik.")
    if first.loc[:1, "league_home_advantage"].nunique(dropna=True) > 1:
        raise AssertionError("Selvtest fejlede: kampe samme dag fik forskellig ligatilstand.")


def _missing_summary(frame):
    return int(frame.isna().sum().sum()), frame.isna().sum().loc[lambda values: values > 0].to_dict()


def _comparison_report(old, new):
    keys = MATCH_KEYS
    old_key_dups = int(old.duplicated(keys).sum())
    new_key_dups = int(new.duplicated(keys).sum())
    merged = old[keys + ["league_home_advantage"] + [f"{s}_{c}" for s in ("home", "away") for c in FEATURE_COLUMNS]].merge(
        new[keys + ["league_home_advantage"] + [f"{s}_{c}" for s in ("home", "away") for c in FEATURE_COLUMNS]],
        on=keys,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    matched = merged[merged["_merge"] == "both"]
    value_cols = ["league_home_advantage"] + [f"{s}_{c}" for s in ("home", "away") for c in FEATURE_COLUMNS]
    differing_rows = set()
    cell_differences = {}
    max_abs = {}
    for column in value_cols:
        a = matched[f"{column}_old"]
        b = matched[f"{column}_new"]
        equal = (a.eq(b) | (a.isna() & b.isna()))
        numeric_a = pd.to_numeric(a, errors="coerce")
        numeric_b = pd.to_numeric(b, errors="coerce")
        close = np.isclose(numeric_a, numeric_b, rtol=1e-10, atol=1e-10, equal_nan=True)
        changed = ~(equal | close)
        cell_differences[column] = int(changed.sum())
        differing_rows.update(matched.index[changed].tolist())
        delta = (numeric_a - numeric_b).abs()
        max_abs[column] = float(delta.max()) if delta.notna().any() else 0.0

    old_missing, old_missing_cols = _missing_summary(old)
    new_missing, new_missing_cols = _missing_summary(new)
    print("\n=== Gammel vs dagsvis beregning ===")
    print(f"Rækker: gammel={len(old):,}; ny={len(new):,}")
    print(f"Dubletter på kampnøgler: gammel={old_key_dups:,}; ny={new_key_dups:,}")
    print(f"Nøgler uden match efter sammenkobling: gammel={int((merged['_merge'] == 'right_only').sum()):,}; ny={int((merged['_merge'] == 'left_only').sum()):,}")
    print(f"Manglende værdier (alle kolonner): gammel={old_missing:,}; ny={new_missing:,}")
    print(f"Kolonner med manglende værdier: gammel={len(old_missing_cols)}; ny={len(new_missing_cols)}")
    print(f"Sammenlignede features: {len(value_cols)}; rækker med mindst én ændret feature: {len(differing_rows):,}")
    changed_cols = {key: value for key, value in cell_differences.items() if value}
    print("Ændrede celler pr. feature:")
    print(pd.Series(changed_cols, dtype="int64").to_string() if changed_cols else "Ingen")
    print("Maksimal absolut forskel pr. feature:")
    print(pd.Series({key: value for key, value in max_abs.items() if cell_differences[key]}, dtype="float64").to_string() if changed_cols else "Ingen")


def main():
    # Guard before any expensive work. The exclusive create below protects the
    # final write against a file appearing after this initial check.
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Output findes allerede; intet overskrives: {OUTPUT_PATH}")
    _temporal_boundary_self_check()

    fixtures = pd.read_csv(DATA_PATH, parse_dates=["date"])
    features = build_features_asof_day(fixtures)
    if len(features) != len(fixtures):
        raise AssertionError("Rækketallet ændrede sig under featureberegningen.")

    try:
        with OUTPUT_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
            features.to_csv(handle, index=False)
    except FileExistsError:
        raise FileExistsError(f"Output blev oprettet under kørslen; intet overskrevet: {OUTPUT_PATH}")

    print("PASS: dagens features er beregnet før dagens resultater blev føjet til historikken.")
    print("PASS: in-memory test bekræftede, at ændring af dagens resultat ikke ændrer samme dags features, men ændrer senere historik.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rækker: {len(features):,}; kolonner: {len(features.columns)}")

    if REFERENCE_PATH.exists():
        old = pd.read_csv(REFERENCE_PATH, parse_dates=["date"])
        new = pd.read_csv(OUTPUT_PATH, parse_dates=["date"])
        _comparison_report(old, new)
    else:
        print(f"Sammenligning sprunget over: referencefil mangler ({REFERENCE_PATH}).")


if __name__ == "__main__":
    main()
