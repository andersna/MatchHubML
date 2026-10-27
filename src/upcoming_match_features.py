"""Calculate the saved MatchHubML model's 27 sports features for future matches.

The existing as-of-day builders remain the single source of formulas. For a
requested date this module replays only fixtures strictly before that date
into the builders' start-of-date state, then reads every requested match from
that one state. The state is cached per fixtures file and date, so repeated
requests do not replay the full history. No odds are loaded or calculated here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import threading
import unittest
import time

import numpy as np
import pandas as pd

try:  # Works both as ``python src/upcoming_match_features.py`` and as a module.
    from .sync_football_data import _norm_name
    from .build_features_asof_day_v1 import FormState, replay_form_state
    from .build_elo_features_asof_day_v1 import (
        ATTACK_DEFENSE_COLUMNS,
        ELO_COLUMNS,
        RatingState,
        replay_rating_state,
    )
    from .experiment_nn_odds_asof_day_v1 import NN_FEATURES
except ImportError:  # pragma: no cover - exercised by direct script execution
    from sync_football_data import _norm_name
    from build_features_asof_day_v1 import FormState, replay_form_state
    from build_elo_features_asof_day_v1 import (
        ATTACK_DEFENSE_COLUMNS,
        ELO_COLUMNS,
        RatingState,
        replay_rating_state,
    )
    from experiment_nn_odds_asof_day_v1 import NN_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
MATCH_KEYS = ["date", "league", "home_team", "away_team"]
REQUEST_KEYS = ["date", "league", "home_team", "away_team"]
HISTORY_COLUMNS = [
    "date", "time", "league", "home_team", "away_team", "home_goals",
    "away_goals", "home_shots", "away_shots", "home_shots_on_target",
    "away_shots_on_target",
]
OUTPUT_COLUMNS = tuple(NN_FEATURES)

# football-data.org's canonical names often differ from the historical
# fixtures.csv vocabulary. These targets are existing labels in fixtures.csv;
# this mapping is used only to resolve a request against the in-memory history.
# It never changes database match names, fixture keys, odds identities, or the
# model's 27 numeric features.
HISTORY_TEAM_ALIASES = {
    "Premier League": {
        _norm_name("Brighton & Hove Albion FC"): "Brighton",
        _norm_name("Coventry City FC"): "Coventry",
        _norm_name("Hull City AFC"): "Hull",
        _norm_name("Ipswich Town FC"): "Ipswich",
        _norm_name("Leeds United FC"): "Leeds",
        _norm_name("Manchester City FC"): "Man City",
        _norm_name("Manchester United FC"): "Man United",
        _norm_name("Newcastle United FC"): "Newcastle",
        _norm_name("Nottingham Forest FC"): "Nott'm Forest",
        _norm_name("Tottenham Hotspur FC"): "Tottenham",
    },
    "La Liga": {
        _norm_name("Athletic Club"): "Ath Bilbao",
        _norm_name("CA Osasuna"): "Osasuna",
        _norm_name("Club Atlético de Madrid"): "Ath Madrid",
        _norm_name("Deportivo Alavés"): "Alaves",
        _norm_name("Levante UD"): "Levante",
        _norm_name("RC Celta de Vigo"): "Celta",
        _norm_name("RC Deportivo La Coruña"): "La Coruna",
        _norm_name("RCD Espanyol de Barcelona"): "Espanol",
        _norm_name("Rayo Vallecano de Madrid"): "Vallecano",
        _norm_name("Real Betis Balompié"): "Betis",
        _norm_name("Real Racing Club de Santander"): "Santander",
        _norm_name("Real Sociedad de Fútbol"): "Sociedad",
    },
    "Bundesliga": {
        _norm_name("1. FC Köln"): "FC Koln",
        _norm_name("1. FC Union Berlin"): "Union Berlin",
        _norm_name("1. FSV Mainz 05"): "Mainz",
        _norm_name("Bayer 04 Leverkusen"): "Leverkusen",
        _norm_name("Borussia Dortmund"): "Dortmund",
        _norm_name("Borussia Mönchengladbach"): "M'Gladbach",
        _norm_name("Eintracht Frankfurt"): "Ein Frankfurt",
        _norm_name("SV Werder Bremen"): "Werder Bremen",
        _norm_name("TSG 1899 Hoffenheim"): "Hoffenheim",
        _norm_name("VfB Stuttgart"): "Stuttgart",
        _norm_name("SC Paderborn 07"): "Paderborn",
        _norm_name("Hamburger SV"): "Hamburg",
        _norm_name("SV 07 Elversberg"): "Elversberg",
        _norm_name("FC Bayern München"): "Bayern Munich",
    },
    "Serie A": {
        _norm_name("AC Milan"): "Milan",
        _norm_name("AC Monza"): "Monza",
        _norm_name("ACF Fiorentina"): "Fiorentina",
        _norm_name("AS Roma"): "Roma",
        _norm_name("Atalanta BC"): "Atalanta",
        _norm_name("Bologna FC 1909"): "Bologna",
        _norm_name("Cagliari Calcio"): "Cagliari",
        _norm_name("Como 1907"): "Como",
        _norm_name("FC Internazionale Milano"): "Inter",
        _norm_name("Frosinone Calcio"): "Frosinone",
        _norm_name("Genoa CFC"): "Genoa",
        _norm_name("Parma Calcio 1913"): "Parma",
        _norm_name("SS Lazio"): "Lazio",
        _norm_name("SSC Napoli"): "Napoli",
        _norm_name("US Lecce"): "Lecce",
        _norm_name("US Sassuolo Calcio"): "Sassuolo",
        _norm_name("Udinese Calcio"): "Udinese",
        _norm_name("Venezia FC"): "Venezia",
    },
    "Ligue 1": {
        _norm_name("AJ Auxerre"): "Auxerre",
        _norm_name("AS Monaco FC"): "Monaco",
        _norm_name("Angers SCO"): "Angers",
        _norm_name("ES Troyes AC"): "Troyes",
        _norm_name("Le Havre AC"): "Le Havre",
        _norm_name("Lille OSC"): "Lille",
        _norm_name("OGC Nice"): "Nice",
        _norm_name("Olympique Lyonnais"): "Lyon",
        _norm_name("Olympique de Marseille"): "Marseille",
        _norm_name("Paris Saint-Germain FC"): "Paris SG",
        _norm_name("RC Strasbourg Alsace"): "Strasbourg",
        _norm_name("Racing Club de Lens"): "Lens",
        _norm_name("Stade Brestois 29"): "Brest",
        _norm_name("Stade Rennais FC 1901"): "Rennes",
    },
}

if len(OUTPUT_COLUMNS) != 27 or len(set(OUTPUT_COLUMNS)) != 27:
    raise RuntimeError("Modelens sportsfeatureliste skal indeholde 27 unikke kolonner.")
EXPECTED_ELO_ORDER = [
    name
    for k in (16, 32, 48)
    for name in (f"home_elo_k{k}", f"away_elo_k{k}")
] + [f"elo_difference_k{k}" for k in (16, 32, 48)]
if list(OUTPUT_COLUMNS) != [
    "home_avg_goals_for", "home_avg_goals_against", "home_avg_points",
    "away_avg_goals_for", "away_avg_goals_against", "away_avg_points",
    "home_last5_goals_for", "home_last5_goals_against", "home_last5_points",
    "away_last5_goals_for", "away_last5_goals_against", "away_last5_points",
    *ATTACK_DEFENSE_COLUMNS,
    *EXPECTED_ELO_ORDER,
]:
    raise RuntimeError("Modelens 27 sportsfeatures har ændret navn eller rækkefølge.")


def _read_history(fixtures: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if fixtures is None:
        frame = pd.read_csv(FIXTURES_PATH, low_memory=False)
    elif isinstance(fixtures, (str, Path)):
        frame = pd.read_csv(fixtures, low_memory=False)
    elif isinstance(fixtures, pd.DataFrame):
        frame = fixtures.copy()
    else:
        raise TypeError("fixtures skal være en DataFrame, sti eller None.")

    missing = [column for column in HISTORY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Historiske fixtures mangler nødvendige kolonner: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Historiske fixtures indeholder manglende kampdatoer.")
    # Keep sorting deterministic even though the source CSV has mixed time
    # representations and many missing kickoff times.
    frame["time"] = frame["time"].map(lambda value: None if pd.isna(value) else str(value))
    for column in ("league", "home_team", "away_team"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Historiske fixtures indeholder tomme værdier i {column}.")
        frame[column] = frame[column].astype(str)

    duplicate_count = int(frame.duplicated(MATCH_KEYS).sum())
    if duplicate_count:
        raise ValueError(f"Historikken har {duplicate_count} dublerede kampnøgler.")
    return frame


def _normalize_requests(
    matches: Mapping[str, Any] | pd.Series | pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if isinstance(matches, pd.DataFrame):
        requests = matches.copy()
    elif isinstance(matches, pd.Series):
        requests = matches.to_frame().T
    elif isinstance(matches, Mapping):
        requests = pd.DataFrame([dict(matches)])
    else:
        requests = pd.DataFrame(list(matches))
    if requests.empty:
        raise ValueError("Der skal angives mindst én kommende kamp.")
    if "date" not in requests.columns and "match_date" in requests.columns:
        requests = requests.rename(columns={"match_date": "date"})
    missing = [column for column in REQUEST_KEYS if column not in requests.columns]
    if missing:
        raise ValueError(f"Kommende kampe mangler påkrævede oplysninger: {missing}")

    requests = requests.reset_index(drop=True)
    requests["date"] = pd.to_datetime(requests["date"], errors="raise").dt.normalize()
    if requests["date"].isna().any():
        raise ValueError("Kommende kamp mangler dato.")
    for column in ("league", "home_team", "away_team"):
        if requests[column].isna().any() or requests[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Kommende kamp har tom værdi i {column}.")
        requests[column] = requests[column].astype(str)
    requests["__request_id"] = np.arange(len(requests), dtype=np.int64)
    return requests


@dataclass(frozen=True)
class _HistorySnapshot:
    """Read-only feature state at the start of one calculation date."""

    calculation_date: pd.Timestamp
    match_count: int
    teams_by_league: dict[str, frozenset[str]]
    # league -> normalized name -> sorted historical labels with that key
    normalized_teams: dict[str, dict[str, tuple[str, ...]]]
    form: FormState
    ratings: RatingState


def _build_snapshot(history: pd.DataFrame, calculation_date: pd.Timestamp) -> _HistorySnapshot:
    day_history = history.loc[history["date"] < calculation_date]
    if day_history.empty:
        raise ValueError(
            f"Ingen historiske kampe før {calculation_date.date()}; "
            "featureberegningen afvises."
        )
    t = time.perf_counter()
    form = replay_form_state(day_history)
    ratings = replay_rating_state(day_history)

    appearances = pd.concat([
        day_history[["league", "home_team"]].rename(columns={"home_team": "team"}),
        day_history[["league", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).drop_duplicates()
    teams_by_league = {
        league: frozenset(group["team"])
        for league, group in appearances.groupby("league", sort=False)
    }
    normalized_teams: dict[str, dict[str, tuple[str, ...]]] = {}
    for league, teams in teams_by_league.items():
        by_key: dict[str, list[str]] = {}
        for name in sorted(teams):
            by_key.setdefault(_norm_name(name), []).append(name)
        normalized_teams[league] = {key: tuple(names) for key, names in by_key.items()}

    print(
        f"[FEATURE-TID] Historiktilstand bygget: {time.perf_counter() - t:.3f} sek. "
        f"({len(day_history)} historiske kampe, skæringsdato {calculation_date.date()})",
        flush=True,
    )
    return _HistorySnapshot(
        calculation_date=calculation_date,
        match_count=len(day_history),
        teams_by_league=teams_by_league,
        normalized_teams=normalized_teams,
        form=form,
        ratings=ratings,
    )


@lru_cache(maxsize=4)
def _cached_snapshot(
    path: str, mtime_ns: int, size: int, calculation_date: pd.Timestamp
) -> _HistorySnapshot:
    # mtime_ns/size are part of the cache key only, so a rewritten CSV
    # (e.g. by update_historical_data) is never served from a stale state.
    return _build_snapshot(_read_history(path), calculation_date)


_snapshot_lock = threading.Lock()


def _history_snapshot(
    fixtures: pd.DataFrame | str | Path | None, calculation_date: pd.Timestamp
) -> _HistorySnapshot:
    """Return the start-of-date state, reusing it while the fixtures file is unchanged."""
    if isinstance(fixtures, pd.DataFrame):
        return _build_snapshot(_read_history(fixtures), calculation_date)
    if fixtures is not None and not isinstance(fixtures, (str, Path)):
        raise TypeError("fixtures skal være en DataFrame, sti eller None.")
    path = Path(FIXTURES_PATH if fixtures is None else fixtures).resolve()
    stat = path.stat()
    # One replay at a time; concurrent requests for the same date wait for it.
    with _snapshot_lock:
        return _cached_snapshot(str(path), stat.st_mtime_ns, stat.st_size, calculation_date)


def _calculation_date(as_of_date: str | pd.Timestamp | None) -> pd.Timestamp:
    if as_of_date is None:
        return (
            pd.Timestamp.now(tz="Europe/Copenhagen")
            .normalize()
            .tz_localize(None)
        )
    calculation_date = pd.Timestamp(as_of_date).normalize()
    if calculation_date.tz is not None:
        calculation_date = calculation_date.tz_convert(
            "Europe/Copenhagen"
        ).tz_localize(None)
    return calculation_date


def _check_history_availability(snapshot: _HistorySnapshot, request: pd.Series) -> None:
    """Reject cold starts that the batch builders would otherwise fill by priors."""
    league = request["league"]
    teams = snapshot.teams_by_league.get(league)
    if not teams:
        raise ValueError(
            f"Liga {league!r} har ingen historik før {request['date'].date()}; "
            "featureberegningen afvises frem for at bruge startværdier alene."
        )
    for side in ("home", "away"):
        team = request[f"{side}_team"]
        if team not in teams:
            raise ValueError(
                f"{side.title()}hold {team!r} har ingen historik i liga {league!r} "
                f"før {request['date'].date()}; forudsigelsen afvises."
            )


def _resolve_history_team_name(
    snapshot: _HistorySnapshot, league: str, requested_name: str
) -> str:
    """Resolve an input name to one exact historical label without fuzzy matching."""
    available = snapshot.teams_by_league.get(league, frozenset())
    if requested_name in available:
        return requested_name

    requested_key = _norm_name(requested_name)
    aliased_name = HISTORY_TEAM_ALIASES.get(league, {}).get(requested_key)
    if aliased_name in available:
        return aliased_name

    normalized_matches = snapshot.normalized_teams.get(league, {}).get(requested_key, ())
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    if len(normalized_matches) > 1:
        raise ValueError(
            f"Holdnavnet {requested_name!r} matcher flere historiske navne i "
            f"{league!r}; forudsigelsen afvises."
        )
    return requested_name


def history_team_name(league: str, name: str) -> str:
    """fixtures.csv / football-data.co.uk label for a football-data.org team name.

    Uses the same exact/alias/normalized lookup as the feature calculation;
    an unknown or ambiguous name is returned unchanged.
    """
    snapshot = _history_snapshot(None, _calculation_date(None))
    try:
        return _resolve_history_team_name(snapshot, league, name)
    except ValueError:
        return name


def _calculate_for_date(
    snapshot: _HistorySnapshot, day_requests: pd.DataFrame
) -> pd.DataFrame:
    # Every request reads the same start-of-date state, so no candidate can
    # affect another, exactly as when they shared one replay date.
    rows = []
    for request in day_requests.itertuples(index=False):
        league, home, away = request.league, request.home_team, request.away_team
        values = {
            f"home_{name}": value
            for name, value in snapshot.form.team_features(league, home, 1).items()
        }
        values.update(
            (f"away_{name}", value)
            for name, value in snapshot.form.team_features(league, away, 0).items()
        )
        values.update(snapshot.ratings.match_features(league, home, away))
        rows.append([values[column] for column in OUTPUT_COLUMNS])

    values = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS), dtype=np.float64)
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        bad = values.columns[values.isna().any()].tolist()
        raise ValueError(
            "Featureberegningen gav manglende/ikke-endelige værdier; "
            f"ingen medianimputering udføres her. Kolonner: {bad}"
        )
    return values


def build_upcoming_match_features_partial(
    matches: Mapping[str, Any] | pd.Series | pd.DataFrame | Sequence[Mapping[str, Any]],
    fixtures: pd.DataFrame | str | Path | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Calculate one shared feature batch while rejecting invalid requests individually.

    Returns a 27-column DataFrame indexed by each input match's zero-based
    request position, plus errors containing request_id and reason. History is
    replayed at most once per fixtures file and calculation date.
    """
    total_start = time.perf_counter()
    requests = _normalize_requests(matches)
    if requests.duplicated(REQUEST_KEYS).any():
        # Preserve the existing strict duplicate policy. A duplicated request
        # batch is a caller/data issue rather than an individual cold start.
        raise ValueError("Kommende kampe indeholder dublerede kampnøgler.")

    calculation_date = _calculation_date(as_of_date)
    snapshot = _history_snapshot(fixtures, calculation_date)

    valid_requests: list[pd.Series] = []
    errors: list[dict[str, Any]] = []

    for _, request in requests.iterrows():
        resolved = request.copy()
        try:
            for side in ("home", "away"):
                resolved[f"{side}_team"] = _resolve_history_team_name(
                    snapshot,
                    str(request["league"]),
                    str(request[f"{side}_team"]),
                )
            _check_history_availability(snapshot, resolved)
            resolved["date"] = calculation_date
            valid_requests.append(resolved)
        except ValueError as exc:
            errors.append({
                "request_id": int(request["__request_id"]),
                "reason": str(exc),
            })

    if not valid_requests:
        print(
            f"[FEATURE-TID] Ingen gyldige kampe; {len(errors)} sprunget over.",
            flush=True,
        )
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS, dtype=np.float64)
        empty.index.name = "request_id"
        return empty, errors

    replay_requests = pd.DataFrame(valid_requests).reset_index(drop=True)
    calculated = _calculate_for_date(snapshot, replay_requests)
    calculated.index = pd.Index(
        replay_requests["__request_id"].to_numpy(dtype=np.int64), name="request_id"
    )
    result = calculated.sort_index().loc[:, OUTPUT_COLUMNS].astype(np.float64)

    if result.shape != (len(valid_requests), 27):
        raise AssertionError(
            "Resultatet skal indeholde præcis 27 features pr. gyldig kamp."
        )

    print(
        f"[FEATURE-TID] Features for {len(valid_requests)} gyldige kampe "
        f"({len(errors)} sprunget over): "
        f"{time.perf_counter() - total_start:.3f} sek.",
        flush=True,
    )
    return result, errors


def calculate_upcoming_match_features(
    match: Mapping[str, Any] | pd.Series,
    fixtures: pd.DataFrame | str | Path | None = None,
) -> pd.Series:
    """Return exactly the model's ordered 27 numeric sportsfeatures for one match."""
    result = build_upcoming_match_features(match, fixtures=fixtures)
    return result.iloc[0].copy()


def build_upcoming_match_features(
    matches: Mapping[str, Any] | pd.Series | pd.DataFrame | Sequence[Mapping[str, Any]],
    fixtures: pd.DataFrame | str | Path | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calculate 27 features from one shared historical snapshot.

    All requested matches use the same as-of date, regardless of their
    actual match dates. Historical results on or after the as-of date
    are excluded. Candidate matches never contribute to one another.
    """
    total_start = time.perf_counter()
    requests = _normalize_requests(matches)
    if requests.duplicated(REQUEST_KEYS).any():
        raise ValueError("Kommende kampe indeholder dublerede kampnøgler.")

    calculation_date = _calculation_date(as_of_date)
    snapshot = _history_snapshot(fixtures, calculation_date)

    replay_requests = requests.copy()
    for request_index, request in replay_requests.iterrows():
        for side in ("home", "away"):
            replay_requests.at[
                request_index, f"{side}_team"
            ] = _resolve_history_team_name(
                snapshot,
                request["league"],
                request[f"{side}_team"],
            )

    for _, request in replay_requests.iterrows():
        _check_history_availability(snapshot, request)

    result = _calculate_for_date(snapshot, replay_requests)

    if result.shape != (len(requests), 27):
        raise AssertionError(
            "Resultatet skal indeholde præcis 27 features pr. kamp."
        )

    print(
        f"[FEATURE-TID] Features for {len(requests)} kampe: "
        f"{time.perf_counter() - total_start:.3f} sek.",
        flush=True,
    )
    return result


def _historical_reference_sample(sample_size: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute selected historical rows through the upcoming-match API and compare references."""
    history = _read_history(None)
    feature_reference = pd.read_csv(PROJECT_ROOT / "data" / "features_asof_day_v1.csv", low_memory=False)
    elo_reference = pd.read_csv(PROJECT_ROOT / "data" / "elo_features_asof_day_v1.csv", low_memory=False)
    if "date" not in feature_reference or "date" not in elo_reference:
        raise ValueError("Historiske referencefiler mangler date-kolonnen.")
    feature_reference["date"] = pd.to_datetime(feature_reference["date"]).dt.normalize()
    elo_reference["date"] = pd.to_datetime(elo_reference["date"]).dt.normalize()
    if feature_reference.duplicated(MATCH_KEYS).any() or elo_reference.duplicated(MATCH_KEYS).any():
        raise ValueError("Historiske referencefiler indeholder dublerede kampnøgler.")

    # Spread sample dates across the full history; omit cold-start rows because
    # this API intentionally rejects teams without any earlier league history.
    candidates = history.copy()
    candidates["__row"] = np.arange(len(candidates))
    appearances = pd.concat([
        history[["date", "league", "home_team"]].rename(columns={"home_team": "team"}),
        history[["date", "league", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).drop_duplicates(["date", "league", "team"])
    first_seen = appearances.groupby(["league", "team"], sort=False)["date"].min().to_dict()
    eligible_mask = [
        first_seen.get((row.league, row.home_team), row.date) < row.date
        and first_seen.get((row.league, row.away_team), row.date) < row.date
        for row in candidates.itertuples(index=False)
    ]
    eligible = candidates.loc[eligible_mask]
    if eligible.empty:
        raise AssertionError("Ingen historiske kampe kan bruges til parity-test.")
    # Use several real fixtures from one date so both existing builders replay
    # the history only once, while the sample still exercises a same-day batch.
    day_sizes = eligible.groupby("date").size()
    eligible_dates = day_sizes[day_sizes >= sample_size].index
    if len(eligible_dates):
        # The earliest eligible date keeps the parity test fast while still
        # comparing real rows against both materialized reference datasets.
        chosen_day = eligible_dates[0]
        sample = eligible.loc[eligible["date"].eq(chosen_day)].head(sample_size).copy()
    else:
        chosen_day = day_sizes.index[0]
        sample = eligible.loc[eligible["date"].eq(chosen_day)].copy()

    requests = sample[MATCH_KEYS].rename(columns={"date": "match_date"}).reset_index(drop=True)
    calculated = build_upcoming_match_features(requests, fixtures=history, as_of_date=chosen_day)
    keys = requests.rename(columns={"match_date": "date"})[MATCH_KEYS]
    expected = keys.merge(
        feature_reference[MATCH_KEYS + [
            "home_avg_goals_for", "home_avg_goals_against", "home_avg_points",
            "away_avg_goals_for", "away_avg_goals_against", "away_avg_points",
            "home_last5_goals_for", "home_last5_goals_against", "home_last5_points",
            "away_last5_goals_for", "away_last5_goals_against", "away_last5_points",
        ]], on=MATCH_KEYS, how="left", validate="one_to_one"
    ).merge(
        elo_reference[MATCH_KEYS + ATTACK_DEFENSE_COLUMNS + ELO_COLUMNS],
        on=MATCH_KEYS, how="left", validate="one_to_one"
    )
    differences = []
    for column in OUTPUT_COLUMNS:
        actual = calculated[column].to_numpy(dtype=np.float64)
        reference = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=np.float64)
        differences.append({
            "feature": column,
            "different_values": int((~np.isclose(actual, reference, rtol=1e-10, atol=1e-10, equal_nan=True)).sum()),
            "max_abs_difference": float(np.nanmax(np.abs(actual - reference))) if len(actual) else 0.0,
        })
    return keys, calculated, pd.DataFrame(differences)


def _temporal_isolation_check() -> None:
    """Changing a same-day result leaves that day's prediction unchanged and alters later state."""
    fixtures = pd.DataFrame([
        {"date": "2020-01-01", "time": None, "league": "L", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "home_shots": 4, "away_shots": 2, "home_shots_on_target": 2, "away_shots_on_target": 1},
        {"date": "2020-01-01", "time": None, "league": "L", "home_team": "C", "away_team": "D", "home_goals": 0, "away_goals": 1, "home_shots": 2, "away_shots": 5, "home_shots_on_target": 1, "away_shots_on_target": 3},
        {"date": "2020-01-02", "time": None, "league": "L", "home_team": "A", "away_team": "C", "home_goals": 1, "away_goals": 1, "home_shots": 3, "away_shots": 3, "home_shots_on_target": 1, "away_shots_on_target": 1},
    ])
    changed = fixtures.copy()
    changed.loc[2, "home_goals"] = 4
    changed.loc[2, "away_goals"] = 0
    same_day = {"date": "2020-01-02", "league": "L", "home_team": "B", "away_team": "D"}
    next_day = {"date": "2020-01-03", "league": "L", "home_team": "A", "away_team": "C"}

    def features_on_match_date(match, history):
        # Pin the as-of date to the match date; the default as-of date is today.
        return build_upcoming_match_features(match, fixtures=history, as_of_date=match["date"]).iloc[0]

    pd.testing.assert_series_equal(
        features_on_match_date(same_day, fixtures),
        features_on_match_date(same_day, changed),
        check_names=False,
    )
    before = features_on_match_date(next_day, fixtures)
    after = features_on_match_date(next_day, changed)
    if np.allclose(before.to_numpy(), after.to_numpy(), rtol=1e-10, atol=1e-10):
        raise AssertionError("Ændret kampresultat påvirkede ikke features for senere dato.")


def _self_tests() -> None:
    _temporal_isolation_check()
    keys, _, differences = _historical_reference_sample()
    print(f"Historisk parity-sample: {len(keys)} kampe.")
    print(f"Features med mindst én forskel: {int(differences['different_values'].gt(0).sum())}/27.")
    changed = differences.loc[differences["different_values"].gt(0)]
    if not changed.empty:
        print("Afvigelser mod historiske referencefiler:")
        print(changed.to_string(index=False))
    else:
        print("Historiske sportsfeatures matcher referencefilerne inden for tolerance.")
    print("PASS: samme-dags-resultater påvirker ikke dagens features; ændringer kan påvirke senere datoer.")


class TestUpcomingMatchFeatures(unittest.TestCase):
    """Runnable read-only checks for temporal isolation and historical parity."""

    def test_same_day_isolation_and_later_update(self) -> None:
        _temporal_isolation_check()

    def test_historical_feature_parity_sample(self) -> None:
        keys, calculated, differences = _historical_reference_sample()
        self.assertEqual(calculated.shape, (len(keys), 27))
        self.assertEqual(tuple(calculated.columns), OUTPUT_COLUMNS)
        self.assertFalse(differences["different_values"].gt(0).any(), differences.to_string(index=False))

    def test_unknown_team_is_rejected(self) -> None:
        fixtures = pd.DataFrame([
            {"date": "2020-01-01", "time": None, "league": "L", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "home_shots": 1, "away_shots": 1, "home_shots_on_target": 1, "away_shots_on_target": 1},
        ])
        with self.assertRaisesRegex(ValueError, "ingen historik"):
            calculate_upcoming_match_features(
                {"date": "2020-01-02", "league": "L", "home_team": "NEW", "away_team": "B"},
                fixtures,
            )

    def test_official_names_resolve_to_legacy_history_names_across_leagues(self) -> None:
        # The input names come from football-data.org while fixtures.csv uses
        # its older, shorter team labels. These cases failed before the lookup
        # normalization/alias step was added.
        cases = [
            ("Premier League", "Aston Villa FC", "Aston Villa"),
            ("Premier League", "Manchester City FC", "Man City"),
            ("La Liga", "Club Atlético de Madrid", "Ath Madrid"),
            ("Bundesliga", "VfB Stuttgart", "Stuttgart"),
            ("Serie A", "AC Milan", "Milan"),
            ("Ligue 1", "Paris Saint-Germain FC", "Paris SG"),
        ]
        for index, (league, requested_name, history_name) in enumerate(cases):
            with self.subTest(league=league, requested_name=requested_name):
                opponent = f"Opponent {index}"
                fixtures = pd.DataFrame([{
                    "date": "2020-01-01", "time": None, "league": league,
                    "home_team": history_name, "away_team": opponent,
                    "home_goals": 1, "away_goals": 0, "home_shots": 4,
                    "away_shots": 2, "home_shots_on_target": 2,
                    "away_shots_on_target": 1,
                }])
                result = build_upcoming_match_features(
                    {
                        "date": "2020-01-02", "league": league,
                        "home_team": requested_name, "away_team": opponent,
                    },
                    fixtures=fixtures,
                )
                self.assertEqual(result.shape, (1, 27))
                self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_ambiguous_normalized_history_name_is_rejected(self) -> None:
        fixtures = pd.DataFrame([
            {
                "date": "2020-01-01", "time": None, "league": "Premier League",
                "home_team": "AFC Bournemouth", "away_team": "Opponent A",
                "home_goals": 1, "away_goals": 0, "home_shots": 4,
                "away_shots": 2, "home_shots_on_target": 2, "away_shots_on_target": 1,
            },
            {
                "date": "2020-01-01", "time": None, "league": "Premier League",
                "home_team": "Bournemouth FC", "away_team": "Opponent B",
                "home_goals": 0, "away_goals": 1, "home_shots": 2,
                "away_shots": 5, "home_shots_on_target": 1, "away_shots_on_target": 3,
            },
        ])
        with self.assertRaisesRegex(ValueError, "matcher flere historiske navne"):
            build_upcoming_match_features(
                {
                    "date": "2020-01-02", "league": "Premier League",
                    "home_team": "Bournemouth", "away_team": "Opponent A",
                },
                fixtures=fixtures,
            )


if __name__ == "__main__":
    _self_tests()
