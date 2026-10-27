"""Synchronize upcoming football-data.org fixtures and Bet365 odds into SQLite.

No requests are retried. Dry-run is strictly offline and never initializes or
changes SQLite. Cross-provider matching is deliberately conservative: same
competition, UTC match date, exact normalized home/away names, and kickoff
within the configured tolerance; ambiguous matches are skipped.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata
import uuid
from typing import Any, Mapping

import requests

try:
    from .data_store import DEFAULT_DATABASE_PATH, DataStore, utc_now
    from .test_api_football import _read_dotenv_value
except ImportError:  # pragma: no cover - direct script execution
    from data_store import DEFAULT_DATABASE_PATH, DataStore, utc_now
    from test_api_football import _read_dotenv_value


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
FIVE_DOLLAR_BASE = "https://api.5dollarfootballapi.com/v1"
FOOTBALL_DATA_SOURCE = "football-data.org"
FIVE_DOLLAR_SOURCE = "5dollarfootballapi"
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_DAYS = 30
DEFAULT_MATCH_REFRESH_HOURS = 12.0
DEFAULT_ODDS_REFRESH_HOURS = 6.0
DEFAULT_MAX_ODDS_CALLS = 8
KICKOFF_TOLERANCE_MINUTES = 30

# Exact, source-specific canonicalization for verified 5DollarFootballAPI names.
# These aliases are applied only to the provider-side names, before the existing
# exact normalized comparison. They do not relax date or kickoff requirements.
FIVE_DOLLAR_TEAM_ALIASES: dict[str, str] = {
    "ipswich": "ipswich town",
    "nottm forest": "nottingham forest",
    "man city": "manchester city",
    "coventry": "coventry city",
    "tottenham": "tottenham hotspur",
    "newcastle": "newcastle united",
    "hull": "hull city",
    "brighton": "brighton hove albion",
    "man utd": "manchester united",
    "leeds": "leeds united",
    # La Liga: exact, audit-verified 5DollarFootballAPI name variants.
    "levante": "levante ud",
    "cd alaves": "deportivo alaves",
    "rayo vallecano": "rayo vallecano de madrid",
    "racing santander": "real racing club de santander",
    "espanyol": "rcd espanyol de barcelona",
    "atletico madrid": "club atletico de madrid",
    "deportivo a coruna": "rc deportivo la coruna",
    "celta vigo": "rc celta de vigo",
    "real betis": "real betis balompie",
    "real sociedad": "real sociedad de futbol",
    "osasuna": "ca osasuna",
    # Bundesliga.
    "borussia m gladbach": "borussia monchengladbach",
    "union berlin": "1 union berlin",
    "cologne": "1 koln",
    "schalke": "schalke 04",
    "mainz": "1 fsv mainz 05",
    "werder bremen": "sv werder bremen",
    "paderborn": "paderborn 07",
    "hamburg": "hamburger sv",
    "elversberg": "sv 07 elversberg",
    "tsg hoffenheim": "tsg 1899 hoffenheim",
    "bayer leverkusen": "bayer 04 leverkusen",
    "bayern munich": "bayern munchen",
    # Serie A.
    "monza": "ac monza",
    "cagliari": "cagliari calcio",
    "bologna": "bologna 1909",
    "como": "como 1907",
    "sassuolo": "us sassuolo calcio",
    "napoli": "ssc napoli",
    "roma": "as roma",
    "lazio": "ss lazio",
    "parma": "parma calcio 1913",
    "inter milan": "internazionale milano",
    "fiorentina": "acf fiorentina",
    "atalanta": "atalanta bc",
    "frosinone": "frosinone calcio",
    "genoa": "genoa cfc",
    "lecce": "us lecce",
    "udinese": "udinese calcio",
    # Ligue 1.
    "brest": "stade brestois 29",
    "nice": "ogc nice",
    "troyes": "es troyes ac",
    "lens": "racing club de lens",
    "monaco": "as monaco",
    "lille": "lille osc",
    "angers": "angers sco",
    "auxerre": "aj auxerre",
    "marseille": "olympique de marseille",
    "le havre": "le havre ac",
    "rennes": "stade rennais 1901",
    "strasbourg": "rc strasbourg alsace",
    "psg": "paris saint germain",
    "lyon": "olympique lyonnais",
}

COMPETITIONS: dict[str, dict[str, str]] = {
    "PL": {"league": "Premier League", "search": "Premier League", "country": "england"},
    "PD": {"league": "La Liga", "search": "La Liga", "country": "spain"},
    "BL1": {"league": "Bundesliga", "search": "Bundesliga", "country": "germany"},
    "SA": {"league": "Serie A", "search": "Serie A", "country": "italy"},
    "FL1": {"league": "Ligue 1", "search": "Ligue 1", "country": "france"},
}


class SyncError(RuntimeError):
    """An API, response-validation, or synchronization failure."""


@dataclass
class SyncConfig:
    database_path: Path = DEFAULT_DATABASE_PATH
    leagues: tuple[str, ...] = tuple(COMPETITIONS)
    date_from: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    date_to: date | None = None
    match_refresh_hours: float = DEFAULT_MATCH_REFRESH_HOURS
    odds_refresh_hours: float = DEFAULT_ODDS_REFRESH_HOURS
    max_odds_calls: int = DEFAULT_MAX_ODDS_CALLS
    dry_run: bool = False
    skip_odds: bool = False

    def __post_init__(self) -> None:
        if self.date_to is None:
            self.date_to = self.date_from + timedelta(days=DEFAULT_DAYS)
        if self.date_to < self.date_from:
            raise ValueError("date_to skal være samme dato eller senere end date_from.")
        unknown = [league for league in self.leagues if league not in COMPETITIONS]
        if unknown:
            raise ValueError(f"Ukendte turneringskoder: {', '.join(unknown)}")
        if len(set(self.leagues)) != len(self.leagues):
            raise ValueError("En liga må kun angives én gang pr. synkronisering.")
        if self.match_refresh_hours < 0 or self.odds_refresh_hours < 0:
            raise ValueError("Cacheintervaller må ikke være negative.")
        if self.max_odds_calls < 0:
            raise ValueError("max_odds_calls må ikke være negativ.")


@dataclass
class SyncReport:
    date_from: date
    date_to: date
    sync_run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    matches_inserted: int = 0
    matches_updated: int = 0
    matches_unchanged: int = 0
    matches_rejected: int = 0
    odds_added: int = 0
    odds_duplicate: int = 0
    odds_no_quote: int = 0
    odds_cached: int = 0
    odds_unmatched: int = 0
    odds_ambiguous: int = 0
    odds_deferred: int = 0
    odds_detail_calls_by_league: dict[str, int] = field(default_factory=dict)
    # Per-league operational counts. The outcome buckets below describe the
    # first applicable state for a fixture, so a fixture is never counted in
    # multiple odds-handling buckets.
    odds_collection_by_league: dict[str, dict[str, int]] = field(default_factory=dict)


ODDS_COLLECTION_METRICS = (
    "upcoming_fixtures",
    "opening_reused",
    "opening_obtained",
    "opening_missing_after_lookup",
    "fixture_cache_missing",
    "odds_deferred",
    "no_usable_kickoff",
    "range_cache_skipped",
    "lookup_failed",
)


def _record_odds_metric(report: SyncReport, league: str, metric: str, count: int = 1) -> None:
    if metric not in ODDS_COLLECTION_METRICS:
        raise ValueError(f"Ukendt oddsrapportmåling: {metric}")
    bucket = report.odds_collection_by_league.setdefault(
        league, {name: 0 for name in ODDS_COLLECTION_METRICS}
    )
    bucket[metric] += count


def _token(variable: str) -> str:
    value = os.environ.get(variable) or _read_dotenv_value(variable, PROJECT_ROOT / ".env")
    if not value:
        raise SyncError(f"Token mangler: angiv {variable} i projektets .env-fil.")
    return value


def _safe(value: Any, token: str, limit: int = 400) -> str:
    return str(value).replace(token, "[REDACTED]")[:limit]


def _response_error(response: requests.Response, token: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return "API'et returnerede ikke JSON-fejloplysninger."
    if isinstance(payload, dict):
        # football-data.org's documented error is a message string.
        if isinstance(payload.get("message"), str):
            return _safe(payload["message"], token)
        error = payload.get("error")
        if isinstance(error, dict):
            pieces = []
            for key in ("type", "code", "message", "param", "request_id"):
                if error.get(key) not in (None, ""):
                    pieces.append(f"{key}={_safe(error[key], token, 200)}")
            if pieces:
                return "; ".join(pieces)
    return "API'et returnerede ingen kendte fejlbeskedfelter."


def _request_json(
    *,
    url: str,
    source: str,
    token: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
    report: SyncReport,
    label: str,
) -> dict[str, Any]:
    """Make one bounded GET with no retry, redirect, or secret-bearing output."""
    try:
        response = requests.get(
            url,
            headers=dict(headers),
            params=dict(params),
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except requests.Timeout:
        report.api_calls.append({"source": source, "label": label, "status": None})
        raise SyncError(f"{source}: timeout ved {label}.") from None
    except requests.ConnectionError:
        report.api_calls.append({"source": source, "label": label, "status": None})
        raise SyncError(f"{source}: netværksforbindelsen fejlede ved {label}.") from None
    except requests.RequestException:
        report.api_calls.append({"source": source, "label": label, "status": None})
        raise SyncError(f"{source}: HTTP-kaldet fejlede ved {label}.") from None

    rate_remaining = (
        response.headers.get("X-RateLimit-Remaining")
        or response.headers.get("X-Requests-Available-Minute")
    )
    report.api_calls.append({
        "source": source,
        "label": label,
        "status": response.status_code,
        "rate_remaining": rate_remaining,
        # Optional quota headers, shown in the dashboard when the provider sends them.
        "rate_limit": response.headers.get("X-RateLimit-Limit"),
        "rate_reset": response.headers.get("X-RateLimit-Reset"),
        "retry_after": response.headers.get("Retry-After"),
    })
    if response.status_code == 429:
        retry = response.headers.get("Retry-After")
        suffix = f" retry_after={retry}" if retry else ""
        raise SyncError(f"{source}: rate limit (HTTP 429){suffix}; intet retry udføres.")
    if response.status_code in (401, 403):
        raise SyncError(
            f"{source}: adgang afvist (HTTP {response.status_code}): "
            f"{_response_error(response, token)}"
        )
    if 300 <= response.status_code < 400:
        raise SyncError(f"{source}: uventet redirect (HTTP {response.status_code}); ikke fulgt.")
    if not response.ok:
        raise SyncError(
            f"{source}: HTTP {response.status_code} ved {label}: "
            f"{_response_error(response, token)}"
        )
    try:
        payload = response.json()
    except ValueError:
        raise SyncError(f"{source}: ugyldigt JSON ved {label}.") from None
    if not isinstance(payload, dict):
        raise SyncError(f"{source}: uventet JSON-struktur ved {label}.")
    if source == FIVE_DOLLAR_SOURCE and payload.get("success") != 1:
        raise SyncError(f"{source}: API-fejl ved {label}: {_response_error(response, token)}")
    return payload


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} mangler.")
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        raise ValueError(f"{label} er ugyldig.") from None


def _parse_kickoff(raw: Any) -> tuple[date, str | None]:
    """Parse a provider datetime; a date-only value keeps kickoff NULL."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("utcDate mangler.")
    value = raw.strip()
    if len(value) == 10:
        return _parse_date(value, "utcDate"), None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("utcDate er ugyldig.") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("utcDate har intet tidszone-offset.")
    utc = parsed.astimezone(timezone.utc)
    return utc.date(), utc.isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_football_data_match(item: Any, league_code: str) -> dict[str, Any]:
    """Validate and normalize one v4 match to the existing matches schema."""
    if league_code not in COMPETITIONS or not isinstance(item, dict):
        raise ValueError("ukendt turnering eller ugyldigt kampobjekt")
    external_id = item.get("id")
    if isinstance(external_id, bool) or external_id is None or not str(external_id).isdigit():
        raise ValueError("kampen mangler et gyldigt eksternt kamp-ID")
    raw_kickoff = item.get("utcDate")
    if raw_kickoff:
        match_date, kickoff_at = _parse_kickoff(raw_kickoff)
    else:
        match_date = _parse_date(item.get("date"), "kampdato")
        kickoff_at = None
    home_team = (item.get("homeTeam") or {}).get("name")
    away_team = (item.get("awayTeam") or {}).get("name")
    if not isinstance(home_team, str) or not home_team.strip():
        raise ValueError("hjemmehold mangler")
    if not isinstance(away_team, str) or not away_team.strip():
        raise ValueError("udehold mangler")
    if home_team.strip().casefold() == away_team.strip().casefold():
        raise ValueError("hjemmehold og udehold er ens")

    status = str(item.get("status") or "UNKNOWN").strip().upper()
    full_time = ((item.get("score") or {}).get("fullTime") or {})
    home_score, away_score = full_time.get("home"), full_time.get("away")
    if not isinstance(home_score, int) or isinstance(home_score, bool) or home_score < 0:
        home_score = None
    if not isinstance(away_score, int) or isinstance(away_score, bool) or away_score < 0:
        away_score = None
    if (home_score is None) != (away_score is None):
        home_score = away_score = None
    result = None
    if home_score is not None and away_score is not None:
        result = "H" if home_score > away_score else "A" if home_score < away_score else "D"

    season_obj = item.get("season") or {}
    season = season_obj.get("id") or season_obj.get("startDate")
    source_payload = dict(item)
    return {
        "data_source": FOOTBALL_DATA_SOURCE,
        "external_match_id": str(external_id),
        "source_match_key": f"external:{external_id}",
        "league": COMPETITIONS[league_code]["league"],
        "match_date": match_date.isoformat(),
        "kickoff_at": kickoff_at,
        "home_team": home_team.strip(),
        "away_team": away_team.strip(),
        "match_status": status,
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
        "season": str(season) if season is not None else None,
        "source_payload": source_payload,
    }


def _norm_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", plain)
    # Remove only generic club markers, not meaningful city/team words.
    tokens = [token for token in tokens if token not in {"fc", "afc", "cf", "sc"}]
    return " ".join(tokens)


def _team_fields(item: Mapping[str, Any]) -> tuple[str, str] | None:
    teams = item.get("teams") or {}
    home = (teams.get("home") or {}).get("name")
    away = (teams.get("away") or {}).get("name")
    if not isinstance(home, str) or not isinstance(away, str):
        return None
    if not home.strip() or not away.strip():
        return None
    return home.strip(), away.strip()


def _provider_team_key(value: str, *, source: str) -> str:
    """Return a strict canonical key, applying aliases only for 5Dollar names."""
    normalized = _norm_name(value)
    if source == FIVE_DOLLAR_SOURCE:
        return FIVE_DOLLAR_TEAM_ALIASES.get(normalized, normalized)
    return normalized


def _match_provider_fixture(
    fixture: Mapping[str, Any],
    football_matches: list[dict[str, Any]],
    *,
    provider_source: str,
) -> tuple[dict[str, Any] | None, str]:
    """Match same date and exact normalized home/away names; reject ambiguity."""
    candidate, reason, _ = _match_provider_fixture_with_diagnostics(
        fixture, football_matches, provider_source=provider_source
    )
    if reason == "multiple_candidates":
        reason = "ambiguous"
    elif reason not in {"matched", "invalid_teams", "invalid_kickoff", "invalid_date"}:
        reason = "unmatched"
    return candidate, reason


def _match_provider_fixture_with_diagnostics(
    fixture: Mapping[str, Any],
    football_matches: list[dict[str, Any]],
    *,
    provider_source: str,
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    """Apply the existing conservative rules and retain evidence for an audit row."""
    teams = _team_fields(fixture)
    if teams is None:
        return None, "invalid_teams", []
    home_key, away_key = (
        _provider_team_key(team, source=provider_source) for team in teams
    )
    raw_kickoff = fixture.get("kickoff_utc")
    provider_kickoff: datetime | None = None
    if raw_kickoff:
        try:
            provider_kickoff = datetime.fromisoformat(str(raw_kickoff).replace("Z", "+00:00"))
            if provider_kickoff.tzinfo is None:
                return None, "invalid_kickoff", []
            provider_kickoff = provider_kickoff.astimezone(timezone.utc)
        except ValueError:
            return None, "invalid_kickoff", []
    raw_date = fixture.get("kickoff_utc") or fixture.get("date")
    try:
        fixture_date = _parse_date(str(raw_date), "fixture-dato")
    except ValueError:
        return None, "invalid_date", []

    candidates: list[dict[str, Any]] = []
    audit_candidates: list[dict[str, Any]] = []
    same_date_count = 0
    same_team_count = 0
    kickoff_rejections = 0
    for match in football_matches:
        same_date = match["match_date"] == fixture_date.isoformat()
        same_teams = (
            _norm_name(match["home_team"]) == home_key
            and _norm_name(match["away_team"]) == away_key
        )
        if same_date:
            same_date_count += 1
        if same_teams:
            same_team_count += 1
        if not same_date and not same_teams:
            continue
        fd_kickoff = match.get("kickoff_at")
        kickoff_delta_minutes = None
        kickoff_compatible = True
        if provider_kickoff is not None and fd_kickoff:
            try:
                parsed_fd = datetime.fromisoformat(fd_kickoff.replace("Z", "+00:00"))
            except ValueError:
                kickoff_compatible = False
            else:
                kickoff_delta_minutes = abs((parsed_fd - provider_kickoff).total_seconds()) / 60
                kickoff_compatible = kickoff_delta_minutes <= KICKOFF_TOLERANCE_MINUTES
        diagnostic = {
            "football_data_match_id": match.get("match_id"),
            "external_match_id": match.get("external_match_id"),
            "league": match.get("league"),
            "match_date": match.get("match_date"),
            "kickoff_at": fd_kickoff,
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "match_status": match.get("match_status"),
            "same_date": same_date,
            "normalized_teams_match": same_teams,
            "provider_home_match_key": home_key,
            "provider_away_match_key": away_key,
            "kickoff_delta_minutes": kickoff_delta_minutes,
            "kickoff_compatible": kickoff_compatible,
        }
        audit_candidates.append(diagnostic)
        if not same_date or not same_teams:
            continue
        if not kickoff_compatible:
            kickoff_rejections += 1
            continue
        candidates.append(match)
    if not same_date_count:
        reason = "no_same_date_candidate"
    elif not same_team_count:
        reason = "team_names_mismatch"
    elif kickoff_rejections and not candidates:
        reason = "kickoff_difference_over_30_minutes"
    else:
        reason = "matched" if len(candidates) == 1 else "multiple_candidates"
    if not candidates:
        return None, reason, audit_candidates
    if len(candidates) != 1:
        return None, "multiple_candidates", audit_candidates
    return candidates[0], "matched", audit_candidates


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _state(store: DataStore, source: str, dataset: str) -> sqlite3.Row | None:
    return store.connection.execute(
        "SELECT * FROM sync_state WHERE data_source = ? AND dataset = ?",
        (source, dataset),
    ).fetchone()


def _state_is_fresh(
    row: sqlite3.Row | None,
    *,
    now: datetime,
    ttl_hours: float,
) -> bool:
    if row is None or ttl_hours <= 0:
        return False
    try:
        updated = datetime.fromisoformat(row["last_success_at_utc"].replace("Z", "+00:00"))
        return (now - updated).total_seconds() < ttl_hours * 3600
    except (TypeError, ValueError):
        return False


def _range_state_is_fresh(
    row: sqlite3.Row | None,
    *,
    now: datetime,
    ttl_hours: float,
    date_from: date,
    date_to: date,
) -> bool:
    """A fresh state only covers requests wholly contained in its saved range."""
    if not _state_is_fresh(row, now=now, ttl_hours=ttl_hours) or not row["cursor"]:
        return False
    try:
        start_text, end_text = row["cursor"].split("..", 1)
        covered_from = date.fromisoformat(start_text)
        covered_to = date.fromisoformat(end_text)
    except (ValueError, AttributeError):
        return False
    return covered_from <= date_from and covered_to >= date_to


def _readonly_sync_states(db_path: Path) -> dict[tuple[str, str], sqlite3.Row]:
    if not db_path.is_file():
        return {}
    # URI read-only mode avoids DataStore's schema initialization in dry-run.
    uri_path = db_path.resolve().as_posix()
    connection = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sync_state'"
        ).fetchone()
        if not exists:
            return {}
        return {
            (row["data_source"], row["dataset"]): row
            for row in connection.execute("SELECT * FROM sync_state")
        }
    finally:
        connection.close()


def build_dry_run_plan(config: SyncConfig, *, now: datetime | None = None) -> dict[str, Any]:
    """Build an offline call plan without creating/opening the DB for writes."""
    now = now or _utc_now_dt()
    states = _readonly_sync_states(config.database_path)
    plan: list[dict[str, Any]] = []
    total_fd = total_lookup = total_fixture = 0
    any_fixture_call = False
    for code in config.leagues:
        match_state = states.get((FOOTBALL_DATA_SOURCE, f"matches:{code}"))
        match_call = not _range_state_is_fresh(
            match_state, now=now, ttl_hours=config.match_refresh_hours,
            date_from=config.date_from, date_to=config.date_to,
        )
        odds_state = states.get((FIVE_DOLLAR_SOURCE, f"odds:{code}"))
        odds_fresh = (not config.skip_odds) and _range_state_is_fresh(
            odds_state, now=now, ttl_hours=config.odds_refresh_hours,
            date_from=config.date_from, date_to=config.date_to,
        )
        lookup_state = states.get((FIVE_DOLLAR_SOURCE, f"league_id:{code}"))
        lookup_call = not config.skip_odds and not odds_fresh and lookup_state is None
        fixture_call = not config.skip_odds and not odds_fresh
        total_fd += int(match_call)
        total_lookup += int(lookup_call)
        total_fixture += int(fixture_call)
        any_fixture_call = any_fixture_call or fixture_call
        plan.append({
            "league": code,
            "football_data_match_call": match_call,
            "five_dollar_league_lookup": lookup_call,
            "five_dollar_fixture_list": fixture_call,
            "odds_cache_fresh": odds_fresh,
        })
    return {
        "leagues": plan,
        "football_data_calls": total_fd,
        "five_dollar_league_lookups": total_lookup,
        "five_dollar_fixture_lists": total_fixture,
        # The detail-call cap is invocation-wide; runtime allocations divide it
        # fairly between selected leagues.
        "odds_detail_calls_upper_bound": config.max_odds_calls if any_fixture_call else 0,
        "total_calls_upper_bound": total_fd + total_lookup + total_fixture + (
            config.max_odds_calls if any_fixture_call else 0
        ),
        "database_exists": config.database_path.is_file(),
    }


def _load_football_matches(
    store: DataStore,
    league: str,
    date_from: date,
    date_to: date,
) -> list[dict[str, Any]]:
    rows = store.connection.execute(
        """SELECT * FROM matches
        WHERE data_source = ? AND league = ? AND match_date >= ? AND match_date <= ?
        ORDER BY match_date, kickoff_at, home_team, away_team""",
        (FOOTBALL_DATA_SOURCE, league, date_from.isoformat(), date_to.isoformat()),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_football_data(
    code: str,
    config: SyncConfig,
    *,
    token: str,
    report: SyncReport,
) -> list[dict[str, Any]]:
    payload = _request_json(
        url=f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches",
        source=FOOTBALL_DATA_SOURCE,
        token=token,
        headers={"X-Auth-Token": token},
        params={"dateFrom": config.date_from.isoformat(), "dateTo": config.date_to.isoformat()},
        report=report,
        label=f"matches:{code}",
    )
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise SyncError(f"{FOOTBALL_DATA_SOURCE}: `matches` mangler for {code}.")
    normalized = []
    for item in matches:
        try:
            match = normalize_football_data_match(item, code)
            if not (config.date_from <= date.fromisoformat(match["match_date"]) <= config.date_to):
                report.matches_rejected += 1
                continue
            normalized.append(match)
        except (TypeError, ValueError) as exc:
            report.matches_rejected += 1
            report.failures.append(f"{code}: kamp afvist ({exc}).")
    return normalized


def _persist_matches(
    store: DataStore,
    code: str,
    matches: list[dict[str, Any]],
    *,
    date_from: date,
    date_to: date,
    now: datetime,
    report: SyncReport,
) -> None:
    with store.transaction():
        for match in matches:
            _, status = store.upsert_match(match)
            if status == "inserted":
                report.matches_inserted += 1
            elif status == "updated":
                report.matches_updated += 1
            else:
                report.matches_unchanged += 1
        store.set_sync_state(
            FOOTBALL_DATA_SOURCE,
            f"matches:{code}",
            last_success_at=now,
            cursor=f"{date_from.isoformat()}..{date_to.isoformat()}",
            details={"returned_matches": len(matches), "league_code": code},
        )


def _resolve_five_dollar_league(
    store: DataStore,
    code: str,
    *,
    token: str,
    report: SyncReport,
    now: datetime,
) -> str:
    cached = _state(store, FIVE_DOLLAR_SOURCE, f"league_id:{code}")
    if cached and cached["cursor"] and str(cached["cursor"]).isdigit():
        return str(cached["cursor"])
    query = COMPETITIONS[code]["search"]
    payload = _request_json(
        url=f"{FIVE_DOLLAR_BASE}/leagues",
        source=FIVE_DOLLAR_SOURCE,
        token=token,
        headers={"Authorization": f"Bearer {token}"},
        params={"search": query, "page": 1, "per_page": 100},
        report=report,
        label=f"league_lookup:{code}",
    )
    items = payload.get("data")
    if not isinstance(items, list):
        raise SyncError(f"{FIVE_DOLLAR_SOURCE}: ligaopslag mangler en `data`-liste ({code}).")
    wanted = COMPETITIONS[code]
    expected_name = _norm_name(wanted["search"])
    expected_country = _norm_name(wanted["country"])
    candidates = []
    excluded_tokens = {"women", "womens", "u21", "u19", "reserve", "cup", "2", "ii"}
    for item in items:
        if not isinstance(item, dict) or not str(item.get("id", "")).isdigit():
            continue
        name = item.get("name")
        country = item.get("country")
        if isinstance(country, dict):
            country = country.get("name") or country.get("code")
        name_norm = _norm_name(str(name or ""))
        country_norm = _norm_name(str(country or ""))
        if expected_name not in name_norm or expected_country not in country_norm:
            continue
        if set(name_norm.split()) & excluded_tokens:
            continue
        candidates.append(item)
    if len(candidates) != 1:
        raise SyncError(
            f"{FIVE_DOLLAR_SOURCE}: ligaopslag for {code} gav {len(candidates)} "
            "entydige kandidater; ID'et caches ikke."
        )
    league_id = str(candidates[0]["id"])
    with store.transaction():
        store.set_sync_state(
            FIVE_DOLLAR_SOURCE,
            f"league_id:{code}",
            last_success_at=now,
            cursor=league_id,
            details={"league_name": candidates[0].get("name"), "country": wanted["country"]},
        )
    return league_id


def _fetch_five_dollar_fixtures(
    league_id: str,
    code: str,
    config: SyncConfig,
    *,
    token: str,
    report: SyncReport,
    now: datetime,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    start = datetime.combine(config.date_from, time.min, tzinfo=timezone.utc)
    end = datetime.combine(config.date_to + timedelta(days=1), time.min, tzinfo=timezone.utc)
    payload = _request_json(
        url=f"{FIVE_DOLLAR_BASE}/leagues/{league_id}/fixtures",
        source=FIVE_DOLLAR_SOURCE,
        token=token,
        headers={"Authorization": f"Bearer {token}"},
        params={
            "status": "scheduled",
            "start_time": int(start.timestamp()),
            "end_time": int(end.timestamp()),
            "order": "asc",
            "page": 1,
            "per_page": 100,
        },
        report=report,
        label=f"fixtures:{code}",
    )
    fixtures = payload.get("data")
    if not isinstance(fixtures, list):
        raise SyncError(f"{FIVE_DOLLAR_SOURCE}: fixture-listen mangler `data` ({code}).")
    output = []
    rejected: list[tuple[dict[str, Any], str]] = []
    for item in fixtures:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "scheduled":
            rejected.append((item, "fixture_status_not_scheduled"))
            continue
        if not str(item.get("id", "")).isdigit() or int(item["id"]) <= 0:
            rejected.append((item, "invalid_fixture_id"))
            continue
        kickoff = item.get("kickoff_utc")
        try:
            parsed = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            rejected.append((item, "invalid_or_missing_kickoff"))
            continue
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            rejected.append((item, "kickoff_missing_timezone"))
            continue
        if parsed.astimezone(timezone.utc) <= now:
            rejected.append((item, "kickoff_not_future"))
            continue
        if not (config.date_from <= parsed.astimezone(timezone.utc).date() <= config.date_to):
            rejected.append((item, "kickoff_outside_requested_date_range"))
            continue
        output.append(item)
    return output, rejected


def _has_stored_opening_odds(
    store: DataStore,
    *,
    match_id: int,
) -> bool:
    row = store.connection.execute(
        """SELECT 1 FROM odds_observations
        WHERE match_id = ? AND lower(replace(bookmaker, ' ', '')) = 'bet365'
          AND market = '1X2:opening'""",
        (match_id,),
    ).fetchone()
    # An opening quote is a historical observation, not a refreshable cache
    # value. Once stored, do not spend another API call trying to fetch it.
    return row is not None


def _latest_audited_fixture_id(store: DataStore, *, match_id: int) -> str | None:
    """Find the most recently matched provider fixture for a cached match."""
    row = store.connection.execute(
        """SELECT external_fixture_id FROM fixture_match_audit
        WHERE selected_match_id = ? AND outcome = 'matched'
          AND external_fixture_id IS NOT NULL
        ORDER BY audit_id DESC LIMIT 1""",
        (match_id,),
    ).fetchone()
    return str(row["external_fixture_id"]) if row else None


def _fixture_lookup_fresh(
    store: DataStore,
    fixture_id: str,
    *,
    now: datetime,
    ttl_hours: float,
) -> bool:
    return _state_is_fresh(
        _state(store, FIVE_DOLLAR_SOURCE, f"odds_fixture:{fixture_id}"),
        now=now,
        ttl_hours=ttl_hours,
    )


def _extract_odds(data: Any) -> tuple[str | None, dict[str, dict[str, float]]]:
    if not isinstance(data, dict) or not isinstance(data.get("bookmakers"), list):
        return None, {}
    for book in data["bookmakers"]:
        if not isinstance(book, dict) or str(book.get("slug", "")).casefold() != "bet365":
            continue
        name = str(book.get("name") or "Bet365")
        market = ((book.get("odds") or {}).get("1x2"))
        stages: dict[str, dict[str, float]] = {}
        if isinstance(market, dict):
            for stage in ("opening", "closing"):
                prices = market.get(stage)
                if not isinstance(prices, dict):
                    continue
                values: dict[str, float] = {}
                valid = True
                for key in ("home", "draw", "away"):
                    value = prices.get(key)
                    if isinstance(value, bool):
                        valid = False
                        break
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        valid = False
                        break
                    if not math.isfinite(number) or number <= 1.0:
                        valid = False
                        break
                    values[key] = number
                if valid:
                    stages[stage] = values
        return name, stages
    return None, {}


def _store_odds(
    store: DataStore,
    *,
    match_id: int,
    fixture_id: str,
    bookmaker: str,
    stages: dict[str, dict[str, float]],
    fetched_at: datetime,
    code: str,
    opening_status: str = "available",
) -> tuple[int, int]:
    added = duplicates = 0
    stamp = fetched_at.isoformat(timespec="seconds").replace("+00:00", "Z")
    with store.transaction():
        for stage, odds in stages.items():
            inserted = store.add_odds_observation(
                match_id=match_id,
                bookmaker=bookmaker,
                market=f"1X2:{stage}",
                odds_home=odds["home"],
                odds_draw=odds["draw"],
                odds_away=odds["away"],
                # Single-fixture endpoint has no per-stage recorded_at. This is
                # explicitly our retrieval time, not the bookmaker quote time.
                observed_at=fetched_at,
                data_source=FIVE_DOLLAR_SOURCE,
                fetched_at=fetched_at,
            )
            added += int(inserted)
            duplicates += int(not inserted)
        store.set_sync_state(
            FIVE_DOLLAR_SOURCE,
            f"odds_fixture:{fixture_id}",
            last_success_at=fetched_at,
            cursor=fixture_id,
            details={
                "league_code": code,
                "stages": sorted(stages),
                "opening_status": opening_status,
                "retrieved_at_utc": stamp,
            },
        )
    return added, duplicates


def _record_no_odds(
    store: DataStore,
    fixture_id: str,
    *,
    code: str,
    now: datetime,
) -> None:
    with store.transaction():
        store.set_sync_state(
            FIVE_DOLLAR_SOURCE,
            f"odds_fixture:{fixture_id}",
            last_success_at=now,
            cursor=fixture_id,
            details={
                "league_code": code,
                "result": "no_valid_bet365_opening_1x2",
                "opening_status": "missing_or_invalid",
            },
        )


def run_sync(
    config: SyncConfig,
    *,
    football_data_token: str | None = None,
    five_dollar_token: str | None = None,
    now: datetime | None = None,
) -> SyncReport:
    """Run a sync. Dry-run returns before token lookup, DB creation, or HTTP."""
    now = now or _utc_now_dt()
    report = SyncReport(config.date_from, config.date_to)
    if config.dry_run:
        plan = build_dry_run_plan(config, now=now)
        print(f"Dry-run; database exists: {plan['database_exists']}")
        print(f"Interval: {config.date_from} til {config.date_to} (UTC-datoer)")
        for item in plan["leagues"]:
            print(
                f"{item['league']}: football-data-kald={item['football_data_match_call']}; "
                f"ligaopslag={item['five_dollar_league_lookup']}; "
                f"5Dollar-fixtureliste={item['five_dollar_fixture_list']}"
            )
        print(
            "Estimeret maksimalt antal kald: "
            f"{plan['total_calls_upper_bound']} "
            f"(football-data {plan['football_data_calls']}, "
            f"5Dollar ligaopslag {plan['five_dollar_league_lookups']}, "
            f"fixturelister {plan['five_dollar_fixture_lists']}, "
            f"oddsdetaljer højst {plan['odds_detail_calls_upper_bound']} delt mellem ligaerne)."
        )
        if not plan["database_exists"]:
            print("Ingen lokal cache endnu; konkrete kampnavne/odds-kald kræver fixture-responser.")
        return report

    fd_token = football_data_token or _token("FOOTBALL_DATA_API_KEY")
    odds_token = None if config.skip_odds else (
        five_dollar_token or _token("5DOLLARFOOTBALL_API_KEY")
    )
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now skal være timezone-aware.")

    # Schema and migrations were inspected before creating the production DB.
    # DataStore creates its schema transactionally if the DB is absent.
    with DataStore(config.database_path) as store:
        odds_detail_calls = 0
        football_data_rate_limited = False
        five_dollar_rate_limited = False
        for league_index, code in enumerate(config.leagues):
            report.odds_collection_by_league.setdefault(
                code, {name: 0 for name in ODDS_COLLECTION_METRICS}
            )
            league_odds_detail_calls = 0
            league_name = COMPETITIONS[code]["league"]
            match_state = _state(store, FOOTBALL_DATA_SOURCE, f"matches:{code}")
            if _range_state_is_fresh(
                match_state, now=now, ttl_hours=config.match_refresh_hours,
                date_from=config.date_from, date_to=config.date_to,
            ):
                report.skipped.append(f"{code}: kamp-cache stadig frisk")
            elif football_data_rate_limited:
                report.skipped.append(f"{code}: football-data-kald sprunget over efter rate limit")
                continue
            else:
                try:
                    normalized = _fetch_football_data(
                        code, config, token=fd_token, report=report
                    )
                    _persist_matches(
                        store, code, normalized, date_from=config.date_from,
                        date_to=config.date_to, now=now, report=report,
                    )
                except SyncError as exc:
                    report.failures.append(str(exc))
                    if "HTTP 429" in str(exc):
                        football_data_rate_limited = True
                    continue

            if config.skip_odds:
                continue
            odds_state = _state(store, FIVE_DOLLAR_SOURCE, f"odds:{code}")
            if _range_state_is_fresh(
                odds_state, now=now, ttl_hours=config.odds_refresh_hours,
                date_from=config.date_from, date_to=config.date_to,
            ):
                report.skipped.append(f"{code}: odds-cache stadig frisk")
                # The range cache suppresses provider calls, so report only
                # fixtures already known locally. Do not imply that this is a
                # fresh provider fixture-list count.
                cached_matches = _load_football_matches(
                    store, league_name, config.date_from, config.date_to
                )
                for cached_match in cached_matches:
                    if cached_match.get("match_status") in {"FINISHED", "CANCELED", "CANCELLED"}:
                        continue
                    kickoff = cached_match.get("kickoff_at")
                    if not kickoff:
                        if date.fromisoformat(cached_match["match_date"]) >= now.astimezone(timezone.utc).date():
                            _record_odds_metric(report, code, "no_usable_kickoff")
                        continue
                    try:
                        cached_kickoff = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                    except ValueError:
                        _record_odds_metric(report, code, "no_usable_kickoff")
                        continue
                    if cached_kickoff <= now:
                        continue
                    _record_odds_metric(report, code, "upcoming_fixtures")
                    match_id = store.find_match_id(
                        FOOTBALL_DATA_SOURCE, cached_match["source_match_key"]
                    )
                    if match_id is not None and _has_stored_opening_odds(store, match_id=match_id):
                        _record_odds_metric(report, code, "opening_reused")
                    else:
                        fixture_id = (
                            _latest_audited_fixture_id(store, match_id=match_id)
                            if match_id is not None else None
                        )
                        if fixture_id and _fixture_lookup_fresh(
                            store, fixture_id, now=now, ttl_hours=config.odds_refresh_hours
                        ):
                            _record_odds_metric(report, code, "fixture_cache_missing")
                        else:
                            _record_odds_metric(report, code, "range_cache_skipped")
                continue
            if five_dollar_rate_limited:
                report.skipped.append(f"{code}: 5Dollar-kald sprunget over efter rate limit")
                continue

            matches = _load_football_matches(
                store, league_name, config.date_from, config.date_to
            )
            if not matches:
                report.skipped.append(f"{code}: ingen football-data-kampe i intervallet til matchning")
                continue

            try:
                league_id = _resolve_five_dollar_league(
                    store, code, token=odds_token, report=report, now=now
                )
                provider_fixtures, rejected_fixtures = _fetch_five_dollar_fixtures(
                    league_id, code, config, token=odds_token, report=report, now=now
                )
            except SyncError as exc:
                report.failures.append(str(exc))
                if "HTTP 429" in str(exc):
                    five_dollar_rate_limited = True
                continue

            _record_odds_metric(report, code, "upcoming_fixtures", len(provider_fixtures))
            _record_odds_metric(
                report,
                code,
                "no_usable_kickoff",
                sum(
                    reason in {"invalid_or_missing_kickoff", "kickoff_missing_timezone"}
                    for _, reason in rejected_fixtures
                ),
            )

            joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
            fixture_audits: list[dict[str, Any]] = []
            for rejected_fixture, rejection_reason in rejected_fixtures:
                fixture_audits.append({
                    "fixture": rejected_fixture,
                    "candidate": None,
                    "reason": rejection_reason,
                    "candidate_details": [],
                    "outcome": "rejected",
                })
            collisions: dict[str, int] = {}
            for fixture in provider_fixtures:
                candidate, reason, candidate_details = _match_provider_fixture_with_diagnostics(
                    fixture, matches, provider_source=FIVE_DOLLAR_SOURCE
                )
                fixture_audits.append({
                    "fixture": fixture,
                    "candidate": candidate,
                    "reason": reason,
                    "candidate_details": candidate_details,
                    "outcome": "matched" if candidate is not None else (
                        "ambiguous" if reason == "multiple_candidates" else "unmatched"
                    ),
                })
                if candidate is None and reason != "multiple_candidates":
                    report.odds_unmatched += 1
                elif candidate is None:
                    report.odds_ambiguous += 1
                if candidate is not None:
                    joined.append((fixture, candidate))
                    collisions[candidate["source_match_key"]] = collisions.get(
                        candidate["source_match_key"], 0
                    ) + 1
            safe_joined = []
            for fixture, match in joined:
                if collisions[match["source_match_key"]] > 1:
                    report.odds_ambiguous += 1
                    for audit in fixture_audits:
                        if audit["fixture"] is fixture:
                            audit["outcome"] = "ambiguous"
                            audit["reason"] = "multiple_provider_fixtures_map_to_same_match"
                            break
                    continue
                safe_joined.append((fixture, match))

            # Persist the decision and the candidate evidence before odds-detail
            # calls, so a later provider error cannot erase match diagnostics.
            with store.transaction():
                for audit in fixture_audits:
                    candidate = audit["candidate"]
                    store.add_fixture_match_audit(
                        sync_run_id=report.sync_run_id,
                        audited_at=now,
                        data_source=FIVE_DOLLAR_SOURCE,
                        fixture=audit["fixture"],
                        candidate_matches=audit["candidate_details"],
                        selected_match_id=(
                            int(candidate["match_id"])
                            if audit["outcome"] == "matched" and candidate and candidate.get("match_id") is not None
                            else None
                        ),
                        outcome=audit["outcome"],
                        reason=audit["reason"],
                    )

            odds_failed = False
            odds_deferred = False
            leagues_remaining = len(config.leagues) - league_index
            calls_remaining = config.max_odds_calls - odds_detail_calls
            league_call_budget = (
                (calls_remaining + leagues_remaining - 1) // leagues_remaining
                if leagues_remaining > 0 else 0
            )
            for fixture, match in sorted(
                safe_joined, key=lambda pair: str(pair[0].get("kickoff_utc", ""))
            ):
                fixture_id = str(fixture["id"])
                kickoff = match.get("kickoff_at")
                if match.get("match_status") in {"FINISHED", "CANCELED", "CANCELLED"}:
                    continue
                if not kickoff:
                    report.odds_unmatched += 1
                    _record_odds_metric(report, code, "upcoming_fixtures", -1)
                    _record_odds_metric(report, code, "no_usable_kickoff")
                    continue
                try:
                    match_kickoff = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                except ValueError:
                    report.odds_unmatched += 1
                    _record_odds_metric(report, code, "upcoming_fixtures", -1)
                    _record_odds_metric(report, code, "no_usable_kickoff")
                    continue
                if match_kickoff <= now:
                    continue
                match_id = store.find_match_id(
                    FOOTBALL_DATA_SOURCE, match["source_match_key"]
                )
                if match_id is None:
                    report.odds_unmatched += 1
                    continue
                if _has_stored_opening_odds(store, match_id=match_id):
                    report.odds_cached += 1
                    _record_odds_metric(report, code, "opening_reused")
                    continue
                if _fixture_lookup_fresh(
                    store, fixture_id, now=now, ttl_hours=config.odds_refresh_hours
                ):
                    report.odds_cached += 1
                    _record_odds_metric(report, code, "fixture_cache_missing")
                    continue
                # This share was calculated once at the start of the league.
                # If earlier leagues had only cached odds (or were skipped),
                # their unused capacity flows to the remaining leagues.
                if (
                    league_odds_detail_calls >= league_call_budget
                    or odds_detail_calls >= config.max_odds_calls
                ):
                    report.odds_deferred += 1
                    _record_odds_metric(report, code, "odds_deferred")
                    odds_deferred = True
                    continue
                odds_detail_calls += 1
                league_odds_detail_calls += 1
                report.odds_detail_calls_by_league[code] = league_odds_detail_calls
                try:
                    odds_payload = _request_json(
                        url=f"{FIVE_DOLLAR_BASE}/fixtures/{fixture_id}/odds",
                        source=FIVE_DOLLAR_SOURCE,
                        token=odds_token,
                        headers={"Authorization": f"Bearer {odds_token}"},
                        params={"bookmakers": "bet365", "market": "1x2"},
                        report=report,
                        label=f"odds:{fixture_id}",
                    )
                    odds_data = odds_payload.get("data")
                    response_fixture_id = (
                        odds_data.get("fixture_id") if isinstance(odds_data, dict) else None
                    )
                    if response_fixture_id is not None and str(response_fixture_id) != fixture_id:
                        raise SyncError("5Dollar: odds-responsens kamp-ID afviger fra forespørgslen.")
                    bookmaker, stages = _extract_odds(odds_data)
                    fetched_at = _utc_now_dt()
                    if bookmaker is None or not stages:
                        _record_no_odds(store, fixture_id, code=code, now=fetched_at)
                        report.odds_no_quote += 1
                        _record_odds_metric(report, code, "opening_missing_after_lookup")
                        continue
                    opening_status = "available" if "opening" in stages else "missing_or_invalid"
                    added, duplicates = _store_odds(
                        store, match_id=match_id, fixture_id=fixture_id,
                        bookmaker=bookmaker, stages=stages, fetched_at=fetched_at, code=code,
                        opening_status=opening_status,
                    )
                    report.odds_added += added
                    report.odds_duplicate += duplicates
                    if opening_status == "available":
                        _record_odds_metric(report, code, "opening_obtained")
                    else:
                        report.odds_no_quote += 1
                        _record_odds_metric(report, code, "opening_missing_after_lookup")
                except SyncError as exc:
                    report.failures.append(str(exc))
                    odds_failed = True
                    _record_odds_metric(report, code, "lookup_failed")
                    if "HTTP 429" in str(exc):
                        five_dollar_rate_limited = True
                        break

            if not odds_failed and not odds_deferred:
                with store.transaction():
                    store.set_sync_state(
                        FIVE_DOLLAR_SOURCE,
                        f"odds:{code}",
                        last_success_at=_utc_now_dt(),
                        cursor=f"{config.date_from.isoformat()}..{config.date_to.isoformat()}",
                        details={
                            "fixtures_returned": len(provider_fixtures),
                            "matched": len(safe_joined),
                            "odds_detail_calls": league_odds_detail_calls,
                        },
                    )
            elif odds_deferred:
                report.skipped.append(f"{code}: resterende odds-kald udskudt af max-odds-calls")

    return report


def print_report(report: SyncReport) -> None:
    print("\nSynkroniseringsrapport")
    print(f"Synkroniserings-ID: {report.sync_run_id}")
    print(f"Dato-interval (UTC): {report.date_from} til {report.date_to}")
    print(f"Faktiske API-kald: {len(report.api_calls)}")
    for call in report.api_calls:
        status = call["status"] if call["status"] is not None else "intet HTTP-svar"
        quota = f", kvote tilbage={call['rate_remaining']}" if call.get("rate_remaining") else ""
        print(f"  {call['source']} {call['label']}: {status}{quota}")
    print(
        "Kampe: "
        f"nye={report.matches_inserted}, opdaterede={report.matches_updated}, "
        f"uændrede={report.matches_unchanged}, afviste={report.matches_rejected}"
    )
    print(
        "Odds: "
        f"observationer gemt={report.odds_added}, dubletter={report.odds_duplicate}, "
        f"manglende/ugyldige opening odds efter opslag={report.odds_no_quote}, "
        f"umatchede={report.odds_unmatched}, tvetydige={report.odds_ambiguous}, "
        f"udskudte={report.odds_deferred}"
    )
    if report.odds_collection_by_league:
        metrics = ODDS_COLLECTION_METRICS
        totals = {
            name: sum(values.get(name, 0) for values in report.odds_collection_by_league.values())
            for name in metrics
        }
        print("Oddsstatus pr. liga (fixtures uden dobbelttælling i statuskategorierne):")
        header = (
            "Liga | kommende | opening genbrugt | opening hentet | mangler efter opslag | "
            "frisk fixture-cache | udskudt | uden kickoff | frisk interval-cache | opslag fejlede"
        )
        print("  " + header)
        for league, values in report.odds_collection_by_league.items():
            print(
                "  " + league + " | " + " | ".join(str(values.get(name, 0)) for name in metrics)
            )
        print(
            "  TOTAL | " + " | ".join(str(totals[name]) for name in metrics)
        )
        print(
            "  Definitioner: 'opening genbrugt' = allerede gemt Bet365 1X2:opening; "
            "'frisk fixture-cache' = fixtureopslag stadig inden for TTL uden gemt opening; "
            "'mangler efter opslag' = faktisk detailkald uden gyldig opening. "
            "'frisk interval-cache' tælles fra lokalt kendte kampe, fordi fixturelisten ikke hentes. "
            "Prioritet ved klassifikation: intet brugbart kickoff, gemt opening, frisk fixture-cache, "
            "udskudt, faktisk opslag (opening hentet/mangler), opslagfejl. 'Kommende' tæller "
            "fixtures med fremtidigt kickoff, som kan vurderes; fixtures uden brugbart kickoff "
            "står kun i deres særskilte tæller."
        )
    if report.odds_detail_calls_by_league:
        print("Oddsdetailkald pr. liga: " + ", ".join(
            f"{code}={count}" for code, count in report.odds_detail_calls_by_league.items()
        ))
    for item in report.skipped:
        print(f"Sprunget over: {item}")
    for failure in report.failures:
        print(f"Fejl: {failure}")
    print(
        "Tidsstempelnote: oddsens observed_at_utc er hentetidspunktet, når API'et "
        "ikke returnerer quote-tidsstempel. 1X2:opening/closing forbliver adskilt; "
        "closing er ikke behandlet som tidligere tilgængelige odds."
    )


def _parse_args(argv: list[str] | None = None) -> SyncConfig:
    parser = argparse.ArgumentParser(description="Synkronisér kampe og Bet365-odds til SQLite-cache.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Dage frem fra dagens UTC-dato.")
    parser.add_argument("--date-from", type=date.fromisoformat, default=None, help="Startdato YYYY-MM-DD (UTC).")
    parser.add_argument("--date-to", type=date.fromisoformat, default=None, help="Slutdato YYYY-MM-DD (UTC).")
    parser.add_argument("--leagues", nargs="+", choices=tuple(COMPETITIONS), default=list(COMPETITIONS))
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--match-refresh-hours", type=float, default=DEFAULT_MATCH_REFRESH_HOURS)
    parser.add_argument("--odds-refresh-hours", type=float, default=DEFAULT_ODDS_REFRESH_HOURS)
    parser.add_argument("--max-odds-calls", type=int, default=DEFAULT_MAX_ODDS_CALLS)
    parser.add_argument("--dry-run", action="store_true", help="Offline plan; no API or database writes.")
    parser.add_argument("--skip-odds", action="store_true", help="Synkronisér kun kampdata.")
    args = parser.parse_args(argv)
    start = args.date_from or datetime.now(timezone.utc).date()
    end = args.date_to or start + timedelta(days=args.days)
    return SyncConfig(
        database_path=args.database,
        leagues=tuple(args.leagues),
        date_from=start,
        date_to=end,
        match_refresh_hours=args.match_refresh_hours,
        odds_refresh_hours=args.odds_refresh_hours,
        max_odds_calls=args.max_odds_calls,
        dry_run=args.dry_run,
        skip_odds=args.skip_odds,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        report = run_sync(config)
        if not config.dry_run:
            print_report(report)
        return 1 if report.failures else 0
    except (OSError, sqlite3.Error, ValueError, SyncError) as exc:
        # API tokens/headers are never included in these errors.
        print(f"Synkronisering kunne ikke startes: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
