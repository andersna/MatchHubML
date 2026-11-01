"""Local prediction API for MatchHubML.

Run from the project root:
    uvicorn src.api:app --reload

The match cache is opened read-only. Predictions are logged to a separate
SQLite file (src/prediction_log.py) so they can be scored after kickoff.
Only `/api/sync` makes external calls.
"""
from __future__ import annotations

from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from urllib.parse import quote

import pandas as pd
from fastapi import Cookie, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .upcoming_match_features import OUTPUT_COLUMNS, build_upcoming_match_features_partial, history_team_name
from .predict_upcoming_match import (
    MODEL_FILES,
    assemble_model_input,
    latest_v3_dir,
    ODDS_MARKET_LABELS,
    ODDS_MARKETS,
    ODDS_PREFERENCE_SQL,
    odds_probabilities,
)
from . import game, prediction_log
from .backfill_prediction_log import backfill as backfill_predictions
from .update_historical_data import update_historical_data
from .build_elo_features_asof_day_v1 import replay_form_ratings
from .collect_understat_xg import update_understat_matches
from .sync_football_data import COMPETITIONS, SyncConfig, SyncError, run_sync
from .sync_football_data import print_report as print_sync_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "matchhub_cache.sqlite"
MODEL_DIRS = {
    "v1": PROJECT_ROOT / "models" / "nn_odds_asof_day_v1",
    "v2": PROJECT_ROOT / "models" / "nn_odds_asof_day_v2",
    "v3": PROJECT_ROOT / "models" / "nn_market_residual_v3",
}
DEFAULT_MODEL_VERSION = "v3"
SUPPORTED_LEAGUES = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
SYNC_DAYS_BACK = 14  # re-fetch recent matches so finished results reach the cache
SYNC_DAYS_AHEAD = 30
# football-data.co.uk publishes results a couple of times a week; refresh the
# history on a sync when the last download is older than this.
HISTORY_REFRESH_INTERVAL = timedelta(hours=12)
OUTCOMES = ("H", "D", "A")
# Set MATCHHUB_HIDE_MODEL=1 when others play the prediction league: /api/matches
# is refused and the frontend hides the Kampe tab, so nobody can copy the model
# before kickoff. Predictions are still made and logged (the bot's entries).
HIDE_MODEL_ENV = "MATCHHUB_HIDE_MODEL"
SESSION_COOKIE = "matchhub_session"
BOT_REFRESH_INTERVAL = timedelta(minutes=15)
# Started matches stay in the Kampe tab this long after kickoff (or until the
# cache marks them finished), shown as in play. Their predictions are the
# pre-match ones and are not logged, so the evaluation stays pre-kickoff only.
IN_PLAY_WINDOW = timedelta(hours=2)


def hide_model() -> bool:
    return os.environ.get(HIDE_MODEL_ENV, "").strip().lower() in {"1", "true", "yes"}


_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from .predict_match import load_model_bundle, predict_match
except ImportError:  # pragma: no cover
    from predict_match import load_model_bundle, predict_match

_sync_lock = threading.Lock()

app = FastAPI(
    title="MatchHubML API",
    version="1.1.0",
    description="Local football predictions using saved MatchHubML models.",
)


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the local MatchHubML frontend."""
    frontend_path = PROJECT_ROOT / "matchhubml_frontend_live.html"
    if not frontend_path.is_file():
        raise HTTPException(status_code=404, detail="Frontendfilen blev ikke fundet i projektmappen.")
    return FileResponse(frontend_path, media_type="text/html")


def available_versions() -> list[str]:
    return [
        version for version, path in model_dirs().items()
        if all((path / name).is_file() for name in MODEL_FILES)
    ]


def model_dirs() -> dict[str, Path]:
    """Artifact folder per version; v3 is the newest nn_market_residual_v3* folder,
    so a model from src/retrain_v3.py is used without code changes."""
    return {**MODEL_DIRS, "v3": latest_v3_dir(MODEL_DIRS["v3"])}


@lru_cache(maxsize=8)
def _load_bundle(artifact_dir: str):
    return load_model_bundle(artifact_dir=Path(artifact_dir), device="cpu")


def get_bundle(model_version: str):
    """Load a saved model once per artifact folder; never train or overwrite artifacts."""
    version = model_version.lower()
    dirs = model_dirs()
    if version not in dirs:
        raise ValueError("model_version skal være 'v1', 'v2' eller 'v3'.")
    artifact_dir = dirs[version]
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Modelmappen findes ikke: {artifact_dir}")
    return _load_bundle(str(artifact_dir))


def open_database_read_only() -> sqlite3.Connection:
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(f"SQLite-cachen findes ikke: {DATABASE_PATH}")
    uri = "file:" + quote(DATABASE_PATH.resolve().as_posix(), safe="/:\\") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


ODDS_CALLS_PER_SYNC = 5
# football-data.org's free tier allows 10 calls a minute and a sync uses one per
# league, so fixtures/results are fetched at most once an hour; presses in
# between only collect odds.
SYNC_MATCH_REFRESH_HOURS = 1.0
# A match looked up without a Bet365 price is retried after 6 hours,
# and a league's 5DollarFootball fixture list is re-read at least this often to
# discover newly listed matches.
SYNC_ODDS_REFRESH_HOURS = 6.0
LAST_SYNC_PATH = DATA_DIR / "last_sync.json"
CODE_BY_LEAGUE = {info["league"]: code for code, info in COMPETITIONS.items()}


def plan_odds_leagues(coverage: dict) -> tuple[str, ...]:
    """Leagues whose 5DollarFootball fixture list this press should download.

    Every league in the odds run costs one fixture-list call, so the 5 odds
    calls are concentrated: leagues are taken by earliest-kickoff missing match
    until they hold 5 fetchable matches. Leagues whose list is older than
    SYNC_ODDS_REFRESH_HOURS are added so newly listed matches are found.
    The order is smallest backlog first: run_sync gives each league
    ceil(calls left / leagues left), so the last league takes all spare calls.
    """
    leagues = {bucket["code"]: bucket for bucket in coverage.get("leagues", [])}
    chosen, budget = [], 0
    for code in sorted(
        (code for code, bucket in leagues.items() if bucket["fetchable"]),
        key=lambda code: leagues[code]["earliest_fetchable"],
    ):
        if budget >= ODDS_CALLS_PER_SYNC:
            break
        chosen.append(code)
        budget += leagues[code]["fetchable"]
    stale_before = (datetime.now(timezone.utc) - timedelta(hours=SYNC_ODDS_REFRESH_HOURS)).isoformat()
    for code in COMPETITIONS:
        last = leagues.get(code, {}).get("list_checked_utc")
        if code not in chosen and (last is None or last.replace("Z", "+00:00") < stale_before):
            chosen.append(code)
    return tuple(sorted(chosen, key=lambda code: leagues.get(code, {}).get("fetchable", 0)))


def odds_coverage() -> dict:
    """Upcoming cached matches grouped by why they do or don't have Bet365 odds.

    has_odds    - a Bet365 price stored (lookup or opening), ready for predictions
    fetchable   - 5DollarFootball lists the match; one odds call fetches it
    no_odds_yet - looked up, but Bet365 had no price (retried after 6 h)
    not_listed  - the provider does not list the match yet (usually far-off fixtures)
    """
    empty = {"total": 0, "has_odds": 0, "fetchable": 0, "no_odds_yet": 0, "not_listed": 0,
             "leagues": [], "missing": [], "presses_needed": 0, "next_press_calls": 0,
             "calls_per_sync": ODDS_CALLS_PER_SYNC}
    if not DATABASE_PATH.is_file():
        return empty
    with closing(open_database_read_only()) as connection:
        rows = connection.execute(
            f"""
            WITH listed AS (
                SELECT selected_match_id AS match_id, external_fixture_id, max(audited_at_utc)
                FROM fixture_match_audit
                WHERE outcome = 'matched' AND selected_match_id IS NOT NULL
                GROUP BY selected_match_id
            )
            SELECT m.league, m.kickoff_at, m.home_team, m.away_team,
                   EXISTS (
                       SELECT 1 FROM odds_observations AS o
                       WHERE o.match_id = m.match_id AND o.market IN {ODDS_MARKETS}
                         AND lower(replace(o.bookmaker, ' ', '')) = 'bet365'
                   ) AS has_odds,
                   l.external_fixture_id, s.details_json, s.last_success_at_utc AS looked_up_utc
            FROM matches AS m
            LEFT JOIN listed AS l ON l.match_id = m.match_id
            LEFT JOIN sync_state AS s
              ON s.data_source = '5dollarfootballapi'
             AND s.dataset = 'odds_fixture:' || l.external_fixture_id
            WHERE m.data_source = 'football-data.org'
              AND upper(m.match_status) NOT IN ('FINISHED', 'CANCELED', 'CANCELLED')
              AND m.kickoff_at > ?
            ORDER BY m.kickoff_at
            """,
            (_utc_now_iso(),),
        ).fetchall()
        list_checked = dict(connection.execute(
            """SELECT m.league, max(a.audited_at_utc) FROM fixture_match_audit AS a
            JOIN matches AS m ON m.match_id = a.selected_match_id
            WHERE a.data_source = '5dollarfootballapi' GROUP BY m.league"""
        ).fetchall())

    states = ("has_odds", "fetchable", "no_odds_yet", "not_listed")
    retry_before = (datetime.now(timezone.utc) - timedelta(hours=SYNC_ODDS_REFRESH_HOURS)).isoformat()
    leagues = {
        league: {"league": league, "code": code, "total": 0, **dict.fromkeys(states, 0),
                 "earliest_fetchable": None, "list_checked_utc": list_checked.get(league)}
        for league, code in CODE_BY_LEAGUE.items()
    }
    missing = []
    for row in rows:
        if row["league"] not in leagues:
            continue
        if row["has_odds"]:
            state = "has_odds"
        elif row["external_fixture_id"] is None:
            state = "not_listed"
        elif (
            row["details_json"]
            and json.loads(row["details_json"]).get("opening_status") != "available"
            and (row["looked_up_utc"] or "").replace("Z", "+00:00") >= retry_before
        ):
            state = "no_odds_yet"
        else:
            state = "fetchable"
        bucket = leagues[row["league"]]
        bucket["total"] += 1
        bucket[state] += 1
        if state == "fetchable" and bucket["earliest_fetchable"] is None:
            bucket["earliest_fetchable"] = row["kickoff_at"]  # rows are in kickoff order
        if state != "has_odds":
            missing.append({key: row[key] for key in ("league", "kickoff_at", "home_team", "away_team")} | {"state": state})

    totals = {state: sum(bucket[state] for bucket in leagues.values()) for state in states}
    coverage = {
        **empty,
        "total": len(rows),
        **totals,
        # Each press spends all 5 odds calls while 5 or more matches are fetchable.
        "presses_needed": math.ceil(totals["fetchable"] / ODDS_CALLS_PER_SYNC),
        "leagues": [bucket for bucket in leagues.values() if bucket["total"]],
        "missing": missing[:40],
    }
    # 5DollarFootball calls the next press uses: one fixture list per planned league + odds calls.
    coverage["next_press_calls"] = len(plan_odds_leagues(coverage)) + min(totals["fetchable"], ODDS_CALLS_PER_SYNC)
    return coverage


@app.get("/api/odds-coverage")
def get_odds_coverage():
    """How many upcoming matches have odds, and how many presses fetch the rest."""
    try:
        return odds_coverage()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Kunne ikke læse SQLite-cachen: {exc}") from exc


def _plain_failure(message: str) -> str:
    """Turn sync errors into something a user can act on."""
    if "HTTP 429" in message:
        source = "football-data.org" if "football-data" in message else "5DollarFootball"
        return f"{source} har nået sin grænse for antal kald. Vent et minut (eller til i morgen, hvis dagskvoten er brugt) og prøv igen."
    if "HTTP 401" in message or "HTTP 403" in message or "Token mangler" in message:
        return f"Adgang afvist – tjek API-nøglen i .env. ({message})"
    if "timeout" in message or "netværks" in message:
        return f"Ingen forbindelse til API'et – prøv igen om lidt. ({message})"
    return message


def _reset_time(value) -> str | None:
    """Quota reset header -> UTC ISO time. Accepts epoch seconds or seconds from now."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    moment = (
        datetime.fromtimestamp(seconds, tz=timezone.utc) if seconds > 1e9
        else datetime.now(timezone.utc) + timedelta(seconds=seconds)
    )
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _sync_summary(reports, history, before: dict, after: dict, started: float) -> dict:
    by_source: dict[str, dict] = {}
    for report in reports:
        for call in report.api_calls:
            source = by_source.setdefault(
                call["source"], {"calls": 0, "quota_remaining": None, "quota_limit": None, "quota_reset_utc": None}
            )
            source["calls"] += 1
            # Keep the latest response's values.
            if call.get("rate_remaining") not in (None, ""):
                source["quota_remaining"] = call["rate_remaining"]
            if call.get("rate_limit") not in (None, ""):
                source["quota_limit"] = call["rate_limit"]
            reset = _reset_time(call.get("retry_after")) or _reset_time(call.get("rate_reset"))
            if reset:
                source["quota_reset_utc"] = reset
    football = by_source.get("football-data.org", {})
    try:
        football_left = int(football.get("quota_remaining"))
    except (TypeError, ValueError):
        football_left = 0
    # Another press within the minute would exceed football-data.org's per-minute limit.
    cooldown = 60 if football.get("calls") and football_left < len(COMPETITIONS) else 0

    failures = [message for report in reports for message in report.failures]
    obtained = sum(
        values.get("opening_obtained", 0)
        for report in reports for values in report.odds_collection_by_league.values()
    )
    if history.skipped_by_interval:
        history_text = "Allerede opdateret inden for 12 timer."
    elif history.error:
        history_text = f"Kunne ikke opdateres; de eksisterende data bruges ({history.error})."
    else:
        history_text = f"{history.new_matches} nye resultater hentet fra football-data.co.uk."
    matches_fetched = football.get("calls", 0) > 0
    inserted = sum(report.matches_inserted for report in reports)
    updated = sum(report.matches_updated for report in reports)
    return {
        "status": "ok" if not failures else "partial",
        "finished_utc": _utc_now_iso(),
        "seconds": round(time.perf_counter() - started, 1),
        "odds_fetched": obtained,
        "odds_without_price": sum(report.odds_no_quote for report in reports),
        "matches": {
            "fetched": matches_fetched,
            "new": inserted,
            "updated": updated,
            "text": (
                f"{inserted} nye kampe og {updated} opdaterede (inkl. resultater for afsluttede kampe)."
                if matches_fetched else "Kampe og resultater er hentet inden for den seneste time – sprunget over for at spare kald."
            ),
        },
        "history": {"text": history_text, "new_matches": history.new_matches, "error": history.error},
        "api_calls": by_source,
        "cooldown_seconds": cooldown,
        "coverage_before": {k: before[k] for k in ("total", "has_odds", "fetchable", "presses_needed")},
        "coverage_after": {k: after[k] for k in ("total", "has_odds", "fetchable", "presses_needed", "next_press_calls")},
        "failures": [_plain_failure(message) for message in failures],
    }


def _sync_config(**overrides) -> SyncConfig:
    today = datetime.now(timezone.utc).date()
    settings = {
        "date_from": today - timedelta(days=SYNC_DAYS_BACK),
        "date_to": today + timedelta(days=SYNC_DAYS_AHEAD),
        "match_refresh_hours": SYNC_MATCH_REFRESH_HOURS,
        "odds_refresh_hours": SYNC_ODDS_REFRESH_HOURS,
        "max_odds_calls": ODDS_CALLS_PER_SYNC,
    }
    return SyncConfig(**(settings | overrides))


@app.post("/api/sync")
def sync_data():
    """Update history, then fixtures/results, then up to 5 Bet365 odds lookups."""
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="En synkronisering kører allerede. Vent, til den er færdig.")
    started = time.perf_counter()
    try:
        before = odds_coverage()
        # Fail-safe: on any error the existing fixtures.csv is kept and the match sync still runs.
        history = update_historical_data(refresh_interval=HISTORY_REFRESH_INTERVAL)
        try:
            # 1) Fixtures and results for every league (football-data.org, at most hourly).
            reports = [run_sync(_sync_config(skip_odds=True))]
            # 2) Odds only for the planned leagues; their matches were refreshed in step 1.
            odds_leagues = plan_odds_leagues(before)
            if odds_leagues:
                reports.append(run_sync(_sync_config(leagues=odds_leagues, match_refresh_hours=24 * 365)))
        except (SyncError, ValueError, OSError, sqlite3.Error) as exc:
            raise HTTPException(status_code=502, detail=_plain_failure(str(exc))) from exc
        if history.attempted and not history.error:
            _refresh_understat_xg()
            # A fresh season file can give played matches odds for the Resultater tab.
            try:
                backfill_predictions({v: get_bundle(v) for v in available_versions()}, DATABASE_PATH)
            except (sqlite3.Error, ValueError, OSError) as exc:
                print(f"[ADVARSEL] Rekonstruktion efter synkronisering fejlede: {exc}", flush=True)
        technical = io.StringIO()
        with redirect_stdout(technical):
            for report in reports:
                print_sync_report(report)
        summary = _sync_summary(reports, history, before, odds_coverage(), started)
        summary["odds_leagues"] = list(odds_leagues)
        summary["technical_report"] = technical.getvalue().strip()[-8000:]
    finally:
        _sync_lock.release()

    try:
        LAST_SYNC_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[ADVARSEL] Synkroniseringsresultat kunne ikke gemmes: {exc}", flush=True)
    return summary


@app.get("/api/sync/last")
def last_sync():
    """The result of the most recent sync, so it survives a page reload."""
    if not LAST_SYNC_PATH.is_file():
        return {"status": "none"}
    return json.loads(LAST_SYNC_PATH.read_text(encoding="utf-8"))


LAST_HISTORY_REFRESH_PATH = DATA_DIR / "last_history_refresh.json"


@lru_cache(maxsize=2)
def _fixtures_results(mtime_ns: int) -> dict[tuple, dict]:
    """match_identity -> result from fixtures.csv, with football-data.co.uk team names."""
    fixtures = pd.read_csv(
        DATA_DIR / "fixtures.csv", low_memory=False,
        usecols=["date", "league", "home_team", "away_team", "home_goals", "away_goals", "result"],
    ).dropna(subset=["result", "home_goals", "away_goals"])
    fixtures = fixtures[fixtures["result"].isin(OUTCOMES)]
    return {
        prediction_log.match_identity(row.league, str(row.date)[:10], row.home_team, row.away_team): {
            "match_date": str(row.date)[:10],
            "home_score": int(row.home_goals), "away_score": int(row.away_goals), "result": row.result,
        }
        for row in fixtures.itertuples(index=False)
    }


@lru_cache(maxsize=1)
def _form_ratings(mtime_ns: int) -> dict[tuple[str, str], float]:
    return replay_form_ratings(pd.read_csv(DATA_DIR / "fixtures.csv", low_memory=False))


def form_rating(league: str, team: str) -> float | None:
    """The app's form rating for a team (display only; not a model input)."""
    path = DATA_DIR / "fixtures.csv"
    if not path.is_file():
        return None
    return _form_ratings(path.stat().st_mtime_ns).get((league, history_team_name(league, team)))


def history_results(keys: list[tuple[str, str, str, str]]) -> dict[tuple, dict]:
    """Results from fixtures.csv for logged matches the match cache does not have."""
    path = DATA_DIR / "fixtures.csv"
    if not path.is_file():
        return {}
    known = _fixtures_results(path.stat().st_mtime_ns)
    found = {}
    for league, match_date, home, away in keys:
        result = known.get(prediction_log.match_identity(
            league, match_date, history_team_name(league, home), history_team_name(league, away)))
        if result:
            found[(league, match_date, home, away)] = result
    return found


@lru_cache(maxsize=4)
def _workbook_summary(path_text: str, mtime_ns: int) -> dict:
    path = Path(path_text)
    excel = pd.ExcelFile(path)
    played = with_odds = 0
    last_played = None
    for sheet in ("E0", "SP1", "D1", "I1", "F1"):
        if sheet not in excel.sheet_names:
            continue
        frame = pd.read_excel(excel, sheet_name=sheet)
        done = frame.dropna(subset=["FTR"]) if "FTR" in frame else frame.iloc[0:0]
        played += len(done)
        if {"B365H", "B365D", "B365A"} <= set(done.columns):
            with_odds += int(done[["B365H", "B365D", "B365A"]].notna().all(axis=1).sum())
        if len(done):
            day = pd.to_datetime(done["Date"], dayfirst=True, errors="coerce").max()
            if pd.notna(day) and (last_played is None or day.strftime("%Y-%m-%d") > last_played):
                last_played = day.strftime("%Y-%m-%d")
    modified = datetime.fromtimestamp(mtime_ns / 1e9, tz=timezone.utc)
    return {
        "file": path.name,
        "downloaded_utc": modified.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "played_matches": played,
        "played_with_odds": with_odds,
        "last_played_date": last_played,
    }


def workbook_summary() -> dict | None:
    """The newest saved football-data.co.uk season workbook: played matches and Bet365 coverage."""
    files = sorted((DATA_DIR / "raw").glob("all-euro-data-*.xlsx"))
    if not files:
        return None
    return _workbook_summary(str(files[-1]), files[-1].stat().st_mtime_ns)


def _history_text(history) -> str:
    if history.error:
        return f"Download fejlede; de eksisterende data bruges ({history.error})."
    if history.skipped_by_interval:
        return "Allerede opdateret for nylig."
    return (
        f"{history.new_matches} nye kampe og {history.updated_matches} spillede kampe opdateret "
        "med resultat og statistik."
    )


def _refresh_understat_xg() -> str | None:
    """Refresh Understat xG (v3's xG features) after a history update.

    Finished seasons are cached and the current season is re-downloaded at most
    daily. A failure only logs a warning: predictions then use the existing xG.
    """
    try:
        update_understat_matches()
        return None
    except Exception as exc:  # network/parse problems must never break a sync
        print(f"[ADVARSEL] Understat-xG kunne ikke opdateres: {exc}", flush=True)
        return str(exc)


@app.post("/api/history/refresh")
def refresh_history():
    """Download the newest football-data.co.uk season file now and use it.

    The file is free and has no quota. It adds new matches and fills results
    and stats for played matches in fixtures.csv (form, Elo and xG features),
    is saved with its Bet365 odds for retraining, and gives played matches
    without a 5DollarFootball quote odds for the Resultater tab.
    """
    if not _sync_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="En synkronisering kører allerede. Vent, til den er færdig.")
    started = time.perf_counter()
    try:
        history = update_historical_data(force=True)
        backfilled = {"logged": 0, "matches_added": 0, "from_workbook": 0, "from_5dollar": 0, "no_odds": 0, "skipped": []}
        xg_error = None
        if not history.error:
            xg_error = _refresh_understat_xg()
            try:
                bundles = {version: get_bundle(version) for version in available_versions()}
                backfilled = backfill_predictions(bundles, DATABASE_PATH)
            except (sqlite3.Error, ValueError, OSError) as exc:
                backfilled["skipped"] = [f"Rekonstruktion fejlede: {exc}"]
    finally:
        _sync_lock.release()

    summary = {
        "status": "error" if history.error else "ok",
        "finished_utc": _utc_now_iso(),
        "seconds": round(time.perf_counter() - started, 1),
        "text": _history_text(history),
        "new_matches": history.new_matches,
        "updated_matches": history.updated_matches,
        "unchanged_matches": history.unchanged_matches,
        "missing_odds": history.missing_odds,
        "error": history.error,
        "backfill": backfilled,
        "xg_error": xg_error,
        "workbook": workbook_summary(),
    }
    try:
        LAST_HISTORY_REFRESH_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[ADVARSEL] Resultatet kunne ikke gemmes: {exc}", flush=True)
    return summary


@app.get("/api/history")
def history_status():
    """When the season file was last downloaded and what it contains."""
    state_path = DATA_DIR / "historical_update_state.json"
    return {
        "update_state": json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None,
        "workbook": workbook_summary(),
        "last_refresh": (
            json.loads(LAST_HISTORY_REFRESH_PATH.read_text(encoding="utf-8"))
            if LAST_HISTORY_REFRESH_PATH.is_file() else None
        ),
    }


@app.get("/api/health")
def health():
    """Status check without loading a model or making network calls."""
    return {
        "status": "ok",
        "database_exists": DATABASE_PATH.is_file(),
        "available_models": {
            version: all((path / name).is_file() for name in MODEL_FILES)
            for version, path in model_dirs().items()
        },
        "default_model": DEFAULT_MODEL_VERSION,
        "external_sync": False,
        "hide_model": hide_model(),
    }


def _upcoming_with_odds(league: str | None, limit: int) -> list[dict]:
    """Future and in-play fixtures (kicked off less than IN_PLAY_WINDOW ago, not
    finished) joined with their preferred stored Bet365 quote (see ODDS_MARKETS),
    in one query. Each row gets in_play = kickoff has passed."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    cutoff_iso = (now - IN_PLAY_WINDOW).isoformat(timespec="seconds").replace("+00:00", "Z")
    params: list[object] = [cutoff_iso]
    league_clause = ""
    if league:
        league_clause = " AND m.league = ?"
        params.append(league)
    params.append(limit)
    with open_database_read_only() as connection:
        rows = connection.execute(
            f"""
            SELECT m.league, m.match_date, m.kickoff_at, m.home_team, m.away_team,
                   o.odds_home, o.odds_draw, o.odds_away, o.observed_at_utc, o.market
            FROM matches AS m
            JOIN odds_observations AS o ON o.odds_observation_id = (
                SELECT o2.odds_observation_id FROM odds_observations AS o2
                WHERE o2.match_id = m.match_id
                  AND lower(replace(o2.bookmaker, ' ', '')) = 'bet365'
                  AND o2.market IN {ODDS_MARKETS}
                ORDER BY {ODDS_PREFERENCE_SQL.format(alias='o2')}
                LIMIT 1
            )
            WHERE m.data_source = 'football-data.org'
              AND m.league IN ('Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1')
              AND upper(m.match_status) NOT IN ('FINISHED', 'CANCELED', 'CANCELLED')
              AND m.kickoff_at IS NOT NULL
              AND m.kickoff_at > ?
              {league_clause}
            ORDER BY m.kickoff_at, m.league, m.home_team
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) | {"in_play": row["kickoff_at"] <= now_iso} for row in rows]


def _skip(fixture: dict, reason: str) -> dict:
    return {key: fixture[key] for key in ("league", "match_date", "home_team", "away_team")} | {"reason": reason}


@app.get("/api/matches")
def list_predictions(
    model_version: str = Query(default=DEFAULT_MODEL_VERSION, pattern="^(v1|v2|v3)$"),
    league: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    if hide_model():
        raise HTTPException(status_code=403, detail="Modellens forudsigelser er skjult, mens spillet kører.")
    return predict_upcoming(model_version, league, limit)


def predict_upcoming(model_version: str = DEFAULT_MODEL_VERSION, league: str | None = None, limit: int = 50):
    """Predict future and in-play cached fixtures with stored Bet365 odds (lookup price, else opening).

    Every available model predicts every match (cheap once features exist);
    all predictions are logged for later scoring, the selected one is returned.
    """
    selected_version = model_version.lower()
    if league is not None and league not in SUPPORTED_LEAGUES:
        raise HTTPException(status_code=400, detail="Ukendt liga.")
    versions = available_versions()
    if selected_version not in versions:
        raise HTTPException(status_code=500, detail=f"Model {selected_version} er ikke tilgængelig.")
    try:
        bundles = {version: get_bundle(version) for version in versions}
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Modellen kunne ikke indlæses: {exc}") from exc

    started = time.perf_counter()
    try:
        rows = _upcoming_with_odds(league, limit)
    except (sqlite3.Error, FileNotFoundError) as exc:
        raise HTTPException(status_code=500, detail=f"Kunne ikke læse SQLite-cachen: {exc}") from exc

    skipped: list[dict] = []
    fixtures: list[dict] = []
    for row in rows:
        try:
            row["odds_probabilities"] = odds_probabilities(
                row["odds_home"], row["odds_draw"], row["odds_away"]
            )
            fixtures.append(row)
        except ValueError as exc:
            skipped.append(_skip(row, str(exc)))

    predictions: list[dict] = []
    log_records: list[dict] = []
    if fixtures:
        try:
            sports_batch, feature_errors = build_upcoming_match_features_partial([
                {key: fixture[key] for key in ("league", "home_team", "away_team", "match_date")}
                for fixture in fixtures
            ])
        except (ValueError, FileNotFoundError, RuntimeError, KeyError, AssertionError) as exc:
            # A failure in the shared feature engine is reported per match.
            print(f"[FEJL] Samlet feature-beregning fejlede: {exc}", flush=True)
            sports_batch, feature_errors = pd.DataFrame(columns=OUTPUT_COLUMNS), [
                {"request_id": index, "reason": f"Samlet feature-beregning fejlede: {exc}"}
                for index in range(len(fixtures))
            ]
        for error in feature_errors:
            skipped.append(_skip(fixtures[int(error["request_id"])], str(error["reason"])))

        for index, fixture in enumerate(fixtures):
            if index not in sports_batch.index:
                continue
            try:
                sports = {column: float(value) for column, value in sports_batch.loc[index].items()}
                if not all(math.isfinite(value) for value in sports.values()):
                    raise ValueError("Feature-beregningen indeholder ikke-endelige værdier.")
                model_input, parsed_date = assemble_model_input(
                    fixture["league"], fixture["home_team"], fixture["away_team"],
                    fixture["match_date"], *fixture["odds_probabilities"],
                    sports_features=sports,
                )
                all_models = {}
                for version, bundle in bundles.items():
                    result = predict_match(model_input, bundle=bundle)
                    probs = [result["probabilities"][label]["decimal"] for label in OUTCOMES]
                    all_models[version] = {"artifact": result["model_version"], "probs": probs}
                    if fixture["in_play"]:
                        continue  # only pre-kickoff predictions are logged and scored
                    log_records.append({
                        "source": "live", "model_version": version,
                        "model_artifact": result["model_version"],
                        "league": fixture["league"], "match_date": str(parsed_date.date()),
                        "kickoff_at": fixture["kickoff_at"], "home_team": fixture["home_team"],
                        "away_team": fixture["away_team"],
                        "odds_observed_at_utc": fixture["observed_at_utc"],
                        "odds_market": fixture["market"],
                        "odds": fixture["odds_probabilities"], "probs": probs,
                        "expected_goals": (sports["home_expected_goals"], sports["away_expected_goals"]),
                    })
            except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
                skipped.append(_skip(fixture, str(exc)))
                continue

            market = fixture["odds_probabilities"]
            decimal = (fixture["odds_home"], fixture["odds_draw"], fixture["odds_away"])
            chosen = all_models[selected_version]["probs"]
            predictions.append({
                "league": fixture["league"],
                "match_date": str(parsed_date.date()),
                "kickoff_at": fixture["kickoff_at"],
                "in_play": fixture["in_play"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "model_version_selected": selected_version,
                "model_version": all_models[selected_version]["artifact"],
                "probabilities": {label: chosen[i] for i, label in enumerate(OUTCOMES)},
                "odds_source": ODDS_MARKET_LABELS.get(fixture["market"], fixture["market"]),
                "odds_market": fixture["market"],
                "odds_observed_at_utc": fixture["observed_at_utc"],
                "odds_decimal": {label: decimal[i] for i, label in enumerate(OUTCOMES)},
                "odds_probabilities": {label: market[i] for i, label in enumerate(OUTCOMES)},
                # Model minus market (probability points) and expected value at Bet365's price.
                "edge": {label: chosen[i] - market[i] for i, label in enumerate(OUTCOMES)},
                "expected_value": {label: chosen[i] * decimal[i] - 1.0 for i, label in enumerate(OUTCOMES)},
                "all_models": {
                    version: {label: info["probs"][i] for i, label in enumerate(OUTCOMES)}
                    for version, info in all_models.items()
                },
                "features": sports,
                "form_rating": {
                    "home": form_rating(fixture["league"], fixture["home_team"]),
                    "away": form_rating(fixture["league"], fixture["away_team"]),
                },
            })

    try:
        logged = prediction_log.log_predictions(log_records)
    except sqlite3.Error as exc:  # logging must never break predictions
        print(f"[ADVARSEL] Forudsigelser kunne ikke logges: {exc}", flush=True)
        logged = 0
    print(
        f"[TID] /api/matches: {time.perf_counter() - started:.3f} sek. "
        f"({len(predictions)} kampe, {len(skipped)} sprunget over, {logged} nye logrækker)",
        flush=True,
    )
    return {
        "model_version_selected": selected_version,
        "count": len(predictions),
        "matches": predictions,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "logged_new": logged,
        "note": "Bet365-prisen ved seneste opslag bruges (ellers åbningsprisen). Alle modellers forudsigelser logges til senere evaluering.",
    }


@app.get("/api/performance")
def performance():
    """Logged predictions scored against finished results, per model and vs. the odds."""
    try:
        return prediction_log.evaluate(DATABASE_PATH, fallback_results=history_results)
    except (sqlite3.Error, FileNotFoundError) as exc:
        raise HTTPException(status_code=500, detail=f"Kunne ikke evaluere forudsigelser: {exc}") from exc


def _read_csv_rows(path: Path, **filters) -> list[dict]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path, low_memory=False)
    for column, value in filters.items():
        frame = frame[frame[column].astype(str) == str(value)]
    return json.loads(frame.to_json(orient="records"))


@app.get("/api/models")
def models():
    """Model metadata plus the backtest and experiment results behind them."""
    residual = DATA_DIR / "nn_market_residual_summary.csv"
    backtest_rows = {
        row["variant"]: row
        for row in _read_csv_rows(residual, scope="overall")
        if row["seed"] in ("single", "mean5", "42")
    }

    def backtest(variant: str) -> dict | None:
        row = backtest_rows.get(variant)
        if not row:
            return None
        return {key: row.get(key) for key in (
            "matches", "accuracy", "log_loss", "brier_score", "recall_D",
            "delta_log_loss_vs_odds", "delta_log_loss_vs_odds_se",
        )}

    descriptions = {
        "v1": ("Neuralt netværk på 27 sportsfeatures + 3 odds-sandsynligheder.", "current_v1"),
        "v2": ("Som V1, gentrænet med 48 kampe fra 2026-27.", "current_v1"),
        "v3": ("Retter Bet365-markedet: log(odds) + korrektion fra et netværk. Ensemble af 5 seeds, ingen klassevægte.", "residual_27_ensemble"),
    }
    result = []
    for version, path in model_dirs().items():
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training = metadata.get("training_data", {})
        optimizer = metadata.get("optimizer", {})
        description, variant = descriptions[version]
        result.append({
            "version": version,
            "default": version == DEFAULT_MODEL_VERSION,
            "artifact": metadata.get("artifact_name"),
            "description": description,
            "model_class": metadata.get("model_class"),
            "members": metadata.get("member_count", 1),
            "created_utc": metadata.get("created_utc"),
            "training_rows": training.get("rows"),
            "training_start": training.get("start_date"),
            "training_end": training.get("end_date"),
            "class_weights": "ingen" if version == "v3" else "alpha=0,25",
            "epochs": optimizer.get("epochs_by_seed") or optimizer.get("epochs"),
            "features": metadata.get("feature_columns_in_order", []),
            # A retrained v3 carries its own holdout test (src/retrain_v3.py).
            "backtest": own_backtest if (own_backtest := metadata.get("backtest", {})).get("season") else backtest(variant),
            "backtest_note": (
                f"Holdout: trænet før sæson {own_backtest['season']}, testet på dens "
                f"{own_backtest['matches']:,} kampe.".replace(f"{own_backtest['matches']:,}", f"{own_backtest['matches']:,}".replace(",", "."))
                if own_backtest.get("season")
                else None if version == "v3"
                else "Backtest for V1-arkitekturen (enkelt seed); V2 deler arkitektur."
            ),
        })
    return {
        "models": result,
        "odds_backtest": backtest("odds_only"),
        "backtest_scope": "Kronologiske testfolds 2010-2026 (fold 2-3), identiske kampe for alle varianter.",
        "experiments": {
            "market_residual": [
                {k: row.get(k) for k in ("variant", "matches", "accuracy", "log_loss",
                                         "delta_log_loss_vs_odds", "delta_log_loss_vs_odds_se")}
                for row in backtest_rows.values() if row["seed"] in ("single", "mean5")
            ],
            "pinnacle_xg": _read_csv_rows(DATA_DIR / "nn_pinnacle_xg_summary.csv"),
        },
    }


def _file_info(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {"exists": True, "modified_utc": modified.isoformat(timespec="seconds").replace("+00:00", "Z")}


@app.get("/api/status")
def status():
    """Freshness of every data source the predictions depend on."""
    info: dict = {"generated_utc": _utc_now_iso()}

    fixtures_path = DATA_DIR / "fixtures.csv"
    info["history"] = _file_info(fixtures_path)
    if fixtures_path.is_file():
        fixtures = pd.read_csv(fixtures_path, usecols=["date", "league"], low_memory=False)
        info["history"].update({
            "matches": int(len(fixtures)),
            "first_date": str(fixtures["date"].min()),
            "last_date": str(fixtures["date"].max()),
            "by_league": {k: int(v) for k, v in fixtures["league"].value_counts().items()},
        })
    state_path = DATA_DIR / "historical_update_state.json"
    if state_path.is_file():
        info["history"]["update_state"] = json.loads(state_path.read_text(encoding="utf-8"))

    info["cache"] = _file_info(DATABASE_PATH)
    if DATABASE_PATH.is_file():
        now = _utc_now_iso()
        with open_database_read_only() as connection:
            info["cache"].update({
                "upcoming": connection.execute(
                    "SELECT count(*) FROM matches WHERE upper(match_status) NOT IN ('FINISHED','CANCELED','CANCELLED') AND kickoff_at > ?",
                    (now,),
                ).fetchone()[0],
                "upcoming_with_odds": connection.execute(
                    """SELECT count(DISTINCT m.match_id) FROM matches m JOIN odds_observations o ON o.match_id = m.match_id
                    WHERE o.market IN ('1X2:closing', '1X2:opening') AND upper(m.match_status) NOT IN ('FINISHED','CANCELED','CANCELLED')
                      AND m.kickoff_at > ?""",
                    (now,),
                ).fetchone()[0],
                "finished": connection.execute(
                    "SELECT count(*) FROM matches WHERE upper(match_status) = 'FINISHED'"
                ).fetchone()[0],
                "odds_observations": connection.execute("SELECT count(*) FROM odds_observations").fetchone()[0],
                "last_syncs": [dict(row) for row in connection.execute(
                    "SELECT data_source, dataset, last_success_at_utc FROM sync_state ORDER BY last_success_at_utc DESC LIMIT 12"
                )],
            })

    info["understat"] = _file_info(DATA_DIR / "understat_matches.csv")
    info["prediction_log"] = _file_info(prediction_log.LOG_PATH)
    if prediction_log.LOG_PATH.is_file():
        with closing(sqlite3.connect(prediction_log.LOG_PATH)) as connection:
            info["prediction_log"]["rows"] = connection.execute("SELECT count(*) FROM predictions").fetchone()[0]
    info["models"] = available_versions()
    return info


# ---------- Prediction league (src/game.py) ----------

class Credentials(BaseModel):
    username: str
    password: str


class EntryIn(BaseModel):
    league: str
    match_date: str
    home_team: str
    away_team: str
    pick: str
    home_goals: int | None = None
    away_goals: int | None = None


class EntriesIn(BaseModel):
    entries: list[EntryIn]


class LeagueIn(BaseModel):
    name: str


class JoinIn(BaseModel):
    code: str


_bot_lock = threading.Lock()
_bot_refreshed_at: datetime | None = None


def _refresh_bot_entries() -> None:
    """Predict and log upcoming matches (the bot's entries) at most every 15 minutes, in the background."""
    global _bot_refreshed_at
    now = datetime.now(timezone.utc)
    if (_bot_refreshed_at and now - _bot_refreshed_at < BOT_REFRESH_INTERVAL) or not _bot_lock.acquire(blocking=False):
        return
    _bot_refreshed_at = now

    def run() -> None:
        try:
            predict_upcoming(game.BOT_MODEL_VERSION, None, 200)
        except Exception as exc:  # the game must work even when predictions fail
            print(f"[ADVARSEL] Botten kunne ikke forudsige: {exc}", flush=True)
        finally:
            _bot_lock.release()

    threading.Thread(target=run, daemon=True).start()


def game_results(keys: list[tuple[str, str, str, str]]) -> dict[tuple, dict]:
    """Finished results from the match cache, with fixtures.csv as fallback; matched on
    match_identity, so a key with a match's old date still finds its result."""
    found = prediction_log._finished_results(DATABASE_PATH) if DATABASE_PATH.is_file() else {}
    identity = prediction_log.match_identity
    wanted = {key: found[identity(*key)] for key in keys if identity(*key) in found}
    missing = [key for key in keys if key not in wanted]
    if missing:
        wanted.update(history_results(missing))
    return wanted


def _player(session: str | None) -> dict:
    user = game.user_for_token(session)
    if user is None:
        raise HTTPException(status_code=401, detail="Log ind for at spille.")
    return user


def _is_https(request: Request) -> bool:
    """True when the browser talks HTTPS: directly, or through Cloudflare's tunnel
    (cloudflared -> Traefik -> app is plain HTTP, so Cloudflare's CF-Visitor header
    is what still says https; Traefik passes it through untouched)."""
    return request.url.scheme == "https" or '"https"' in request.headers.get("cf-visitor", "")


def _set_session(request: Request, response: Response, token: str) -> None:
    # Secure only over HTTPS, so local http://...localhost logins keep working.
    response.set_cookie(
        SESSION_COOKIE, token, max_age=game.SESSION_DAYS * 86400, httponly=True, samesite="lax",
        secure=_is_https(request),
    )


def _game_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except game.GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (sqlite3.Error, FileNotFoundError) as exc:
        raise HTTPException(status_code=500, detail=f"Spillets data kunne ikke læses: {exc}") from exc


@app.post("/api/game/register")
def game_register(credentials: Credentials, request: Request, response: Response):
    _set_session(request, response, _game_call(game.register, credentials.username, credentials.password))
    return {"ok": True}


@app.post("/api/game/login")
def game_login(credentials: Credentials, request: Request, response: Response):
    _set_session(request, response, _game_call(game.login, credentials.username, credentials.password))
    return {"ok": True}


@app.post("/api/game/logout")
def game_logout(response: Response, session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    if session:
        game.logout(session)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/game/me")
def game_me(session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    user = game.user_for_token(session)
    if user is None:
        return {"user": None}
    return {"user": user["username"], "leagues": game.user_leagues(user["user_id"])}


@app.get("/api/game/round")
def game_round(
    week: str | None = Query(default=None),
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    """The week's matches with the player's picks; the bot's pick only after kickoff."""
    user = _player(session)
    _refresh_bot_entries()
    week = week or game.iso_week(datetime.now(timezone.utc))
    return _game_call(
        game.round_view, user["user_id"], week,
        cache_path=DATABASE_PATH, log_path=prediction_log.LOG_PATH, results=game_results,
    )


@app.put("/api/game/entries")
def game_entries(payload: EntriesIn, session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    user = _player(session)
    return _game_call(
        game.save_entries, user["user_id"], [entry.model_dump() for entry in payload.entries],
        cache_path=DATABASE_PATH,
    )


@app.post("/api/game/leagues")
def game_create_league(payload: LeagueIn, session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    return _game_call(game.create_league, _player(session)["user_id"], payload.name)


@app.post("/api/game/leagues/join")
def game_join_league(payload: JoinIn, session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    return _game_call(game.join_league, _player(session)["user_id"], payload.code)


@app.get("/api/game/table")
def game_table(
    league: str = Query(default=game.GLOBAL_LEAGUE),
    week: str | None = Query(default=None),
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    """Standings with MatchHub V3 as a player; private leagues only for members."""
    user = _player(session)
    if league != game.GLOBAL_LEAGUE and league not in {l["code"] for l in game.user_leagues(user["user_id"])}:
        raise HTTPException(status_code=403, detail="Du er ikke med i den liga.")
    return _game_call(
        game.league_table, league, cache_path=DATABASE_PATH, log_path=prediction_log.LOG_PATH,
        results=game_results, week=week or game.iso_week(datetime.now(timezone.utc)),
    )
