"""Local SQLite storage for MatchHubML matches and timestamped odds snapshots."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "matchhub_cache.sqlite"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_timestamp_utc(value: str | datetime, *, field_name: str) -> str:
    """Convert an aware timestamp to UTC; reject naive values rather than guessing."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} skal indeholde tidszone, fx 2026-10-20T12:00:00Z.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def deterministic_match_key(
    data_source: str,
    match_date: str,
    league: str,
    home_team: str,
    away_team: str,
) -> str:
    """Stable key for sources without IDs; kickoff is omitted so time changes do not duplicate."""
    parts = [str(data_source).strip(), str(match_date).strip(), str(league).strip(),
             str(home_team).strip(), str(away_team).strip()]
    canonical = "\x1f".join(part.casefold() for part in parts)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_dumps(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Payload skal kunne serialiseres som gyldig JSON.") from exc


def _optional_raw_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class DataStore:
    """SQLite connection and transaction-aware persistence helpers."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.initialize_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a group atomically; any exception rolls back every statement in it."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def initialize_schema(self) -> None:
        try:
            self.connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS matches (
                    match_id INTEGER PRIMARY KEY,
                    data_source TEXT NOT NULL,
                    external_match_id TEXT,
                    source_match_key TEXT NOT NULL,
                    league TEXT NOT NULL,
                    match_date TEXT NOT NULL,
                    kickoff_at TEXT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    match_status TEXT NOT NULL,
                    home_score INTEGER CHECK (home_score IS NULL OR home_score >= 0),
                    away_score INTEGER CHECK (away_score IS NULL OR away_score >= 0),
                    result TEXT CHECK (result IS NULL OR result IN ('H','D','A')),
                    season TEXT,
                    source_payload_json TEXT NOT NULL CHECK (json_valid(source_payload_json)),
                    imported_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    CHECK (home_team <> away_team),
                    UNIQUE (data_source, source_match_key)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_matches_external_id
                    ON matches(data_source, external_match_id)
                    WHERE external_match_id IS NOT NULL AND external_match_id <> '';
                CREATE INDEX IF NOT EXISTS ix_matches_date_league
                    ON matches(match_date, league);
                CREATE INDEX IF NOT EXISTS ix_matches_teams_date
                    ON matches(home_team, away_team, match_date);

                CREATE TABLE IF NOT EXISTS odds_observations (
                    odds_observation_id INTEGER PRIMARY KEY,
                    match_id INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE RESTRICT,
                    bookmaker TEXT NOT NULL,
                    market TEXT NOT NULL,
                    odds_home REAL NOT NULL CHECK (odds_home > 1),
                    odds_draw REAL NOT NULL CHECK (odds_draw > 1),
                    odds_away REAL NOT NULL CHECK (odds_away > 1),
                    observed_at_utc TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    fetched_at_utc TEXT,
                    observation_key TEXT NOT NULL UNIQUE,
                    created_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_odds_match_observed
                    ON odds_observations(match_id, observed_at_utc);
                CREATE INDEX IF NOT EXISTS ix_odds_bookmaker_market_time
                    ON odds_observations(bookmaker, market, observed_at_utc);

                CREATE TABLE IF NOT EXISTS sync_state (
                    sync_state_id INTEGER PRIMARY KEY,
                    data_source TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    last_success_at_utc TEXT NOT NULL,
                    cursor TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(details_json)),
                    UNIQUE (data_source, dataset)
                );
                CREATE TABLE IF NOT EXISTS fixture_match_audit (
                    audit_id INTEGER PRIMARY KEY,
                    sync_run_id TEXT NOT NULL,
                    audited_at_utc TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    external_fixture_id TEXT,
                    raw_league_json TEXT NOT NULL CHECK (json_valid(raw_league_json)),
                    raw_home_team_json TEXT NOT NULL CHECK (json_valid(raw_home_team_json)),
                    raw_away_team_json TEXT NOT NULL CHECK (json_valid(raw_away_team_json)),
                    raw_date TEXT,
                    raw_kickoff TEXT,
                    candidate_matches_json TEXT NOT NULL CHECK (json_valid(candidate_matches_json)),
                    selected_match_id INTEGER REFERENCES matches(match_id) ON DELETE RESTRICT,
                    outcome TEXT NOT NULL CHECK (outcome IN ('matched','unmatched','ambiguous','rejected')),
                    reason TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_fixture_match_audit_run
                    ON fixture_match_audit(sync_run_id, audit_id);
                CREATE INDEX IF NOT EXISTS ix_fixture_match_audit_fixture
                    ON fixture_match_audit(data_source, external_fixture_id);
                COMMIT;
                """
            )
        except Exception:
            if self.connection.in_transaction:
                self.connection.rollback()
            raise

    def add_fixture_match_audit(
        self,
        *,
        sync_run_id: str,
        audited_at: str | datetime,
        data_source: str,
        fixture: Mapping[str, Any],
        candidate_matches: list[Mapping[str, Any]],
        selected_match_id: int | None,
        outcome: str,
        reason: str,
    ) -> int:
        """Append one per-fixture cross-provider matching decision."""
        if outcome not in {"matched", "unmatched", "ambiguous", "rejected"}:
            raise ValueError("Ugyldigt udfald for matchningsaudit.")
        for label, value in (("sync_run_id", sync_run_id), ("data_source", data_source), ("reason", reason)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} skal være en ikke-tom tekstværdi.")
        timestamp = normalize_timestamp_utc(audited_at, field_name="audited_at")
        league = fixture.get("league")
        teams = fixture.get("teams")
        teams = teams if isinstance(teams, Mapping) else {}
        home = teams.get("home")
        away = teams.get("away")
        external_id = fixture.get("id")
        external_id = str(external_id) if external_id is not None else None
        raw_values = [league, home, away]
        raw_json = [json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    for value in raw_values]
        candidates_json = json.dumps(
            candidate_matches, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        cursor = self.connection.execute(
            """INSERT INTO fixture_match_audit (
                sync_run_id, audited_at_utc, data_source, external_fixture_id,
                raw_league_json, raw_home_team_json, raw_away_team_json,
                raw_date, raw_kickoff, candidate_matches_json, selected_match_id,
                outcome, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sync_run_id.strip(), timestamp, data_source.strip(), external_id,
             *raw_json, _optional_raw_text(fixture.get("date")),
             _optional_raw_text(fixture.get("kickoff_utc")), candidates_json,
             selected_match_id, outcome, reason.strip()),
        )
        return int(cursor.lastrowid)

    def find_match_id(self, data_source: str, source_match_key: str) -> int | None:
        row = self.connection.execute(
            "SELECT match_id FROM matches WHERE data_source = ? AND source_match_key = ?",
            (data_source, source_match_key),
        ).fetchone()
        return int(row["match_id"]) if row else None

    def get_match_by_key(self, data_source: str, source_match_key: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM matches WHERE data_source = ? AND source_match_key = ?",
            (data_source, source_match_key),
        ).fetchone()

    def upsert_match(self, match: Mapping[str, Any]) -> tuple[int, str]:
        """Insert/update mutable match facts while leaving odds observations untouched."""
        data_source = str(match["data_source"])
        source_match_key = str(match["source_match_key"])
        raw_json = _json_dumps(match["source_payload"])
        previous = self.get_match_by_key(data_source, source_match_key)
        now = utc_now()
        imported_at = previous["imported_at_utc"] if previous else now
        values = (
            data_source,
            match.get("external_match_id"),
            source_match_key,
            str(match["league"]),
            str(match["match_date"]),
            match.get("kickoff_at"),
            str(match["home_team"]),
            str(match["away_team"]),
            str(match["match_status"]),
            match.get("home_score"),
            match.get("away_score"),
            match.get("result"),
            match.get("season"),
            raw_json,
            imported_at,
            now,
        )
        self.connection.execute(
            """INSERT INTO matches (
                data_source, external_match_id, source_match_key, league, match_date,
                kickoff_at, home_team, away_team, match_status, home_score, away_score,
                result, season, source_payload_json, imported_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(data_source, source_match_key) DO UPDATE SET
                external_match_id = excluded.external_match_id,
                league = excluded.league,
                match_date = excluded.match_date,
                kickoff_at = excluded.kickoff_at,
                home_team = excluded.home_team,
                away_team = excluded.away_team,
                match_status = excluded.match_status,
                home_score = excluded.home_score,
                away_score = excluded.away_score,
                result = excluded.result,
                season = excluded.season,
                source_payload_json = excluded.source_payload_json,
                updated_at_utc = excluded.updated_at_utc""",
            values,
        )
        row = self.get_match_by_key(data_source, source_match_key)
        assert row is not None
        status = "inserted" if previous is None else (
            "unchanged" if previous["source_payload_json"] == raw_json else "updated"
        )
        return int(row["match_id"]), status

    def add_odds_observation(
        self,
        *,
        match_id: int,
        bookmaker: str,
        market: str,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        observed_at: str | datetime,
        data_source: str,
        fetched_at: str | datetime | None = None,
    ) -> bool:
        odds = tuple(float(value) for value in (odds_home, odds_draw, odds_away))
        if not all(value > 1 and value < float("inf") for value in odds):
            raise ValueError("Decimalodds for H/D/A skal være endelige tal større end 1.")
        for label, value in (("bookmaker", bookmaker), ("market", market), ("data_source", data_source)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} skal være en ikke-tom tekstværdi.")
        observed_utc = normalize_timestamp_utc(observed_at, field_name="observed_at")
        fetched_utc = normalize_timestamp_utc(fetched_at, field_name="fetched_at") if fetched_at is not None else None
        identity = {
            "match_id": int(match_id), "bookmaker": bookmaker.strip(), "market": market.strip(),
            "odds": odds, "observed_at_utc": observed_utc,
            "data_source": data_source.strip(), "fetched_at_utc": fetched_utc,
        }
        observation_key = hashlib.sha256(_json_dumps(identity).encode("utf-8")).hexdigest()
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO odds_observations (
                match_id, bookmaker, market, odds_home, odds_draw, odds_away,
                observed_at_utc, data_source, fetched_at_utc, observation_key, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (int(match_id), bookmaker.strip(), market.strip(), *odds, observed_utc,
             data_source.strip(), fetched_utc, observation_key, utc_now()),
        )
        return cursor.rowcount == 1

    def set_sync_state(
        self,
        data_source: str,
        dataset: str,
        *,
        last_success_at: str | datetime | None = None,
        cursor: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        success = normalize_timestamp_utc(last_success_at, field_name="last_success_at") if last_success_at else utc_now()
        details_json = _json_dumps(details or {})
        self.connection.execute(
            """INSERT INTO sync_state(data_source, dataset, last_success_at_utc, cursor, details_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(data_source, dataset) DO UPDATE SET
                last_success_at_utc = excluded.last_success_at_utc,
                cursor = excluded.cursor,
                details_json = excluded.details_json""",
            (data_source, dataset, success, cursor, details_json),
        )

    def iter_match_payloads(self, data_source: str | None = None):
        if data_source is None:
            rows = self.connection.execute(
                "SELECT source_payload_json FROM matches ORDER BY match_date, league, home_team, away_team"
            )
        else:
            rows = self.connection.execute(
                "SELECT source_payload_json FROM matches WHERE data_source = ? ORDER BY match_date, league, home_team, away_team",
                (data_source,),
            )
        for row in rows:
            # JSON is data-only; no Python object deserialization is used.
            yield json.loads(row["source_payload_json"])

