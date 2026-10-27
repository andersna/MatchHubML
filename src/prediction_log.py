"""Log every pre-kickoff prediction and score it once the result is known.

Predictions live in their own SQLite file (data/prediction_log.sqlite), so the
match cache stays read-only for the API. One row is kept per model artifact,
match and odds snapshot (the first time it was predicted); results are read
from the match cache, which `/api/sync` refreshes for the last 14 days.

source = 'live'     predicted by the API before kickoff
source = 'backfill' reconstructed afterwards for matches that finished before
                    logging existed: features as of the match date, odds stored
                    before kickoff, models trained on data before the match.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterable
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "prediction_log.sqlite"
OUTCOMES = ("H", "D", "A")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    logged_at_utc TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('live', 'backfill')),
    model_version TEXT NOT NULL,
    model_artifact TEXT NOT NULL,
    league TEXT NOT NULL,
    match_date TEXT NOT NULL,
    kickoff_at TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    odds_observed_at_utc TEXT NOT NULL,
    odds_market TEXT NOT NULL DEFAULT '1X2:opening',
    odds_prob_home REAL NOT NULL,
    odds_prob_draw REAL NOT NULL,
    odds_prob_away REAL NOT NULL,
    prob_home REAL NOT NULL,
    prob_draw REAL NOT NULL,
    prob_away REAL NOT NULL,
    UNIQUE (model_artifact, league, match_date, home_team, away_team, odds_observed_at_utc, odds_market)
);
CREATE INDEX IF NOT EXISTS ix_predictions_match
    ON predictions(league, match_date, home_team, away_team);
"""


def _migrate_add_odds_market(connection: sqlite3.Connection) -> None:
    """Logs from before odds_market existed: every live row used the opening price.

    The column is part of the unique key (one lookup stores the opening and the
    lookup price with the same timestamp), so the table is rebuilt.
    """
    columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
    if not columns or "odds_market" in columns:
        return
    connection.executescript(
        """
        BEGIN;
        ALTER TABLE predictions RENAME TO predictions_before_odds_market;
        DROP INDEX IF EXISTS ix_predictions_match;
        """ + SCHEMA + """
        INSERT INTO predictions (
            prediction_id, logged_at_utc, source, model_version, model_artifact, league,
            match_date, kickoff_at, home_team, away_team, odds_observed_at_utc, odds_market,
            odds_prob_home, odds_prob_draw, odds_prob_away, prob_home, prob_draw, prob_away
        )
        SELECT prediction_id, logged_at_utc, source, model_version, model_artifact, league,
            match_date, kickoff_at, home_team, away_team, odds_observed_at_utc,
            CASE WHEN odds_observed_at_utc LIKE 'football-data.co.uk:%'
                 THEN odds_observed_at_utc ELSE '1X2:opening' END,
            odds_prob_home, odds_prob_draw, odds_prob_away, prob_home, prob_draw, prob_away
        FROM predictions_before_odds_market;
        DROP TABLE predictions_before_odds_market;
        COMMIT;
        """
    )


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    _migrate_add_odds_market(connection)
    connection.executescript(SCHEMA)
    return connection


def log_predictions(records: Iterable[dict[str, Any]], *, path: Path = LOG_PATH) -> int:
    """Insert predictions; repeats of the same model/match/odds snapshot are ignored.

    Each record needs: source, model_version, model_artifact, league,
    match_date, kickoff_at, home_team, away_team, odds_observed_at_utc,
    odds_market, odds (H, D, A) and probs (H, D, A). Returns the number of new rows.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = [
        (
            record.get("logged_at_utc", now), record["source"], record["model_version"],
            record["model_artifact"], record["league"], record["match_date"],
            record.get("kickoff_at"), record["home_team"], record["away_team"],
            record["odds_observed_at_utc"], record["odds_market"],
            *map(float, record["odds"]), *map(float, record["probs"]),
        )
        for record in records
    ]
    if not rows:
        return 0
    with closing(_connect(path)) as connection, connection:
        before = connection.total_changes
        connection.executemany(
            """INSERT OR IGNORE INTO predictions (
                logged_at_utc, source, model_version, model_artifact, league, match_date,
                kickoff_at, home_team, away_team, odds_observed_at_utc, odds_market,
                odds_prob_home, odds_prob_draw, odds_prob_away,
                prob_home, prob_draw, prob_away
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        return connection.total_changes - before


def logged_matches(*, path: Path = LOG_PATH) -> set[tuple[str, str, str, str, str]]:
    """(model_version, league, match_date, home_team, away_team) already in the log."""
    if not Path(path).is_file():
        return set()
    with closing(_connect(path)) as connection:
        return {tuple(row) for row in connection.execute(
            "SELECT DISTINCT model_version, league, match_date, home_team, away_team FROM predictions"
        )}


def _finished_results(cache_path: Path) -> dict[tuple, dict[str, Any]]:
    uri = "file:" + quote(Path(cache_path).resolve().as_posix(), safe="/:\\") + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT league, match_date, home_team, away_team, home_score, away_score, result
            FROM matches
            WHERE data_source = 'football-data.org' AND upper(match_status) = 'FINISHED'
              AND result IN ('H', 'D', 'A')"""
        ).fetchall()
    return {
        (row["league"], row["match_date"], row["home_team"], row["away_team"]): dict(row)
        for row in rows
    }


def _scores(probs: list[float], result: str) -> tuple[float, float, bool]:
    index = OUTCOMES.index(result)
    log_loss = -math.log(max(probs[index], 1e-15))
    brier = sum((p - (i == index)) ** 2 for i, p in enumerate(probs))
    return log_loss, brier, max(range(3), key=probs.__getitem__) == index


def _summary(label: str, entries: list[dict[str, Any]], key: str) -> dict[str, Any]:
    count = len(entries)
    if not count:
        return {"model": label, "matches": 0}
    log_losses = [entry[key]["log_loss"] for entry in entries]
    deltas = [entry[key]["log_loss"] - entry["odds"]["log_loss"] for entry in entries]
    mean_delta = sum(deltas) / count
    se = (
        math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / (count - 1) / count)
        if count > 1 else None
    )
    return {
        "model": label,
        "matches": count,
        "accuracy": sum(entry[key]["hit"] for entry in entries) / count,
        "log_loss": sum(log_losses) / count,
        "brier_score": sum(entry[key]["brier"] for entry in entries) / count,
        "delta_log_loss_vs_odds": mean_delta,
        "delta_log_loss_vs_odds_se": se,
    }


def evaluate(
    cache_path: Path,
    *,
    path: Path = LOG_PATH,
    fallback_results: Callable[[list[tuple]], dict[tuple, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score logged predictions against finished matches.

    Results come from the match cache; ``fallback_results`` can supply results
    for logged matches the cache no longer covers (e.g. from fixtures.csv).
    """
    if not Path(path).is_file():
        return {"models": [], "odds": {"model": "odds", "matches": 0}, "matches": [], "pending": 0, "logged": 0}
    with closing(_connect(path)) as connection:
        predictions = [dict(row) for row in connection.execute(
            "SELECT * FROM predictions ORDER BY logged_at_utc, prediction_id"
        )]
    results = _finished_results(cache_path)
    if fallback_results is not None:
        missing = sorted({
            (row["league"], row["match_date"], row["home_team"], row["away_team"]) for row in predictions
        } - set(results))
        if missing:
            results.update(fallback_results(missing))

    # Latest prediction per model and match (later rows overwrite earlier ones).
    by_match: dict[tuple, dict[str, Any]] = {}
    pending = set()
    for row in predictions:
        key = (row["league"], row["match_date"], row["home_team"], row["away_team"])
        if key not in results:
            pending.add(key)
            continue
        entry = by_match.setdefault(key, {"models": {}})
        entry["models"][row["model_version"]] = row

    scored = []
    for key, entry in by_match.items():
        result = results[key]
        first = next(iter(entry["models"].values()))
        odds = [first["odds_prob_home"], first["odds_prob_draw"], first["odds_prob_away"]]
        odds_ll, odds_brier, odds_hit = _scores(odds, result["result"])
        item = {
            "league": key[0], "match_date": key[1], "home_team": key[2], "away_team": key[3],
            "kickoff_at": first["kickoff_at"],
            "home_score": result["home_score"], "away_score": result["away_score"],
            "result": result["result"],
            "odds": {"probs": odds, "log_loss": odds_ll, "brier": odds_brier, "hit": odds_hit},
            "sources": sorted({row["source"] for row in entry["models"].values()}),
        }
        for version, row in entry["models"].items():
            probs = [row["prob_home"], row["prob_draw"], row["prob_away"]]
            log_loss, brier, hit = _scores(probs, result["result"])
            item[version] = {
                "probs": probs, "log_loss": log_loss, "brier": brier, "hit": hit,
                "source": row["source"], "logged_at_utc": row["logged_at_utc"],
            }
        scored.append(item)
    scored.sort(key=lambda item: (item["kickoff_at"] or item["match_date"]))

    versions = sorted({version for item in scored for version in item if version.startswith("v")})
    models = []
    for version in versions:
        entries = [item for item in scored if version in item]
        summary = _summary(version, entries, version)
        # Cumulative log-loss difference vs. odds, in kickoff order, for charts.
        running, series = 0.0, []
        for entry in entries:
            running += entry[version]["log_loss"] - entry["odds"]["log_loss"]
            series.append(round(running, 6))
        summary["cumulative_delta_log_loss"] = series
        models.append(summary)
    odds_summary = _summary("odds", scored, "odds")
    odds_summary["delta_log_loss_vs_odds"] = 0.0
    return {
        "models": models,
        "odds": odds_summary,
        "matches": scored,
        "pending": len(pending),
        "logged": len(predictions),
    }
