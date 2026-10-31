"""Prediction league: players pick a result (1, X or 2) before kickoff, and
optionally the exact score, with MatchHub V3 as a player.

Scoring: 1 point for the right 1X2, plus 2 bonus points when the exact score
is also right (3 at most).

Accounts, leagues and picks live in their own SQLite file
(data/game.sqlite). Fixtures and results come from the match cache. The bot
picks the outcome V3 rated most likely in its latest live prediction logged
before kickoff (src/prediction_log.py), and as its score the most likely
scoreline with that outcome under Poisson goal counts: the teams' expected
goals (logged with the prediction) set the total, V3's 1X2 probabilities the
split (experiment_bot_score.py). It is never shown before kickoff.

A league's slate is every finished match that at least one of its members
picked; a member (or the bot) without a pick for a slate match gets 0.
Matches are identified by prediction_log.match_identity (league season and
teams), so a pick survives when a match is moved to another day, and the
match's current kickoff from the cache decides locking and the bot's entry.
The open league "Alle spillere" contains every player.
"""

from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Any, Callable, Iterable
from urllib.parse import quote

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

try:
    from .prediction_log import match_identity
except ImportError:  # run as a script from src/ (experiments)
    from prediction_log import match_identity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MATCHHUB_GAME_DB moves the game's database, e.g. to persistent storage online.
GAME_PATH = Path(os.environ.get("MATCHHUB_GAME_DB") or PROJECT_ROOT / "data" / "game.sqlite")
OUTCOMES = ("H", "D", "A")
LEAGUES = ("Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1")
EXCLUDED_STATUSES = ("POSTPONED", "CANCELED", "CANCELLED", "SUSPENDED")
BOT_NAME = "MatchHub V3"
BOT_MODEL_VERSION = "v3"
GLOBAL_LEAGUE = "alle"
SESSION_DAYS = 30
USERNAME_PATTERN = re.compile(r"^[\w.\-]{3,20}$")
MIN_PASSWORD_LENGTH = 6
INVITE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I
INVITE_LENGTH = 6
POINTS_PER_HIT = 1
POINTS_EXACT_SCORE = 2  # bonus on top of the hit
MAX_GOALS = 15
BOT_MAX_GOALS = 10  # size of the bot's Poisson score grid
# Goals in a match = a + b * (home + away expected_goals feature), fitted on 2022-23
# in experiment_bot_score.py; it beat V3-only rates on 2023-24 .. 2025-26.
TOTAL_INTERCEPT, TOTAL_SLOPE = 1.05, 0.60
TOTAL_RANGE = (1.0, 5.0)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    expires_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leagues (
    league_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    invite_code TEXT NOT NULL UNIQUE,
    owner_id INTEGER NOT NULL REFERENCES users(user_id),
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS league_members (
    league_id INTEGER NOT NULL REFERENCES leagues(league_id),
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    joined_at_utc TEXT NOT NULL,
    PRIMARY KEY (league_id, user_id)
);
CREATE TABLE IF NOT EXISTS entries (
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    league TEXT NOT NULL,
    match_date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff_at TEXT NOT NULL,
    pick TEXT NOT NULL CHECK (pick IN ('H', 'D', 'A')),
    home_goals INTEGER,
    away_goals INTEGER,
    submitted_at_utc TEXT NOT NULL,
    PRIMARY KEY (user_id, league, match_date, home_team, away_team)
);
"""

MatchKey = tuple[str, str, str, str]  # league, match_date, home_team, away_team
ResultsFn = Callable[[list[MatchKey]], dict[MatchKey, dict[str, Any]]]


class GameError(ValueError):
    """A request the game refuses; the message is shown to the player."""


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    _migrate_entries(connection)
    connection.executescript(SCHEMA)
    return connection


def _migrate_entries(connection: sqlite3.Connection) -> None:
    """Bring an entries table from an earlier version up to date.

    Slider version (probabilities): each entry becomes the outcome rated highest.
    Pick-only version: the score columns are added.
    """
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(entries)")}
    if "prob_home" in columns:
        connection.execute("ALTER TABLE entries RENAME TO entries_probabilities")
        connection.executescript(SCHEMA)
        with connection:
            connection.execute(
                """INSERT INTO entries (user_id, league, match_date, home_team, away_team, kickoff_at,
                    pick, submitted_at_utc)
                SELECT user_id, league, match_date, home_team, away_team, kickoff_at,
                    CASE WHEN prob_home >= prob_draw AND prob_home >= prob_away THEN 'H'
                         WHEN prob_draw >= prob_away THEN 'D' ELSE 'A' END,
                    submitted_at_utc
                FROM entries_probabilities"""
            )
            connection.execute("DROP TABLE entries_probabilities")
    elif columns and "home_goals" not in columns:
        with connection:
            connection.execute("ALTER TABLE entries ADD COLUMN home_goals INTEGER")
            connection.execute("ALTER TABLE entries ADD COLUMN away_goals INTEGER")


def _read_only(path: Path) -> sqlite3.Connection:
    uri = "file:" + quote(Path(path).resolve().as_posix(), safe="/:\\") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


# ---------- Accounts ----------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    _, salt, digest = stored.split("$")
    candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1)
    return hmac.compare_digest(candidate.hex(), digest)


def register(username: str, password: str, *, path: Path = GAME_PATH, now: datetime | None = None) -> str:
    """Create a user and return a session token."""
    username = username.strip()
    if not USERNAME_PATTERN.match(username):
        raise GameError("Brugernavnet skal være 3–20 tegn: bogstaver, tal, punktum, bindestreg eller _.")
    if username.casefold() == BOT_NAME.casefold():
        raise GameError("Det navn er reserveret til modellen.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise GameError(f"Adgangskoden skal være mindst {MIN_PASSWORD_LENGTH} tegn.")
    now = now or datetime.now(timezone.utc)
    with closing(_connect(path)) as connection, connection:
        try:
            cursor = connection.execute(
                "INSERT INTO users (username, password_hash, created_at_utc) VALUES (?, ?, ?)",
                (username, hash_password(password), _iso(now)),
            )
        except sqlite3.IntegrityError:
            raise GameError("Brugernavnet er taget.") from None
        return _new_session(connection, cursor.lastrowid, now)


def login(username: str, password: str, *, path: Path = GAME_PATH, now: datetime | None = None) -> str:
    with closing(_connect(path)) as connection, connection:
        row = connection.execute(
            "SELECT user_id, password_hash FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            raise GameError("Forkert brugernavn eller adgangskode.")
        return _new_session(connection, row["user_id"], now or datetime.now(timezone.utc))


def _new_session(connection: sqlite3.Connection, user_id: int, now: datetime) -> str:
    token = secrets.token_urlsafe(32)
    connection.execute("DELETE FROM sessions WHERE expires_at_utc < ?", (_iso(now),))
    connection.execute(
        "INSERT INTO sessions (token, user_id, expires_at_utc) VALUES (?, ?, ?)",
        (token, user_id, _iso(now + timedelta(days=SESSION_DAYS))),
    )
    return token


def logout(token: str, *, path: Path = GAME_PATH) -> None:
    with closing(_connect(path)) as connection, connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))


def user_for_token(token: str | None, *, path: Path = GAME_PATH, now: datetime | None = None) -> dict | None:
    if not token or not Path(path).is_file():
        return None
    with closing(_connect(path)) as connection:
        row = connection.execute(
            """SELECT u.user_id, u.username FROM sessions AS s JOIN users AS u USING (user_id)
            WHERE s.token = ? AND s.expires_at_utc > ?""",
            (token, _iso(now or datetime.now(timezone.utc))),
        ).fetchone()
    return dict(row) if row else None


# ---------- Leagues ----------

def create_league(user_id: int, name: str, *, path: Path = GAME_PATH, now: datetime | None = None) -> dict:
    name = " ".join(name.split())
    if not 2 <= len(name) <= 40:
        raise GameError("Liganavnet skal være 2–40 tegn.")
    now_text = _iso(now or datetime.now(timezone.utc))
    with closing(_connect(path)) as connection, connection:
        for _ in range(20):
            code = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(INVITE_LENGTH))
            try:
                cursor = connection.execute(
                    "INSERT INTO leagues (name, invite_code, owner_id, created_at_utc) VALUES (?, ?, ?, ?)",
                    (name, code, user_id, now_text),
                )
                break
            except sqlite3.IntegrityError:
                continue
        else:  # pragma: no cover - 32**6 codes
            raise GameError("Kunne ikke lave en invitationskode; prøv igen.")
        connection.execute(
            "INSERT INTO league_members (league_id, user_id, joined_at_utc) VALUES (?, ?, ?)",
            (cursor.lastrowid, user_id, now_text),
        )
    return {"name": name, "code": code}


def join_league(user_id: int, code: str, *, path: Path = GAME_PATH, now: datetime | None = None) -> dict:
    with closing(_connect(path)) as connection, connection:
        league = connection.execute(
            "SELECT league_id, name, invite_code FROM leagues WHERE invite_code = ?", (code.strip().upper(),)
        ).fetchone()
        if league is None:
            raise GameError("Ukendt invitationskode.")
        connection.execute(
            "INSERT OR IGNORE INTO league_members (league_id, user_id, joined_at_utc) VALUES (?, ?, ?)",
            (league["league_id"], user_id, _iso(now or datetime.now(timezone.utc))),
        )
    return {"name": league["name"], "code": league["invite_code"]}


def user_leagues(user_id: int, *, path: Path = GAME_PATH) -> list[dict]:
    with closing(_connect(path)) as connection:
        rows = connection.execute(
            """SELECT l.name, l.invite_code AS code, l.owner_id = ? AS owner,
                      (SELECT count(*) FROM league_members m2 WHERE m2.league_id = l.league_id) AS members
            FROM leagues AS l JOIN league_members AS m USING (league_id)
            WHERE m.user_id = ? ORDER BY l.name""",
            (user_id, user_id),
        ).fetchall()
    return [{**dict(row), "owner": bool(row["owner"])} for row in rows]


def _members(connection: sqlite3.Connection, code: str) -> tuple[str, list[dict]]:
    if code == GLOBAL_LEAGUE:
        rows = connection.execute("SELECT user_id, username FROM users ORDER BY username").fetchall()
        return "Alle spillere", [dict(row) for row in rows]
    league = connection.execute("SELECT league_id, name FROM leagues WHERE invite_code = ?", (code,)).fetchone()
    if league is None:
        raise GameError("Ukendt liga.")
    rows = connection.execute(
        """SELECT u.user_id, u.username FROM league_members AS m JOIN users AS u USING (user_id)
        WHERE m.league_id = ? ORDER BY u.username""",
        (league["league_id"],),
    ).fetchall()
    return league["name"], [dict(row) for row in rows]


# ---------- Weeks and fixtures ----------

def iso_week(moment: datetime) -> str:
    year, week, _ = moment.astimezone(timezone.utc).isocalendar()
    return f"{year}-W{week:02d}"


def week_bounds(week: str) -> tuple[datetime, datetime]:
    """UTC Monday 00:00 of ``week`` ("2026-W45") and of the following week."""
    match = re.fullmatch(r"(\d{4})-W(\d{2})", week)
    if not match:
        raise GameError("Ugen skal angives som ÅÅÅÅ-Uxx, fx 2026-W45.")
    try:
        monday = date.fromisocalendar(int(match[1]), int(match[2]), 1)
    except ValueError:
        raise GameError("Ugyldig uge.") from None
    start = datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)
    return start, start + timedelta(days=7)


def week_fixtures(cache_path: Path, week: str) -> list[dict]:
    start, end = week_bounds(week)
    with closing(_read_only(cache_path)) as connection:
        rows = connection.execute(
            f"""SELECT league, match_date, kickoff_at, home_team, away_team, match_status
            FROM matches
            WHERE data_source = 'football-data.org'
              AND league IN ({','.join('?' * len(LEAGUES))})
              AND kickoff_at >= ? AND kickoff_at < ?
              AND upper(match_status) NOT IN ({','.join('?' * len(EXCLUDED_STATUSES))})
            ORDER BY kickoff_at, league, home_team""",
            (*LEAGUES, _iso(start), _iso(end), *EXCLUDED_STATUSES),
        ).fetchall()
    return [dict(row) for row in rows]


# ---------- Picks ----------

def outcome_of(home_goals: int, away_goals: int) -> str:
    return "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"


def _score(entry: dict, pick: str) -> tuple[int | None, int | None]:
    """The optional exact score of an entry; it must agree with the pick."""
    home_goals, away_goals = entry.get("home_goals"), entry.get("away_goals")
    if home_goals is None and away_goals is None:
        return None, None
    if home_goals is None or away_goals is None:
        raise GameError("Angiv mål for begge hold – eller ingen.")
    try:
        home_goals, away_goals = int(home_goals), int(away_goals)
    except (TypeError, ValueError):
        raise GameError("Mål skal være hele tal.") from None
    if not (0 <= home_goals <= MAX_GOALS and 0 <= away_goals <= MAX_GOALS):
        raise GameError(f"Mål skal være mellem 0 og {MAX_GOALS}.")
    if outcome_of(home_goals, away_goals) != pick:
        raise GameError("Resultatet passer ikke med dit 1/X/2.")
    return home_goals, away_goals


def save_entries(
    user_id: int, entries: Iterable[dict], *, cache_path: Path, path: Path = GAME_PATH,
    now: datetime | None = None,
) -> dict:
    """Save picks for matches that have not kicked off; a later pick replaces the earlier one.

    Each entry: league, match_date, home_team, away_team, pick ("H", "D" or "A") and
    optionally home_goals and away_goals (the exact score, for a bonus).
    Returns {"saved": n, "rejected": [{match..., "reason"}]}.
    """
    now = now or datetime.now(timezone.utc)
    saved, rejected, rows = 0, [], []
    with closing(_read_only(cache_path)) as cache:
        for entry in entries:
            key = tuple(str(entry.get(name, "")) for name in ("league", "match_date", "home_team", "away_team"))
            label = {"league": key[0], "match_date": key[1], "home_team": key[2], "away_team": key[3]}
            fixture = cache.execute(
                f"""SELECT kickoff_at FROM matches WHERE data_source = 'football-data.org'
                AND league = ? AND match_date = ? AND home_team = ? AND away_team = ?
                AND upper(match_status) NOT IN ({','.join('?' * len(EXCLUDED_STATUSES))})""",
                (*key, *EXCLUDED_STATUSES),
            ).fetchone()
            try:
                if fixture is None or not fixture["kickoff_at"]:
                    raise GameError("Kampen findes ikke.")
                if _parse(fixture["kickoff_at"]) <= now:
                    raise GameError("Kampen er gået i gang – dit bud er låst.")
                pick = str(entry.get("pick", "")).upper()
                if pick not in OUTCOMES:
                    raise GameError("Vælg 1, X eller 2.")
                home_goals, away_goals = _score(entry, pick)
            except GameError as exc:
                rejected.append({**label, "reason": str(exc)})
                continue
            rows.append((user_id, *key, fixture["kickoff_at"], pick, home_goals, away_goals, _iso(now)))
    if rows:
        with closing(_connect(path)) as connection, connection:
            for user, league, match_date, home_team, away_team, *_ in rows:
                identity = match_identity(league, match_date, home_team, away_team)
                connection.executemany(
                    "DELETE FROM entries WHERE user_id = ? AND league = ? AND match_date = ? AND home_team = ? AND away_team = ?",
                    [(user, league, row["match_date"], home_team, away_team) for row in connection.execute(
                        """SELECT match_date FROM entries WHERE user_id = ? AND league = ? AND home_team = ?
                        AND away_team = ? AND match_date != ?""",
                        (user, league, home_team, away_team, match_date),
                    ) if match_identity(league, row["match_date"], home_team, away_team) == identity],
                )
            connection.executemany(
                """INSERT INTO entries (user_id, league, match_date, home_team, away_team, kickoff_at,
                    pick, home_goals, away_goals, submitted_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, league, match_date, home_team, away_team) DO UPDATE SET
                    kickoff_at = excluded.kickoff_at, pick = excluded.pick,
                    home_goals = excluded.home_goals, away_goals = excluded.away_goals,
                    submitted_at_utc = excluded.submitted_at_utc""",
                rows,
            )
        saved = len(rows)
    return {"saved": saved, "rejected": rejected}


def _entry(row: sqlite3.Row) -> dict:
    return {"pick": row["pick"], "home_goals": row["home_goals"], "away_goals": row["away_goals"]}


def match_total(home_expected_goals: float | None, away_expected_goals: float | None) -> float | None:
    """Expected goals in the match from the logged features, or None if they were not logged."""
    if home_expected_goals is None or away_expected_goals is None:
        return None
    total = TOTAL_INTERCEPT + TOTAL_SLOPE * (home_expected_goals + away_expected_goals)
    return round(min(max(total, TOTAL_RANGE[0]), TOTAL_RANGE[1]), 3)


BotEntry = tuple[tuple[float, float, float], float | None]  # 1X2 probabilities, expected total goals


def bot_entries(log_path: Path, kickoffs: dict[MatchKey, str]) -> dict[MatchKey, BotEntry]:
    """V3's latest live prediction logged before each match's actual kickoff (``kickoffs``:
    key -> current kickoff), found under any date the match had when it was logged."""
    wanted = {match_identity(*key): key for key in kickoffs}
    if not wanted or not Path(log_path).is_file():
        return {}
    found: dict[MatchKey, BotEntry] = {}
    with closing(_read_only(log_path)) as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(predictions)")}
        goals = ("home_expected_goals, away_expected_goals" if "home_expected_goals" in columns
                 else "NULL AS home_expected_goals, NULL AS away_expected_goals")
        rows = connection.execute(
            f"""SELECT league, match_date, home_team, away_team, logged_at_utc, prob_home, prob_draw, prob_away, {goals}
            FROM predictions
            WHERE model_version = ? AND source = 'live' AND kickoff_at IS NOT NULL AND logged_at_utc < kickoff_at
            ORDER BY logged_at_utc, prediction_id""",
            (BOT_MODEL_VERSION,),
        ).fetchall()
    for row in rows:  # later rows overwrite earlier ones
        key = wanted.get(match_identity(row["league"], row["match_date"], row["home_team"], row["away_team"]))
        if key is not None and row["logged_at_utc"] < kickoffs[key]:
            found[key] = ((row["prob_home"], row["prob_draw"], row["prob_away"]),
                          match_total(row["home_expected_goals"], row["away_expected_goals"]))
    return found


def _current_fixtures(cache_path: Path, identities: set) -> dict[tuple, tuple[MatchKey, str]]:
    """match_identity -> (current key, current kickoff) in the match cache."""
    if not identities:
        return {}
    found = {}
    with closing(_read_only(cache_path)) as connection:
        for row in connection.execute(
            f"""SELECT league, match_date, kickoff_at, home_team, away_team FROM matches
            WHERE data_source = 'football-data.org' AND kickoff_at IS NOT NULL
              AND upper(match_status) NOT IN ({','.join('?' * len(EXCLUDED_STATUSES))})""",
            EXCLUDED_STATUSES,
        ):
            key = (row["league"], row["match_date"], row["home_team"], row["away_team"])
            if match_identity(*key) in identities:
                found[match_identity(*key)] = (key, row["kickoff_at"])
    return found


def bot_pick(probs: Iterable[float]) -> str:
    """The outcome the model rates most likely (ties go to the earlier of H, D, A)."""
    probs = list(probs)
    return OUTCOMES[probs.index(max(probs))]


def _poisson_grid(home_rate: float, away_rate: float) -> np.ndarray:
    """P(home goals = row, away goals = column) for independent Poisson goal counts."""
    goals = np.arange(BOT_MAX_GOALS + 1)
    return np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))


def _outcome_probabilities(grid: np.ndarray) -> np.ndarray:
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


@lru_cache(maxsize=4096)
def goal_rates(probs: tuple[float, float, float], total: float | None = None) -> tuple[float, float]:
    """Poisson goal rates (home, away) that reproduce V3's 1X2 probabilities.

    Without ``total`` both rates are fitted. With ``total`` (expected goals in the match,
    from the teams' own scoring) the rates sum to it and only the split is fitted, so
    the 1X2 probabilities decide who scores more and the teams decide how many.
    """
    target = np.asarray(probs, dtype=float) / sum(probs)

    def error(home_rate: float, away_rate: float) -> float:
        return float(((_outcome_probabilities(_poisson_grid(home_rate, away_rate)) - target) ** 2).sum())

    if total is None:
        fit = minimize(lambda rates: error(*rates), x0=(1.5, 1.2),
                       bounds=((0.05, 6.0), (0.05, 6.0)), method="L-BFGS-B")
        return float(fit.x[0]), float(fit.x[1])
    share = minimize_scalar(lambda s: error(total * s, total * (1 - s)), bounds=(0.02, 0.98), method="bounded").x
    return total * share, total * (1 - share)


@lru_cache(maxsize=4096)
def bot_score(probs: tuple[float, float, float], total: float | None = None) -> tuple[int, int]:
    """The bot's exact score: the most likely score under goal_rates whose outcome is the
    bot's pick, so the score and the pick always agree."""
    grid = _poisson_grid(*goal_rates(probs, total))
    home, away = np.indices(grid.shape)
    allowed = {"H": home > away, "D": home == away, "A": home < away}[bot_pick(probs)]
    best = np.unravel_index(np.where(allowed, grid, -1.0).argmax(), grid.shape)
    return int(best[0]), int(best[1])


def bot_prediction(probs: tuple[float, float, float], total: float | None = None) -> dict:
    home_goals, away_goals = bot_score(tuple(probs), total)
    return {"pick": bot_pick(probs), "probability": max(probs), "home_goals": home_goals, "away_goals": away_goals}


def points(entry: dict, result: dict) -> int:
    """1 point for the right 1X2, plus a bonus when the exact score is also right."""
    if entry["pick"] != result["result"]:
        return 0
    exact = entry.get("home_goals") is not None and (
        (entry["home_goals"], entry["away_goals"]) == (result["home_score"], result["away_score"])
    )
    return POINTS_PER_HIT + (POINTS_EXACT_SCORE if exact else 0)


# ---------- Views ----------

def round_view(
    user_id: int, week: str, *, cache_path: Path, log_path: Path, results: ResultsFn,
    path: Path = GAME_PATH, now: datetime | None = None,
) -> dict:
    """The week's matches with the player's picks; the bot's pick only after kickoff."""
    now = now or datetime.now(timezone.utc)
    fixtures = week_fixtures(cache_path, week)
    keys = [(f["league"], f["match_date"], f["home_team"], f["away_team"]) for f in fixtures]
    with closing(_connect(path)) as connection:
        by_identity = {
            match_identity(row["league"], row["match_date"], row["home_team"], row["away_team"]): _entry(row)
            for row in connection.execute("SELECT * FROM entries WHERE user_id = ?", (user_id,))
        }
    mine = {key: by_identity[match_identity(*key)] for key in keys if match_identity(*key) in by_identity}
    kickoff_of = {key: f["kickoff_at"] for key, f in zip(keys, fixtures)}
    locked_keys = [key for key in keys if _parse(kickoff_of[key]) <= now]
    bot = {key: bot_prediction(*entry)
           for key, entry in bot_entries(log_path, {key: kickoff_of[key] for key in locked_keys}).items()}
    finished = results(locked_keys) if locked_keys else {}

    matches, my_points, bot_points, exact = [], 0, 0, 0
    locked_set = set(locked_keys)
    for key, fixture in zip(keys, fixtures):
        locked = key in locked_set
        item = {**fixture, "locked": locked, "my_entry": mine.get(key)}
        if locked:
            item["bot_entry"] = bot.get(key)
        result = finished.get(key)
        if result:
            item.update(result=result["result"], home_score=result["home_score"], away_score=result["away_score"])
            item["my_points"] = points(mine[key], result) if key in mine else None
            item["bot_points"] = points(bot[key], result) if key in bot else None
            if key in mine:
                my_points += item["my_points"]
                bot_points += item["bot_points"] or 0
                exact += item["my_points"] > POINTS_PER_HIT
        matches.append(item)
    start, _ = week_bounds(week)
    return {
        "week": week,
        "previous_week": iso_week(start - timedelta(days=7)),
        "next_week": iso_week(start + timedelta(days=7)),
        "current_week": iso_week(now),
        "matches": matches,
        "summary": {
            "entered": sum(1 for m in matches if m["my_entry"]),
            "open": sum(1 for m in matches if not m["locked"]),
            "scored": sum(1 for m in matches if m.get("my_points") is not None),
            "my_points": my_points,
            "exact": exact,
            "bot_points_same_matches": bot_points,
        },
    }


def league_table(
    code: str, *, cache_path: Path, log_path: Path, results: ResultsFn, week: str | None = None,
    path: Path = GAME_PATH, now: datetime | None = None,
) -> dict:
    """Standings over the league's slate (finished matches any member picked), all time
    and for ``week``; missing picks score 0. The bot is a row."""
    now = now or datetime.now(timezone.utc)
    with closing(_connect(path)) as connection:
        name, members = _members(connection, code)
        ids = [m["user_id"] for m in members]
        entry_rows = connection.execute(
            f"SELECT * FROM entries WHERE user_id IN ({','.join('?' * len(ids))})", ids,
        ).fetchall() if ids else []
    # Picks by match_identity; each match's key and kickoff come from the cache when it has
    # the match (it may have been moved since the pick), else from the pick.
    entries: dict[int, dict[tuple, dict]] = {uid: {} for uid in ids}
    fixtures: dict[tuple, tuple[MatchKey, str]] = {}
    for row in entry_rows:
        key = (row["league"], row["match_date"], row["home_team"], row["away_team"])
        entries[row["user_id"]][match_identity(*key)] = _entry(row)
        fixtures.setdefault(match_identity(*key), (key, row["kickoff_at"]))
    fixtures.update(_current_fixtures(cache_path, set(fixtures)))
    kickoffs = {fixtures[i][0]: fixtures[i][1] for i in fixtures if _parse(fixtures[i][1]) <= now}
    slate = sorted(kickoffs, key=kickoffs.get)
    finished = results(slate) if slate else {}
    slate = [key for key in slate if key in finished]
    entries = {uid: {key: picks[match_identity(*key)] for key in slate if match_identity(*key) in picks}
               for uid, picks in entries.items()}
    bot = {key: bot_prediction(*entry)
           for key, entry in bot_entries(log_path, {key: kickoffs[key] for key in slate}).items()}
    week_keys = set()
    if week:
        start, end = week_bounds(week)
        week_keys = {key for key in slate if start <= _parse(kickoffs[key]) < end}

    def row_for(label: str, picks: dict[MatchKey, dict], is_bot: bool, user_id: int | None) -> dict:
        scores = {key: points(picks[key], finished[key]) if key in picks else 0 for key in slate}
        predicted = [key for key in slate if key in picks]
        return {
            "player": label, "is_bot": is_bot, "user_id": user_id,
            "total_points": sum(scores.values()),
            "week_points": sum(scores[key] for key in week_keys),
            "predicted": len(predicted),
            "hit_rate": sum(1 for key in predicted if scores[key]) / len(predicted) if predicted else None,
            "exact": sum(1 for key in predicted if scores[key] > POINTS_PER_HIT),
        }

    rows = [row_for(m["username"], entries[m["user_id"]], False, m["user_id"]) for m in members]
    rows.append(row_for(BOT_NAME, bot, True, None))
    rows.sort(key=lambda r: (-r["total_points"], -r["week_points"], r["player"].casefold()))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {"league": name, "code": code, "week": week, "matches": len(slate),
            "week_matches": len(week_keys), "rows": rows}
