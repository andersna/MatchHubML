"""Weekly, fail-safe merge of Football-Data.co.uk season data into fixtures.csv.

This module never rebuilds model features or odds datasets. It is intentionally
not called from src.main.py; a real application bootstrap can call
``ensure_historical_data_current`` when that entry point exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

try:
    from .collect_excel_data import COLUMNS, HISTORICAL_DIR, LEAGUES, get_season, load_league_sheet
except ImportError:  # pragma: no cover - direct script execution
    from collect_excel_data import COLUMNS, HISTORICAL_DIR, LEAGUES, get_season, load_league_sheet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
STATE_PATH = PROJECT_ROOT / "data" / "historical_update_state.json"
SOURCE_URL_TEMPLATE = "https://football-data.co.uk/mmz4281/{code}/all-euro-data-{start}-{end}.xlsx"
REFRESH_INTERVAL = timedelta(days=7)
REQUEST_TIMEOUT_SECONDS = 30
KEY_COLUMNS = ("season", "league", "date", "home_team", "away_team")
REQUIRED_SOURCE_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
ODDS_SOURCE_COLUMNS = ("B365H", "B365D", "B365A")


def season_source_url(when: datetime | date | None = None) -> str:
    """Workbook URL for the season running at ``when`` (seasons start in July)."""
    when = when or datetime.now(timezone.utc)
    start = when.year if when.month >= 7 else when.year - 1
    return SOURCE_URL_TEMPLATE.format(
        code=f"{start % 100:02d}{(start + 1) % 100:02d}", start=start, end=start + 1
    )


SOURCE_URL = season_source_url()


class HistoricalUpdateError(RuntimeError):
    """Download, source-validation, or merge failure that leaves data untouched."""


@dataclass
class UpdateReport:
    attempted: bool = False
    skipped_by_interval: bool = False
    new_matches: int = 0
    updated_matches: int = 0
    unchanged_matches: int = 0
    duplicate_source_rows: int = 0
    missing_results: int = 0
    missing_odds: int = 0
    error: str | None = None
    updated_at_utc: str | None = None


def _utc_now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now skal være et tidsstempel med tidszone.")
    return current.astimezone(timezone.utc)


def _read_last_success(path: Path) -> datetime | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("last_success_at_utc")
        if not isinstance(raw, str):
            return None
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # Missing or unreadable state causes a safe refresh attempt.
        return None


def _fixture_key(row: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for column in KEY_COLUMNS:
        value = row.get(column)
        if pd.isna(value):
            raise HistoricalUpdateError(f"Kampnøglen mangler feltet {column}.")
        if column == "date":
            if isinstance(value, (date, datetime, pd.Timestamp)):
                parsed_date = pd.Timestamp(value).date()
            else:
                raw_date = str(value).strip()
                try:
                    parsed_date = date.fromisoformat(raw_date[:10])
                except ValueError:
                    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
                    if pd.isna(parsed):
                        raise HistoricalUpdateError("Kampnøglen indeholder en ugyldig dato.")
                    parsed_date = parsed.date()
            text = parsed_date.isoformat()
        else:
            text = " ".join(str(value).strip().casefold().split())
        if not text:
            raise HistoricalUpdateError(f"Kampnøglen har et tomt felt: {column}.")
        values.append(text)
    return tuple(values)


def _source_season(url: str) -> tuple[str, str]:
    filename = Path(urlparse(url).path).name
    if not filename:
        raise HistoricalUpdateError("Kilde-URL'en mangler et Excel-filnavn.")
    try:
        season = get_season(Path(filename))
    except ValueError as exc:
        raise HistoricalUpdateError(str(exc)) from exc
    return filename, season


def _load_and_validate_workbook(content: bytes, *, season: str) -> tuple[pd.DataFrame, int, int]:
    if not content:
        raise HistoricalUpdateError("Excel-downloaden er tom.")
    try:
        workbook = pd.ExcelFile(BytesIO(content))
    except Exception as exc:
        raise HistoricalUpdateError(f"Downloaden er ikke en læsbar Excel-fil: {exc}") from exc

    missing_sheets = sorted(set(LEAGUES) - set(workbook.sheet_names))
    if missing_sheets:
        raise HistoricalUpdateError(
            "Excel-filen mangler forventede ligeark: " + ", ".join(missing_sheets)
        )

    frames: list[pd.DataFrame] = []
    duplicate_count = 0
    missing_odds_count = 0
    for sheet, league_name in LEAGUES.items():
        raw = pd.read_excel(workbook, sheet_name=sheet)
        missing_columns = [name for name in REQUIRED_SOURCE_COLUMNS if name not in raw.columns]
        if missing_columns:
            raise HistoricalUpdateError(
                f"Arket {sheet} mangler nødvendige kolonner: {', '.join(missing_columns)}"
            )
        parsed = load_league_sheet(workbook, sheet, league_name, season)
        if parsed.empty:
            raise HistoricalUpdateError(f"Arket {sheet} indeholder ingen brugbare kampe.")

        # The existing feature importer deliberately strips bookmaker columns.
        # Count missing Bet365 1X2 data for the report, but never merge odds here.
        if all(column in raw.columns for column in ODDS_SOURCE_COLUMNS):
            odds = raw.loc[:, list(ODDS_SOURCE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
            valid_odds = odds.gt(1.0).all(axis=1)
            raw_dates = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
            missing_by_key: dict[tuple[str, ...], bool] = {}
            for index, raw_row in raw.iterrows():
                if pd.isna(raw_dates.loc[index]):
                    continue
                home, away = raw_row.get("HomeTeam"), raw_row.get("AwayTeam")
                if pd.isna(home) or pd.isna(away) or not str(home).strip() or not str(away).strip():
                    continue
                key = _fixture_key({
                    "season": season, "league": league_name,
                    "date": raw_dates.loc[index].strftime("%Y-%m-%d"),
                    "home_team": home, "away_team": away,
                })
                missing_by_key[key] = missing_by_key.get(key, True) and not bool(valid_odds.loc[index])
            parsed_keys = {_fixture_key(row) for row in parsed.to_dict("records")}
            missing_odds_count += sum(missing_by_key.get(key, True) for key in parsed_keys)
        else:
            missing_odds_count += len(parsed)

        frames.append(parsed)

    incoming = pd.concat(frames, ignore_index=True)
    # Collapse repeated source rows by the documented season/league/date/teams
    # identity. For non-key fields, the last non-empty source value wins.
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in incoming.to_dict("records"):
        key = _fixture_key(row)
        if key not in records:
            records[key] = row
            continue
        duplicate_count += 1
        current = records[key]
        for column, value in row.items():
            if column not in KEY_COLUMNS and not pd.isna(value):
                current[column] = value
    deduped = pd.DataFrame(records.values()).reset_index(drop=True)
    if deduped.empty:
        raise HistoricalUpdateError("Excel-filen indeholder ingen unikke kampdata.")
    if not deduped["result"].dropna().isin(["H", "D", "A"]).all():
        raise HistoricalUpdateError("Excel-filen indeholder ugyldige H/D/A-resultater.")
    return deduped, duplicate_count, missing_odds_count


def _merge_fixtures(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    required_existing = set(KEY_COLUMNS)
    if not required_existing.issubset(existing.columns):
        raise HistoricalUpdateError(
            "fixtures.csv mangler kampnøglekolonner: "
            + ", ".join(sorted(required_existing - set(existing.columns)))
        )
    existing_records = existing.to_dict("records")
    index_by_key: dict[tuple[str, ...], int] = {}
    for index, row in enumerate(existing_records):
        key = _fixture_key(row)
        if key in index_by_key:
            raise HistoricalUpdateError(
                "fixtures.csv har allerede dubletter på sæson/liga/dato/hold; "
                "opdateringen afvises uden at ændre filen."
            )
        index_by_key[key] = index

    key_set = set(KEY_COLUMNS)
    source_columns = set(COLUMNS.values()) | {"league", "season"}
    update_columns = [
        column for column in incoming.columns
        if column in source_columns and column not in key_set
    ]
    new_count = updated_count = unchanged_count = 0
    for source_row in incoming.to_dict("records"):
        key = _fixture_key(source_row)
        if key not in index_by_key:
            index_by_key[key] = len(existing_records)
            existing_records.append(source_row)
            new_count += 1
            continue
        target = existing_records[index_by_key[key]]
        changed = False
        for column in update_columns:
            new_value = source_row.get(column)
            if pd.isna(new_value):
                continue  # Empty source cells never erase previously known data.
            old_value = target.get(column)
            old_missing = old_value is None or pd.isna(old_value)
            equal = False
            if not old_missing:
                try:
                    equal = float(old_value) == float(new_value)
                except (TypeError, ValueError):
                    equal = str(old_value) == str(new_value)
            if old_missing or not equal:
                target[column] = new_value
                changed = True
        if changed:
            updated_count += 1
        else:
            unchanged_count += 1

    columns = list(existing.columns)
    for column in incoming.columns:
        if column not in columns:
            columns.append(column)
    merged = pd.DataFrame(existing_records).reindex(columns=columns)
    return merged, new_count, updated_count, unchanged_count


def _validate_merged(frame: pd.DataFrame, *, minimum_rows: int) -> None:
    if len(frame) < minimum_rows:
        raise HistoricalUpdateError("Det flettede datasæt mistede eksisterende rækker.")
    keys = [_fixture_key(row) for row in frame.to_dict("records")]
    if len(keys) != len(set(keys)):
        raise HistoricalUpdateError("Det flettede datasæt indeholder dublerede kampnøgler.")
    if not frame["result"].dropna().isin(["H", "D", "A"]).all():
        raise HistoricalUpdateError("Det flettede datasæt indeholder ugyldige kampresultater.")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def update_historical_data(
    *,
    fixtures_path: str | Path = FIXTURES_PATH,
    state_path: str | Path = STATE_PATH,
    url: str | None = None,
    now: datetime | None = None,
    force: bool = False,
    refresh_interval: timedelta = REFRESH_INTERVAL,
) -> UpdateReport:
    """Download the current workbook if due, validate, and atomically merge it.

    Failure is returned in ``UpdateReport.error``. Existing fixtures and state
    remain unchanged unless the complete validated update commits successfully.
    """
    current = _utc_now(now)
    url = url or season_source_url(current)
    fixtures_file, state_file = Path(fixtures_path), Path(state_path)
    report = UpdateReport()
    last_success = _read_last_success(state_file)
    if not force and last_success is not None and current >= last_success:
        if current - last_success < refresh_interval:
            report.skipped_by_interval = True
            report.updated_at_utc = last_success.isoformat(timespec="seconds").replace("+00:00", "Z")
            return report

    report.attempted = True
    temp_fixture: Path | None = None
    temp_state: Path | None = None
    backup_fixture: Path | None = None
    preserve_backup = False
    original_fixture_existed = fixtures_file.exists()
    try:
        filename, season = _source_season(url)
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        downloaded = bytes(response.content)
        incoming, duplicates, missing_odds = _load_and_validate_workbook(
            downloaded, season=season
        )
        if not fixtures_file.is_file():
            raise HistoricalUpdateError(f"Eksisterende fixtures-fil findes ikke: {fixtures_file}")
        existing = pd.read_csv(fixtures_file, low_memory=False)
        merged, new_count, updated_count, unchanged_count = _merge_fixtures(existing, incoming)
        _validate_merged(merged, minimum_rows=len(existing))

        missing_results = int(incoming["result"].isna().sum())
        stamp = current.isoformat(timespec="seconds").replace("+00:00", "Z")
        checksum = sha256(downloaded).hexdigest()
        state_payload = {
            "last_success_at_utc": stamp,
            "source_url": url,
            "season": season,
            "sha256": checksum,
            "new_matches": new_count,
            "updated_matches": updated_count,
        }

        fixtures_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=fixtures_file.parent,
            prefix=f".{fixtures_file.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_fixture = Path(handle.name)
            merged.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        staged = pd.read_csv(temp_fixture, low_memory=False)
        _validate_merged(staged, minimum_rows=len(existing))
        if not set(existing.columns).issubset(staged.columns):
            raise HistoricalUpdateError("Den midlertidige CSV mistede eksisterende kolonner.")

        temp_state = _atomic_write_json(state_file, state_payload)
        if original_fixture_existed:
            with tempfile.NamedTemporaryFile(
                dir=fixtures_file.parent, prefix=f".{fixtures_file.name}.backup.",
                suffix=".tmp", delete=False,
            ) as backup_handle:
                backup_fixture = Path(backup_handle.name)
            shutil.copy2(fixtures_file, backup_fixture)

        os.replace(temp_fixture, fixtures_file)
        temp_fixture = None
        try:
            os.replace(temp_state, state_file)
            temp_state = None
        except Exception:
            # Roll back the data file if the state commit fails, keeping the
            # two-file update all-or-nothing for ordinary filesystem errors.
            if backup_fixture is not None:
                try:
                    os.replace(backup_fixture, fixtures_file)
                    backup_fixture = None
                except Exception:
                    preserve_backup = True
                    raise HistoricalUpdateError(
                        "Statusfilen kunne ikke gemmes, og fixtures.csv kunne ikke rulles tilbage. "
                        f"Gendannelseskopi er bevaret: {backup_fixture}"
                    )
            elif not original_fixture_existed:
                fixtures_file.unlink(missing_ok=True)
            raise

        # Keep the validated workbook (it carries the Bet365 odds a retrain
        # needs) next to the history as data/raw/<filename>. Best effort: the
        # history update above is already committed.
        workbook_path = fixtures_file.parent / "raw" / filename
        try:
            workbook_path.parent.mkdir(parents=True, exist_ok=True)
            staged_workbook = workbook_path.with_name(f".{filename}.tmp")
            staged_workbook.write_bytes(downloaded)
            os.replace(staged_workbook, workbook_path)
        except OSError as exc:
            print(f"Advarsel: workbook kunne ikke gemmes i {workbook_path}: {exc}")

        report.new_matches = new_count
        report.updated_matches = updated_count
        report.unchanged_matches = unchanged_count
        report.duplicate_source_rows = duplicates
        report.missing_results = missing_results
        report.missing_odds = missing_odds
        report.updated_at_utc = stamp
        return report
    except Exception as exc:
        report.error = str(exc) or exc.__class__.__name__
        return report
    finally:
        if temp_fixture is not None:
            temp_fixture.unlink(missing_ok=True)
        if temp_state is not None:
            temp_state.unlink(missing_ok=True)
        if backup_fixture is not None and not preserve_backup:
            backup_fixture.unlink(missing_ok=True)


def ensure_historical_data_current(**kwargs: Any) -> UpdateReport:
    """Application-bootstrap friendly wrapper; callers may continue on error."""
    return update_historical_data(**kwargs)


def print_report(report: UpdateReport) -> None:
    if report.skipped_by_interval:
        print(f"Historiske data er opdaterede; seneste succes: {report.updated_at_utc}.")
        return
    print("Historisk dataopdatering" if report.error is None else "Historisk dataopdatering fejlede")
    print(f"Download forsøgt: {'ja' if report.attempted else 'nej'}")
    print(
        f"Nye kampe={report.new_matches}, opdaterede={report.updated_matches}, "
        f"uændrede={report.unchanged_matches}, kildedubletter={report.duplicate_source_rows}"
    )
    print(f"Manglende resultater i download={report.missing_results}, manglende B365 1X2-odds={report.missing_odds}")
    if report.updated_at_utc:
        print(f"Seneste succes (UTC): {report.updated_at_utc}")
    if report.error:
        print(f"Fejl: {report.error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opdatér Football-Data.co.uk historiske kampdata sikkert.")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--url", default=None, help="Standard: indeværende sæsons workbook.")
    parser.add_argument("--force", action="store_true", help="Spring syvdagesintervallet over.")
    args = parser.parse_args(argv)
    report = update_historical_data(
        fixtures_path=args.fixtures, state_path=args.state, url=args.url, force=args.force
    )
    print_report(report)
    return 1 if report.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
