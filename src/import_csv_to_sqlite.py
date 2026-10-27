"""Import a fixtures-style CSV into MatchHubML's local SQLite store."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

try:
    from .data_store import DEFAULT_DATABASE_PATH, DataStore, deterministic_match_key
except ImportError:  # pragma: no cover - direct script execution
    from data_store import DEFAULT_DATABASE_PATH, DataStore, deterministic_match_key


REQUIRED_COLUMNS = ("date", "home_team", "away_team", "league")
OPTIONAL_NUMERIC_COLUMNS = {
    "home_goals", "away_goals", "home_goals_ht", "away_goals_ht",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards", "home_xg", "away_xg",
}


@dataclass
class ImportReport:
    source_file: str
    rows_read: int = 0
    imported: int = 0
    updated: int = 0
    duplicates: int = 0
    rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except (TypeError, ValueError):
        raise ValueError("ugyldig dato; forventede YYYY-MM-DD") from None


def _parse_time(value: str | None, match_date: date) -> tuple[str | None, str | None]:
    raw = (value or "").strip()
    if not raw:
        return None, None
    try:
        parsed = datetime.fromisoformat(f"{match_date.isoformat()}T{raw}".replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("ugyldigt klokkeslæt; forventede HH:MM eller HH:MM:SS") from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat(timespec="seconds").replace("+00:00", "Z"), parsed.strftime("%H:%M:%S")
    return parsed.isoformat(timespec="seconds"), parsed.strftime("%H:%M:%S")


def _optional_number(value: str | None, column: str) -> int | float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        raise ValueError(f"'{column}' skal være numerisk") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"'{column}' skal være et endeligt tal mindst 0")
    if column not in {"home_xg", "away_xg"}:
        if not number.is_integer():
            raise ValueError(f"'{column}' skal være et heltal")
        return int(number)
    return number


def _normalize_row(row: dict[str, str], data_source: str) -> dict[str, Any]:
    match_date = _parse_date(row.get("date", ""))
    league = (row.get("league") or "").strip()
    home_team = (row.get("home_team") or "").strip()
    away_team = (row.get("away_team") or "").strip()
    if not league or not home_team or not away_team:
        raise ValueError("liga og begge hold skal være udfyldt")
    if home_team.casefold() == away_team.casefold():
        raise ValueError("hjemmehold og udehold skal være forskellige")
    kickoff_at, normalized_time = _parse_time(row.get("time"), match_date)

    normalized_payload = dict(row)
    normalized_payload["date"] = match_date.isoformat()
    if "time" in normalized_payload:
        # Keep the source's time representation for round-trip export. The
        # normalized kickoff value is stored separately in matches.kickoff_at.
        normalized_payload["time"] = (row.get("time") or "").strip()

    numbers = {
        column: _optional_number(row.get(column), column)
        for column in OPTIONAL_NUMERIC_COLUMNS.intersection(row)
    }
    home_score, away_score = numbers.get("home_goals"), numbers.get("away_goals")
    if (home_score is None) != (away_score is None):
        raise ValueError("begge slutresultater skal være udfyldt eller begge tomme")
    result = (row.get("result") or "").strip().upper() or None
    if result is not None and result not in {"H", "D", "A"}:
        raise ValueError("resultat skal være H, D eller A")
    if home_score is not None:
        expected = "H" if home_score > away_score else "A" if home_score < away_score else "D"
        if result is not None and result != expected:
            raise ValueError("resultat stemmer ikke med slutresultatet")
        result = expected

    status = (row.get("status") or row.get("match_status") or "").strip().upper()
    if not status:
        status = "FINISHED" if home_score is not None or result is not None else "SCHEDULED"
    external_id = (row.get("external_match_id") or row.get("external_id") or "").strip() or None
    source_key = (
        f"external:{external_id}" if external_id else deterministic_match_key(
            data_source, match_date.isoformat(), league, home_team, away_team
        )
    )
    return {
        "data_source": data_source,
        "external_match_id": external_id,
        "source_match_key": source_key,
        "league": league,
        "match_date": match_date.isoformat(),
        "kickoff_at": kickoff_at,
        "home_team": home_team,
        "away_team": away_team,
        "match_status": status,
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
        "season": (row.get("season") or "").strip() or None,
        "source_payload": normalized_payload,
    }


def import_csv(
    csv_path: str | Path,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    data_source: str | None = None,
) -> ImportReport:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV-filen findes ikke: {csv_path}")
    source = data_source or f"csv:{csv_path.name}"
    report = ImportReport(source_file=str(csv_path))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV-filen mangler en header-række.")
        header = [column.strip() for column in reader.fieldnames]
        if len(header) != len(set(header)):
            raise ValueError("CSV-filen indeholder dublerede kolonnenavne.")
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise ValueError(f"CSV mangler obligatoriske kolonner: {missing}")
        reader.fieldnames = header

        accepted: list[dict[str, Any]] = []
        seen: dict[str, str] = {}
        for row_number, raw_row in enumerate(reader, start=2):
            report.rows_read += 1
            if None in raw_row:
                report.reject(f"række {row_number}: flere værdier end kolonner")
                continue
            try:
                match = _normalize_row(raw_row, source)
                key = match["source_match_key"]
                fingerprint = json.dumps(match["source_payload"], ensure_ascii=False, sort_keys=True)
                if key in seen:
                    if seen[key] == fingerprint:
                        report.duplicates += 1
                    else:
                        report.reject(f"række {row_number}: konflikt på kampnøgle i samme CSV")
                    continue
                seen[key] = fingerprint
                accepted.append(match)
            except ValueError as exc:
                report.reject(f"række {row_number}: {exc}")

    # SQLite exceptions roll back the whole accepted batch; rejected source
    # rows are counted separately and never leave half-written match records.
    with DataStore(database_path) as store:
        with store.transaction():
            for match in accepted:
                _, status = store.upsert_match(match)
                if status == "inserted":
                    report.imported += 1
                elif status == "updated":
                    report.updated += 1
                else:
                    report.duplicates += 1
            store.set_sync_state(
                source, "matches", details={"source_file": csv_path.name, "rows_read": report.rows_read}
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Importér fixtures-CSV til lokal SQLite.")
    parser.add_argument("csv_path", nargs="?", type=Path, default=Path("data/fixtures.csv"))
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--source", help="Datakildenavn (standard: csv:<filnavn>).")
    args = parser.parse_args()
    try:
        report = import_csv(args.csv_path, args.database, data_source=args.source)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Import fejlede: {exc}", file=sys.stderr)
        return 1
    print(f"Rækker læst: {report.rows_read}")
    print(f"Nye importerede: {report.imported}; opdaterede: {report.updated}; dubletter: {report.duplicates}; afviste: {report.rejected}")
    for reason, count in sorted(report.rejection_reasons.items()):
        print(f"  {count} × {reason}")
    print(f"Database: {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
