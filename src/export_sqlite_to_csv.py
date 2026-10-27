"""Export source match payloads to a new CSV under data/exports by default."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Iterable

try:
    from .data_store import DEFAULT_DATABASE_PATH, DataStore, PROJECT_ROOT
except ImportError:  # pragma: no cover - direct script execution
    from data_store import DEFAULT_DATABASE_PATH, DataStore, PROJECT_ROOT


DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "exports"


def export_matches_csv(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    output_path: str | Path | None = None,
    *,
    data_source: str | None = None,
) -> Path:
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_EXPORT_DIR / f"fixtures_export_{stamp}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DataStore(database_path) as store:
        payloads = list(store.iter_match_payloads(data_source=data_source))
    if not payloads:
        raise ValueError("Der er ingen kampe at eksportere.")

    fieldnames: list[str] = []
    seen = set()
    for payload in payloads:
        for column in payload:
            if column not in seen:
                seen.add(column)
                fieldnames.append(column)

    # Exclusive creation means a caller can never accidentally overwrite the
    # source fixtures.csv or an earlier export.
    try:
        with output_path.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(payloads)
    except FileExistsError:
        raise FileExistsError(f"Eksportfil findes allerede; intet overskrevet: {output_path}") from None
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Eksportér kampe fra SQLite til en ny CSV.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source", help="Filtrér til én datakilde.")
    args = parser.parse_args()
    try:
        output = export_matches_csv(args.database, args.output, data_source=args.source)
    except (OSError, ValueError) as exc:
        print(f"Eksport fejlede: {exc}", file=sys.stderr)
        return 1
    print(f"CSV eksport gemt: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
