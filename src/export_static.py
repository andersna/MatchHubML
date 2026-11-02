"""Static export of the dashboard for GitHub Pages, and the state archive between runs.

    python -m src.export_static export site      # sync, then write site/index.html + site/data/*.json
    python -m src.export_static pack state.tar.gz
    python -m src.export_static unpack state.tar.gz

The scheduled GitHub Action unpacks the state from the previous run, exports (which
syncs fixtures, results, odds and the season file through the same code as the sync
button), packs the updated state and publishes site/. The frontend detects the static
build and reads data/<name>.json instead of calling the API; static_name() and
staticName() in the HTML must stay in sync.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile
from urllib.parse import parse_qs, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_PATH = PROJECT_ROOT / "matchhubml_frontend_live.html"

LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
MATCH_LIMIT = 200  # what the frontend asks for

# Everything a run reads or updates, relative to data/. Static inputs that are committed
# (feature_reference.json, the experiment summaries) come from the checkout instead.
# game.sqlite is deliberately absent: the game is not part of the static site.
STATE_FILES = [
    "fixtures.csv",
    "features_asof_day_v1.csv",
    "elo_features_asof_day_v1.csv",
    "understat_matches.csv",
    "matchhub_cache.sqlite",
    "prediction_log.sqlite",
    "historical_update_state.json",
    "last_sync.json",
    "last_history_refresh.json",
]
STATE_GLOBS = ["raw/all-euro-data-*.xlsx", "raw/understat/**/*"]


def slug(text: str) -> str:
    return text.strip().lower().replace(" ", "-")


def static_name(path: str) -> str:
    """'/api/matches?model_version=v3&league=La Liga' -> 'matches-v3-la-liga'."""
    parts = urlsplit(path)
    name = parts.path.removeprefix("/api/").strip("/").replace("/", "-")
    if name == "matches":
        query = parse_qs(parts.query)
        version = query.get("model_version", ["v3"])[0]
        league = query.get("league", [""])[0]
        name = f"matches-{version}-{slug(league) if league else 'all'}"
    return name


# ---------- state archive ----------

def state_paths(data_dir: Path = DATA_DIR) -> list[Path]:
    paths = [data_dir / name for name in STATE_FILES if (data_dir / name).is_file()]
    for pattern in STATE_GLOBS:
        paths += [p for p in sorted(data_dir.glob(pattern)) if p.is_file()]
    return paths


def pack(archive: Path, data_dir: Path = DATA_DIR) -> list[str]:
    names = []
    with tarfile.open(archive, "w:gz") as tar:
        for path in state_paths(data_dir):
            name = path.relative_to(data_dir).as_posix()
            tar.add(path, arcname=name)
            names.append(name)
    return names


def unpack(archive: Path, data_dir: Path = DATA_DIR) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # The "data" filter refuses absolute paths, ".." and links outside data_dir.
        tar.extractall(data_dir, filter="data")
        return [m.name for m in tar.getmembers() if m.isfile()]


# ---------- export ----------

def _encode(payload) -> str:
    from fastapi.encoders import jsonable_encoder

    # allow_nan=False: the same rule as FastAPI's JSON responses, so the files match the API.
    return json.dumps(jsonable_encoder(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _call(endpoint, *args, **kwargs):
    """Run an endpoint function; an HTTPException becomes the {"detail": ...} body the API would send."""
    from fastapi import HTTPException

    try:
        return endpoint(*args, **kwargs)
    except HTTPException as exc:
        print(f"[ADVARSEL] {endpoint.__name__}: {exc.detail}", flush=True)
        return {"detail": exc.detail, "status_code": exc.status_code}


def export(out_dir: Path, *, sync: bool = True, now: datetime | None = None) -> list[str]:
    from . import api

    generated = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")

    if sync:
        result = _call(api.sync_data)
        if "detail" in result:
            # Publish anyway: the previous data is still valid, and the Data tab shows the failure.
            print("[ADVARSEL] Synkroniseringen fejlede; siden udgives med de eksisterende data.", flush=True)

    files: dict[str, object] = {
        "health": _call(api.health),
        "status": _call(api.status),
        "odds-coverage": _call(api.get_odds_coverage),
        "sync-last": _call(api.last_sync),
        "history": _call(api.history_status),
        "models": _call(api.models),
    }
    versions = sorted(v for v, ok in files["health"].get("available_models", {}).items() if ok)
    for version in versions:
        for league in [None, *LEAGUES]:
            query = f"model_version={version}" + (f"&league={league}" if league else "")
            files[static_name(f"/api/matches?{query}")] = _call(api.list_predictions, version, league, MATCH_LIMIT)
    # After the predictions: they are logged on the way, and performance scores the log.
    files["performance"] = _call(api.performance)
    files["meta"] = {"generated_utc": generated, "model_versions": versions}

    data_out = out_dir / "data"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    data_out.mkdir(parents=True)
    for name, payload in files.items():
        (data_out / f"{name}.json").write_text(_encode(payload), encoding="utf-8")

    html = FRONTEND_PATH.read_text(encoding="utf-8")
    flag = f"<script>window.MATCHHUB_STATIC = {json.dumps({'generated': generated})};</script>\n"
    html = html.replace("<head>", "<head>\n" + flag, 1)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    # GitHub Pages: serve files as-is (no Jekyll processing).
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    export_cmd = commands.add_parser("export", help="sync and write the static site")
    export_cmd.add_argument("out_dir", type=Path)
    export_cmd.add_argument("--no-sync", action="store_true", help="export the current data without syncing")
    for name in ("pack", "unpack"):
        commands.add_parser(name, help=f"{name} the state archive").add_argument("archive", type=Path)
    args = parser.parse_args(argv)

    if args.command == "export":
        names = export(args.out_dir, sync=not args.no_sync)
        print(f"Skrev {len(names)} datafiler og index.html til {args.out_dir}")
    elif args.command == "pack":
        names = pack(args.archive)
        print(f"Pakkede {len(names)} filer i {args.archive} ({args.archive.stat().st_size / 1e6:.1f} MB)")
    else:
        names = unpack(args.archive)
        print(f"Udpakkede {len(names)} filer til {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
