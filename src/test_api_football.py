"""Make one small API-Football connectivity and Premier League fixture check.

The API key is read from the project-root .env file (or the process
environment) and is never printed. No response body or request headers are
logged. python-dotenv is not required; .env parsing uses the standard library.
"""

from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path
import sys

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
API_URL = "https://v3.football.api-sports.io/fixtures"
PREMIER_LEAGUE_ID = 39
REQUEST_TIMEOUT_SECONDS = 20
MAX_FIXTURES_TO_PRINT = 10


def _read_dotenv_value(name: str, path: Path = ENV_PATH) -> str | None:
    """Read one key without exposing file contents or loading unrelated vars."""
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        return value or None
    return None


def _get_api_key() -> str:
    value = os.environ.get("API_FOOTBALL_KEY") or _read_dotenv_value("API_FOOTBALL_KEY")
    if not value:
        raise RuntimeError(
            "API-nøglen mangler. Angiv API_FOOTBALL_KEY i projektets .env-fil."
        )
    return value


def _quota_summary(headers: requests.structures.CaseInsensitiveDict) -> str:
    parts = []
    daily_remaining = headers.get("x-ratelimit-requests-remaining")
    daily_limit = headers.get("x-ratelimit-requests-limit")
    minute_remaining = headers.get("x-ratelimit-remaining")
    minute_limit = headers.get("x-ratelimit-limit")
    if daily_remaining is not None:
        label = f"Døgnkvote tilbage: {daily_remaining}"
        if daily_limit is not None:
            label += f"/{daily_limit}"
        parts.append(label)
    if minute_remaining is not None:
        label = f"Minutkvote tilbage: {minute_remaining}"
        if minute_limit is not None:
            label += f"/{minute_limit}"
        parts.append(label)
    return " | ".join(parts) if parts else "Kvoteoplysninger: ingen rate-limit-headers i responsen."


def _api_error(response: requests.Response) -> RuntimeError:
    if response.status_code == 429:
        return RuntimeError(
            "API-Football afviste kaldet på grund af rate limit (HTTP 429). "
            + _quota_summary(response.headers)
        )
    if response.status_code in (401, 403):
        return RuntimeError(
            "API-Football afviste adgangen (HTTP "
            f"{response.status_code}). Kontrollér API-nøglen og kontoens adgang."
        )
    if response.status_code >= 500:
        return RuntimeError(
            f"API-Football har en serverfejl (HTTP {response.status_code}). Prøv igen senere."
        )
    return RuntimeError(f"API-Football returnerede HTTP {response.status_code}.")


def _payload_error_message(errors: object) -> str:
    """Describe API error categories without printing arbitrary response values."""
    if isinstance(errors, dict):
        keys = {str(key).lower() for key in errors}
        if "token" in keys or "authorization" in keys:
            return "API-Football afviste API-nøglen eller dens adgang. Kontrollér nøglen i .env."
        if "ratelimit" in keys or "requests" in keys:
            return "API-Football angiver, at en rate limit eller kvote er nået."
        if "plan" in keys or "subscription" in keys:
            return "API-Football angiver, at endpointet eller funktionen ikke er inkluderet i abonnementet."
        if keys:
            safe_keys = ", ".join(sorted(keys))
            return f"API-Football rapporterede en API-fejl (felter: {safe_keys}); kontrollér parametre og konto."
    return "API-Football rapporterede en API-fejl; kontrollér nøgle, konto og parametre."


def _format_fixture(item: dict) -> tuple[str, str, str] | None:
    fixture = item.get("fixture") or {}
    teams = item.get("teams") or {}
    home = (teams.get("home") or {}).get("name")
    away = (teams.get("away") or {}).get("name")
    kickoff = fixture.get("date")
    if not home or not away or not kickoff:
        return None
    try:
        kickoff_local = str(kickoff).replace("T", " ")
        if kickoff_local.endswith("+00:00"):
            kickoff_local = kickoff_local[:-6] + " UTC"
        elif kickoff_local.endswith("Z"):
            kickoff_local = kickoff_local[:-1] + " UTC"
    except Exception:
        kickoff_local = str(kickoff)
    return kickoff_local, str(home), str(away)


def test_api_football() -> int:
    """Send one bounded request and print a small fixture sample."""
    api_key = _get_api_key()
    today = date.today()
    date_end = today + timedelta(days=30)
    params = {
        "league": PREMIER_LEAGUE_ID,
        "season": today.year,
        "from": today.isoformat(),
        "to": date_end.isoformat(),
        "timezone": "Europe/Copenhagen",
    }

    try:
        response = requests.get(
            API_URL,
            params=params,
            headers={"x-apisports-key": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        raise RuntimeError("Forbindelsen til API-Football fik timeout.") from None
    except requests.ConnectionError:
        raise RuntimeError("Kunne ikke oprette netværksforbindelse til API-Football.") from None
    except requests.RequestException:
        # Do not print raw request exceptions: this keeps request details and
        # credentials out of terminal output and log files.
        raise RuntimeError("HTTP-kaldet til API-Football fejlede.") from None

    print(f"API-Football svar: HTTP {response.status_code}")
    print(_quota_summary(response.headers))
    if not response.ok:
        raise _api_error(response)

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("API-Football returnerede ikke gyldig JSON.") from None
    if not isinstance(payload, dict):
        raise RuntimeError("API-Football returnerede et uventet svarformat.")
    errors = payload.get("errors")
    if errors:
        # Report only safe categories/field names, never arbitrary payload values.
        raise RuntimeError(_payload_error_message(errors))
    fixtures = payload.get("response")
    if not isinstance(fixtures, list):
        raise RuntimeError("API-Football-responsen mangler en fixture-liste.")

    print(
        "Kommende Premier League-kampe "
        f"({today.isoformat()} til {date_end.isoformat()}): {len(fixtures)} fundet"
    )
    printed = 0
    for item in fixtures:
        formatted = _format_fixture(item)
        if formatted is None:
            continue
        kickoff, home, away = formatted
        print(f"{kickoff} | {home} – {away}")
        printed += 1
        if printed >= MAX_FIXTURES_TO_PRINT:
            break
    if not fixtures:
        print("API'et svarede korrekt, men returnerede ingen kampe i datointervallet.")
    return len(fixtures)


def main() -> int:
    try:
        test_api_football()
    except RuntimeError as exc:
        print(f"Fejl: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
