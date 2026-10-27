"""Check football-data.org v4 access to upcoming Premier League matches.

Reads FOOTBALL_DATA_API_KEY from the project .env using the parser shared with
test_api_football.py. The token and request headers are never printed.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import sys

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
API_URL = "https://api.football-data.org/v4/competitions/PL/matches"
REQUEST_TIMEOUT_SECONDS = 20
DATE_WINDOW_DAYS = 30
MAX_MATCHES_TO_PRINT = 20

# The existing parser handles quotes, comments and optional `export` without
# importing python-dotenv or exposing any other .env values.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_api_football import _read_dotenv_value  # noqa: E402


def _get_token() -> str:
    token = os.environ.get("FOOTBALL_DATA_API_KEY") or _read_dotenv_value(
        "FOOTBALL_DATA_API_KEY", ENV_PATH
    )
    if not token:
        raise RuntimeError(
            "Token mangler. Angiv FOOTBALL_DATA_API_KEY i projektets .env-fil."
        )
    return token


def _safe_error_text(response: requests.Response, token: str) -> str:
    """Extract only the API's error message and redact the token defensively."""
    try:
        payload = response.json()
    except ValueError:
        return "API'et returnerede ikke et JSON-fejlsvar."
    message = payload.get("message") if isinstance(payload, dict) else None
    if not message:
        return "API'et returnerede ingen fejlbesked i JSON-svaret."
    return str(message).replace(token, "<redacted>")


def _rate_limit_info(headers: requests.structures.CaseInsensitiveDict) -> str | None:
    parts = []
    available = headers.get("X-Requests-Available-Minute")
    reset = headers.get("X-RequestCounter-Reset")
    if available is not None:
        parts.append(f"Kald tilbage dette minut: {available}")
    if reset is not None:
        parts.append(f"Tæller nulstilles om: {reset} sek.")
    return " | ".join(parts) if parts else None


def _raise_http_error(response: requests.Response, token: str) -> None:
    detail = _safe_error_text(response, token)
    if response.status_code == 401:
        raise RuntimeError(f"Token blev afvist (HTTP 401): {detail}")
    if response.status_code == 403:
        raise RuntimeError(f"Adgang til turneringen eller endpointet blev afvist (HTTP 403): {detail}")
    if response.status_code == 429:
        rate = _rate_limit_info(response.headers)
        suffix = f" {rate}." if rate else ""
        raise RuntimeError(f"Rate limit nået (HTTP 429): {detail}.{suffix}")
    if response.status_code >= 500:
        raise RuntimeError(f"football-data.org har en serverfejl (HTTP {response.status_code}): {detail}")
    raise RuntimeError(f"football-data.org returnerede HTTP {response.status_code}: {detail}")


def _parse_match_date(value: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        raise RuntimeError("API-svaret indeholder en kamp med ugyldig dato.") from None


def test_football_data() -> int:
    token = _get_token()
    date_from = date.today()
    date_to = date_from + timedelta(days=DATE_WINDOW_DAYS)
    params = {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()}

    try:
        response = requests.get(
            API_URL,
            params=params,
            headers={"X-Auth-Token": token},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        raise RuntimeError("Forbindelsen til football-data.org fik timeout.") from None
    except requests.ConnectionError:
        raise RuntimeError("Kunne ikke oprette netværksforbindelse til football-data.org.") from None
    except requests.RequestException:
        raise RuntimeError("HTTP-kaldet til football-data.org fejlede.") from None

    print(f"football-data.org svar: HTTP {response.status_code}")
    rate_info = _rate_limit_info(response.headers)
    if rate_info:
        print(rate_info)
    if not response.ok:
        _raise_http_error(response, token)

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("football-data.org returnerede ikke gyldig JSON.") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise RuntimeError("API-svaret mangler den forventede 'matches'-liste.")

    now_utc = datetime.now(timezone.utc)
    upcoming = []
    for match in payload["matches"]:
        utc_text = match.get("utcDate")
        if not utc_text:
            continue
        try:
            utc_date = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            raise RuntimeError("API-svaret indeholder en kamp med ugyldigt utcDate.") from None
        match_day = utc_date.date()
        if date_from <= match_day <= date_to and utc_date > now_utc:
            upcoming.append(match)

    print(f"Forespurgt interval: {date_from.isoformat()} til {date_to.isoformat()}")
    print(f"Kampe i API-svaret: {len(payload['matches'])}; kommende kampe i intervallet: {len(upcoming)}")
    if not upcoming:
        raise RuntimeError(
            "API-kaldet lykkedes, men der blev ikke returneret kommende kampe i det ønskede interval."
        )

    for match in upcoming[:MAX_MATCHES_TO_PRINT]:
        home = (match.get("homeTeam") or {}).get("name") or "Ukendt hjemmehold"
        away = (match.get("awayTeam") or {}).get("name") or "Ukendt udehold"
        status = match.get("status") or "Ukendt status"
        kickoff = match.get("utcDate") or "Ukendt dato"
        print(f"{kickoff} | {home} – {away} | {status}")
    if len(upcoming) > MAX_MATCHES_TO_PRINT:
        print(f"Viser {MAX_MATCHES_TO_PRINT} af {len(upcoming)} kampe.")
    return len(upcoming)


def main() -> int:
    try:
        test_football_data()
    except RuntimeError as exc:
        print(f"Fejl: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
