"""Test one upcoming Premier League fixture and its Bet365 1X2 odds.

The script makes at most two HTTP requests and does not follow redirects or
retry. It reuses the project's standard-library .env parser and never prints
the API token or request headers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from typing import Any

import requests

from test_api_football import _read_dotenv_value


API_BASE_URL = "https://api.5dollarfootballapi.com/v1"
PREMIER_LEAGUE_ID = 4160026622
TOKEN_VARIABLE = "5DOLLARFOOTBALL_API_KEY"
REQUEST_TIMEOUT_SECONDS = 20
API_CALLS_USED = 0


class ApiTestError(RuntimeError):
    """Safe-to-display API or transport error."""


def _token() -> str:
    value = os.environ.get(TOKEN_VARIABLE) or _read_dotenv_value(TOKEN_VARIABLE)
    if not value:
        raise ApiTestError(
            f"API-token mangler. Angiv {TOKEN_VARIABLE} i projektets .env-fil."
        )
    return value


def _safe_text(value: Any, token: str, limit: int = 300) -> str:
    return str(value).replace(token, "[REDACTED]")[:limit]


def _error_details(payload: Any, token: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return "API'et returnerede ikke et dokumenteret fejlobjekt."
    error = payload["error"]
    parts = []
    for key in ("type", "code", "message", "param", "request_id"):
        value = error.get(key)
        if value is not None and value != "":
            parts.append(f"{key}={_safe_text(value, token)}")
    return "; ".join(parts) if parts else "API-fejl uden yderligere detaljer."


def _json_or_none(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _get_json(path: str, *, token: str, params: dict[str, Any] | None = None) -> Any:
    """Perform one request only; no retries and no automatic redirects."""
    global API_CALLS_USED
    API_CALLS_USED += 1
    call_number = API_CALLS_USED
    try:
        response = requests.get(
            f"{API_BASE_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except requests.Timeout:
        raise ApiTestError(f"Kald {call_number}: netværkskaldet fik timeout.") from None
    except requests.ConnectionError:
        raise ApiTestError(
            f"Kald {call_number}: kunne ikke oprette netværksforbindelse."
        ) from None
    except requests.RequestException:
        raise ApiTestError(f"Kald {call_number}: HTTP-kaldet fejlede.") from None

    print(f"Kald {call_number} HTTP-status: {response.status_code}")
    payload = _json_or_none(response)

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        suffix = f" Retry-After={retry_after}." if retry_after else ""
        raise ApiTestError("Rate limit nået (HTTP 429)." + suffix)
    if response.status_code in (401, 403):
        raise ApiTestError(
            f"Token/adgang afvist. {_error_details(payload, token)}"
        )
    if 300 <= response.status_code < 400:
        raise ApiTestError(
            "API'et returnerede redirect; det blev ikke fulgt, så kaldgrænsen bevares."
        )
    if not response.ok:
        raise ApiTestError(
            f"API-fejl (HTTP {response.status_code}). {_error_details(payload, token)}"
        )
    if not isinstance(payload, dict) or payload.get("success") != 1:
        raise ApiTestError(
            f"API'et markerede kaldet som mislykket. {_error_details(payload, token)}"
        )
    return payload


def _fixture(item: Any) -> tuple[int, str, str, datetime] | None:
    if not isinstance(item, dict):
        return None
    raw_id = item.get("id")
    teams = item.get("teams")
    if not isinstance(teams, dict):
        return None
    home = (teams.get("home") or {}).get("name")
    away = (teams.get("away") or {}).get("name")
    raw_kickoff = item.get("kickoff_utc")
    try:
        fixture_id = int(raw_id)
        if fixture_id <= 0:
            return None
        kickoff = datetime.fromisoformat(str(raw_kickoff).replace("Z", "+00:00"))
        if kickoff.tzinfo is None:
            return None
        kickoff = kickoff.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None
    if not home or not away:
        return None
    league = item.get("league")
    if isinstance(league, dict) and league.get("id") is not None:
        if str(league["id"]) != str(PREMIER_LEAGUE_ID):
            return None
    return fixture_id, str(home), str(away), kickoff


def _valid_odds(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 1.0


def _bet365_1x2(data: Any) -> tuple[str | None, dict[str, dict[str, float]]]:
    if not isinstance(data, dict) or not isinstance(data.get("bookmakers"), list):
        return None, {}
    for bookmaker in data["bookmakers"]:
        if not isinstance(bookmaker, dict):
            continue
        if str(bookmaker.get("slug", "")).casefold() != "bet365":
            continue
        name = str(bookmaker.get("name") or "Bet365")
        market = ((bookmaker.get("odds") or {}).get("1x2"))
        stages: dict[str, dict[str, float]] = {}
        if isinstance(market, dict):
            for stage in ("opening", "closing", "inplay"):
                prices = market.get(stage)
                if isinstance(prices, dict) and all(
                    _valid_odds(prices.get(key)) for key in ("home", "draw", "away")
                ):
                    stages[stage] = {
                        "home": float(prices["home"]),
                        "draw": float(prices["draw"]),
                        "away": float(prices["away"]),
                    }
        return name, stages
    return None, {}


def run_test() -> int:
    token = _token()
    print("5DollarFootballAPI: kommende Premier League-kampe og Bet365 1X2")

    # Documented league fixtures endpoint; chronological scheduled results.
    fixtures_payload = _get_json(
        f"/leagues/{PREMIER_LEAGUE_ID}/fixtures",
        token=token,
        params={"status": "scheduled", "order": "asc", "page": 1, "per_page": 50},
    )
    data = fixtures_payload.get("data")
    if not isinstance(data, list):
        raise ApiTestError("Fixture-svaret indeholder ikke en liste i `data`.")

    now = datetime.now(timezone.utc)
    upcoming = []
    for item in data:
        candidate = _fixture(item)
        if candidate is not None and candidate[3] > now:
            upcoming.append(candidate)
    if not upcoming:
        print("Ingen kommende kamp med fremtidigt kickoff og gyldigt kamp-ID blev fundet.")
        return 0
    fixture_id, home, away, kickoff = min(upcoming, key=lambda match: match[3])
    print(f"Kamp-ID: {fixture_id}")
    print(f"Kamp: {home} – {away}")
    print(f"Kickoff (UTC): {kickoff.isoformat()}")

    # The only other call is the documented single-fixture Bet365 1X2 endpoint.
    odds_payload = _get_json(
        f"/fixtures/{fixture_id}/odds",
        token=token,
        params={"bookmakers": "bet365", "market": "1x2"},
    )
    odds_data = odds_payload.get("data")
    response_fixture_id = (
        odds_data.get("fixture_id") if isinstance(odds_data, dict) else None
    )
    if response_fixture_id is not None and str(response_fixture_id) != str(fixture_id):
        raise ApiTestError("Odds-responsens kamp-ID stemmer ikke med den valgte kamp.")

    bookmaker, stages = _bet365_1x2(odds_data)
    if bookmaker is None:
        print("Bookmaker: Bet365 ikke returneret")
        print("Marked: 1X2")
        print("Odds-status: API-kald lykkedes, men Bet365 blev ikke fundet i svaret.")
        return 0
    print(f"Bookmaker: {bookmaker}")
    print("Marked: 1X2 (hjemmesejr/uafgjort/udesejr)")
    if not stages:
        print("Odds-status: Bet365 findes, men svaret har ingen gyldige H/D/A-odds.")
        return 0
    print("Odds-status: Bet365 1X2-odds fundet")
    for stage, prices in stages.items():
        print(
            f"{stage}: H={prices['home']:.3f}, D={prices['draw']:.3f}, "
            f"A={prices['away']:.3f}"
        )
    return 0


def main() -> int:
    global API_CALLS_USED
    try:
        return run_test()
    except ApiTestError as exc:
        print(f"Fejl: {exc}")
        return 1
    finally:
        print(f"Antal brugte API-kald: {API_CALLS_USED}")


if __name__ == "__main__":
    raise SystemExit(main())
