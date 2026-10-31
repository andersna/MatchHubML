"""Predict H/D/A for a future fixture using manual or cached Bet365 odds.

Example:
    python src/predict_upcoming_match.py --league "Premier League" \
        --home-team Arsenal --away-team Chelsea --match-date 2026-11-01 \
        --odds-prob-home 0.52 --odds-prob-draw 0.26 --odds-prob-away 0.22

Odds are entered by the caller; this script does not collect odds automatically.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from typing import Any
from urllib.parse import quote
import unittest
from datetime import timedelta

# Keep the project's established PyTorch-before-NumPy/Pandas import order.
import torch  # noqa: F401
import numpy as np
import pandas as pd

# `predict_match.py` uses script-local absolute imports, so make those modules
# available both for direct CLI use and when this file is imported as `src.*`.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from .data_store import DEFAULT_DATABASE_PATH, DataStore
    from .predict_match import CLASS_ORDER, ModelBundle, load_model_bundle, predict_match
    from .train_final_nn_odds_asof_day_v1 import MODEL_FEATURES, NN_FEATURES, ODDS_FEATURES
    from .update_historical_data import UpdateReport, ensure_historical_data_current, print_report
    from .upcoming_match_features import build_upcoming_match_features, history_team_name
    from .xg_features import XG_FEATURES, load_xg_history
except ImportError:  # pragma: no cover - direct script execution
    from data_store import DEFAULT_DATABASE_PATH, DataStore
    from predict_match import CLASS_ORDER, ModelBundle, load_model_bundle, predict_match
    from train_final_nn_odds_asof_day_v1 import MODEL_FEATURES, NN_FEATURES, ODDS_FEATURES
    from update_historical_data import UpdateReport, ensure_historical_data_current, print_report
    from upcoming_match_features import build_upcoming_match_features, history_team_name
    from xg_features import XG_FEATURES, load_xg_history


ODDS_TOLERANCE = 1e-8
BASE_V3_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "nn_market_residual_v3"
MODEL_FILES = ("model_state_dict.pt", "preprocessing.json", "metadata.json")


def latest_v3_dir(base: Path = BASE_V3_MODEL_DIR) -> Path:
    """Newest v3 artifact: a dated retrain (nn_market_residual_v3_YYYYMMDD) or the original."""
    retrained = sorted(
        path for path in base.parent.glob(f"{base.name}_*")
        if all((path / name).is_file() for name in MODEL_FILES)
    )
    return retrained[-1] if retrained else base


def check_historical_data_on_startup(update_func=None) -> UpdateReport | None:
    """Refresh historical fixtures when due, but never block prediction on failure.

    This is called only by the prediction CLI after its arguments and odds have
    been validated. It does not run on module import or for programmatic calls
    to ``predict_upcoming_match``.
    """
    updater = update_func or ensure_historical_data_current
    try:
        report = updater()
    except Exception as exc:  # Defensive boundary around application startup.
        print(
            "Advarsel: historiske data kunne ikke kontrolleres "
            f"({exc}); fortsætter med eksisterende data.",
            file=sys.stderr,
        )
        return None

    print_report(report)
    if report.error:
        print(
            "Advarsel: MatchHubML fortsætter med de eksisterende historiske data.",
            file=sys.stderr,
        )
    return report


def _parse_match_date(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Kampdato skal være en gyldig dato, fx 2026-11-01.") from exc
    if pd.isna(parsed):
        raise ValueError("Kampdato må ikke mangle.")
    if parsed.tz is not None:
        raise ValueError("Brug en lokal kampdato uden tidszone, fx 2026-11-01.")
    parsed = parsed.normalize()
    if parsed.date() < date.today():
        raise ValueError("Kampdatoen ligger i fortiden; angiv en kommende kampdato.")
    return parsed


def _validate_odds(home: Any, draw: Any, away: Any) -> tuple[float, float, float]:
    values = []
    for name, raw in (
        ("odds_prob_home", home),
        ("odds_prob_draw", draw),
        ("odds_prob_away", away),
    ):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} skal være et numerisk decimaltal.") from exc
        if not np.isfinite(value) or value <= 0 or value > 1:
            raise ValueError(f"{name} skal være endelig og ligge i intervallet (0, 1].")
        values.append(value)
    if not np.isclose(sum(values), 1.0, atol=ODDS_TOLERANCE, rtol=0):
        raise ValueError(
            "Odds-sandsynlighederne skal summere til 1 "
            f"(tolerance {ODDS_TOLERANCE:g}); modtog {sum(values):.12f}."
        )
    return tuple(values)


# Stored Bet365 1X2 prices, most preferred first. 5DollarFootball's "closing"
# stage, fetched before kickoff, is Bet365's price at lookup time: newer than the
# opening price, so it includes team news since opening. Opening is the fallback.
LOOKUP_MARKET = "1X2:closing"
ODDS_MARKETS = (LOOKUP_MARKET, "1X2:opening")
ODDS_MARKET_LABELS = {LOOKUP_MARKET: "Bet365 ved opslag", "1X2:opening": "Bet365 åbning"}
# SQL ordering for an odds_observations alias: preferred market first, then newest.
ODDS_PREFERENCE_SQL = "({alias}.market = '1X2:closing') DESC, {alias}.observed_at_utc DESC, {alias}.odds_observation_id DESC"


def odds_probabilities(home: Any, draw: Any, away: Any) -> tuple[float, float, float]:
    """Decimal odds -> implied probabilities with the overround removed proportionally."""
    decimal_odds = np.asarray([home, draw, away], dtype=np.float64)
    if not np.isfinite(decimal_odds).all() or (decimal_odds <= 1.0).any():
        raise ValueError("Den gemte oddsrække er ugyldig.")
    implied = 1.0 / decimal_odds
    return _validate_odds(*(implied / implied.sum()))


def load_stored_odds(
    league: str,
    home_team: str,
    away_team: str,
    match_date: str | date | datetime | pd.Timestamp,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> tuple[tuple[float, float, float], str, str]:
    """Read the preferred stored Bet365 1X2 quote for one exact future match.

    Returns (probabilities, observed_at_utc, market): the lookup-time price when
    stored, otherwise the opening price (see ODDS_MARKETS).

    The SQLite connection is read-only. Decimal odds are converted with the
    project's existing implied-probability rule: reciprocal odds, normalized
    to sum to one (overround removed by proportional normalization).
    """
    parsed_date = _parse_match_date(match_date).date().isoformat()
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Odds-cachedatabasen findes ikke: {path}")
    database_uri = "file:" + quote(path.as_posix(), safe="/:\\") + "?mode=ro"
    try:
        connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise RuntimeError(f"Odds-cachedatabasen kunne ikke åbnes read-only: {exc}") from exc
    try:
        matches = connection.execute(
            """SELECT match_id, match_status, kickoff_at FROM matches
            WHERE data_source = ? AND league = ? AND match_date = ?
              AND home_team = ? COLLATE NOCASE AND away_team = ? COLLATE NOCASE""",
            ("football-data.org", league.strip(), parsed_date,
             home_team.strip(), away_team.strip()),
        ).fetchall()
        if not matches:
            raise ValueError("Kampen findes ikke entydigt i den lokale kampcache.")
        if len(matches) != 1:
            raise ValueError("Flere kampe matcher liga, dato og hold; oddsopslag afvist.")
        match = matches[0]
        if str(match["match_status"]).upper() in {"FINISHED", "CANCELED", "CANCELLED"}:
            raise ValueError("Der hentes ikke gemte odds til en afsluttet eller aflyst kamp.")
        if not match["kickoff_at"]:
            raise ValueError("Kampens kickoff mangler; kommende kamp kan ikke verificeres.")
        try:
            kickoff = datetime.fromisoformat(match["kickoff_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Kampens kickoff i databasen er ugyldigt.") from exc
        if kickoff.tzinfo is None or kickoff.utcoffset() is None:
            raise ValueError("Kampens kickoff mangler tidszone i databasen.")
        if kickoff.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            raise ValueError("Kampen er allerede startet; odds bruges kun før kickoff.")

        observation = connection.execute(
            f"""SELECT o.odds_home, o.odds_draw, o.odds_away, o.observed_at_utc, o.market
            FROM odds_observations AS o
            WHERE o.match_id = ?
              AND lower(replace(o.bookmaker, ' ', '')) = 'bet365'
              AND o.market IN {ODDS_MARKETS}
            ORDER BY {ODDS_PREFERENCE_SQL.format(alias="o")} LIMIT 1""",
            (match["match_id"],),
        ).fetchone()
        if observation is None:
            raise ValueError("Der findes ingen gemt Bet365 1X2-observation for kampen.")
        probabilities = odds_probabilities(
            observation["odds_home"], observation["odds_draw"], observation["odds_away"]
        )
        return probabilities, str(observation["observed_at_utc"]), str(observation["market"])
    except sqlite3.Error as exc:
        raise RuntimeError(f"Fejl ved læsning af odds-cache: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()


def assemble_model_input(
    league: str,
    home_team: str,
    away_team: str,
    match_date: str | date | datetime | pd.Timestamp,
    odds_prob_home: float,
    odds_prob_draw: float,
    odds_prob_away: float,
    fixtures: pd.DataFrame | str | Path | None = None,
    sports_features: dict[str, float] | None = None,
) -> tuple[dict[str, Any], pd.Timestamp]:
    """Build and validate the model row without fitting preprocessing.

    The row holds the 30 v1/v2 columns followed by the 10 xG columns; each
    model reads only the columns it was trained on. xG is NaN when a team has
    no Understat history yet, and the saved median imputer fills it.
    """
    for label, value in (("liga", league), ("hjemmehold", home_team), ("udehold", away_team)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label.capitalize()} skal angives som en ikke-tom tekstværdi.")
    league, home_team, away_team = league.strip(), home_team.strip(), away_team.strip()
    if home_team == away_team:
        raise ValueError("Hjemmehold og udehold skal være forskellige.")
    parsed_date = _parse_match_date(match_date)
    odds = _validate_odds(odds_prob_home, odds_prob_draw, odds_prob_away)

    if sports_features is None:
        sports = build_upcoming_match_features(
            {
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": parsed_date,
            },
            fixtures=fixtures,
        )
    else:
        sports = pd.DataFrame(
            [{name: float(sports_features[name]) for name in NN_FEATURES}],
            columns=NN_FEATURES,
        )
    if tuple(sports.columns) != tuple(NN_FEATURES) or sports.shape != (1, 27):
        raise RuntimeError("Featureberegneren returnerede ikke de forventede 27 sportsfeatures.")

    row = {name: float(sports.at[0, name]) for name in NN_FEATURES}
    row.update(dict(zip(ODDS_FEATURES, odds)))
    row.update(load_xg_history().match_features(
        league, history_team_name(league, home_team), history_team_name(league, away_team), parsed_date,
    ))
    if tuple(row) != tuple(MODEL_FEATURES) + tuple(XG_FEATURES) or len(row) != 40:
        raise RuntimeError("Samlet input afviger fra modellens 40 features eller rækkefølge.")
    return row, parsed_date



def predict_upcoming_match(
    league: str,
    home_team: str,
    away_team: str,
    match_date: str | date | datetime | pd.Timestamp,
    odds_prob_home: float,
    odds_prob_draw: float,
    odds_prob_away: float,
    bundle: ModelBundle | None = None,
    fixtures: pd.DataFrame | str | Path | None = None,
    sports_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate features and predict the match, with timing diagnostics."""

    total_start = time.perf_counter()

    print("[TID-DETALJE] Starter assemble_model_input", flush=True)
    start = time.perf_counter()

    model_input, parsed_date = assemble_model_input(
        league,
        home_team,
        away_team,
        match_date,
        odds_prob_home,
        odds_prob_draw,
        odds_prob_away,
        fixtures=fixtures,
        sports_features=sports_features,
    )

    print(
        f"[TID-DETALJE] assemble_model_input: "
        f"{time.perf_counter() - start:.3f} sek.",
        flush=True,
    )

    print("[TID-DETALJE] Starter predict_match", flush=True)
    start = time.perf_counter()

    prediction = predict_match(model_input, bundle=bundle)

    print(
        f"[TID-DETALJE] predict_match: "
        f"{time.perf_counter() - start:.3f} sek.",
        flush=True,
    )

    print(
        f"[TID-DETALJE] Total: "
        f"{time.perf_counter() - total_start:.3f} sek.",
        flush=True,
    )

    return {
        "league": league.strip(),
        "home_team": home_team.strip(),
        "away_team": away_team.strip(),
        "date": parsed_date,
        **prediction,
    }

def _cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Forudsig en kommende kamp med MatchHubMLs gemte neuralnetværk."
    )
    parser.add_argument("--league", required=True, help="Ligaens præcise navn i historikken.")
    parser.add_argument("--home-team", required=True, help="Hjemmeholdets præcise navn i historikken.")
    parser.add_argument("--away-team", required=True, help="Udeholdets præcise navn i historikken.")
    parser.add_argument("--match-date", required=True, help="Kampdato i formatet YYYY-MM-DD.")
    parser.add_argument("--odds-prob-home", type=float, help="Manuelt indtastet Bet H-sandsynlighed.")
    parser.add_argument("--odds-prob-draw", type=float, help="Manuelt indtastet Bet D-sandsynlighed.")
    parser.add_argument("--odds-prob-away", type=float, help="Manuelt indtastet Bet A-sandsynlighed.")
    parser.add_argument(
        "--use-stored-odds", "--use-stored-opening-odds", dest="use_stored_odds", action="store_true",
        help="Brug gemte Bet365-odds fra SQLite-cachen (prisen ved opslag, ellers åbningsprisen).",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH,
                        help="SQLite-cache (bruges read-only til gemte odds).")
    parser.add_argument("--fixtures", type=Path, default=None, help="Alternativ historisk fixtures-CSV.")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Modelartefakt-mappe (standard: nyeste v3 market-residual-ensemble).")
    args = parser.parse_args()

    manual_odds = (args.odds_prob_home, args.odds_prob_draw, args.odds_prob_away)
    odds_observed_at = odds_market = None
    try:
        if args.use_stored_odds:
            if any(value is not None for value in manual_odds):
                parser.error("Angiv enten --use-stored-odds eller alle tre manuelle sandsynligheder.")
            probabilities, odds_observed_at, odds_market = load_stored_odds(
                args.league, args.home_team, args.away_team, args.match_date, args.database,
            )
        else:
            if any(value is None for value in manual_odds):
                parser.error(
                    "Angiv alle tre --odds-prob-* værdier eller brug --use-stored-odds."
                )
            probabilities = _validate_odds(*manual_odds)
        # The real user-facing entry point in this repository is the
        # prediction CLI. Check the weekly historical refresh immediately
        # before loading historical features; its fail-safe wrapper allows the
        # prediction to continue with the existing CSV if the source is down.
        check_historical_data_on_startup()
        bundle = load_model_bundle(artifact_dir=args.model_dir or latest_v3_dir())
        result = predict_upcoming_match(
            args.league, args.home_team, args.away_team, args.match_date,
            *probabilities,
            bundle=bundle, fixtures=args.fixtures,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    if odds_observed_at:
        print(f"Odds: gemte {ODDS_MARKET_LABELS.get(odds_market, odds_market)}; sandsynligheder beregnet fra inverse odds og normaliseret.")
        print(f"Odds registreret i lokal cache: {odds_observed_at} (hentetid, ikke bookmakerens publiceringstid).")
    else:
        print("Odds: manuelt indtastede sandsynligheder; automatisk oddsindsamling udføres ikke.")
    print("Odds-snapshot-tidspunkter er ikke dokumenteret, så oprindelig publicering før kickoff er ukendt.")
    print(f"Modelversion: {result['model_version']}")
    print(
        f"Kamp: {result['home_team']} vs. {result['away_team']} | "
        f"Dato: {result['date'].date()} | Liga: {result['league']}"
    )
    for label in CLASS_ORDER:
        print(f"{label}: {result['probabilities'][label]['percent']:.4f}%")
    print(f"Sandsynlighedssum: {result['sum']:.10f}")


class TestUpcomingPredictionAssembly(unittest.TestCase):
    def test_assembles_exact_model_input_order(self) -> None:
        from unittest.mock import patch

        fake_sports = pd.DataFrame([{name: float(index) for index, name in enumerate(NN_FEATURES)}])
        fake_xg = {name: 1.0 for name in XG_FEATURES}
        with patch(__name__ + ".build_upcoming_match_features", return_value=fake_sports), \
                patch(__name__ + ".history_team_name", side_effect=lambda league, name: name), \
                patch(__name__ + ".load_xg_history") as xg_history:
            xg_history.return_value.match_features.return_value = fake_xg
            row, _ = assemble_model_input(
                "League", "Home", "Away", date.today(), 0.5, 0.25, 0.25,
                fixtures=pd.DataFrame(),
            )
        self.assertEqual(tuple(row), tuple(MODEL_FEATURES) + tuple(XG_FEATURES))
        self.assertEqual(len(row), 40)
        self.assertEqual([row[name] for name in ODDS_FEATURES], [0.5, 0.25, 0.25])
        self.assertEqual([row[name] for name in XG_FEATURES], [1.0] * len(XG_FEATURES))

    def test_rejects_invalid_odds_before_feature_lookup(self) -> None:
        with self.assertRaisesRegex(ValueError, "summere til 1"):
            assemble_model_input(
                "League", "Home", "Away", date.today(), 0.6, 0.3, 0.2,
                fixtures=pd.DataFrame(),
            )

    def test_prefers_lookup_price_and_normalizes_inverse_prices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "cache.sqlite"
            future_date = date.today() + timedelta(days=4)
            kickoff = datetime.combine(future_date, datetime.min.time(), tzinfo=timezone.utc)
            with DataStore(database) as store:
                with store.transaction():
                    match_id, _ = store.upsert_match({
                        "data_source": "football-data.org", "external_match_id": "sample-1",
                        "source_match_key": "external:sample-1", "league": "La Liga",
                        "match_date": future_date.isoformat(),
                        "kickoff_at": kickoff.isoformat().replace("+00:00", "Z"),
                        "home_team": "Levante UD", "away_team": "Athletic Club",
                        "match_status": "TIMED", "source_payload": {"id": "sample-1"},
                    })
                    for market, prices in (
                        ("1X2:opening", (2.0, 4.0, 4.0)),
                        ("1X2:closing", (9.0, 2.0, 3.0)),
                    ):
                        store.add_odds_observation(
                            match_id=match_id, bookmaker="Bet 365", market=market,
                            odds_home=prices[0], odds_draw=prices[1], odds_away=prices[2],
                            observed_at=datetime.now(timezone.utc), data_source="5dollarfootballapi",
                        )
                    store.add_odds_observation(
                        match_id=match_id, bookmaker="Bet365", market="1X2:opening",
                        odds_home=2.0, odds_draw=4.0, odds_away=4.0,
                        observed_at=datetime.now(timezone.utc), data_source="5dollarfootballapi",
                    )
                    # Same teams in another future fixture must not leak into
                    # the exact league/date lookup used for this prediction.
                    other_date = future_date + timedelta(days=7)
                    other_kickoff = datetime.combine(
                        other_date, datetime.min.time(), tzinfo=timezone.utc
                    )
                    other_match_id, _ = store.upsert_match({
                        "data_source": "football-data.org", "external_match_id": "sample-other",
                        "source_match_key": "external:sample-other", "league": "La Liga",
                        "match_date": other_date.isoformat(),
                        "kickoff_at": other_kickoff.isoformat().replace("+00:00", "Z"),
                        "home_team": "Levante UD", "away_team": "Athletic Club",
                        "match_status": "TIMED", "source_payload": {"id": "sample-other"},
                    })
                    store.add_odds_observation(
                        match_id=other_match_id, bookmaker="Bet 365", market="1X2:opening",
                        odds_home=8.0, odds_draw=9.0, odds_away=10.0,
                        observed_at=datetime.now(timezone.utc), data_source="5dollarfootballapi",
                    )
            probabilities, observed_at, market = load_stored_odds(
                "La Liga", "Levante UD", "Athletic Club", future_date, database,
            )
            # The lookup price (9, 2, 3) wins over opening (2, 4, 4); the other fixture never leaks.
            implied = np.array([1 / 9, 1 / 2, 1 / 3])
            np.testing.assert_allclose(probabilities, implied / implied.sum())
            self.assertEqual(market, "1X2:closing")
            self.assertTrue(observed_at.endswith("Z"))

    def test_opening_price_is_the_fallback_and_no_price_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Path(temp_dir) / "cache.sqlite"
            future_date = date.today() + timedelta(days=4)
            kickoff = datetime.combine(future_date, datetime.min.time(), tzinfo=timezone.utc)
            with DataStore(database) as store:
                with store.transaction():
                    match_id, _ = store.upsert_match({
                        "data_source": "football-data.org", "external_match_id": "sample-2",
                        "source_match_key": "external:sample-2", "league": "La Liga",
                        "match_date": future_date.isoformat(),
                        "kickoff_at": kickoff.isoformat().replace("+00:00", "Z"),
                        "home_team": "Levante UD", "away_team": "Athletic Club",
                        "match_status": "TIMED", "source_payload": {"id": "sample-2"},
                    })
            with self.assertRaisesRegex(ValueError, "ingen gemt Bet365"):
                load_stored_odds("La Liga", "Levante UD", "Athletic Club", future_date, database)
            with DataStore(database) as store:
                with store.transaction():
                    store.add_odds_observation(
                        match_id=match_id, bookmaker="Bet 365", market="1X2:opening",
                        odds_home=2.0, odds_draw=4.0, odds_away=4.0,
                        observed_at=datetime.now(timezone.utc), data_source="5dollarfootballapi",
                    )
            probabilities, _, market = load_stored_odds(
                "La Liga", "Levante UD", "Athletic Club", future_date, database,
            )
            self.assertEqual((probabilities, market), ((0.5, 0.25, 0.25), "1X2:opening"))


if __name__ == "__main__":
    _cli()
