"""Assemble the v2 NN+odds dataset from local, already available data only.

The 48 newly imported 2026-27 fixtures receive Bet365 probabilities from the
matching local Football-Data workbook. No network access is used. Historical
odds have no snapshot timestamps, so their pre-kickoff availability is unknown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from collect_excel_data import LEAGUES
from prepare_odds_model_dataset_asof_day_v1 import (
    ATTACK_DEFENSE_FEATURES,
    BASE_FEATURES,
    COMMON_FEATURES,
    ELO_COLUMNS,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    MODEL_FEATURES,
    NN_FEATURES,
    ODDS_FEATURES,
    ODDS_MATCH_KEYS,
    _assert_unique,
    _require_columns,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATURES_PATH = DATA / "features_asof_day_v2.csv"
ELO_PATH = DATA / "elo_features_asof_day_v2.csv"
EXISTING_ODDS_PATH = DATA / "odds_model_dataset.csv"
V1_FEATURES_PATH = DATA / "features_asof_day_v1.csv"
RAW_CURRENT_SEASON_PATH = DATA / "raw" / "all-euro-data-2026-2027.xlsx"
OUTPUT_PATH = DATA / "odds_model_dataset_asof_day_v2.csv"
EXPECTED_NEW_MATCHES = 48
RAW_ODDS_COLUMNS = ["B365H", "B365D", "B365A"]


def _normalized_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[ODDS_MATCH_KEYS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    for column in ODDS_MATCH_KEYS[1:]:
        result[column] = result[column].astype("string").str.strip().str.casefold()
    return result


def _new_fixture_keys(features: pd.DataFrame) -> set[tuple]:
    current = _normalized_key_frame(features)
    previous = pd.read_csv(V1_FEATURES_PATH, parse_dates=["date"], low_memory=False)
    _require_columns(previous, ODDS_MATCH_KEYS, str(V1_FEATURES_PATH))
    old_keys = set(map(tuple, _normalized_key_frame(previous).to_numpy()))
    return set(map(tuple, current.to_numpy())) - old_keys


def _load_new_workbook_odds(features: pd.DataFrame) -> pd.DataFrame:
    """Read only the 48 new fixture keys from the local current-season workbook."""
    if not RAW_CURRENT_SEASON_PATH.is_file():
        raise FileNotFoundError(f"Lokal odds-kilde mangler: {RAW_CURRENT_SEASON_PATH}")
    workbook = pd.ExcelFile(RAW_CURRENT_SEASON_PATH)
    missing_sheets = sorted(set(LEAGUES) - set(workbook.sheet_names))
    if missing_sheets:
        raise ValueError(f"Excel-filen mangler ligeark: {missing_sheets}")

    wanted = _new_fixture_keys(features)
    if len(wanted) != EXPECTED_NEW_MATCHES:
        raise ValueError(
            f"Forventede {EXPECTED_NEW_MATCHES} nye kampnøgler mod v1, fandt {len(wanted)}."
        )
    records: list[pd.DataFrame] = []
    for sheet, league in LEAGUES.items():
        raw = pd.read_excel(workbook, sheet_name=sheet)
        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", *RAW_ODDS_COLUMNS]
        _require_columns(raw, required, f"Excel-ark {sheet}")
        part = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True),
            "league": league,
            "home_team": raw["HomeTeam"],
            "away_team": raw["AwayTeam"],
            "home_goals": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(raw["FTAG"], errors="coerce"),
            "odds_home": pd.to_numeric(raw["B365H"], errors="coerce"),
            "odds_draw": pd.to_numeric(raw["B365D"], errors="coerce"),
            "odds_away": pd.to_numeric(raw["B365A"], errors="coerce"),
        })
        part = part.dropna(subset=ODDS_MATCH_KEYS + ["home_goals", "away_goals"])
        for column in ("league", "home_team", "away_team"):
            part[column] = part[column].astype("string").str.strip()
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
        part["result"] = np.select(
            [part.home_goals > part.away_goals, part.home_goals == part.away_goals],
            ["H", "D"], default="A",
        )
        records.append(part[ODDS_MATCH_KEYS + ["result", "odds_home", "odds_draw", "odds_away"]])

    raw_odds = pd.concat(records, ignore_index=True)
    raw_keys = _normalized_key_frame(raw_odds)
    raw_odds = raw_odds.loc[raw_keys.apply(tuple, axis=1).isin(wanted)].copy()
    normalized = _normalized_key_frame(raw_odds)
    if len(raw_odds) != EXPECTED_NEW_MATCHES or normalized.duplicated().any():
        raise ValueError(
            "Den lokale Excel-fil gav ikke præcis én oddsrække for hver ny fixture."
        )

    # Require exact names as well as the normalized key, preventing a normalized
    # collision from silently joining odds to a different team spelling.
    target = features[ODDS_MATCH_KEYS].copy()
    target["date"] = pd.to_datetime(target["date"], errors="raise").dt.normalize()
    target_norm = _normalized_key_frame(target)
    target_new = target.loc[target_norm.apply(tuple, axis=1).isin(wanted)].copy()
    exact = target_new.merge(
        raw_odds, on=ODDS_MATCH_KEYS, how="inner", validate="one_to_one",
        suffixes=("", "_raw"),
    )
    if len(exact) != EXPECTED_NEW_MATCHES:
        raise ValueError("De nye fixture- og Excel-kampnøgler matcher ikke eksakt.")

    prices = exact[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 1.0).any():
        raise ValueError("B365-decimalodds skal være endelige og større end 1.")
    implied = 1.0 / prices
    probabilities = implied / implied.sum(axis=1, keepdims=True)
    for i, name in enumerate(ODDS_FEATURES):
        exact[name] = probabilities[:, i]
    if not np.allclose(exact[ODDS_FEATURES].sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Nye odds-sandsynligheder summerer ikke til 1.")
    return exact[ODDS_MATCH_KEYS + ["result"] + ODDS_FEATURES]


def build_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"], low_memory=False)
    existing_odds = pd.read_csv(EXISTING_ODDS_PATH, parse_dates=["date"], low_memory=False)
    _require_columns(
        features,
        KEY_COLUMNS + BASE_FEATURES + [
            "home_form_rating", "away_form_rating",
            "home_venue_form_rating", "away_venue_form_rating", "result",
        ],
        str(FEATURES_PATH),
    )
    _require_columns(elo, KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES, str(ELO_PATH))
    _require_columns(existing_odds, ODDS_MATCH_KEYS + ["result"] + ODDS_FEATURES, str(EXISTING_ODDS_PATH))
    _assert_unique(features, KEY_COLUMNS, str(FEATURES_PATH))
    _assert_unique(elo, KEY_COLUMNS, str(ELO_PATH))
    _assert_unique(existing_odds, ODDS_MATCH_KEYS, str(EXISTING_ODDS_PATH))

    joined = features.merge(
        elo[KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES],
        on=KEY_COLUMNS, how="left", validate="one_to_one", indicator="_elo_merge",
    )
    if not joined["_elo_merge"].eq("both").all():
        raise ValueError("Feature/Elo-join mangler kampnøgler.")
    joined = joined.drop(columns="_elo_merge").sort_values(
        ["date", "time"], kind="stable", na_position="last"
    )
    before_filters = len(joined)
    joined = joined.dropna(subset=COMMON_FEATURES + ELO_COLUMNS)
    joined = joined[joined.result.isin(LABEL_TO_INDEX)].copy()

    new_odds = _load_new_workbook_odds(features)
    combined_odds = pd.concat(
        [existing_odds[ODDS_MATCH_KEYS + ["result"] + ODDS_FEATURES], new_odds],
        ignore_index=True,
    )
    combined_odds["date"] = pd.to_datetime(combined_odds["date"], errors="raise").dt.normalize()
    _assert_unique(combined_odds, ODDS_MATCH_KEYS, "samlet eksisterende + nye odds")
    if not combined_odds["result"].isin(LABEL_TO_INDEX).all():
        raise ValueError("Oddsgrundlaget indeholder ugyldige H/D/A-resultater.")
    probs = combined_odds[ODDS_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all() or (probs <= 0).any() or (probs > 1).any():
        raise ValueError("Odds-sandsynlighederne indeholder ugyldige værdier.")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Odds-sandsynlighederne summerer ikke til 1.")

    left = joined.copy()
    left["_date_key"] = left.date.dt.normalize()
    right = combined_odds.copy().drop(columns="date")
    right["_date_key"] = combined_odds.date
    output = left.merge(
        right, on=["_date_key", "league", "home_team", "away_team"],
        how="inner", validate="one_to_one", suffixes=("", "_odds"),
    )
    if not output["result"].eq(output["result_odds"]).all():
        raise ValueError("Resultaterne i feature- og oddsdata afviger.")
    output = output.drop(columns=["result_odds", "_date_key"])
    output = output[KEY_COLUMNS + ["result"] + MODEL_FEATURES].copy()
    _assert_unique(output, KEY_COLUMNS, "v2 odds-modeldatasæt")
    if output[MODEL_FEATURES].isna().any().any():
        raise ValueError("V2-datasættet har manglende modelinputs.")
    if len(NN_FEATURES) != 27 or len(MODEL_FEATURES) != 30:
        raise AssertionError("Forventede 27 sportsfeatures og 3 oddsfeatures.")

    new_key_count = len(_new_fixture_keys(features))
    if len(new_odds) != EXPECTED_NEW_MATCHES:
        raise AssertionError("Ikke alle nye fixture-odds blev indlæst.")
    report = {
        "features_input": len(features),
        "elo_input": len(elo),
        "existing_odds_rows": len(existing_odds),
        "new_workbook_odds_rows": len(new_odds),
        "new_fixture_keys": new_key_count,
        "rows_after_feature_and_result_filters": len(joined),
        "rows_after_odds_join": len(output),
        "feature_rows_removed_by_filters": before_filters - len(joined),
    }
    return output, report


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Output findes allerede; intet overskrives: {OUTPUT_PATH}")
    print("Historiske odds har ikke dokumenterede snapshot-tidspunkter; før-kickoff-tilgængelighed er ukendt.")
    result, report = build_dataset()
    if len(result) != 43174:
        raise ValueError(f"Forventede 43.174 rækker efter validering; fandt {len(result)}.")
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Output blev oprettet under kørslen; intet overskrevet: {OUTPUT_PATH}")
    try:
        with OUTPUT_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
            result.to_csv(handle, index=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Output findes allerede; intet overskrevet: {OUTPUT_PATH}") from exc
    except Exception:
        OUTPUT_PATH.unlink(missing_ok=True)
        raise
    print("V2-oddsdatasæt valideret:")
    for key, value in report.items():
        print(f"  {key}={value:,}")
    print(f"  kolonner={len(result.columns)}; modelinputs={len(MODEL_FEATURES)}")
    print(f"Gemt: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
