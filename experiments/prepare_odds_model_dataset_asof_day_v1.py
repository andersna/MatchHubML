"""Assemble a versioned 30-input dataset from day-replay features and odds.

The joins and validity filters mirror load_joined_data() in
experiment_nn_odds_modern.py. No model is trained here. Historical bookmaker
snapshot times are unavailable, so this dataset cannot establish that odds
were known before kickoff.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features_asof_day_v1.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features_asof_day_v1.csv"
ODDS_DATASET_PATH = PROJECT_ROOT / "data" / "odds_model_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "odds_model_dataset_asof_day_v1.csv"

KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]
ODDS_MATCH_KEYS = ["date", "league", "home_team", "away_team"]
BASE_FEATURES = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
    "home_last5_goals_for",
    "home_last5_goals_against",
    "home_last5_points",
    "away_last5_goals_for",
    "away_last5_goals_against",
    "away_last5_points",
]
ATTACK_DEFENSE_FEATURES = [
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "home_expected_goals",
    "away_expected_goals",
]
ELO_COLUMNS = [
    f"{prefix}_elo_k{k}"
    for k in [16, 32, 48]
    for prefix in ["home", "away"]
] + [f"elo_difference_k{k}" for k in [16, 32, 48]]
NN_FEATURES = BASE_FEATURES + ATTACK_DEFENSE_FEATURES + ELO_COLUMNS
ODDS_FEATURES = ["odds_prob_home", "odds_prob_draw", "odds_prob_away"]
MODEL_FEATURES = NN_FEATURES + ODDS_FEATURES
COMMON_FEATURES = [
    *BASE_FEATURES,
    "home_form_rating",
    "away_form_rating",
    "home_venue_form_rating",
    "away_venue_form_rating",
    *ATTACK_DEFENSE_FEATURES,
]
LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
# Keep the intended modern neural-network configuration explicit; this script
# assembles data only and does not train or apply class weights.
MODEL_ALPHA = 0.25


def _require_columns(frame, required, name):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} mangler kolonner: {missing}")


def _assert_unique(frame, keys, name):
    duplicates = frame.duplicated(keys)
    if duplicates.any():
        sample = frame.loc[duplicates, keys].head(10)
        raise ValueError(
            f"{name} har {int(duplicates.sum())} dubletter på kampnøgler {keys}:\n"
            + sample.to_string(index=False)
        )


def _read_and_join():
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"], low_memory=False)
    odds = pd.read_csv(ODDS_DATASET_PATH, parse_dates=["date"], low_memory=False)

    required_features = KEY_COLUMNS + BASE_FEATURES + [
        "home_form_rating",
        "away_form_rating",
        "home_venue_form_rating",
        "away_venue_form_rating",
        "result",
    ]
    required_elo = KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES
    required_odds = ODDS_MATCH_KEYS + ["result"] + ODDS_FEATURES
    _require_columns(features, required_features, "features_asof_day_v1.csv")
    _require_columns(elo, required_elo, "elo_features_asof_day_v1.csv")
    _require_columns(odds, required_odds, "odds_model_dataset.csv")

    _assert_unique(features, KEY_COLUMNS, "features_asof_day_v1.csv")
    _assert_unique(elo, KEY_COLUMNS, "elo_features_asof_day_v1.csv")
    _assert_unique(odds, ODDS_MATCH_KEYS, "odds_model_dataset.csv")

    print("=== Rækkeantal gennem databehandlingen ===")
    print(f"Features input: {len(features):,}")
    print(f"Elo input: {len(elo):,}")
    print(f"Odds-model input (allerede filtreret til positive odds): {len(odds):,}")

    # Match the modern experiment: the 27 model columns are sourced from the
    # two versioned files, joined by the same five-column fixture key.
    combined = features.merge(
        elo[KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator="_elo_merge",
    )
    elo_unmatched = int((combined["_elo_merge"] != "both").sum())
    print(f"Efter feature/Elo-join: {len(combined):,}; uden Elo-match: {elo_unmatched:,}")
    if elo_unmatched:
        raise ValueError("Ikke alle features-rækker fik et entydigt Elo-match.")
    combined = combined.drop(columns="_elo_merge")
    combined = combined.sort_values(["date", "time"], kind="stable", na_position="last")

    before_feature_filter = len(combined)
    combined = combined.dropna(subset=COMMON_FEATURES + ELO_COLUMNS).reset_index(drop=True)
    print(
        f"Efter moderne complete-feature-filter: {len(combined):,} "
        f"(fjernet {before_feature_filter - len(combined):,})"
    )

    before_result_filter = len(combined)
    combined = combined[combined["result"].isin(LABEL_TO_INDEX)].copy()
    print(
        f"Efter H/D/A-resultatfilter: {len(combined):,} "
        f"(fjernet {before_result_filter - len(combined):,})"
    )
    if combined.empty:
        raise ValueError("Ingen rækker er tilbage efter baseline-filtrering.")
    _assert_unique(combined, KEY_COLUMNS, "renset feature/Elo-datasæt")

    odds = odds.copy()
    odds[ODDS_FEATURES] = odds[ODDS_FEATURES].apply(pd.to_numeric, errors="coerce")
    if odds[ODDS_FEATURES].isna().any().any():
        raise ValueError("Odds-sandsynligheder indeholder manglende/ugyldige værdier.")
    probabilities = odds[ODDS_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any() or (probabilities > 1).any():
        raise ValueError("Odds-sandsynligheder skal være endelige værdier i (0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Odds-sandsynligheder summerer ikke til 1.")
    if not odds["result"].isin(LABEL_TO_INDEX).all():
        raise ValueError("Odds-datasættet indeholder ukendte udfald.")

    # The existing odds experiment joins on normalized date + league + teams,
    # intentionally excluding time because odds_model_dataset uses that key.
    baseline_for_join = combined.copy()
    baseline_for_join["_date_key"] = baseline_for_join["date"].dt.normalize()
    odds_for_join = odds[ODDS_MATCH_KEYS + ["result"] + ODDS_FEATURES].copy()
    odds_for_join["_date_key"] = odds_for_join["date"].dt.normalize()
    odds_for_join = odds_for_join.drop(columns="date")

    joined = baseline_for_join.merge(
        odds_for_join,
        on=["_date_key", "league", "home_team", "away_team"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_odds"),
    )
    print(f"Efter odds-dækning (indre join): {len(joined):,}")
    if joined.empty:
        raise ValueError("Ingen fælles odds-dækkede kampe blev fundet.")
    if not (joined["result"] == joined["result_odds"]).all():
        mismatched = joined.loc[
            joined["result"] != joined["result_odds"],
            KEY_COLUMNS + ["result", "result_odds"],
        ].head(10)
        raise ValueError(
            "Faktiske udfald afviger mellem feature- og oddsdata:\n"
            + mismatched.to_string(index=False)
        )
    joined = joined.drop(columns=["result_odds", "_date_key"])

    output_columns = KEY_COLUMNS + ["result"] + MODEL_FEATURES
    _require_columns(joined, output_columns, "samlet datasæt")
    result = joined[output_columns].copy()
    _assert_unique(result, KEY_COLUMNS, "nyt odds-modeldatasæt")

    missing_inputs = result[MODEL_FEATURES].isna().sum()
    if missing_inputs.any():
        raise ValueError(
            "Manglende værdier i modelinputs:\n"
            + missing_inputs[missing_inputs > 0].to_string()
        )
    if len(NN_FEATURES) != 27 or len(MODEL_FEATURES) != 30:
        raise AssertionError(
            f"Forkert featureantal: NN={len(NN_FEATURES)}, samlet={len(MODEL_FEATURES)}"
        )

    return result, odds


def _compare_with_old_dataset(new_data, odds_source):
    old_keys = odds_source[ODDS_MATCH_KEYS].copy()
    old_keys["date"] = pd.to_datetime(old_keys["date"], errors="raise").dt.normalize()
    old_keys = old_keys.drop_duplicates()

    new_keys = new_data[ODDS_MATCH_KEYS].copy()
    new_keys["date"] = pd.to_datetime(new_keys["date"], errors="raise").dt.normalize()
    new_keys = new_keys.drop_duplicates()

    comparison = old_keys.merge(
        new_keys,
        on=ODDS_MATCH_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    print("\n=== Sammenligning med tidligere odds_model_dataset.csv ===")
    print(f"Tidligere datasæt: {len(odds_source):,} rækker")
    print(f"Nyt dagsvist datasæt: {len(new_data):,} rækker")
    print(f"Fælles kampnøgler: {int((comparison['_merge'] == 'both').sum()):,}")
    print(f"Kun i tidligere odds-modeldatasæt: {int((comparison['_merge'] == 'left_only').sum()):,}")
    print(f"Kun i nyt dagsvist datasæt: {int((comparison['_merge'] == 'right_only').sum()):,}")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Output findes allerede; intet overskrives: {OUTPUT_PATH}")
    print(
        "Odds-begrænsning: kilderne indeholder ingen snapshot-tidspunkter; "
        "historiske odds kan ikke dokumenteres som tilgængelige før kickoff."
    )
    result, _ = _read_and_join()
    old_odds = pd.read_csv(ODDS_DATASET_PATH, parse_dates=["date"], low_memory=False)
    _compare_with_old_dataset(result, old_odds)

    # Exclusive creation protects against overwriting if another process
    # creates the versioned output after the startup check.
    try:
        with OUTPUT_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
            result.to_csv(handle, index=False)
    except FileExistsError:
        raise FileExistsError(f"Output opstod under kørslen; intet overskrevet: {OUTPUT_PATH}")

    print("\n=== Validering ===")
    print(f"NN-features: {len(NN_FEATURES)}; odds-features: {len(ODDS_FEATURES)}; samlet: {len(MODEL_FEATURES)}")
    print(f"Tiltænkt modelkonfiguration: alpha={MODEL_ALPHA} (ingen træning udført)")
    print(f"Alle modelinputs til stede uden manglende værdier: {not result[MODEL_FEATURES].isna().any().any()}")
    print(f"Resultatlabels følger H=0, D=1, A=2: {LABEL_TO_INDEX}")
    print(f"Dubletter på fulde kampnøgler: {int(result.duplicated(KEY_COLUMNS).sum())}")
    print(f"Gemt: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
