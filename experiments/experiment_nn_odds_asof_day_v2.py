"""Frozen-fold v2 backtest plus separate later-holdout check for the v1 model.

Historical v1 folds are reconstructed from their original full-baseline files;
the 48 post-2026-10-12 matches never redefine those boundaries. Those 48 are
evaluated separately with the already saved v1 artifact, before v2 final
training. Historical bookmaker odds have no snapshot timestamps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_nn_odds_asof_day_v1 import (
    ALPHA,
    DATA,
    DEVICE,
    ELO_COLUMNS,
    INDEX_TO_LABEL,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    METRICS,
    MODEL_FEATURES,
    NN_FEATURES,
    ODDS_FEATURES,
    SEEDS,
    _append_predictions,
    _assert_same_fold_keys,
    _baseline_dates,
    _comparison_with_previous,
    _keys,
    calculate_metrics,
    normalize_probabilities,
    train_variant,
)
from predict_match import load_model_bundle
from train_final_nn_odds_asof_day_v1 import ARTIFACT_DIR as V1_ARTIFACT_DIR


ROOT = Path(__file__).resolve().parent.parent
V2_DATA_PATH = DATA / "odds_model_dataset_asof_day_v2.csv"
V1_DATA_PATH = DATA / "odds_model_dataset_asof_day_v1.csv"
V1_OOF_PATH = DATA / "oof_predictions_nn_odds_asof_day_v1.csv"
V1_SUMMARY_PATH = DATA / "nn_odds_asof_day_v1_summary.csv"
OOF_PATH = DATA / "oof_predictions_nn_odds_asof_day_v2.csv"
SUMMARY_PATH = DATA / "nn_odds_asof_day_v2_summary.csv"
EXPECTED_HISTORICAL_ROWS = 43126
EXPECTED_LATER_ROWS = 48
FOLD_VARIANTS = ("baseline_27", "nn_with_odds", "odds_only")


def check_outputs_absent() -> None:
    existing = [path for path in (OOF_PATH, SUMMARY_PATH) if path.exists()]
    if existing:
        raise FileExistsError("V2 backtest-output findes; intet overskrives: " + ", ".join(map(str, existing)))


def _read_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[list]]:
    v1_full, old_blocks = _baseline_dates(DATA / "features.csv", DATA / "elo_features.csv")
    v1_day_full, v1_blocks = _baseline_dates(
        DATA / "features_asof_day_v1.csv", DATA / "elo_features_asof_day_v1.csv"
    )
    if old_blocks != v1_blocks:
        raise RuntimeError("De oprindelige moderne baseline-foldgrænser er allerede inkonsistente.")

    data = pd.read_csv(V2_DATA_PATH, parse_dates=["date"], low_memory=False)
    v1_data = pd.read_csv(V1_DATA_PATH, parse_dates=["date"], low_memory=False)
    previous_oof = pd.read_csv(V1_OOF_PATH, parse_dates=["date"], low_memory=False)
    previous_summary = pd.read_csv(V1_SUMMARY_PATH, low_memory=False)
    required = KEY_COLUMNS + ["result"] + MODEL_FEATURES
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"V2-datasættet mangler kolonner: {missing}")
    if data.duplicated(KEY_COLUMNS).any() or data[MODEL_FEATURES].isna().any().any():
        raise ValueError("V2-datasættet har dublerede kampnøgler eller manglende modelinputs.")
    if not data.result.isin(LABEL_TO_INDEX).all():
        raise ValueError("V2-datasættet indeholder ugyldige H/D/A-resultater.")
    if len(NN_FEATURES) != 27 or len(MODEL_FEATURES) != 30:
        raise AssertionError("Forventede de eksisterende 27 sports- og 3 oddsfeatures.")

    allowed_dates = {day for block in old_blocks for day in block}
    historical = data[data.date.dt.date.isin(allowed_dates)].copy()
    later = data[~data.date.dt.date.isin(allowed_dates)].copy()
    if len(historical) != EXPECTED_HISTORICAL_ROWS or len(later) != EXPECTED_LATER_ROWS:
        raise ValueError(
            f"Datoafgrænsningen gav historisk={len(historical)} og senere={len(later)}; "
            f"forventede {EXPECTED_HISTORICAL_ROWS} og {EXPECTED_LATER_ROWS}."
        )
    if (later.date.dt.date <= max(allowed_dates)).any():
        raise AssertionError("Senere holdout indeholder datoer inden for de frosne folds.")

    # The v2 backtest must reproduce the exact v1 eligible sample for its paired
    # folds; only the 48 later fixtures may be outside that sample.
    if not _keys(historical).reset_index(drop=True).equals(_keys(v1_data).reset_index(drop=True)):
        historical_sorted = historical.sort_values(["date", "time"], kind="stable")
        v1_sorted = v1_data.sort_values(["date", "time"], kind="stable")
        if not _keys(historical_sorted).reset_index(drop=True).equals(_keys(v1_sorted).reset_index(drop=True)):
            raise RuntimeError("Frosne folds har ikke præcis de samme eligible kampnøgler som v1.")
        historical = historical_sorted

    return historical.sort_values(["date", "time"], kind="stable").reset_index(drop=True), \
        later.sort_values(["date", "time"], kind="stable").reset_index(drop=True), \
        previous_oof, old_blocks


def _assert_v1_fold_keys(test: pd.DataFrame, previous_oof: pd.DataFrame, fold: int) -> None:
    reference = _keys(test).reset_index(drop=True)
    for variant in FOLD_VARIANTS:
        seeds = SEEDS if variant != "odds_only" else (None,)
        for seed in seeds:
            old = previous_oof[
                (previous_oof.fold == fold) & (previous_oof.variant == variant)
            ].copy()
            if seed is not None:
                old = old[pd.to_numeric(old.seed, errors="coerce") == seed]
            if not _keys(old).reset_index(drop=True).equals(reference):
                raise RuntimeError(
                    f"Testkampnøgler afviger fra v1 for {variant}, seed={seed}, fold={fold}."
                )


def _assert_v2_fold_keys(test: pd.DataFrame, collected: pd.DataFrame, fold: int) -> None:
    """Check ordered test keys for every NN seed and the single odds reference."""
    reference = _keys(test).reset_index(drop=True)
    for variant in ("baseline_27", "nn_with_odds"):
        for seed in SEEDS:
            group = collected[
                (collected.fold == fold)
                & (collected.variant == variant)
                & (pd.to_numeric(collected.seed, errors="coerce") == seed)
            ]
            if not _keys(group).reset_index(drop=True).equals(reference):
                raise RuntimeError(
                    f"Testnøgler afviger for {variant}, seed={seed}, fold={fold}."
                )
    odds = collected[(collected.fold == fold) & (collected.variant == "odds_only")]
    if not _keys(odds).reset_index(drop=True).equals(reference):
        raise RuntimeError(f"Odds-only-testnøgler afviger i fold {fold}.")


def _later_v1_artifact_predictions(later: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Evaluate the frozen v1 artifact before any v2 training includes these rows."""
    bundle = load_model_bundle(V1_ARTIFACT_DIR, device=DEVICE)
    scaled = bundle.preprocessor.transform(later[MODEL_FEATURES])
    inputs = torch.as_tensor(scaled.copy(), dtype=torch.float32, device=bundle.device)
    bundle.model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(bundle.model(inputs), dim=1).detach().cpu().numpy()
    probabilities = normalize_probabilities(probabilities)
    metrics = calculate_metrics(later.result.to_numpy(), probabilities)
    return probabilities, metrics


def _write_outputs(oof: pd.DataFrame, summary: pd.DataFrame) -> None:
    check_outputs_absent()
    handles = []
    created: list[Path] = []
    try:
        for path in (OOF_PATH, SUMMARY_PATH):
            handle = path.open("x", encoding="utf-8-sig", newline="")
            handles.append(handle)
            created.append(path)
        oof.to_csv(handles[0], index=False)
        summary.to_csv(handles[1], index=False)
        for handle in handles:
            handle.flush()
            handle.close()
    except Exception:
        for handle in handles:
            if not handle.closed:
                handle.close()
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    check_outputs_absent()
    print("Odds-caveat: historiske B365-priser mangler snapshot-tidspunkter; før-kickoff-tilgængelighed er uverificeret.")
    historical, later, previous_oof, old_blocks = _read_and_validate()
    print(f"Frosne v1-folds: {old_blocks[0][0]}..{old_blocks[-1][-1]}; historical rows={len(historical):,}")
    print(f"Adskilt nyere holdout: {len(later):,} kampe; datoer={later.date.min().date()}..{later.date.max().date()}")

    prediction_rows: list[dict] = []
    seed_fold_rows: list[dict] = []
    evaluated_folds: list[int] = []
    v1_folds = sorted(pd.to_numeric(previous_oof.fold, errors="coerce").dropna().astype(int).unique())
    for fold in range(1, 4):
        if fold not in v1_folds:
            print(f"Fold {fold} udeladt: findes ikke i den gennemførte v1-OOF-evaluering.")
            continue
        train_dates = [day for block in old_blocks[:fold] for day in block]
        test_dates = old_blocks[fold]
        train_dates = sorted(set(train_dates))
        split = int(len(train_dates) * 0.85)
        subtrain_dates, validation_dates = train_dates[:split], train_dates[split:]
        subtrain = historical[historical.date.dt.date.isin(subtrain_dates)].copy()
        validation = historical[historical.date.dt.date.isin(validation_dates)].copy()
        test = historical[historical.date.dt.date.isin(test_dates)].copy().sort_values(
            ["date", "time"], kind="stable"
        )
        if subtrain.empty or validation.empty or test.empty:
            raise RuntimeError(
                f"Fold {fold} har en tom subtrain/validation/test-del: "
                f"subtrain={len(subtrain)}, validation={len(validation)}, test={len(test)}"
            )
        evaluated_folds.append(fold)
        print(
            f"Fold {fold}: subtrain={len(subtrain):,}, validation={len(validation):,}, "
            f"test={len(test):,}, test_dates={test_dates[0]}..{test_dates[-1]}"
        )
        for seed in SEEDS:
            for variant, columns in (("baseline_27", NN_FEATURES), ("nn_with_odds", MODEL_FEATURES)):
                probabilities, validation_ce = train_variant(
                    subtrain, validation, test, columns, fold, seed
                )
                metrics = calculate_metrics(test.result.to_numpy(), probabilities)
                seed_fold_rows.append({
                    "scope": "seed_fold", "variant": variant, "seed": seed,
                    "fold": fold, "matches": len(test), "alpha": ALPHA,
                    "best_validation_ce": validation_ce, **metrics,
                })
                _append_predictions(prediction_rows, test, variant, seed, fold, probabilities)
        odds_probs = normalize_probabilities(test[ODDS_FEATURES].to_numpy(dtype=np.float64))
        _append_predictions(prediction_rows, test, "odds_only", np.nan, fold, odds_probs)
        seed_fold_rows.append({
            "scope": "seed_fold", "variant": "odds_only", "seed": "single",
            "fold": fold, "matches": len(test), "alpha": np.nan,
            **calculate_metrics(test.result.to_numpy(), odds_probs),
        })
        current = pd.DataFrame(prediction_rows)
        _assert_v2_fold_keys(test.reset_index(drop=True), current, fold)
        _assert_v1_fold_keys(test.reset_index(drop=True), previous_oof, fold)

    if evaluated_folds != v1_folds:
        raise RuntimeError(
            f"Evaluerede folds {evaluated_folds} afviger fra v1 OOF-folds {v1_folds}."
        )

    oof = pd.DataFrame(prediction_rows)
    summary_rows = list(seed_fold_rows)
    seed_metrics = pd.DataFrame(seed_fold_rows)
    for (variant, fold), group in seed_metrics[seed_metrics.variant != "odds_only"].groupby(["variant", "fold"]):
        summary_rows.append({
            "scope": "fold_seed_summary", "variant": variant, "seed": "all",
            "fold": fold, "matches": group.matches.mean(), "alpha": ALPHA,
            **{metric: group[metric].mean() for metric in METRICS},
            **{f"{metric}_std": group[metric].std(ddof=1) for metric in METRICS},
        })
    for variant in ("baseline_27", "nn_with_odds"):
        for seed in SEEDS:
            group = oof[(oof.variant == variant) & (oof.seed == seed)]
            summary_rows.append({
                "scope": "seed_overall", "variant": variant, "seed": seed,
                "fold": "all", "matches": len(group), "alpha": ALPHA,
                **calculate_metrics(group.actual.to_numpy(), group[["prob_H", "prob_D", "prob_A"]].to_numpy()),
            })
        rows = [row for row in summary_rows if row.get("scope") == "seed_overall" and row["variant"] == variant]
        summary_rows.append({
            "scope": "overall_seed_summary", "variant": variant, "seed": "all",
            "fold": "all", "matches": float(np.mean([row["matches"] for row in rows])),
            "alpha": ALPHA,
            **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
            **{f"{metric}_std": float(np.std([row[metric] for row in rows], ddof=1)) for metric in METRICS},
        })
    odds_oof = oof[oof.variant == "odds_only"]
    summary_rows.append({
        "scope": "overall_reference", "variant": "odds_only", "seed": "single",
        "fold": "all", "matches": len(odds_oof), "alpha": np.nan,
        **calculate_metrics(odds_oof.actual.to_numpy(), odds_oof[["prob_H", "prob_D", "prob_A"]].to_numpy()),
    })

    paired = _comparison_with_previous(previous_oof, oof)
    summary_rows.extend(paired)
    paired_frame = pd.DataFrame(paired)
    for (variant, fold), group in paired_frame.groupby(["variant", "fold"]):
        row = {"scope": "paired_v1_delta_mean", "variant": variant, "seed": "mean_over_seeds", "fold": fold, "matches": int(group.matches.min())}
        for metric in METRICS:
            row[f"delta_{metric}"] = group[f"delta_{metric}"].mean()
            row[f"delta_{metric}_std"] = group[f"delta_{metric}"].std(ddof=1)
        summary_rows.append(row)

    # This is an untouched later-period check of the saved v1 model, not a v2
    # final-model score. It is completed before final v2 training uses these rows.
    later_probs, later_metrics = _later_v1_artifact_predictions(later)
    for index, (_, match) in enumerate(later.iterrows()):
        prediction_rows.append({
            "variant": "v1_saved_artifact_later_holdout", "alpha": ALPHA,
            "seed": "saved_v1_model", "fold": "later_holdout_after_2026-10-12",
            "date": match.date, "time": match.time, "league": match.league,
            "home_team": match.home_team, "away_team": match.away_team,
            "actual": match.result, "predicted": INDEX_TO_LABEL[int(later_probs[index].argmax())],
            "prob_H": later_probs[index, 0], "prob_D": later_probs[index, 1], "prob_A": later_probs[index, 2],
        })
    summary_rows.append({
        "scope": "later_holdout_v1_saved_artifact", "variant": "v1_saved_artifact_later_holdout",
        "seed": "saved_v1_model", "fold": "after_2026-10-12", "matches": len(later),
        "alpha": ALPHA, **later_metrics,
        "note": "48 senere kampe evalueret med den gemte v1-model før de indgår i v2-finaletræning; oddsens før-kickoff-tidspunkt er ukendt.",
    })
    summary_rows.append({
        "scope": "dataset_check", "variant": "v2", "seed": "all", "fold": "all",
        "matches": len(historical) + len(later), "v1_frozen_fold_matches": len(historical),
        "later_holdout_matches": len(later), "frozen_fold_dates_match_v1": True,
    })
    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "odds_timestamp_caveat", "No documented bookmaker snapshot timestamps; pre-kickoff availability unverified")
    check_outputs_absent()
    _write_outputs(pd.DataFrame(prediction_rows), summary)
    print("V2 frozen-fold backtest and separate v1 later-holdout evaluation saved.")
    print(f"OOF: {OOF_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
