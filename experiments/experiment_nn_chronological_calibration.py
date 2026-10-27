"""Chronological post-hoc calibration of existing repeated-seed OOF predictions.

This script does not train models. It fits one scalar temperature per seed/alpha
using only earlier OOF folds, then evaluates on the next fold. Temperature
scaling works directly from saved probabilities by applying softmax(log(p) / T),
so logits are not required. A single parameter is a reasonable low-variance
choice for these grouped multiclass data; more flexible calibration methods
would be easier to overfit. Fold 1 is calibration data only and is never scored.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "oof_predictions_nn_repeated_seeds_three_weights.csv"
OOF_OUTPUT_PATH = ROOT / "data" / "oof_predictions_nn_chronological_calibration.csv"
SUMMARY_OUTPUT_PATH = ROOT / "data" / "nn_chronological_calibration_summary.csv"

LABELS = ["H", "D", "A"]
PROB_COLUMNS = ["prob_H", "prob_D", "prob_A"]
KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]
REQUIRED_COLUMNS = [
    "seed", "alpha", "fold", *KEY_COLUMNS, "actual", "predicted", *PROB_COLUMNS
]
EPSILON = 1e-12


def check_outputs_do_not_exist():
    existing = [str(path) for path in (OOF_OUTPUT_PATH, SUMMARY_OUTPUT_PATH) if path.exists()]
    if existing:
        raise FileExistsError(
            "Afviser at overskrive eksisterende outputfil(er): " + ", ".join(existing)
        )


def validate_and_normalize(probabilities, context):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(f"{context}: forventede en Nx3 sandsynlighedsmatrice.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{context}: sandsynligheder indeholder NaN eller uendelige værdier.")
    if (probabilities < -1e-8).any() or (probabilities > 1.0 + 1e-8).any():
        raise ValueError(f"{context}: sandsynligheder ligger uden for [0, 1].")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{context}: rækkerne summerer ikke til 1.")
    # Clipping only protects log() and preserves a normalized valid distribution.
    probabilities = np.clip(probabilities, EPSILON, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def temperature_transform(probabilities, log_temperature):
    logits = np.log(np.clip(probabilities, EPSILON, 1.0)) / np.exp(log_temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, actual):
    """Minimize multiclass NLL on prior folds with bounded 1-D golden search."""
    actual_indices = np.array([LABELS.index(value) for value in actual], dtype=np.int64)
    lo, hi = np.log(0.05), np.log(20.0)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)

    def objective(log_temperature):
        calibrated = temperature_transform(probabilities, log_temperature)
        return -np.mean(np.log(calibrated[np.arange(len(actual_indices)), actual_indices]))

    f_left, f_right = objective(left), objective(right)
    for _ in range(100):
        if f_left <= f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo)
            f_left = objective(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo)
            f_right = objective(right)
    return float(np.exp((lo + hi) / 2.0))


def calculate_metrics(actual, probabilities):
    probabilities = validate_and_normalize(probabilities, "metrikberegning")
    actual = np.asarray(actual)
    actual_indices = np.array([LABELS.index(value) for value in actual], dtype=np.int64)
    predicted_indices = probabilities.argmax(axis=1)
    predicted = np.asarray(LABELS)[predicted_indices]
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (actual_indices, predicted_indices), 1)
    true_positive = np.diag(confusion)
    denominators = confusion.sum(axis=1) + confusion.sum(axis=0)
    class_f1 = np.divide(
        2.0 * true_positive,
        denominators,
        out=np.zeros(3, dtype=np.float64),
        where=denominators != 0,
    )
    one_hot = np.eye(3, dtype=np.float64)[actual_indices]
    metrics = {
        "accuracy": float(np.mean(predicted_indices == actual_indices)),
        "macro_f1": float(class_f1.mean()),
        "log_loss": float(-np.mean(np.log(probabilities[np.arange(len(actual_indices)), actual_indices]))),
        # Same multiclass Brier definition as previous experiment scripts:
        # mean over rows of the sum of squared one-hot probability errors.
        "brier_score": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
    }
    return metrics, predicted, probabilities


def key_frame(frame):
    result = frame[KEY_COLUMNS].copy()
    # Normalize missing time values and mixed CSV dtypes before exact comparison.
    for column in KEY_COLUMNS:
        result[column] = result[column].astype("string").fillna("")
    return result.reset_index(drop=True)


def validate_input(frame):
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input mangler kolonner: {missing}")
    if frame.empty:
        raise ValueError("Inputfilen indeholder ingen OOF-rækker.")
    frame = frame.copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise")
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="raise")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    if not set(frame["actual"].dropna().unique()).issubset(LABELS):
        raise ValueError("Input indeholder ukendte eller manglende faktiske klasser.")
    if frame["actual"].isna().any():
        raise ValueError("Input indeholder manglende faktiske klasser.")
    frame[PROB_COLUMNS] = frame[PROB_COLUMNS].apply(pd.to_numeric, errors="raise")
    validate_and_normalize(frame[PROB_COLUMNS].to_numpy(), "inputfil")
    folds = sorted(frame["fold"].unique().tolist())
    if folds != [1, 2, 3]:
        raise ValueError(f"Forventede fold 1, 2 og 3; fandt {folds}.")

    reference = None
    reference_actual = None
    for (seed, alpha, fold), group in frame.groupby(["seed", "alpha", "fold"], sort=False):
        keys = key_frame(group)
        if keys.duplicated().any():
            raise ValueError(f"Gentagne kampnøgler for seed={seed}, alpha={alpha}, fold={fold}.")
        actual = group["actual"].reset_index(drop=True)
        if reference is None:
            reference, reference_actual = {}, {}
        group_key = (fold,)
        if group_key not in reference:
            reference[group_key] = keys
            reference_actual[group_key] = actual
        elif not keys.equals(reference[group_key]) or not actual.equals(reference_actual[group_key]):
            raise ValueError(
                "Testkampenes nøgler, rækkefølge eller faktiske udfald afviger "
                f"mellem seed/alpha-grupper ved fold {fold}."
            )
    expected = {(seed, alpha, fold) for seed in frame["seed"].unique()
                for alpha in frame["alpha"].unique() for fold in (1, 2, 3)}
    actual_groups = set(map(tuple, frame[["seed", "alpha", "fold"]].drop_duplicates().to_numpy()))
    if actual_groups != expected:
        raise ValueError("Seed/alpha-grupperne har ikke alle de samme tre folds.")
    return frame


def main():
    check_outputs_do_not_exist()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"OOF-inputfilen blev ikke fundet: {INPUT_PATH}")
    data = validate_input(pd.read_csv(INPUT_PATH))
    oof_rows = []
    summary_rows = []

    for (seed, alpha), group in data.groupby(["seed", "alpha"], sort=True):
        group = group.sort_values("fold", kind="stable")
        for eval_fold in (2, 3):
            eval_frame = group[group["fold"] == eval_fold].reset_index(drop=True)
            calibration_frame = group[group["fold"] < eval_fold].reset_index(drop=True)
            if eval_frame.empty or calibration_frame.empty:
                raise ValueError(f"Tomt evaluerings- eller kalibreringssæt: seed={seed}, alpha={alpha}.")
            calibration_probs = validate_and_normalize(
                calibration_frame[PROB_COLUMNS].to_numpy(), "kalibreringsdata"
            )
            eval_probs = validate_and_normalize(eval_frame[PROB_COLUMNS].to_numpy(), "evalueringsdata")
            temperature = fit_temperature(calibration_probs, calibration_frame["actual"].to_numpy())
            calibrated_probs = temperature_transform(eval_probs, np.log(temperature))
            calibrated_probs = validate_and_normalize(calibrated_probs, "kalibrerede sandsynligheder")

            # Both probability matrices are derived row-for-row from this exact
            # evaluation frame; explicitly verify their row counts and labels.
            if len(eval_probs) != len(calibrated_probs):
                raise RuntimeError("Kampidentiteten ændrede sig mellem før/efter-beregningen.")
            before_keys = key_frame(eval_frame)
            after_frame = eval_frame.iloc[: len(calibrated_probs)].reset_index(drop=True)
            after_keys = key_frame(after_frame)
            if not before_keys.equals(after_keys) or not np.array_equal(
                eval_frame["actual"].to_numpy(), after_frame["actual"].to_numpy()
            ):
                raise RuntimeError("Kampnøgler eller faktiske udfald afviger før/efter kalibrering.")
            before_metrics, pred_before, before_probs = calculate_metrics(eval_frame["actual"], eval_probs)
            after_metrics, pred_after, after_probs = calculate_metrics(eval_frame["actual"], calibrated_probs)
            calibration_folds = ",".join(map(str, sorted(calibration_frame["fold"].unique())))
            summary = {
                "scope": "seed_alpha_eval_fold",
                "seed": seed,
                "alpha": alpha,
                "eval_fold": eval_fold,
                "calibration_folds": calibration_folds,
                "temperature": temperature,
                "n_games": len(eval_frame),
            }
            for metric in ("accuracy", "macro_f1", "log_loss", "brier_score"):
                summary[f"{metric}_before"] = before_metrics[metric]
                summary[f"{metric}_after"] = after_metrics[metric]
            summary_rows.append(summary)

            for index, row in eval_frame.iterrows():
                item = {column: row[column] for column in ["seed", "alpha", "fold", *KEY_COLUMNS, "actual"]}
                item.update({
                    "calibration_folds": calibration_folds,
                    "temperature": temperature,
                    "predicted_before": pred_before[index],
                    "predicted_after": pred_after[index],
                })
                for class_index, label in enumerate(LABELS):
                    item[f"prob_{label}_before"] = before_probs[index, class_index]
                    item[f"prob_{label}_after"] = after_probs[index, class_index]
                oof_rows.append(item)

    summary_df = pd.DataFrame(summary_rows)
    # Descriptive seed averages; repeated seeds reuse the same matches and are not
    # independent observations. These rows must not be interpreted as such.
    metric_names = ["accuracy", "macro_f1", "log_loss", "brier_score"]
    aggregate_rows = []
    for (alpha, eval_fold), subset in summary_df.groupby(["alpha", "eval_fold"], sort=True):
        row = {
            "scope": "alpha_eval_fold_mean_across_seeds",
            "seed": "mean_across_seeds",
            "alpha": alpha,
            "eval_fold": eval_fold,
            "calibration_folds": "descriptive mean; repeated seeds share games",
            "temperature": subset["temperature"].mean(),
            "n_games": int(subset["n_games"].iloc[0]),
        }
        for metric in metric_names:
            for stage in ("before", "after"):
                values = subset[f"{metric}_{stage}"]
                row[f"{metric}_{stage}"] = values.mean()
                row[f"{metric}_{stage}_std_across_seeds"] = values.std(ddof=1)
        aggregate_rows.append(row)
    for alpha, subset in summary_df.groupby("alpha", sort=True):
        row = {
            "scope": "alpha_overall_mean_across_seeds",
            "seed": "mean_across_seeds",
            "alpha": alpha,
            "eval_fold": "2+3",
            "calibration_folds": "fold 1->2; folds 1+2->3",
            "temperature": np.nan,
            "n_games": int(
                subset[["eval_fold", "n_games"]].drop_duplicates()["n_games"].sum()
            ),
        }
        # Compute pooled-by-seed metrics from concatenated OOF rows, then summarize
        # those five seed-level values; do not count repeated seeds as new games.
        alpha_oof = pd.DataFrame(oof_rows)
        alpha_oof = alpha_oof[alpha_oof["alpha"] == alpha]
        for metric in metric_names:
            per_seed = []
            for seed, seed_rows in alpha_oof.groupby("seed", sort=True):
                for stage in ("before", "after"):
                    probs = seed_rows[[f"prob_{label}_{stage}" for label in LABELS]].to_numpy()
                    metrics, _, _ = calculate_metrics(seed_rows["actual"], probs)
                    per_seed.append((seed, stage, metrics[metric]))
            for stage in ("before", "after"):
                values = [value for _, current_stage, value in per_seed if current_stage == stage]
                row[f"{metric}_{stage}"] = float(np.mean(values))
                row[f"{metric}_{stage}_std_across_seeds"] = float(np.std(values, ddof=1))
        aggregate_rows.append(row)
    summary_df = pd.concat([summary_df, pd.DataFrame(aggregate_rows)], ignore_index=True, sort=False)
    oof_df = pd.DataFrame(oof_rows)

    # Final collision check immediately before either new file is written.
    check_outputs_do_not_exist()
    oof_handle = None
    summary_handle = None
    try:
        # Exclusive creation is the final no-overwrite guard, including races
        # where a file appears after check_outputs_do_not_exist().
        oof_handle = OOF_OUTPUT_PATH.open("x", newline="", encoding="utf-8")
        summary_handle = SUMMARY_OUTPUT_PATH.open("x", newline="", encoding="utf-8")
        oof_df.to_csv(oof_handle, index=False)
        summary_df.to_csv(summary_handle, index=False)
    except Exception:
        if oof_handle is not None:
            oof_handle.close()
        if summary_handle is not None:
            summary_handle.close()
        # Remove only a file exclusively created by this run if paired creation fails.
        if oof_handle is not None and OOF_OUTPUT_PATH.exists():
            OOF_OUTPUT_PATH.unlink()
        if summary_handle is not None and SUMMARY_OUTPUT_PATH.exists():
            SUMMARY_OUTPUT_PATH.unlink()
        raise
    finally:
        if oof_handle is not None and not oof_handle.closed:
            oof_handle.close()
        if summary_handle is not None and not summary_handle.closed:
            summary_handle.close()

    print("Kronologisk kalibrering: fold 1 er kun kalibrering og evalueres ikke.")
    print("Seeds gentager de samme kampe; gennemsnit på tværs af seeds er deskriptive, ikke uafhængige stikprøver.")
    print("Metrikretning: accuracy og Macro F1 højere er bedre; log loss og Brier score lavere er bedre.")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"OOF-resultater: {OOF_OUTPUT_PATH}")
    print(f"Sammenfatning: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fejl: {exc}", file=sys.stderr)
        raise
