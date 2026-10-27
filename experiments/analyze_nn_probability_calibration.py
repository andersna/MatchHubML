"""Analyze calibration and decision rules using existing neural-net OOF data.

This script does not train models or tune thresholds. It reads only variants
A/B/C from the saved fold-2 and fold-3 OOF CSVs, verifies that their raw
probabilities match per match and seed, then uses one copy of each prediction.
For match-level calibration, probabilities are averaged over the five model
seeds for each match. Match-level bootstrap intervals resample matches only;
seed-to-seed standard deviations are reported separately as model-run
variability, not as independent match observations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS = {
    2: PROJECT_ROOT / "data" / "oof_predictions_nn_dynamic_threshold_fold2.csv",
    3: PROJECT_ROOT / "data" / "oof_predictions_nn_dynamic_threshold_isolation.csv",
}
ANALYSIS_PATH = PROJECT_ROOT / "data" / "nn_probability_calibration_analysis.csv"
BINS_PATH = PROJECT_ROOT / "data" / "nn_probability_calibration_bins.csv"
DECISIONS_PATH = PROJECT_ROOT / "data" / "nn_probability_decision_rules_comparison.csv"
OUTPUTS = (ANALYSIS_PATH, BINS_PATH, DECISIONS_PATH)

EXPECTED_SEEDS = (42, 123, 2026, 3407, 9876)
VARIANTS = (
    "A_baseline_argmax",
    "B_baseline_fixed_0.85",
    "C_27_features_dynamic_threshold",
)
LABELS = ("H", "D", "A")
PROB_COLUMNS = ("prob_H", "prob_D", "prob_A")
# Time is missing throughout fold 2 and for some fold-3 rows. These columns
# were verified to identify a unique match within each fold/seed.
MATCH_KEYS = ("date", "league", "home_team", "away_team")
REQUIRED_COLUMNS = set(
    ["variant", "seed", "fold", "actual", "predicted", *MATCH_KEYS,
     "goal_gap_cutoff", "low_draw_threshold", "high_draw_threshold",
     "abs_expected_goals_difference", *PROB_COLUMNS]
)
N_CALIBRATION_BINS = 10
N_BOOTSTRAPS = 2000
BOOTSTRAP_SEEDS = {2: 902102, 3: 902103}
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def check_outputs_absent():
    existing = [path for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError(
            "Analysefil findes allerede; intet blev overskrevet: "
            + ", ".join(str(path) for path in existing)
        )


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Forventede tre sandsynligheder pr. kamp.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("OOF-filen indeholder ugyldige sandsynligheder.")
    sums = probabilities.sum(axis=1, keepdims=True)
    if (sums <= 0).any() or not np.allclose(sums, 1.0, atol=1e-5, rtol=0):
        raise ValueError("Sandsynlighederne summerer ikke til 1.")
    return probabilities / sums


def read_fold(fold, path):
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} mangler nødvendige kolonner: {missing}")
    frame = frame[frame.variant.isin(VARIANTS)].copy()
    if frame.empty:
        raise ValueError(f"{path} indeholder ikke varianterne A/B/C.")
    folds = set(frame.fold.astype(int).unique())
    if folds != {fold}:
        raise ValueError(f"{path} skulle kun indeholde fold {fold}; fandt {folds}.")
    if set(frame.seed.astype(int).unique()) != set(EXPECTED_SEEDS):
        raise ValueError(f"{path} har ikke præcis de forventede fem seeds.")
    if not set(frame.actual.dropna().unique()).issubset(LABELS):
        raise ValueError(f"{path} har ukendte eller manglende faktiske udfald.")

    probs = normalize_probabilities(frame[list(PROB_COLUMNS)].to_numpy())
    frame.loc[:, list(PROB_COLUMNS)] = probs
    # Use a stable textual key representation, including explicit missing
    # values, for exact one-to-one comparisons across variants.
    for column in MATCH_KEYS:
        frame[column] = frame[column].astype("string").fillna("")
    frame["seed"] = frame.seed.astype(int)
    frame["fold"] = frame.fold.astype(int)
    frame = frame.sort_values(["seed", *MATCH_KEYS, "variant"], kind="stable")

    for variant in VARIANTS:
        subset = frame[frame.variant == variant]
        if subset.duplicated(["seed", *MATCH_KEYS]).any():
            raise ValueError(
                f"Dobbelt kampnøgle for {variant}, fold {fold}; kan ikke deduplikere sikkert."
            )
    reference = frame[frame.variant == VARIANTS[0]].sort_values(
        ["seed", *MATCH_KEYS], kind="stable"
    ).reset_index(drop=True)
    compare_columns = ["seed", *MATCH_KEYS, "actual", *PROB_COLUMNS]
    for variant in VARIANTS[1:]:
        candidate = frame[frame.variant == variant].sort_values(
            ["seed", *MATCH_KEYS], kind="stable"
        ).reset_index(drop=True)
        if not reference[compare_columns].equals(candidate[compare_columns]):
            raise ValueError(
                f"A/B/C kampnøgler, udfald eller sandsynligheder afviger i fold {fold}."
            )

    reference_counts = reference.groupby("seed", dropna=False).size()
    if reference_counts.nunique() != 1:
        raise ValueError(f"Seedene har ikke samme antal testkampe i fold {fold}.")
    if not reference.groupby(list(MATCH_KEYS), dropna=False).size().eq(len(EXPECTED_SEEDS)).all():
        raise ValueError(f"Ikke alle fold {fold}-kampe har præcis fem seed-forudsigelser.")
    if not reference.groupby(list(MATCH_KEYS), dropna=False).actual.nunique().eq(1).all():
        raise ValueError(f"Et kampresultat varierer mellem seeds i fold {fold}.")
    return frame, reference


def multiclass_probability_metrics(actual, probabilities):
    probabilities = normalize_probabilities(probabilities)
    indices = np.asarray([LABEL_TO_INDEX[value] for value in actual], dtype=int)
    log_loss = float(-np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-15, 1)).mean())
    one_hot = np.eye(3, dtype=np.float64)[indices]
    brier = float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1)))
    return log_loss, brier


def decision_metrics(actual, predicted, probabilities):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    log_loss, brier = multiclass_probability_metrics(actual, probabilities)
    result = {
        "accuracy": float(np.mean(actual == predicted)),
        "macro_f1": np.nan,
        "log_loss": log_loss,
        "brier_score": brier,
    }
    f1_values = []
    for label in LABELS:
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"recall_{label}"] = float(recall)
        result[f"share_pred_{label}"] = float(np.mean(predicted == label))
        f1_values.append(f1)
    result["macro_f1"] = float(np.mean(f1_values))
    return result


def reconstruct_predictions(frame, variant):
    probabilities = frame[list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
    home_away_best = np.maximum(probabilities[:, 0], probabilities[:, 2])
    if variant == VARIANTS[0]:
        return np.asarray(LABELS)[probabilities.argmax(axis=1)]
    if variant == VARIANTS[1]:
        threshold = 0.85
        return np.where(
            probabilities[:, 1] >= threshold * home_away_best,
            "D",
            np.where(probabilities[:, 0] >= probabilities[:, 2], "H", "A"),
        )
    if variant == VARIANTS[2]:
        gap = frame.abs_expected_goals_difference.to_numpy(dtype=float)
        cutoff = frame.goal_gap_cutoff.to_numpy(dtype=float)
        low = frame.low_draw_threshold.to_numpy(dtype=float)
        high = frame.high_draw_threshold.to_numpy(dtype=float)
        if not (np.isfinite(gap).all() and np.isfinite(cutoff).all()
                and np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError("Foldens dynamiske threshold-parametre eller goal gap mangler.")
        threshold = np.where(gap <= cutoff, low, high)
        return np.where(
            probabilities[:, 1] >= threshold * home_away_best,
            "D",
            np.where(probabilities[:, 0] >= probabilities[:, 2], "H", "A"),
        )
    raise ValueError(f"Ukendt variant: {variant}")


def bootstrap_interval(actual, probabilities, draws, rng):
    """Percentile intervals with matches as the only resampling unit."""
    actual = np.asarray(actual)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    draws = np.asarray(draws, dtype=bool)
    n = len(actual)
    draw_differences, losses, briers = [], [], []
    indices = np.arange(n)
    actual_idx = np.asarray([LABEL_TO_INDEX[value] for value in actual], dtype=int)
    one_hot = np.eye(3, dtype=np.float64)[actual_idx]
    for _ in range(N_BOOTSTRAPS):
        sample = rng.choice(indices, size=n, replace=True)
        sample_draws = draws[sample]
        sample_non_draws = ~sample_draws
        # A bootstrap sample of this size will practically always contain both;
        # skip only pathological samples rather than inventing a value.
        if not sample_draws.any() or not sample_non_draws.any():
            continue
        draw_differences.append(
            probabilities[sample[sample_draws], 1].mean()
            - probabilities[sample[sample_non_draws], 1].mean()
        )
        sampled_actual_idx = actual_idx[sample]
        sampled_probs = probabilities[sample]
        losses.append(float(-np.log(np.clip(
            sampled_probs[np.arange(n), sampled_actual_idx], 1e-15, 1.0
        )).mean()))
        briers.append(float(np.mean(np.sum((one_hot[sample] - sampled_probs) ** 2, axis=1))))

    def interval(values):
        return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))

    return {
        "draw_probability_difference": interval(draw_differences),
        "log_loss": interval(losses),
        "brier_score": interval(briers),
        "valid_replicates": len(losses),
    }


def analyze_fold(fold, full_frame, reference, analysis_rows, bin_rows, decision_rows):
    # Keep one A record per match/seed; B/C are validated then discarded here.
    base = reference.copy().sort_values([*MATCH_KEYS, "seed"], kind="stable")
    match_level = base.groupby(list(MATCH_KEYS), as_index=False, dropna=False).agg(
        actual=("actual", "first"),
        prob_H=("prob_H", "mean"),
        prob_D=("prob_D", "mean"),
        prob_A=("prob_A", "mean"),
        seed_count=("seed", "nunique"),
    )
    if not match_level.seed_count.eq(len(EXPECTED_SEEDS)).all():
        raise ValueError(f"Fold {fold}: seed-aggregation check failed.")
    p_match = normalize_probabilities(match_level[list(PROB_COLUMNS)].to_numpy())
    y_match = match_level.actual.to_numpy()
    actual_shares = {label: float(np.mean(y_match == label)) for label in LABELS}
    predicted_means = {label: float(p_match[:, i].mean()) for i, label in enumerate(LABELS)}
    for label in LABELS:
        analysis_rows.append({
            "fold": fold, "scope": "match_seed_average", "metric": "class_frequency_vs_probability",
            "class": label, "seed": "all", "n_matches": len(match_level),
            "actual_frequency": actual_shares[label], "mean_predicted_probability": predicted_means[label],
            "value": predicted_means[label] - actual_shares[label],
            "ci_lower": np.nan, "ci_upper": np.nan, "seed_std": np.nan,
        })

    match_log_loss, match_brier = multiclass_probability_metrics(y_match, p_match)
    rng = np.random.default_rng(BOOTSTRAP_SEEDS[fold])
    draw_mask = y_match == "D"
    draw_difference = float(p_match[draw_mask, 1].mean() - p_match[~draw_mask, 1].mean())
    ci = bootstrap_interval(y_match, p_match, draw_mask, rng)
    for metric, value, bounds in (
        ("log_loss_match_seed_average", match_log_loss, ci["log_loss"]),
        ("brier_score_match_seed_average", match_brier, ci["brier_score"]),
        ("prob_D_draw_minus_non_draw", draw_difference, ci["draw_probability_difference"]),
    ):
        analysis_rows.append({
            "fold": fold, "scope": "match_bootstrap", "metric": metric,
            "class": "D" if metric == "prob_D_draw_minus_non_draw" else "all",
            "seed": "all", "n_matches": len(match_level), "actual_frequency": np.nan,
            "mean_predicted_probability": np.nan, "value": value,
            "ci_lower": bounds[0], "ci_upper": bounds[1],
            "seed_std": np.nan, "bootstrap_replicates": ci["valid_replicates"],
            "bootstrap_seed": BOOTSTRAP_SEEDS[fold],
        })
    for condition, mask in (("actual_draw", draw_mask), ("actual_non_draw", ~draw_mask)):
        analysis_rows.append({
            "fold": fold, "scope": "match_seed_average", "metric": "mean_prob_D_by_actual_outcome",
            "class": condition, "seed": "all", "n_matches": int(mask.sum()),
            "actual_frequency": np.nan, "mean_predicted_probability": float(p_match[mask, 1].mean()),
            "value": float(p_match[mask, 1].mean()), "ci_lower": np.nan,
            "ci_upper": np.nan, "seed_std": np.nan,
        })

    # Class-wise equal-count calibration bins on match-level seed-averaged p.
    n_bins = min(N_CALIBRATION_BINS, len(match_level))
    for i, label in enumerate(LABELS):
        sort_order = np.argsort(p_match[:, i], kind="stable")
        bin_id = np.empty(len(match_level), dtype=int)
        bin_id[sort_order] = np.minimum(
            np.arange(len(match_level)) * n_bins // len(match_level), n_bins - 1
        )
        for b in range(n_bins):
            selected = bin_id == b
            bin_rows.append({
                "fold": fold, "class": label, "bin": b + 1,
                "n_matches": int(selected.sum()),
                "mean_predicted_probability": float(p_match[selected, i].mean()),
                "actual_frequency": float(np.mean(y_match[selected] == label)),
                "min_predicted_probability": float(p_match[selected, i].min()),
                "max_predicted_probability": float(p_match[selected, i].max()),
            })

    # Seed variability is reported separately; each seed is evaluated on the
    # same match set and is not added to the match-level bootstrap sample size.
    seed_probability_rows = []
    for seed, group in base.groupby("seed", sort=True):
        probabilities = group[list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
        actual = group.actual.to_numpy()
        ll, bs = multiclass_probability_metrics(actual, probabilities)
        draws = actual == "D"
        row = {"seed": int(seed), "log_loss": ll, "brier_score": bs,
               "mean_prob_D_draw": float(probabilities[draws, 1].mean()),
               "mean_prob_D_non_draw": float(probabilities[~draws, 1].mean())}
        seed_probability_rows.append(row)
        for metric, value in row.items():
            if metric == "seed":
                continue
            analysis_rows.append({
                "fold": fold, "scope": "per_seed", "metric": metric,
                "class": "D" if "prob_D" in metric else "all", "seed": int(seed),
                "n_matches": group[list(MATCH_KEYS)].drop_duplicates().shape[0],
                "actual_frequency": np.nan, "mean_predicted_probability": np.nan,
                "value": value, "ci_lower": np.nan, "ci_upper": np.nan,
                "seed_std": np.nan,
            })
    seed_prob_df = pd.DataFrame(seed_probability_rows)
    for metric in ("log_loss", "brier_score", "mean_prob_D_draw", "mean_prob_D_non_draw"):
        analysis_rows.append({
            "fold": fold, "scope": "seed_mean_and_sd", "metric": metric,
            "class": "D" if "prob_D" in metric else "all", "seed": "all",
            "n_matches": len(match_level), "actual_frequency": np.nan,
            "mean_predicted_probability": np.nan, "value": float(seed_prob_df[metric].mean()),
            "ci_lower": np.nan, "ci_upper": np.nan, "seed_std": float(seed_prob_df[metric].std(ddof=1)),
        })

    # Reconstruct each existing decision rule and compare on the identical
    # per-seed test matches. This verifies rule semantics without retuning.
    for seed in EXPECTED_SEEDS:
        seed_frame = full_frame[full_frame.seed == seed]
        seed_metrics = {}
        for variant in VARIANTS:
            group = seed_frame[seed_frame.variant == variant].sort_values(
                list(MATCH_KEYS), kind="stable"
            ).reset_index(drop=True)
            reconstructed = reconstruct_predictions(group, variant)
            if not np.array_equal(reconstructed, group.predicted.to_numpy()):
                raise ValueError(f"Fold {fold}, seed {seed}: {variant} label rule cannot be reconstructed.")
            metrics = decision_metrics(
                group.actual.to_numpy(), reconstructed,
                group[list(PROB_COLUMNS)].to_numpy(dtype=np.float64),
            )
            seed_metrics[variant] = metrics
            decision_rows.append({
                "fold": fold, "scope": "per_seed_rule", "variant": variant,
                "seed": seed, "n_matches": len(group), **metrics,
            })
        for comparison, left, right in (
            ("B_minus_A", VARIANTS[1], VARIANTS[0]),
            ("C_minus_A", VARIANTS[2], VARIANTS[0]),
        ):
            delta = {name: seed_metrics[left][name] - seed_metrics[right][name]
                     for name in seed_metrics[left]}
            decision_rows.append({
                "fold": fold, "scope": "paired_delta_per_seed", "variant": comparison,
                "seed": seed, "n_matches": len(seed_frame[seed_frame.variant == VARIANTS[0]]),
                **delta,
            })

    decision_frame = pd.DataFrame([
        row for row in decision_rows if row["fold"] == fold
        and row["scope"] == "per_seed_rule"
    ])
    for variant, group in decision_frame.groupby("variant", sort=True):
        metric_names = [name for name in decision_metrics(
            np.array(["H"]), np.array(["H"]), np.array([[1.0, 0.0, 0.0]])
        )]
        summary = {name: float(group[name].mean()) for name in metric_names}
        summary.update({f"{name}_std": float(group[name].std(ddof=1)) for name in metric_names})
        decision_rows.append({
            "fold": fold, "scope": "seed_mean_and_sd", "variant": variant,
            "seed": "all", "n_matches": int(group.n_matches.mean()), **summary,
        })
    delta_frame = pd.DataFrame([
        row for row in decision_rows if row["fold"] == fold
        and row["scope"] == "paired_delta_per_seed"
    ])
    for comparison, group in delta_frame.groupby("variant", sort=True):
        metric_names = [name for name in decision_metrics(
            np.array(["H"]), np.array(["H"]), np.array([[1.0, 0.0, 0.0]])
        )]
        summary = {name: float(group[name].mean()) for name in metric_names}
        summary.update({f"{name}_std": float(group[name].std(ddof=1)) for name in metric_names})
        decision_rows.append({
            "fold": fold, "scope": "paired_delta_mean_and_sd", "variant": comparison,
            "seed": "all", "n_matches": int(group.n_matches.mean()), **summary,
        })


def write_new_outputs(analysis_frame, bins_frame, decisions_frame):
    check_outputs_absent()
    handles = {}
    try:
        for path in OUTPUTS:
            handles[path] = path.open("x", newline="", encoding="utf-8-sig")
        for path, frame in zip(OUTPUTS, (analysis_frame, bins_frame, decisions_frame)):
            frame.to_csv(handles[path], index=False)
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in handles:
            if path.exists():
                path.unlink()
        raise
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()


def main():
    # Check all destinations before reading inputs and before any output write.
    check_outputs_absent()
    analysis_rows, bin_rows, decision_rows = [], [], []
    for fold, input_path in INPUTS.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Mangler eksisterende OOF-fil: {input_path}")
        full_frame, reference = read_fold(fold, input_path)
        analyze_fold(fold, full_frame, reference, analysis_rows, bin_rows, decision_rows)

    analysis_frame = pd.DataFrame(analysis_rows)
    bins_frame = pd.DataFrame(bin_rows)
    decisions_frame = pd.DataFrame(decision_rows)
    write_new_outputs(analysis_frame, bins_frame, decisions_frame)
    print("Kalibreringsanalyse gemt:", ANALYSIS_PATH)
    print("Kalibreringsbins gemt:", BINS_PATH)
    print("Beslutningsregelsammenligning gemt:", DECISIONS_PATH)
    print("Bootstrapen resamplede kampe; seeds blev kun opsummeret som modelvariation.")
    print("Log loss og Brier score er beregnet fra sandsynlighederne, ikke labels.")


if __name__ == "__main__":
    main()
