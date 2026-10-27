"""Evaluate one draw threshold fixed from fold-1 validation data.

The model and ratio rule are copied from experiment_nn_draw_threshold.py. A
single threshold is selected by mean validation Macro F1 across the five seeds
on fold 1 only, then frozen for all seeds and all three test folds. The rule is
P(D) >= threshold * max(P(H), P(A)); otherwise it chooses the larger of H/A.
It changes predicted labels only; probabilities remain the model's outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_fixed_draw_threshold.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_fixed_draw_threshold_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHA = 0.25
# Exactly the same candidate ratios and rule as experiment_nn_draw_threshold.py.
DRAW_THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
ARGMAX_RULE = "argmax"
FIXED_RULE = "fixed_draw_threshold"
RULES = (ARGMAX_RULE, FIXED_RULE)
LABELS = ["H", "D", "A"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = np.asarray(LABELS)
KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]

BASE_FEATURES = [
    "home_avg_goals_for", "home_avg_goals_against", "home_avg_points",
    "away_avg_goals_for", "away_avg_goals_against", "away_avg_points",
    "home_last5_goals_for", "home_last5_goals_against", "home_last5_points",
    "away_last5_goals_for", "away_last5_goals_against", "away_last5_points",
]
ATTACK_DEFENSE_FEATURES = [
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "home_expected_goals", "away_expected_goals",
]
COMMON_FEATURES = [
    "home_avg_goals_for", "home_avg_goals_against", "home_avg_points",
    "away_avg_goals_for", "away_avg_goals_against", "away_avg_points",
    "home_last5_goals_for", "home_last5_goals_against", "home_last5_points",
    "away_last5_goals_for", "away_last5_goals_against", "away_last5_points",
    "home_form_rating", "away_form_rating",
    "home_venue_form_rating", "away_venue_form_rating",
    "home_attack_strength", "away_attack_strength",
    "home_defense_strength", "away_defense_strength",
    "home_expected_goals", "away_expected_goals",
]
ELO_COLUMNS = [
    f"{prefix}_elo_k{k}"
    for k in [16, 32, 48]
    for prefix in ["home", "away"]
] + [f"elo_difference_k{k}" for k in [16, 32, 48]]
NN_FEATURES = BASE_FEATURES + ATTACK_DEFENSE_FEATURES + ELO_COLUMNS

METRIC_NAMES = [
    "accuracy", "macro_f1", "log_loss", "brier_score",
    "precision_H", "recall_H", "f1_H", "precision_D", "recall_D", "f1_D",
    "precision_A", "recall_A", "f1_A",
    "share_pred_H", "share_pred_D", "share_pred_A",
]
REPORT_METRICS = [
    "accuracy", "macro_f1", "log_loss", "brier_score",
    "precision_H", "recall_H", "precision_D", "recall_D",
    "precision_A", "recall_A", "share_pred_H", "share_pred_D", "share_pred_A",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchNet(nn.Module):
    """The existing baseline network: 27 -> 32 -> 16 -> 3."""

    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.layers(x)


def check_outputs_absent():
    existing = [path for path in (OOF_PATH, SUMMARY_PATH) if path.exists()]
    if existing:
        raise FileExistsError(
            "Resultatfil findes allerede; intet blev overskrevet: "
            + ", ".join(str(path) for path in existing)
        )


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        raise ValueError("Ugyldige sandsynligheder blev produceret.")
    return probabilities / row_sums


def make_alpha_025_weights(y_subtrain):
    """Same alpha=.25 inverse-frequency weights and missing-class handling."""
    labels = np.asarray(y_subtrain, dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    present = counts > 0
    baseline = np.zeros(3, dtype=np.float64)
    baseline[present] = len(labels) / (present.sum() * counts[present])
    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline[present] ** ALPHA
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def threshold_predictions(probabilities, threshold):
    """Same label-only decision rule as the per-fold threshold experiment."""
    probabilities = np.asarray(probabilities)
    top_non_draw = np.maximum(probabilities[:, 0], probabilities[:, 2])
    return np.where(
        probabilities[:, 1] >= threshold * top_non_draw,
        "D",
        np.where(probabilities[:, 0] >= probabilities[:, 2], "H", "A"),
    )


def validation_candidate_score(actual, probabilities, threshold):
    predicted = threshold_predictions(probabilities, threshold)
    return float(f1_score(
        actual, predicted, labels=LABELS, average="macro", zero_division=0
    ))


def calculate_metrics(actual, predictions, probabilities):
    """Class metrics use labels; probability metrics use original probabilities."""
    probabilities = normalize_probabilities(probabilities)
    actual = np.asarray(actual)
    actual_indices = np.asarray([LABEL_TO_INDEX[value] for value in actual])
    precision, recall, class_f1, _ = precision_recall_fscore_support(
        actual, predictions, labels=LABELS, zero_division=0
    )
    one_hot = np.eye(3, dtype=np.float64)[actual_indices]
    metrics = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(f1_score(
            actual, predictions, labels=LABELS,
            average="macro", zero_division=0,
        )),
        "log_loss": float(log_loss(actual_indices, probabilities, labels=[0, 1, 2])),
        # Same multiclass Brier definition used by the existing experiments.
        "brier_score": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
    }
    for index, label in enumerate(LABELS):
        metrics[f"precision_{label}"] = float(precision[index])
        metrics[f"recall_{label}"] = float(recall[index])
        metrics[f"f1_{label}"] = float(class_f1[index])
        metrics[f"share_pred_{label}"] = float(np.mean(predictions == label))
    return metrics


def train_fold_model(subtrain, validation, test, fold, seed, class_weights):
    # Preprocessing fits on subtrain only and transforms validation/test.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_subtrain = scaler.fit_transform(imputer.fit_transform(subtrain[NN_FEATURES]))
    x_validation = scaler.transform(imputer.transform(validation[NN_FEATURES]))
    x_test = scaler.transform(imputer.transform(test[NN_FEATURES]))
    y_subtrain = subtrain.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    y_validation = validation.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)

    x_subtrain_tensor = torch.tensor(x_subtrain, dtype=torch.float32, device=DEVICE)
    y_subtrain_tensor = torch.tensor(y_subtrain, dtype=torch.long, device=DEVICE)
    x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32, device=DEVICE)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.long, device=DEVICE)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)

    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)

    network = MatchNet(len(NN_FEATURES)).to(DEVICE)
    training_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    # This is the same unweighted early-stopping objective as the source script.
    validation_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
    max_epochs, batch_size, patience = 100, 256, 8
    best_loss, best_state, best_epoch = float("inf"), None, 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        network.train()
        order = torch.randperm(len(x_subtrain_tensor), device=DEVICE)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = training_loss(
                network(x_subtrain_tensor[indices]), y_subtrain_tensor[indices]
            )
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            current_val_loss = validation_loss(
                network(x_validation_tensor), y_validation_tensor
            ).item()
        if current_val_loss < best_loss:
            best_loss = current_val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in network.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("Ingen gyldig model blev gemt.")

    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        validation_probabilities = torch.softmax(
            network(x_validation_tensor), dim=1
        ).cpu().numpy()
    validation_probabilities = normalize_probabilities(validation_probabilities)
    print(f"    seed={seed}: bedste epoke={best_epoch}, uvægtet val-loss={best_loss:.6f}")
    # The test tensor is retained but not scored until the shared threshold is frozen.
    return validation_probabilities, network, x_test_tensor


def infer_test_probabilities(network, x_test_tensor):
    network.eval()
    with torch.no_grad():
        probabilities = torch.softmax(network(x_test_tensor), dim=1).cpu().numpy()
    return normalize_probabilities(probabilities)


def make_summary_row(scope, rule, seed, fold, metrics=None, matches=np.nan,
                     threshold=np.nan, validation_macro_f1=np.nan, stds=None):
    row = {
        "scope": scope,
        "decision_rule": rule,
        "seed": seed,
        "fold": fold,
        "kampe": matches,
        "fixed_threshold": threshold,
        "validation_macro_f1": validation_macro_f1,
    }
    metrics = metrics or {}
    stds = stds or {}
    for name in METRIC_NAMES:
        row[name] = metrics.get(name, np.nan)
        row[f"{name}_std"] = stds.get(name, np.nan)
    return row


def print_table(title, frame, columns):
    print(f"\n=== {title} ===")
    if frame.empty:
        print("Ingen resultater.")
    else:
        print(frame[columns].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))


def print_report(summary, candidate_frame, paired_seed_metrics, fixed_threshold):
    print(f"\nFælles tærskel valgt på fold 1-validering: {fixed_threshold:.2f}")
    candidate_summary = candidate_frame[
        candidate_frame.scope == "fold1_validation_candidate_summary"
    ].rename(columns={
        "threshold": "candidate_threshold",
        "validation_macro_f1": "validation_macro_f1_mean",
    })
    print_table(
        "Macro F1 for hver kandidat på fold 1-validering (gennemsnit over seeds)",
        candidate_summary.sort_values("candidate_threshold", ascending=False),
        ["candidate_threshold", "validation_macro_f1_mean", "validation_macro_f1_std", "selected"],
    )
    print("Metrikretning: accuracy, Macro F1, precision og recall højere er bedre; "
          "log loss og Brier score lavere er bedre. Klasseandele er deskriptive.")
    print("Tærsklen ændrer kun labels; log loss og Brier score bruger de oprindelige sandsynligheder.")

    report_metrics = [
        "accuracy", "macro_f1", "log_loss", "brier_score",
        "precision_H", "recall_H", "precision_D", "recall_D",
        "precision_A", "recall_A", "share_pred_H", "share_pred_D", "share_pred_A",
    ]
    fold_rows = summary[summary.scope == "seed_fold"].sort_values(
        ["seed", "fold", "decision_rule"]
    )
    print_table(
        "Resultater pr. seed og fold",
        fold_rows,
        ["seed", "fold", "decision_rule", "fixed_threshold", *report_metrics],
    )
    overall_rows = summary[summary.scope == "seed_overall"].sort_values(
        ["seed", "decision_rule"]
    )
    print_table(
        "Samlet resultat pr. seed",
        overall_rows,
        ["seed", "decision_rule", *report_metrics],
    )
    summary_metrics = [
        column for metric in report_metrics for column in (metric, f"{metric}_std")
    ]
    print_table(
        "Gennemsnit og standardafvigelse på tværs af seeds",
        summary[summary.scope == "overall_seed_summary"].sort_values("decision_rule"),
        ["decision_rule", *summary_metrics],
    )

    fold_summary = summary[summary.scope == "fold_seed_summary"]
    fold_wide = fold_summary.pivot(index="fold", columns="decision_rule", values=["macro_f1", "recall_D"])
    change_rows = []
    for fold in fold_wide.index:
        change_rows.append({
            "fold": fold,
            "delta_macro_f1": fold_wide.loc[fold, ("macro_f1", FIXED_RULE)]
            - fold_wide.loc[fold, ("macro_f1", ARGMAX_RULE)],
            "delta_recall_D": fold_wide.loc[fold, ("recall_D", FIXED_RULE)]
            - fold_wide.loc[fold, ("recall_D", ARGMAX_RULE)],
        })
    print_table(
        "Ændring pr. testfold (fast tærskel minus argmax; gennemsnit over seeds)",
        pd.DataFrame(change_rows),
        ["fold", "delta_macro_f1", "delta_recall_D"],
    )

    wide = paired_seed_metrics.pivot(
        index="seed", columns="decision_rule", values=["macro_f1", "recall_D", "log_loss", "brier_score"]
    )
    paired_rows = []
    for seed in wide.index:
        paired_rows.append({
            "seed": seed,
            "delta_macro_f1": wide.loc[seed, ("macro_f1", FIXED_RULE)] - wide.loc[seed, ("macro_f1", ARGMAX_RULE)],
            "delta_recall_D": wide.loc[seed, ("recall_D", FIXED_RULE)] - wide.loc[seed, ("recall_D", ARGMAX_RULE)],
            "delta_log_loss": wide.loc[seed, ("log_loss", FIXED_RULE)] - wide.loc[seed, ("log_loss", ARGMAX_RULE)],
            "delta_brier_score": wide.loc[seed, ("brier_score", FIXED_RULE)] - wide.loc[seed, ("brier_score", ARGMAX_RULE)],
        })
    paired_df = pd.DataFrame(paired_rows)
    avg = {"seed": "mean across seeds"}
    for column in paired_df.columns[1:]:
        avg[column] = paired_df[column].mean()
    paired_df = pd.concat([paired_df, pd.DataFrame([avg])], ignore_index=True)
    print_table(
        "Parrede forskelle samlet pr. seed",
        paired_df,
        ["seed", "delta_macro_f1", "delta_recall_D", "delta_log_loss", "delta_brier_score"],
    )
    print(f"\nOOF-fil: {OOF_PATH}")
    print(f"Summary-fil: {SUMMARY_PATH}")


def main():
    # Stop before any data work if either destination already exists.
    check_outputs_absent()
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"])
    data = data.merge(
        elo[KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES],
        on=KEY_COLUMNS, how="left", validate="one_to_one",
    )
    data = data.sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    data = data.dropna(subset=COMMON_FEATURES + ELO_COLUMNS).reset_index(drop=True)
    data = data[data.result.isin(LABEL_TO_INDEX)].reset_index(drop=True)
    if not data.result.isin(LABEL_TO_INDEX).all():
        raise ValueError("Resultatkolonnen indeholder ukendte klasser.")

    date_blocks = [
        list(block)
        for block in np.array_split(sorted(data.date.dt.date.unique()), 4)
        if len(block)
    ]
    if len(date_blocks) != 4:
        raise RuntimeError("Forventede fire kronologiske datoblokke.")
    print(f"PyTorch: {torch.__version__}; enhed: {DEVICE}; features: {len(NN_FEATURES)}")
    print(f"Kampe: {len(data)}; alpha={ALPHA}; seeds={list(SEEDS)}")

    oof_rows, summary_rows, candidate_rows = [], [], []
    fixed_threshold = None

    for fold in range(1, 4):
        train_dates = [date for block in date_blocks[:fold] for date in block]
        test_dates = date_blocks[fold]
        train = data[data.date.dt.date.isin(train_dates)]
        test = data[data.date.dt.date.isin(test_dates)]
        unique_train_dates = sorted(train.date.dt.date.unique())
        split = int(len(unique_train_dates) * 0.85)
        subtrain_dates, validation_dates = unique_train_dates[:split], unique_train_dates[split:]
        subtrain = train[train.date.dt.date.isin(subtrain_dates)].copy()
        validation = train[train.date.dt.date.isin(validation_dates)].copy()
        if train.empty or test.empty or subtrain.empty or validation.empty:
            raise RuntimeError(f"Fold {fold} har tomt train/validerings/test-sæt.")
        expected_keys = test[KEY_COLUMNS].reset_index(drop=True)
        print(f"\nFold {fold}: subtrain={len(subtrain)}, validation={len(validation)}, test={len(test)}")

        # Fold 1 is a dedicated threshold-selection phase. Store each seed's
        # model and validation probabilities, but do not infer its test fold yet.
        fold1_models = {}
        fold1_validation_scores = {}
        for seed in SEEDS:
            y_subtrain = subtrain.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
            weights = make_alpha_025_weights(y_subtrain)
            validation_probabilities, network, x_test_tensor = train_fold_model(
                subtrain, validation, test, fold, seed, weights
            )
            if fold == 1:
                fold1_models[seed] = (network, x_test_tensor)
                fold1_validation_scores[seed] = {}
                for threshold in DRAW_THRESHOLDS:
                    score = validation_candidate_score(
                        validation.result.to_numpy(), validation_probabilities, threshold
                    )
                    fold1_validation_scores[seed][threshold] = score
                    candidate_rows.append({
                        "scope": "fold1_validation_candidate_seed",
                        "seed": seed,
                        "threshold": threshold,
                        "validation_macro_f1": score,
                        "selected": False,
                    })
            else:
                # The fixed threshold was selected once on fold 1 validation.
                fold_probabilities = infer_test_probabilities(network, x_test_tensor)
                save_test_results(
                    fold, seed, test, expected_keys, fold_probabilities,
                    fixed_threshold, np.nan, oof_rows, summary_rows,
                )

        if fold == 1:
            candidate_means = {
                threshold: float(np.mean([
                    fold1_validation_scores[seed][threshold] for seed in SEEDS
                ]))
                for threshold in DRAW_THRESHOLDS
            }
            # Iterate in predefined descending order; strict improvement keeps
            # the highest threshold if the five-seed mean Macro F1 ties.
            best_score = -1.0
            for threshold in DRAW_THRESHOLDS:
                if candidate_means[threshold] > best_score:
                    fixed_threshold = threshold
                    best_score = candidate_means[threshold]
            for row in candidate_rows:
                row["selected"] = row["threshold"] == fixed_threshold
            for threshold in DRAW_THRESHOLDS:
                scores = [fold1_validation_scores[seed][threshold] for seed in SEEDS]
                candidate_rows.append({
                    "scope": "fold1_validation_candidate_summary",
                    "seed": "mean_across_seeds",
                    "threshold": threshold,
                    "validation_macro_f1": float(np.mean(scores)),
                    "validation_macro_f1_std": float(np.std(scores, ddof=1)),
                    "selected": threshold == fixed_threshold,
                })
            print(f"Fælles tærskel valgt udelukkende på fold 1-validering: {fixed_threshold:.2f}")

            # Only after threshold selection do we infer/evaluate the first test fold.
            for seed in SEEDS:
                network, x_test_tensor = fold1_models[seed]
                fold_probabilities = infer_test_probabilities(network, x_test_tensor)
                save_test_results(
                    fold, seed, test, expected_keys, fold_probabilities,
                    fixed_threshold, fold1_validation_scores[seed][fixed_threshold],
                    oof_rows, summary_rows,
                )

    if fixed_threshold is None:
        raise RuntimeError("Den fælles tærskel blev ikke valgt på fold 1.")
    oof_df = pd.DataFrame(oof_rows)
    if oof_df.empty:
        raise RuntimeError("Ingen OOF-resultater blev produceret.")

    # Per-seed summaries combine the three distinct test folds for each rule.
    seed_overall_rows = []
    for (rule, seed), group in oof_df.groupby(["decision_rule", "seed"], sort=True):
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        metrics = calculate_metrics(group.actual.to_numpy(), group.predicted.to_numpy(), probabilities)
        seed_overall_rows.append({"decision_rule": rule, "seed": seed, **metrics})
        summary_rows.append(make_summary_row(
            "seed_overall", rule, seed, "all", metrics, len(group),
            threshold=fixed_threshold if rule == FIXED_RULE else np.nan,
        ))

    seed_fold_df = pd.DataFrame([row for row in summary_rows if row["scope"] == "seed_fold"])
    for (rule, fold), group in seed_fold_df.groupby(["decision_rule", "fold"], sort=True):
        means = {name: group[name].mean() for name in METRIC_NAMES}
        stds = {name: group[name].std(ddof=1) for name in METRIC_NAMES}
        summary_rows.append(make_summary_row(
            "fold_seed_summary", rule, "all", fold, means, group.kampe.mean(),
            threshold=fixed_threshold if rule == FIXED_RULE else np.nan,
            stds=stds,
        ))

    seed_overall_df = pd.DataFrame(seed_overall_rows)
    for rule, group in seed_overall_df.groupby("decision_rule", sort=True):
        means = {name: group[name].mean() for name in METRIC_NAMES}
        stds = {name: group[name].std(ddof=1) for name in METRIC_NAMES}
        summary_rows.append(make_summary_row(
            "overall_seed_summary", rule, "all", "all", means,
            oof_df[oof_df.decision_rule == rule].groupby("seed").size().mean(),
            threshold=fixed_threshold if rule == FIXED_RULE else np.nan,
            stds=stds,
        ))

    summary_columns = [
        "scope", "decision_rule", "seed", "fold", "kampe", "fixed_threshold",
        *[column for metric in METRIC_NAMES for column in (metric, f"{metric}_std")],
    ]
    candidate_df = pd.DataFrame(candidate_rows)
    summary_df = pd.DataFrame(summary_rows)[summary_columns]
    candidate_output = candidate_df.rename(columns={
        "threshold": "candidate_threshold",
        "selected": "is_selected",
    })
    summary_df = pd.concat([summary_df, candidate_output], ignore_index=True, sort=False)

    # Protect both destinations before and during creation; never overwrite.
    check_outputs_absent()
    oof_handle = summary_handle = None
    try:
        oof_handle = OOF_PATH.open("x", newline="", encoding="utf-8-sig")
        summary_handle = SUMMARY_PATH.open("x", newline="", encoding="utf-8-sig")
        oof_df.to_csv(oof_handle, index=False)
        summary_df.to_csv(summary_handle, index=False)
    except Exception:
        if oof_handle is not None:
            oof_handle.close()
        if summary_handle is not None:
            summary_handle.close()
        if oof_handle is not None and OOF_PATH.exists():
            OOF_PATH.unlink()
        if summary_handle is not None and SUMMARY_PATH.exists():
            SUMMARY_PATH.unlink()
        raise
    finally:
        if oof_handle is not None and not oof_handle.closed:
            oof_handle.close()
        if summary_handle is not None and not summary_handle.closed:
            summary_handle.close()

    print_report(summary_df, candidate_df, seed_overall_df, fixed_threshold)


def save_test_results(fold, seed, test, expected_keys, probabilities, threshold,
                      validation_score, oof_rows, summary_rows):
    """Apply both label rules to the same test rows and original probabilities."""
    probabilities_before = probabilities.copy()
    predictions = {
        ARGMAX_RULE: INDEX_TO_LABEL[probabilities.argmax(axis=1)],
        FIXED_RULE: threshold_predictions(probabilities, threshold),
    }
    if not np.array_equal(probabilities_before, probabilities):
        raise RuntimeError("Beslutningsreglen ændrede sandsynlighederne.")

    actual = test.result.to_numpy()
    metric_results = {}
    for rule in RULES:
        metrics = calculate_metrics(actual, predictions[rule], probabilities)
        metric_results[rule] = metrics
        if rule == FIXED_RULE:
            for name in ("log_loss", "brier_score"):
                if metrics[name] != metric_results[ARGMAX_RULE][name]:
                    raise RuntimeError(f"{name} ændrede sig mellem beslutningsreglerne.")
        summary_rows.append(make_summary_row(
            "seed_fold", rule, seed, fold, metrics, len(test),
            threshold=threshold if rule == FIXED_RULE else np.nan,
            validation_macro_f1=validation_score if rule == FIXED_RULE else np.nan,
        ))
        rows_for_rule = []
        for index, (_, match) in enumerate(test.iterrows()):
            row = {
                "decision_rule": rule,
                "seed": seed,
                "fold": fold,
                "date": match.date,
                "time": match.time,
                "league": match.league,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual": match.result,
                "predicted": predictions[rule][index],
                "fixed_threshold": threshold if rule == FIXED_RULE else np.nan,
                "fold1_validation_macro_f1": validation_score if rule == FIXED_RULE else np.nan,
                "prob_H": float(probabilities[index, 0]),
                "prob_D": float(probabilities[index, 1]),
                "prob_A": float(probabilities[index, 2]),
            }
            rows_for_rule.append(row)
            oof_rows.append(row)
        candidate_keys = pd.DataFrame(rows_for_rule)[KEY_COLUMNS].reset_index(drop=True)
        if not candidate_keys.astype("string").equals(expected_keys.astype("string")):
            raise RuntimeError(
                f"Testkampenes nøgler/rækkefølge afviger for {rule}, seed={seed}, fold={fold}."
            )
    if not np.array_equal(
        metric_results[ARGMAX_RULE]["log_loss"],
        metric_results[FIXED_RULE]["log_loss"],
    ):
        raise RuntimeError("Log loss skal være identisk mellem beslutningsreglerne.")


if __name__ == "__main__":
    main()

