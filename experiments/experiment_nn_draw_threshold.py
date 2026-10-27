"""Tune a draw decision threshold on validation data for the existing NN.

The model always outputs its original H/D/A probabilities. The alternative
decision rule changes only the predicted label: choose D when
P(D) >= threshold * max(P(H), P(A)); otherwise choose the larger of P(H), P(A).
The predefined ratio thresholds are selected separately for each seed/fold by
validation Macro F1. Test labels and test metrics are never used for selection.
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
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_draw_threshold.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_draw_threshold_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHA = 0.25
# Predefined ratios; descending order gives the more conservative threshold
# precedence if validation Macro F1 ties exactly.
DRAW_THRESHOLDS = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
RULES = ("argmax", "draw_threshold")
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchNet(nn.Module):
    """Existing baseline architecture: 27 -> 32 -> 16 -> 3."""

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
    sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(sums).all() or (sums <= 0).any():
        raise ValueError("Ugyldige sandsynligheder blev produceret.")
    return probabilities / sums


def make_alpha_025_weights(y_subtrain):
    """Same present-class handling and normalization as the three-weight script."""
    counts = np.bincount(np.asarray(y_subtrain, dtype=np.int64), minlength=3)
    present = counts > 0
    baseline = np.zeros(3, dtype=np.float64)
    baseline[present] = len(y_subtrain) / (present.sum() * counts[present])
    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline[present] ** ALPHA
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def threshold_predictions(probabilities, threshold):
    """Apply the draw rule without modifying the supplied probabilities."""
    probabilities = np.asarray(probabilities)
    top_non_draw = np.maximum(probabilities[:, 0], probabilities[:, 2])
    predictions = np.where(
        probabilities[:, 1] >= threshold * top_non_draw,
        "D",
        np.where(probabilities[:, 0] >= probabilities[:, 2], "H", "A"),
    )
    return predictions


def choose_threshold(validation_actual, validation_probabilities):
    best_threshold = None
    best_macro_f1 = -1.0
    # Thresholds are fixed before looking at validation/test outcomes.
    for threshold in DRAW_THRESHOLDS:
        predictions = threshold_predictions(validation_probabilities, threshold)
        score = f1_score(
            validation_actual, predictions, labels=LABELS,
            average="macro", zero_division=0,
        )
        # Strict comparison preserves the first (highest) threshold on a tie.
        if score > best_macro_f1:
            best_threshold = threshold
            best_macro_f1 = float(score)
    return best_threshold, best_macro_f1


def calculate_metrics(actual, predictions, probabilities):
    """Class metrics use labels; probability metrics use untouched probabilities."""
    probabilities = normalize_probabilities(probabilities)
    actual = np.asarray(actual)
    actual_indices = np.asarray([LABEL_TO_INDEX[label] for label in actual])
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


def train_and_predict(subtrain, validation, test, fold, seed, class_weights):
    # Fit preprocessing only on subtrain, then transform validation and test.
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
    # Same common checkpoint rule as the three-weight experiment.
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
            loss = training_loss(network(x_subtrain_tensor[indices]), y_subtrain_tensor[indices])
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            val_loss = validation_loss(network(x_validation_tensor), y_validation_tensor).item()
        if val_loss < best_loss:
            best_loss = val_loss
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
    # Keep test inference until after the validation-only threshold is selected.
    return validation_probabilities, network, x_test_tensor


def make_summary_row(scope, rule, seed, fold, metrics, matches,
                     threshold=np.nan, validation_macro_f1=np.nan, stds=None):
    row = {
        "scope": scope,
        "decision_rule": rule,
        "seed": seed,
        "fold": fold,
        "kampe": matches,
        "selected_threshold": threshold,
        "validation_macro_f1": validation_macro_f1,
    }
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


def print_report(summary, paired_seed_metrics):
    report_metrics = [
        "accuracy", "macro_f1", "log_loss", "brier_score",
        "precision_H", "recall_H", "precision_D", "recall_D",
        "precision_A", "recall_A", "share_pred_H", "share_pred_D", "share_pred_A",
    ]
    print("\nMetrikretning: accuracy, Macro F1, precision og recall højere er bedre; "
          "log loss og Brier score lavere er bedre. Klasseandele er deskriptive.")
    print("Beslutningsgrænsen ændrer kun klasselabels; log loss og Brier score bruger de oprindelige sandsynligheder.")
    print_table(
        "Resultater pr. seed og fold",
        summary[summary.scope == "seed_fold"].sort_values(["seed", "fold", "decision_rule"]),
        ["seed", "fold", "decision_rule", "selected_threshold", "validation_macro_f1", *report_metrics],
    )
    print_table(
        "Samlet resultat pr. seed",
        summary[summary.scope == "seed_overall"].sort_values(["seed", "decision_rule"]),
        ["seed", "decision_rule", *report_metrics],
    )
    summary_columns = ["decision_rule"] + [
        column for metric in report_metrics for column in (metric, f"{metric}_std")
    ]
    print_table(
        "Gennemsnit og standardafvigelse på tværs af seeds",
        summary[summary.scope == "overall_seed_summary"].sort_values("decision_rule"),
        summary_columns,
    )

    wide = paired_seed_metrics.pivot(index="seed", columns="decision_rule", values=report_metrics)
    rows = []
    delta_metrics = ["macro_f1", "recall_D", "recall_H", "recall_A", "log_loss", "brier_score"]
    for seed in wide.index:
        row = {"seed": seed}
        for metric in delta_metrics:
            row[f"delta_{metric}"] = (
                wide.loc[seed, (metric, "draw_threshold")]
                - wide.loc[seed, (metric, "argmax")]
            )
        rows.append(row)
    delta_frame = pd.DataFrame(rows)
    mean = {"seed": "mean across seeds"}
    for column in delta_frame.columns[1:]:
        mean[column] = delta_frame[column].mean()
    delta_frame = pd.concat([delta_frame, pd.DataFrame([mean])], ignore_index=True)
    print_table(
        "Parret forskel: tærskel minus argmax",
        delta_frame,
        ["seed", "delta_macro_f1", "delta_recall_D", "delta_recall_H",
         "delta_recall_A", "delta_log_loss", "delta_brier_score"],
    )
    print("For forskelle: højere Macro F1/recall er bedre; lavere log loss/Brier score er bedre.")
    print(f"\nOOF-fil: {OOF_PATH}")
    print(f"Summary-fil: {SUMMARY_PATH}")


def main():
    # Initial guard prevents data processing/training if a destination exists.
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

    # Same four chronological date blocks; each later block is one test fold.
    dates = sorted(data.date.dt.date.unique())
    date_blocks = [list(block) for block in np.array_split(dates, 4) if len(block)]
    print(f"PyTorch: {torch.__version__}; enhed: {DEVICE}; features: {len(NN_FEATURES)}")
    print(f"Kampe i backtesten: {len(data)}; alpha={ALPHA}; seeds={list(SEEDS)}")

    oof_rows, summary_rows = [], []
    for fold in range(1, len(date_blocks)):
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

        for seed in SEEDS:
            y_subtrain = subtrain.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
            class_weights = make_alpha_025_weights(y_subtrain)
            validation_probabilities, network, x_test_tensor = train_and_predict(
                subtrain, validation, test, fold, seed, class_weights
            )
            validation_actual = validation.result.to_numpy()
            selected_threshold, selected_val_macro_f1 = choose_threshold(
                validation_actual, validation_probabilities
            )
            # Only now score the untouched test fold; no test label or metric
            # participates in choosing the threshold.
            network.eval()
            with torch.no_grad():
                test_probabilities = torch.softmax(
                    network(x_test_tensor), dim=1
                ).cpu().numpy()
            test_probabilities = normalize_probabilities(test_probabilities)
            # Thresholding is applied to a copy of labels derived from unchanged
            # test probabilities; it cannot modify probability values.
            probabilities_before = test_probabilities.copy()
            argmax_predictions = INDEX_TO_LABEL[test_probabilities.argmax(axis=1)]
            threshold_predictions_test = threshold_predictions(
                test_probabilities, selected_threshold
            )
            if not np.array_equal(probabilities_before, test_probabilities):
                raise RuntimeError("Beslutningsreglen ændrede test-sandsynlighederne.")

            actual = test.result.to_numpy()
            rule_predictions = {
                "argmax": argmax_predictions,
                "draw_threshold": threshold_predictions_test,
            }
            metric_results = {}
            for rule in RULES:
                metrics = calculate_metrics(actual, rule_predictions[rule], test_probabilities)
                metric_results[rule] = metrics
                # Probability scores must be identical across label-only rules.
                if rule == "draw_threshold":
                    for metric in ("log_loss", "brier_score"):
                        if not np.isclose(
                            metrics[metric], metric_results["argmax"][metric],
                            rtol=0.0, atol=1e-12,
                        ):
                            raise RuntimeError(f"{metric} ændrede sig ved tærskelbeslutningen.")
                summary_rows.append(make_summary_row(
                    "seed_fold", rule, seed, fold, metrics, len(test),
                    threshold=selected_threshold if rule == "draw_threshold" else np.nan,
                    validation_macro_f1=selected_val_macro_f1 if rule == "draw_threshold" else np.nan,
                ))

                group_rows = []
                for index, (_, match) in enumerate(test.iterrows()):
                    record = {
                        "decision_rule": rule,
                        "seed": seed,
                        "fold": fold,
                        "date": match.date,
                        "time": match.time,
                        "league": match.league,
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                        "actual": match.result,
                        "predicted": rule_predictions[rule][index],
                        "selected_threshold": selected_threshold if rule == "draw_threshold" else np.nan,
                        "validation_macro_f1": selected_val_macro_f1 if rule == "draw_threshold" else np.nan,
                        "prob_H": float(test_probabilities[index, 0]),
                        "prob_D": float(test_probabilities[index, 1]),
                        "prob_A": float(test_probabilities[index, 2]),
                    }
                    oof_rows.append(record)
                    group_rows.append(record)
                candidate_keys = pd.DataFrame(group_rows)[KEY_COLUMNS].reset_index(drop=True)
                if not candidate_keys.astype("string").equals(expected_keys.astype("string")):
                    raise RuntimeError(
                        f"Testkampene/rækkefølgen afviger for {rule}, seed={seed}, fold={fold}."
                    )
            if not np.array_equal(
                metric_results["argmax"]["log_loss"],
                metric_results["draw_threshold"]["log_loss"],
            ):
                raise RuntimeError("Log loss er ikke identisk mellem beslutningsreglerne.")

    oof_df = pd.DataFrame(oof_rows)
    if oof_df.empty:
        raise RuntimeError("Ingen OOF-resultater blev produceret.")

    per_seed_overall = []
    for (rule, seed), group in oof_df.groupby(["decision_rule", "seed"], sort=True):
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        metrics = calculate_metrics(
            group.actual.to_numpy(), group.predicted.to_numpy(), probabilities
        )
        per_fold_settings = group[
            ["fold", "selected_threshold", "validation_macro_f1"]
        ].dropna(subset=["selected_threshold"]).drop_duplicates("fold")
        threshold_values = per_fold_settings.selected_threshold.to_numpy()
        threshold_mean = float(np.mean(threshold_values)) if len(threshold_values) else np.nan
        val_scores = per_fold_settings.validation_macro_f1
        val_mean = float(val_scores.mean()) if len(val_scores) else np.nan
        per_seed_overall.append({
            "decision_rule": rule, "seed": seed, **metrics,
            "selected_threshold": threshold_mean,
            "validation_macro_f1": val_mean,
            "kampe": group.shape[0],
        })
        summary_rows.append(make_summary_row(
            "seed_overall", rule, seed, "all", metrics, len(group),
            threshold=threshold_mean, validation_macro_f1=val_mean,
        ))

    seed_fold_df = pd.DataFrame([row for row in summary_rows if row["scope"] == "seed_fold"])
    for (rule, fold), group in seed_fold_df.groupby(["decision_rule", "fold"], sort=True):
        means = {name: group[name].mean() for name in METRIC_NAMES}
        stds = {name: group[name].std(ddof=1) for name in METRIC_NAMES}
        thresholds = group.selected_threshold.dropna()
        val_scores = group.validation_macro_f1.dropna()
        summary_rows.append(make_summary_row(
            "fold_seed_summary", rule, "all", fold, means, group.kampe.mean(),
            threshold=float(thresholds.mean()) if len(thresholds) else np.nan,
            validation_macro_f1=float(val_scores.mean()) if len(val_scores) else np.nan,
            stds=stds,
        ))

    overall_seed_df = pd.DataFrame(per_seed_overall)
    for rule, group in overall_seed_df.groupby("decision_rule", sort=True):
        means = {name: group[name].mean() for name in METRIC_NAMES}
        stds = {name: group[name].std(ddof=1) for name in METRIC_NAMES}
        summary_rows.append(make_summary_row(
            "overall_seed_summary", rule, "all", "all", means,
            group.kampe.mean(),
            threshold=float(group.selected_threshold.mean()),
            validation_macro_f1=float(group.validation_macro_f1.mean()),
            stds=stds,
        ))

    summary_columns = [
        "scope", "decision_rule", "seed", "fold", "kampe",
        "selected_threshold", "validation_macro_f1",
        *[column for metric in METRIC_NAMES for column in (metric, f"{metric}_std")],
    ]
    summary_df = pd.DataFrame(summary_rows)[summary_columns]

    # Final existence check plus exclusive creation prevents overwrite races.
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

    print_report(summary_df, overall_seed_df)


if __name__ == "__main__":
    main()
