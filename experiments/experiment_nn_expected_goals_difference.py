"""Test one explicit pre-match expected-goals parity feature in the NN.

This is an isolated comparison of the 27-feature baseline and the same model
with abs(home_expected_goals - away_expected_goals) appended. Both variants
use alpha=0.25 class weights and unweighted validation cross-entropy for common
checkpoint selection, matching the repeated-seed three-weights experiment.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_expected_goals_difference.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_expected_goals_difference_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHA = 0.25
BASE_VARIANT = "baseline_27"
DIFFERENCE_VARIANT = "expected_goals_difference_28"
VARIANTS = (BASE_VARIANT, DIFFERENCE_VARIANT)
METRIC_NAMES = (
    "accuracy", "macro_f1", "log_loss", "brier_score",
    "precision_H", "recall_H", "f1_H",
    "precision_D", "recall_D", "f1_D",
    "precision_A", "recall_A", "f1_A",
)

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
KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]
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
BASE_FEATURE_COLUMNS = BASE_FEATURES + ATTACK_DEFENSE_FEATURES + ELO_COLUMNS
DIFFERENCE_FEATURE = "abs_expected_goals_difference"
FEATURES_BY_VARIANT = {
    BASE_VARIANT: BASE_FEATURE_COLUMNS,
    DIFFERENCE_VARIANT: BASE_FEATURE_COLUMNS + [DIFFERENCE_FEATURE],
}
LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
LABELS = ["H", "D", "A"]
INDEX_TO_LABEL = np.asarray(LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchNet(nn.Module):
    """Existing architecture: input -> 32 -> 16 -> 3."""

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


def calculate_metrics(y_true, probabilities):
    probabilities = normalize_probabilities(probabilities)
    y_true = np.asarray(y_true)
    actual_indices = np.asarray([LABEL_TO_INDEX[value] for value in y_true])
    predicted_indices = probabilities.argmax(axis=1)
    predictions = INDEX_TO_LABEL[predicted_indices]
    one_hot = np.eye(3, dtype=np.float64)[actual_indices]
    # Same multiclass Brier definition as the existing experiments.
    brier_score = np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, labels=LABELS,
            average="macro", zero_division=0,
        ),
        "log_loss": log_loss(actual_indices, probabilities, labels=[0, 1, 2]),
        "brier_score": brier_score,
    }
    precision, recall, class_f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=LABELS, zero_division=0
    )
    for index, label in enumerate(LABELS):
        metrics[f"precision_{label}"] = precision[index]
        metrics[f"recall_{label}"] = recall[index]
        metrics[f"f1_{label}"] = class_f1[index]
    return metrics, predictions, probabilities


def make_alpha_025_weights(y_subtrain):
    """Use the existing experiment's inverse-frequency weights to alpha 0.25."""
    labels = np.asarray(y_subtrain, dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    present = counts > 0
    baseline_weights = np.zeros(3, dtype=np.float64)
    baseline_weights[present] = len(labels) / (present.sum() * counts[present])
    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline_weights[present] ** ALPHA
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def train_variant(subtrain, validation, test, variant, fold, seed, class_weights):
    feature_columns = FEATURES_BY_VARIANT[variant]
    # Imputer and scaler are fitted independently, using subtrain only.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_subtrain = scaler.fit_transform(imputer.fit_transform(subtrain[feature_columns]))
    x_validation = scaler.transform(imputer.transform(validation[feature_columns]))
    x_test = scaler.transform(imputer.transform(test[feature_columns]))

    y_subtrain = subtrain["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    y_validation = validation["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    x_subtrain_tensor = torch.tensor(x_subtrain, dtype=torch.float32, device=DEVICE)
    y_subtrain_tensor = torch.tensor(y_subtrain, dtype=torch.long, device=DEVICE)
    x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32, device=DEVICE)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.long, device=DEVICE)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)

    # Reset to an identical seed for the two variants in the same fold.
    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)

    network = MatchNet(len(feature_columns)).to(DEVICE)
    training_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    # Common, unweighted validation objective for both model variants.
    validation_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
    max_epochs, batch_size, patience = 100, 256, 8
    best_loss, best_state, best_epoch = float("inf"), None, 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        network.train()
        order = torch.randperm(len(x_subtrain_tensor), device=DEVICE)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            batch_loss = training_loss(
                network(x_subtrain_tensor[indices]), y_subtrain_tensor[indices]
            )
            batch_loss.backward()
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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Ingen gyldig model blev gemt.")
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        probabilities = torch.softmax(network(x_test_tensor), dim=1).cpu().numpy()
    print(
        f"    {variant}: bedste epoke={best_epoch}, "
        f"uvægtet validerings-loss={best_loss:.6f}"
    )
    return normalize_probabilities(probabilities)


def make_summary_row(scope, variant, seed, fold, metrics, matches, stds=None):
    row = {
        "scope": scope,
        "variant": variant,
        "seed": seed,
        "fold": fold,
        "kampe": matches,
    }
    stds = stds or {}
    for metric in METRIC_NAMES:
        row[metric] = metrics.get(metric, np.nan)
        row[f"{metric}_std"] = stds.get(metric, np.nan)
    return row


def format_table(title, frame, columns):
    print(f"\n=== {title} ===")
    if frame.empty:
        print("Ingen resultater.")
    else:
        print(frame[columns].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))


def print_report(summary_df, seed_overall_df):
    detail_metrics = [
        "accuracy", "macro_f1", "log_loss", "brier_score",
        "precision_H", "recall_H", "precision_D", "recall_D",
        "precision_A", "recall_A",
    ]
    print("\nMetrikretning: accuracy, Macro F1, precision og recall højere er bedre; "
          "log loss og Brier score lavere er bedre.")
    print("Særligt fokus: recall_D samt log loss og Brier score.")
    fold_rows = summary_df[summary_df.scope == "seed_fold"].copy()
    format_table(
        "Resultater pr. seed og fold",
        fold_rows.sort_values(["seed", "fold", "variant"]),
        ["seed", "fold", "variant", "kampe", *detail_metrics],
    )
    overall_rows = summary_df[summary_df.scope == "seed_overall"].copy()
    format_table(
        "Samlet resultat pr. seed (alle OOF-folds)",
        overall_rows.sort_values(["seed", "variant"]),
        ["seed", "variant", "kampe", *detail_metrics],
    )
    fold_summary = summary_df[summary_df.scope == "fold_seed_summary"].copy()
    format_table(
        "Foldgennemsnit og standardafvigelse på tværs af seeds",
        fold_summary.sort_values(["fold", "variant"]),
        ["fold", "variant", *[c for m in detail_metrics for c in (m, f"{m}_std")]],
    )
    seed_summary = summary_df[summary_df.scope == "overall_seed_summary"].copy()
    format_table(
        "Samlet gennemsnit og standardafvigelse på tværs af seeds",
        seed_summary.sort_values("variant"),
        ["variant", *[c for m in detail_metrics for c in (m, f"{m}_std")]],
    )

    paired = seed_overall_df.pivot(index="seed", columns="variant", values=detail_metrics)
    delta_rows = []
    for seed in paired.index:
        row = {"seed": seed}
        for metric in ["recall_D", "log_loss", "brier_score", "accuracy", "macro_f1"]:
            row[f"delta_{metric}"] = (
                paired.loc[seed, (metric, DIFFERENCE_VARIANT)]
                - paired.loc[seed, (metric, BASE_VARIANT)]
            )
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    mean_row = {"seed": "mean across seeds"}
    for column in deltas.columns[1:]:
        mean_row[column] = deltas[column].mean()
    deltas = pd.concat([deltas, pd.DataFrame([mean_row])], ignore_index=True)
    format_table(
        "Parret forskel (28 features minus baseline 27), samlet pr. seed",
        deltas,
        ["seed", "delta_recall_D", "delta_log_loss", "delta_brier_score",
         "delta_accuracy", "delta_macro_f1"],
    )
    print("Fortolkning af forskelle: positiv recall/accuracy/Macro F1 er bedre; "
          "negativ log loss/Brier score er bedre.")
    print(f"\nOOF-fil: {OOF_PATH}")
    print(f"Summary-fil: {SUMMARY_PATH}")


def main():
    # Refuse existing destinations before reading data or starting any training.
    check_outputs_absent()
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"])
    data = data.merge(
        elo[KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    # This is exactly the repeated-seed experiment's data cleaning and order.
    data = data.sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    data = data.dropna(subset=COMMON_FEATURES + ELO_COLUMNS).reset_index(drop=True)
    data = data[data.result.isin(LABEL_TO_INDEX)].reset_index(drop=True)
    if not data.result.isin(LABEL_TO_INDEX).all():
        raise ValueError("Resultatkolonnen indeholder ukendte udfald.")

    # Derived only from the stored pre-match expected-goals estimates.
    data[DIFFERENCE_FEATURE] = (
        data["home_expected_goals"] - data["away_expected_goals"]
    ).abs()

    date_blocks = [
        list(block)
        for block in np.array_split(sorted(data.date.dt.date.unique()), 4)
        if len(block) > 0
    ]
    print(f"PyTorch: {torch.__version__}; enhed: {DEVICE}")
    print(f"Kampe: {len(data)}; baseline features: 27; udvidede features: 28")
    oof_rows, summary_rows = [], []

    for fold in range(1, len(date_blocks)):
        train_dates = [date for block in date_blocks[:fold] for date in block]
        test_dates = date_blocks[fold]
        train = data[data.date.dt.date.isin(train_dates)]
        test = data[data.date.dt.date.isin(test_dates)]
        train_dates_unique = sorted(train.date.dt.date.unique())
        split_index = int(len(train_dates_unique) * 0.85)
        subtrain_dates = train_dates_unique[:split_index]
        validation_dates = train_dates_unique[split_index:]
        subtrain = train[train.date.dt.date.isin(subtrain_dates)].copy()
        validation = train[train.date.dt.date.isin(validation_dates)].copy()
        if train.empty or test.empty or subtrain.empty or validation.empty:
            raise RuntimeError(f"Fold {fold} har tomt train/validerings/test-sæt.")

        expected_keys = test[KEY_COLUMNS].reset_index(drop=True)
        print(
            f"\nFold {fold}: subtrain={len(subtrain)}, validering={len(validation)}, "
            f"test={len(test)}"
        )
        for seed in SEEDS:
            class_weights = make_alpha_025_weights(
                subtrain.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
            )
            for variant in VARIANTS:
                probabilities = train_variant(
                    subtrain, validation, test, variant, fold, seed, class_weights
                )
                metrics, predictions, probabilities = calculate_metrics(
                    test.result.to_numpy(), probabilities
                )
                summary_rows.append(make_summary_row(
                    "seed_fold", variant, seed, fold, metrics, len(test)
                ))
                group_rows = []
                for index, (_, match) in enumerate(test.iterrows()):
                    record = {
                        "variant": variant,
                        "seed": seed,
                        "fold": fold,
                        "date": match.date,
                        "time": match.time,
                        "league": match.league,
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                        "actual": match.result,
                        "predicted": predictions[index],
                        DIFFERENCE_FEATURE: float(match[DIFFERENCE_FEATURE]),
                        "prob_H": float(probabilities[index, 0]),
                        "prob_D": float(probabilities[index, 1]),
                        "prob_A": float(probabilities[index, 2]),
                    }
                    oof_rows.append(record)
                    group_rows.append(record)
                candidate_keys = pd.DataFrame(group_rows)[KEY_COLUMNS].reset_index(drop=True)
                if not candidate_keys.astype("string").equals(expected_keys.astype("string")):
                    raise RuntimeError(
                        f"Testkampenes nøgler/rækkefølge afviger: "
                        f"variant={variant}, seed={seed}, fold={fold}."
                    )

    oof_df = pd.DataFrame(oof_rows)
    if oof_df.empty:
        raise RuntimeError("Ingen OOF-forudsigelser blev beregnet.")

    seed_overall_records = []
    for (variant, seed), group in oof_df.groupby(["variant", "seed"], sort=True):
        metrics, _, _ = calculate_metrics(
            group.actual.to_numpy(), group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        )
        seed_overall_records.append({"variant": variant, "seed": seed, **metrics})
        summary_rows.append(make_summary_row(
            "seed_overall", variant, seed, "all", metrics, len(group)
        ))

    seed_fold_df = pd.DataFrame(
        [row for row in summary_rows if row["scope"] == "seed_fold"]
    )
    for (variant, fold), group in seed_fold_df.groupby(["variant", "fold"], sort=True):
        means = {name: group[name].mean() for name in METRIC_NAMES}
        stds = {name: group[name].std(ddof=1) for name in METRIC_NAMES}
        summary_rows.append(make_summary_row(
            "fold_seed_summary", variant, "all", fold, means,
            group.kampe.mean(), stds,
        ))

    seed_overall_df = pd.DataFrame(seed_overall_records)
    for variant, group in seed_overall_df.groupby("variant", sort=True):
        means = {name: group[name].mean() for name in METRIC_NAMES}
        stds = {name: group[name].std(ddof=1) for name in METRIC_NAMES}
        summary_rows.append(make_summary_row(
            "overall_seed_summary", variant, "all", "all", means,
            len(oof_df[oof_df.variant == variant]) / len(SEEDS), stds,
        ))

    summary_columns = [
        "scope", "variant", "seed", "fold", "kampe",
        *[column for metric in METRIC_NAMES for column in (metric, f"{metric}_std")],
    ]
    summary_df = pd.DataFrame(summary_rows)[summary_columns]

    # Recheck immediately before writing and use exclusive creation as a final guard.
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
        # Only remove destinations created exclusively by this run.
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

    print_report(summary_df, seed_overall_df)


if __name__ == "__main__":
    main()
