"""Controlled feature-group ablation for MatchHubML's existing neural net.

This script is intentionally separate from the baseline and other experiments.
It trains the same network from scratch for the full 27-feature baseline and
for four variants, each excluding one predefined feature group.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_feature_ablation.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_feature_ablation_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHA = 0.25
LABELS = ("H", "D", "A")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = np.asarray(LABELS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AVERAGE_HISTORY_FEATURES = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_points",
    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_points",
]

LAST_FIVE_FEATURES = [
    "home_last5_goals_for",
    "home_last5_goals_against",
    "home_last5_points",
    "away_last5_goals_for",
    "away_last5_goals_against",
    "away_last5_points",
]

ATTACK_DEFENSE_EXPECTED_GOALS_FEATURES = [
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "home_expected_goals",
    "away_expected_goals",
]

ELO_FEATURES = [
    f"{prefix}_elo_k{k}"
    for k in (16, 32, 48)
    for prefix in ("home", "away")
] + [f"elo_difference_k{k}" for k in (16, 32, 48)]

BASE_FEATURES = (
    AVERAGE_HISTORY_FEATURES
    + LAST_FIVE_FEATURES
    + ATTACK_DEFENSE_EXPECTED_GOALS_FEATURES
    + ELO_FEATURES
)

FEATURE_GROUPS = {
    "A_remove_average_history": AVERAGE_HISTORY_FEATURES,
    "B_remove_last_five": LAST_FIVE_FEATURES,
    "C_remove_attack_defense_expected_goals": (
        ATTACK_DEFENSE_EXPECTED_GOALS_FEATURES
    ),
    "D_remove_elo": ELO_FEATURES,
}

VARIANT_FEATURES = {"baseline": list(BASE_FEATURES)}
for variant, removed_features in FEATURE_GROUPS.items():
    VARIANT_FEATURES[variant] = [
        feature for feature in BASE_FEATURES if feature not in removed_features
    ]

KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]
DATA_REQUIRED_FEATURES = (
    AVERAGE_HISTORY_FEATURES
    + LAST_FIVE_FEATURES
    + ["home_form_rating", "away_form_rating",
       "home_venue_form_rating", "away_venue_form_rating"]
    + ATTACK_DEFENSE_EXPECTED_GOALS_FEATURES
)
METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "log_loss",
    "brier_score",
    "recall_H",
    "recall_D",
    "recall_A",
)


def validate_feature_groups():
    """Fail fast if groups overlap, omit baseline inputs, or have wrong sizes."""
    expected_sizes = {
        "baseline": 27,
        "A_remove_average_history": 21,
        "B_remove_last_five": 21,
        "C_remove_attack_defense_expected_goals": 21,
        "D_remove_elo": 18,
    }
    flattened = [feature for group in FEATURE_GROUPS.values() for feature in group]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Featuregrupperne overlapper.")
    if set(flattened) != set(BASE_FEATURES) or len(BASE_FEATURES) != 27:
        raise RuntimeError("Featuregrupperne dækker ikke præcis baseline-listen.")
    for variant, features in VARIANT_FEATURES.items():
        if len(features) != expected_sizes[variant]:
            raise RuntimeError(
                f"{variant} har {len(features)} inputfeatures; "
                f"forventede {expected_sizes[variant]}."
            )
        removed = set(BASE_FEATURES) - set(features)
        expected_removed = (
            set() if variant == "baseline" else set(FEATURE_GROUPS[variant])
        )
        if removed != expected_removed:
            raise RuntimeError(
                f"{variant} fjerner utilsigtede features: "
                f"{sorted(removed ^ expected_removed)}"
            )


class MatchNet(nn.Module):
    """Baseline architecture with only its input width varying by variant."""

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


def normalize_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Sandsynligheder skal have præcis tre kolonner H/D/A.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Sandsynligheder indeholder NaN eller uendelige værdier.")
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Sandsynligheder kan ikke normaliseres.")
    return probabilities / row_sums


def calculate_metrics(y_true, probabilities):
    probabilities = normalize_probabilities(probabilities)
    actual_indices = np.asarray(
        [LABEL_TO_INDEX[value] for value in y_true], dtype=np.int64
    )
    predictions = INDEX_TO_LABEL[probabilities.argmax(axis=1)]
    one_hot = np.eye(3, dtype=np.float64)[actual_indices]
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        ),
        "log_loss": log_loss(
            actual_indices, probabilities, labels=[0, 1, 2]
        ),
        "brier_score": np.mean(
            np.sum((one_hot - probabilities) ** 2, axis=1)
        ),
        **{
            f"recall_{label}": recall_score(
                y_true,
                predictions,
                labels=[label],
                average="macro",
                zero_division=0,
            )
            for label in LABELS
        },
        "predictions": predictions,
        "probabilities": probabilities,
    }


def make_alpha_025_weights(y_subtrain):
    """Compute softened inverse-frequency weights on subtrain only."""
    labels = np.asarray(y_subtrain, dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    present = counts > 0
    baseline_weights = np.zeros(3, dtype=np.float64)
    baseline_weights[present] = (
        len(labels) / (present.sum() * counts[present])
    )
    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline_weights[present] ** ALPHA
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def train_variant(subtrain, validation, test, features, fold, seed, variant):
    """Train one fresh model; fit preprocessing on this variant's subtrain."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_subtrain = scaler.fit_transform(
        imputer.fit_transform(subtrain[features])
    )
    x_validation = scaler.transform(
        imputer.transform(validation[features])
    )
    x_test = scaler.transform(imputer.transform(test[features]))

    y_subtrain = subtrain["result"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    y_validation = validation["result"].map(
        LABEL_TO_INDEX
    ).to_numpy(dtype=np.int64)

    x_subtrain_tensor = torch.tensor(
        x_subtrain, dtype=torch.float32, device=DEVICE
    )
    y_subtrain_tensor = torch.tensor(
        y_subtrain, dtype=torch.long, device=DEVICE
    )
    x_validation_tensor = torch.tensor(
        x_validation, dtype=torch.float32, device=DEVICE
    )
    y_validation_tensor = torch.tensor(
        y_validation, dtype=torch.long, device=DEVICE
    )
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)

    # Reset for every variant so seed/fold runs use comparable RNG settings.
    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)

    network = MatchNet(len(features)).to(DEVICE)
    class_weights = make_alpha_025_weights(y_subtrain)
    training_loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    validation_loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=0.001, weight_decay=0.01
    )

    max_epochs = 100
    batch_size = 256
    patience = 8
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        network.train()
        order = torch.randperm(len(x_subtrain_tensor), device=DEVICE)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            training_loss = training_loss_function(
                network(x_subtrain_tensor[indices]), y_subtrain_tensor[indices]
            )
            training_loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            validation_loss = validation_loss_function(
                network(x_validation_tensor), y_validation_tensor
            ).item()
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in network.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"Ingen checkpoint blev gemt for {variant}.")
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        probabilities = torch.softmax(
            network(x_test_tensor), dim=1
        ).cpu().numpy()
    print(
        f"    {variant}: bedste epoke={best_epoch}, "
        f"uvægtet validerings-loss={best_validation_loss:.6f}"
    )
    return normalize_probabilities(probabilities)


def make_summary_row(scope, variant, comparison, seed, fold, games,
                     metrics, metric_stds=None):
    row = {
        "scope": scope,
        "variant": variant,
        "comparison": comparison,
        "seed": seed,
        "fold": fold,
        "games": games,
    }
    metric_stds = metric_stds or {}
    for metric in METRIC_NAMES:
        row[metric] = metrics.get(metric, np.nan)
        row[f"{metric}_std"] = metric_stds.get(metric, np.nan)
    return row


def metrics_only(metrics):
    return {name: metrics[name] for name in METRIC_NAMES}


def add_mean_std_summary(rows, frame, group_columns, scope):
    """Append mean/std rows, preserving paired deltas when present."""
    for group_values, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        identity = dict(zip(group_columns, group_values))
        means = {metric: group[metric].mean() for metric in METRIC_NAMES}
        stds = {
            metric: group[metric].std(ddof=1) for metric in METRIC_NAMES
        }
        rows.append(make_summary_row(
            scope=scope,
            variant=identity.get("variant", "all"),
            comparison=identity.get("comparison", ""),
            seed="all",
            fold=identity.get("fold", "all"),
            games=group["games"].mean(),
            metrics=means,
            metric_stds=stds,
        ))


def initial_output_check():
    existing = [path for path in (OOF_PATH, SUMMARY_PATH) if path.exists()]
    if existing:
        raise FileExistsError(
            "Output findes allerede; intet overskrives: "
            + ", ".join(str(path) for path in existing)
        )


def write_outputs_exclusively(oof_df, summary_df):
    """Create both outputs exclusively and clean up only files created here."""
    initial_output_check()
    created_paths = []
    handles = []
    try:
        for path in (OOF_PATH, SUMMARY_PATH):
            handle = path.open("x", newline="", encoding="utf-8-sig")
            handles.append(handle)
            created_paths.append(path)
        oof_df.to_csv(handles[0], index=False)
        summary_df.to_csv(handles[1], index=False)
        for handle in handles:
            handle.close()
    except Exception:
        for handle in handles:
            if not handle.closed:
                handle.close()
        for path in created_paths:
            if path.exists():
                path.unlink()
        raise


def main():
    # Check before data loading or any training begins.
    initial_output_check()
    validate_feature_groups()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"])
    elo_columns = KEY_COLUMNS + ELO_FEATURES + ATTACK_DEFENSE_EXPECTED_GOALS_FEATURES
    missing_elo_columns = [column for column in elo_columns if column not in elo]
    if missing_elo_columns:
        raise ValueError(f"Kolonner mangler i elo_features.csv: {missing_elo_columns}")
    df = df.merge(
        elo[elo_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    df = df.sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    df = df.dropna(
        subset=DATA_REQUIRED_FEATURES + ELO_FEATURES
    ).reset_index(drop=True)
    df = df[df["result"].isin(LABEL_TO_INDEX)].reset_index(drop=True)
    if df.empty:
        raise ValueError("Ingen gyldige kampe efter datarensning.")
    missing_features = [
        feature
        for feature in BASE_FEATURES
        if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(f"Baselinefeatures mangler i inputdata: {missing_features}")

    dates = sorted(df["date"].dt.date.unique())
    date_blocks = [
        list(block)
        for block in np.array_split(dates, 4)
        if len(block) > 0
    ]
    if len(date_blocks) != 4:
        raise RuntimeError(
            f"Forventede fire datoblokke til tre folds; fandt {len(date_blocks)}."
        )

    print(f"PyTorch: {torch.__version__}; enhed: {DEVICE}")
    print("Inputfeatures pr. variant:")
    for variant, features in VARIANT_FEATURES.items():
        print(f"  {variant}: {len(features)}")

    prediction_rows = []
    seed_fold_rows = []
    delta_seed_fold_rows = []
    variant_names = list(VARIANT_FEATURES)

    for fold in range(1, len(date_blocks)):
        train_dates = [date for block in date_blocks[:fold] for date in block]
        test_dates = date_blocks[fold]
        train = df[df["date"].dt.date.isin(train_dates)]
        test = df[df["date"].dt.date.isin(test_dates)]
        unique_train_dates = sorted(train["date"].dt.date.unique())
        split_index = int(len(unique_train_dates) * 0.85)
        subtrain_dates = unique_train_dates[:split_index]
        validation_dates = unique_train_dates[split_index:]
        subtrain = train[train["date"].dt.date.isin(subtrain_dates)].copy()
        validation = train[train["date"].dt.date.isin(validation_dates)].copy()
        if subtrain.empty or validation.empty or test.empty:
            raise RuntimeError(
                f"Fold {fold} har tomt subtrain, validation eller test."
            )

        print(
            f"\nFold {fold}: subtrain={len(subtrain)}, "
            f"validation={len(validation)}, test={len(test)}"
        )
        reference_keys = test[KEY_COLUMNS].reset_index(drop=True)

        for seed in SEEDS:
            fold_results = {}
            for variant in variant_names:
                probabilities = train_variant(
                    subtrain=subtrain,
                    validation=validation,
                    test=test,
                    features=VARIANT_FEATURES[variant],
                    fold=fold,
                    seed=seed,
                    variant=variant,
                )
                metrics = calculate_metrics(test["result"].to_numpy(), probabilities)
                fold_results[variant] = metrics
                seed_fold_rows.append(make_summary_row(
                    scope="seed_fold",
                    variant=variant,
                    comparison="",
                    seed=seed,
                    fold=fold,
                    games=len(test),
                    metrics=metrics_only(metrics),
                ))

                current_keys = test[KEY_COLUMNS].reset_index(drop=True)
                if not current_keys.astype("string").equals(
                    reference_keys.astype("string")
                ):
                    raise RuntimeError(
                        f"Testkampnøgler afviger for seed={seed}, fold={fold}, "
                        f"variant={variant}."
                    )

                for index, (_, match) in enumerate(test.iterrows()):
                    prediction_rows.append({
                        "seed": seed,
                        "fold": fold,
                        "variant": variant,
                        "date": match["date"],
                        "time": match["time"],
                        "league": match["league"],
                        "home_team": match["home_team"],
                        "away_team": match["away_team"],
                        "actual": match["result"],
                        "predicted": metrics["predictions"][index],
                        "prob_H": float(metrics["probabilities"][index, 0]),
                        "prob_D": float(metrics["probabilities"][index, 1]),
                        "prob_A": float(metrics["probabilities"][index, 2]),
                    })

            baseline_metrics = fold_results["baseline"]
            for variant in variant_names:
                if variant == "baseline":
                    continue
                delta = {
                    metric: fold_results[variant][metric] - baseline_metrics[metric]
                    for metric in METRIC_NAMES
                }
                delta_seed_fold_rows.append(make_summary_row(
                    scope="seed_fold_delta",
                    variant=variant,
                    comparison="baseline",
                    seed=seed,
                    fold=fold,
                    games=len(test),
                    metrics=delta,
                ))

    oof_df = pd.DataFrame(prediction_rows)
    if oof_df.empty:
        raise RuntimeError("Eksperimentet producerede ingen testforudsigelser.")

    # Summary rows for fold-level seed means/std and paired fold deltas.
    summary_rows = list(seed_fold_rows) + list(delta_seed_fold_rows)
    seed_fold_df = pd.DataFrame(seed_fold_rows)
    delta_seed_fold_df = pd.DataFrame(delta_seed_fold_rows)
    add_mean_std_summary(
        summary_rows, seed_fold_df, ["variant", "fold"], "fold_seed_summary"
    )
    add_mean_std_summary(
        summary_rows,
        delta_seed_fold_df,
        ["variant", "comparison", "fold"],
        "fold_delta_seed_summary",
    )

    # Overall OOF metrics are computed separately per seed; seeds are not pooled
    # as if they were additional independent matches.
    seed_overall_rows = []
    delta_seed_overall_rows = []
    for seed in SEEDS:
        per_variant = {}
        for variant in variant_names:
            group = oof_df[
                (oof_df["seed"] == seed) & (oof_df["variant"] == variant)
            ]
            metrics = calculate_metrics(
                group["actual"].to_numpy(),
                group[["prob_H", "prob_D", "prob_A"]].to_numpy(),
            )
            per_variant[variant] = metrics
            seed_overall_rows.append(make_summary_row(
                scope="seed_overall",
                variant=variant,
                comparison="",
                seed=seed,
                fold="all",
                games=len(group),
                metrics=metrics_only(metrics),
            ))
        for variant in variant_names:
            if variant == "baseline":
                continue
            delta = {
                metric: per_variant[variant][metric]
                - per_variant["baseline"][metric]
                for metric in METRIC_NAMES
            }
            delta_seed_overall_rows.append(make_summary_row(
                scope="seed_overall_delta",
                variant=variant,
                comparison="baseline",
                seed=seed,
                fold="all",
                games=len(oof_df[
                    (oof_df["seed"] == seed) & (oof_df["variant"] == variant)
                ]),
                metrics=delta,
            ))

    summary_rows.extend(seed_overall_rows)
    summary_rows.extend(delta_seed_overall_rows)
    seed_overall_df = pd.DataFrame(seed_overall_rows)
    delta_seed_overall_df = pd.DataFrame(delta_seed_overall_rows)
    add_mean_std_summary(
        summary_rows,
        seed_overall_df,
        ["variant"],
        "overall_seed_summary",
    )
    add_mean_std_summary(
        summary_rows,
        delta_seed_overall_df,
        ["variant", "comparison"],
        "overall_delta_seed_summary",
    )

    summary_columns = [
        "scope", "variant", "comparison", "seed", "fold", "games",
        *[column for metric in METRIC_NAMES
          for column in (metric, f"{metric}_std")],
    ]
    summary_df = pd.DataFrame(summary_rows)[summary_columns]
    write_outputs_exclusively(oof_df, summary_df)
    print(f"\nOOF-fil gemt: {OOF_PATH}")
    print(f"Summary-fil gemt: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
