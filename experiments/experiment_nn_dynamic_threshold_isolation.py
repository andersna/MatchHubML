"""Isolate a dynamic draw rule from adding expected-goals gap as an NN feature.

A: 27-feature network with argmax. B: the same 27-feature probabilities with
the fixed 0.85 draw ratio. C: the same 27-feature network with a dynamic rule
that uses the pre-match absolute expected-goals gap only for the decision. D:
the 28-feature network also uses that gap as an input, then applies the same
dynamic rule. For both dynamic variants, if gap <= cutoff use the low draw
ratio; otherwise use the high ratio. Choose D when P(D) >= ratio * max(P(H),
P(A)); otherwise choose the larger of H/A.

The predeclared candidate grid and mean-Macro-F1 selection procedure match the
preceding experiment. To isolate the rule from the extra model input, the
shared rule parameters are selected on the 27-feature dynamic variant's
validation predictions from folds 1 and 2, averaged across seeds. Fold-3 test
predictions are not calculated until these parameters are frozen. Rules change
labels only, never probabilities.
"""

from itertools import product
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
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_dynamic_threshold_isolation.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_dynamic_threshold_isolation_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHA = 0.25
FIXED_BASELINE_THRESHOLD = 0.85
# Small, predeclared grid. Every high threshold is greater than every low one.
GOAL_GAP_CUTOFFS = (0.25, 0.50, 0.75)
LOW_DRAW_THRESHOLDS = (0.65, 0.70, 0.75)
HIGH_DRAW_THRESHOLDS = (0.85, 0.90, 0.95)
CANDIDATES = tuple(product(
    GOAL_GAP_CUTOFFS, LOW_DRAW_THRESHOLDS, HIGH_DRAW_THRESHOLDS
))

VARIANT_A = "A_baseline_argmax"
VARIANT_B = "B_baseline_fixed_0.85"
VARIANT_C = "C_27_features_dynamic_threshold"
VARIANT_D = "D_28_features_dynamic_threshold"
VARIANTS = (VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D)
DYNAMIC_VARIANTS = (VARIANT_C, VARIANT_D)
LABELS = ["H", "D", "A"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = np.asarray(LABELS)
KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]
GOAL_GAP_FEATURE = "abs_expected_goals_difference"

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
BASE_FEATURE_COLUMNS = BASE_FEATURES + ATTACK_DEFENSE_FEATURES + ELO_COLUMNS
GOAL_GAP_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [GOAL_GAP_FEATURE]
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
    """Existing network architecture: input -> 32 -> 16 -> 3."""

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
    """Same present-class logic, inverse frequencies and mean-1 normalization."""
    labels = np.asarray(y_subtrain, dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    present = counts > 0
    baseline = np.zeros(3, dtype=np.float64)
    baseline[present] = len(labels) / (present.sum() * counts[present])
    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline[present] ** ALPHA
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def piecewise_draw_predictions(probabilities, goal_gap, cutoff, low_threshold, high_threshold):
    """Apply the C rule without modifying model probabilities."""
    probabilities = np.asarray(probabilities)
    goal_gap = np.asarray(goal_gap)
    thresholds = np.where(goal_gap <= cutoff, low_threshold, high_threshold)
    top_non_draw = np.maximum(probabilities[:, 0], probabilities[:, 2])
    return np.where(
        probabilities[:, 1] >= thresholds * top_non_draw,
        "D",
        np.where(probabilities[:, 0] >= probabilities[:, 2], "H", "A"),
    )


def candidate_macro_f1(actual, probabilities, goal_gap, candidate):
    cutoff, low_threshold, high_threshold = candidate
    predictions = piecewise_draw_predictions(
        probabilities, goal_gap, cutoff, low_threshold, high_threshold
    )
    return float(f1_score(
        actual, predictions, labels=LABELS, average="macro", zero_division=0
    ))


def calculate_metrics(actual, predictions, probabilities):
    """Same metric definitions as earlier experiments; probabilities stay raw."""
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
        # Same multiclass Brier definition as the existing experiment scripts.
        "brier_score": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
    }
    for index, label in enumerate(LABELS):
        metrics[f"precision_{label}"] = float(precision[index])
        metrics[f"recall_{label}"] = float(recall[index])
        metrics[f"f1_{label}"] = float(class_f1[index])
        metrics[f"share_pred_{label}"] = float(np.mean(predictions == label))
    return metrics


def train_for_validation(subtrain, validation, feature_columns, fold, seed, class_weights):
    """Train/checkpoint on subtrain/validation and return validation probabilities."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_subtrain = scaler.fit_transform(imputer.fit_transform(subtrain[feature_columns]))
    x_validation = scaler.transform(imputer.transform(validation[feature_columns]))
    y_subtrain = subtrain.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    y_validation = validation.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    x_subtrain_tensor = torch.tensor(x_subtrain, dtype=torch.float32, device=DEVICE)
    y_subtrain_tensor = torch.tensor(y_subtrain, dtype=torch.long, device=DEVICE)
    x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32, device=DEVICE)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.long, device=DEVICE)

    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)
    network = MatchNet(len(feature_columns)).to(DEVICE)
    training_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
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
    print(f"    fold={fold}, seed={seed}, bedste epoke={best_epoch}, val-loss={best_loss:.6f}")
    return normalize_probabilities(validation_probabilities)


def train_for_test(subtrain, validation, test, feature_columns, fold, seed, class_weights):
    """Train with the same procedure, then infer test probabilities."""
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_subtrain = scaler.fit_transform(imputer.fit_transform(subtrain[feature_columns]))
    x_validation = scaler.transform(imputer.transform(validation[feature_columns]))
    x_test = scaler.transform(imputer.transform(test[feature_columns]))
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
    network = MatchNet(len(feature_columns)).to(DEVICE)
    training_loss = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
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
        probabilities = torch.softmax(network(x_test_tensor), dim=1).cpu().numpy()
    print(f"    fold={fold}, seed={seed}, bedste epoke={best_epoch}, val-loss={best_loss:.6f}")
    return normalize_probabilities(probabilities)


def key_frame(frame):
    result = frame[KEY_COLUMNS].copy()
    for column in KEY_COLUMNS:
        result[column] = result[column].astype("string").fillna("")
    return result.reset_index(drop=True)


def metric_summary_row(scope, variant, seed, fold, metrics=None, matches=np.nan,
                       stds=None, cutoff=np.nan, low_threshold=np.nan,
                       high_threshold=np.nan):
    row = {
        "scope": scope, "variant": variant, "seed": seed, "fold": fold,
        "kampe": matches, "goal_gap_cutoff": cutoff,
        "low_draw_threshold": low_threshold,
        "high_draw_threshold": high_threshold,
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


def score_candidate_grid(validation_by_seed):
    """Score each rule on folds 1+2 validation, then average Macro F1 by seed."""
    per_seed_rows = []
    for seed in SEEDS:
        parts = validation_by_seed[seed]
        actual = np.concatenate([part["actual"] for part in parts])
        probabilities = np.vstack([part["probabilities"] for part in parts])
        gap = np.concatenate([part["goal_gap"] for part in parts])
        for cutoff, low_threshold, high_threshold in CANDIDATES:
            predictions = piecewise_draw_predictions(
                probabilities, gap, cutoff, low_threshold, high_threshold
            )
            score = float(f1_score(
                actual, predictions, labels=LABELS,
                average="macro", zero_division=0,
            ))
            per_seed_rows.append({
                "seed": seed,
                "goal_gap_cutoff": cutoff,
                "low_draw_threshold": low_threshold,
                "high_draw_threshold": high_threshold,
                "validation_macro_f1": score,
            })
    per_seed = pd.DataFrame(per_seed_rows)
    summary = per_seed.groupby(
        ["goal_gap_cutoff", "low_draw_threshold", "high_draw_threshold"],
        as_index=False,
    ).agg(
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
    )
    # Deterministic tie-break: maximize F1, then prefer the more conservative
    # rule (smaller small-gap region, higher low/high thresholds).
    ordered = summary.sort_values(
        ["validation_macro_f1_mean", "goal_gap_cutoff", "low_draw_threshold", "high_draw_threshold"],
        ascending=[False, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    selected = ordered.iloc[0]
    params = (
        float(selected.goal_gap_cutoff),
        float(selected.low_draw_threshold),
        float(selected.high_draw_threshold),
    )
    summary["selected"] = (
        (summary.goal_gap_cutoff == params[0])
        & (summary.low_draw_threshold == params[1])
        & (summary.high_draw_threshold == params[2])
    )
    return params, per_seed, summary


def save_one_variant(fold, seed, variant, test, expected_keys, probabilities,
                     params, oof_rows, summary_rows):
    cutoff, low_threshold, high_threshold = params
    actual = test.result.to_numpy()
    goal_gap = test[GOAL_GAP_FEATURE].to_numpy(dtype=np.float64)
    probabilities = normalize_probabilities(probabilities)
    unchanged = probabilities.copy()
    if variant == VARIANT_A:
        predictions = INDEX_TO_LABEL[probabilities.argmax(axis=1)]
    elif variant == VARIANT_B:
        predictions = np.where(
            probabilities[:, 1] >= FIXED_BASELINE_THRESHOLD
            * np.maximum(probabilities[:, 0], probabilities[:, 2]),
            "D",
            np.where(probabilities[:, 0] >= probabilities[:, 2], "H", "A"),
        )
    elif variant in DYNAMIC_VARIANTS:
        predictions = piecewise_draw_predictions(
            probabilities, goal_gap, cutoff, low_threshold, high_threshold
        )
    else:
        raise ValueError(f"Ukendt variant: {variant}")
    if not np.array_equal(unchanged, probabilities):
        raise RuntimeError("En beslutningsregel ændrede sandsynlighederne.")

    metrics = calculate_metrics(actual, predictions, probabilities)
    summary_rows.append(metric_summary_row(
        "seed_fold", variant, seed, fold, metrics, len(test),
        cutoff=cutoff if variant in DYNAMIC_VARIANTS else np.nan,
        low_threshold=low_threshold if variant in DYNAMIC_VARIANTS else (
            FIXED_BASELINE_THRESHOLD if variant == VARIANT_B else np.nan
        ),
        high_threshold=high_threshold if variant in DYNAMIC_VARIANTS else np.nan,
    ))
    rows = []
    for index, (_, match) in enumerate(test.iterrows()):
        row = {
            "variant": variant, "seed": seed, "fold": fold,
            "date": match.date, "time": match.time,
            "league": match.league, "home_team": match.home_team,
            "away_team": match.away_team, "actual": match.result,
            "predicted": predictions[index],
            "abs_expected_goals_difference": float(goal_gap[index]),
            "goal_gap_cutoff": cutoff if variant in DYNAMIC_VARIANTS else np.nan,
            "low_draw_threshold": low_threshold if variant in DYNAMIC_VARIANTS else (
                FIXED_BASELINE_THRESHOLD if variant == VARIANT_B else np.nan
            ),
            "high_draw_threshold": high_threshold if variant in DYNAMIC_VARIANTS else np.nan,
            "prob_H": float(probabilities[index, 0]),
            "prob_D": float(probabilities[index, 1]),
            "prob_A": float(probabilities[index, 2]),
        }
        rows.append(row)
        oof_rows.append(row)
    if not key_frame(pd.DataFrame(rows)).equals(key_frame(expected_keys)):
        raise RuntimeError(
            f"Testkampenes nøgler/rækkefølge afviger for {variant}, seed={seed}, fold={fold}."
        )


def main():
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

    # Derived only from the two pre-match expected-goals features.
    data[GOAL_GAP_FEATURE] = (
        data.home_expected_goals - data.away_expected_goals
    ).abs()
    if len(BASE_FEATURE_COLUMNS) != 27 or len(GOAL_GAP_FEATURE_COLUMNS) != 28:
        raise RuntimeError("Featurelisterne afviger fra de forventede 27/28 features.")

    date_blocks = [
        list(block)
        for block in np.array_split(sorted(data.date.dt.date.unique()), 4)
        if len(block)
    ]
    if len(date_blocks) != 4:
        raise RuntimeError("Forventede fire kronologiske datoblokke.")
    print(f"PyTorch: {torch.__version__}; enhed: {DEVICE}; kampe: {len(data)}")
    print(f"Seeds: {list(SEEDS)}; alpha={ALPHA}; features A/B/C=27, D=28")

    # Threshold tuning phase: validation probabilities only from fold 1 and 2.
    validation_by_seed = {seed: [] for seed in SEEDS}
    validation_key_frames = []
    for fold in (1, 2):
        train_dates = [date for block in date_blocks[:fold] for date in block]
        train = data[data.date.dt.date.isin(train_dates)]
        unique_train_dates = sorted(train.date.dt.date.unique())
        split = int(len(unique_train_dates) * 0.85)
        subtrain_dates, validation_dates = unique_train_dates[:split], unique_train_dates[split:]
        subtrain = train[train.date.dt.date.isin(subtrain_dates)].copy()
        validation = train[train.date.dt.date.isin(validation_dates)].copy()
        if subtrain.empty or validation.empty:
            raise RuntimeError(f"Fold {fold} har tomt subtrain/valideringssæt.")
        validation_key_frames.append(key_frame(validation))
        print(f"\nParametervalg fold {fold}: subtrain={len(subtrain)}, validation={len(validation)}")
        for seed in SEEDS:
            y_subtrain = subtrain.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
            weights = make_alpha_025_weights(y_subtrain)
            # Tune on the 27-feature dynamic-rule model so the extra feature
            # in D cannot influence rule selection.
            val_probabilities = train_for_validation(
                subtrain, validation, BASE_FEATURE_COLUMNS, fold, seed, weights
            )
            validation_by_seed[seed].append({
                "actual": validation.result.to_numpy(),
                "probabilities": val_probabilities,
                "goal_gap": validation[GOAL_GAP_FEATURE].to_numpy(dtype=np.float64),
                "fold": fold,
            })

    if validation_key_frames[0]["date"].max() >= validation_key_frames[1]["date"].min():
        raise RuntimeError("Fold 1/2 valideringsdata er ikke kronologisk adskilt.")
    # Ensure the two tuning validation slices do not reuse any match keys.
    overlap = validation_key_frames[0].merge(
        validation_key_frames[1], on=KEY_COLUMNS, how="inner"
    )
    if not overlap.empty:
        raise RuntimeError("Fold 1 og 2 genbruger valideringskampe ved tærskelvalg.")

    fixed_params, candidate_seed_scores, candidate_summary = score_candidate_grid(validation_by_seed)
    cutoff, low_threshold, high_threshold = fixed_params
    print(
        f"\nFrosne C-parametre fra fold 1+2-validering: cutoff={cutoff:.2f}, "
        f"lav D-tærskel={low_threshold:.2f}, høj D-tærskel={high_threshold:.2f}"
    )
    print_table(
        "Kandidater: gennemsnitlig Macro F1 på fold 1+2-validering over seeds",
        candidate_summary.sort_values(
            ["validation_macro_f1_mean", "goal_gap_cutoff", "low_draw_threshold", "high_draw_threshold"],
            ascending=[False, True, False, False], kind="stable",
        ),
        ["goal_gap_cutoff", "low_draw_threshold", "high_draw_threshold",
         "validation_macro_f1_mean", "validation_macro_f1_std", "selected"],
    )

    # Test phase: parameters are frozen before fold 3 test data is accessed.
    fold = 3
    train_dates = [date for block in date_blocks[:fold] for date in block]
    test_dates = date_blocks[fold]
    train = data[data.date.dt.date.isin(train_dates)]
    test = data[data.date.dt.date.isin(test_dates)]
    unique_train_dates = sorted(train.date.dt.date.unique())
    split = int(len(unique_train_dates) * 0.85)
    subtrain_dates, validation_dates = unique_train_dates[:split], unique_train_dates[split:]
    subtrain = train[train.date.dt.date.isin(subtrain_dates)].copy()
    validation = train[train.date.dt.date.isin(validation_dates)].copy()
    if subtrain.empty or validation.empty or test.empty:
        raise RuntimeError("Fold 3 har tomt subtrain/validering/test-sæt.")
    expected_test_keys = key_frame(test)
    print(f"\nFold 3 urørt test: subtrain={len(subtrain)}, validation={len(validation)}, test={len(test)}")

    oof_rows, summary_rows = [], []
    for seed in SEEDS:
        y_subtrain = subtrain.result.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
        weights = make_alpha_025_weights(y_subtrain)
        # A and B share the exact same 27-feature model and probabilities.
        probabilities_a = train_for_test(
            subtrain, validation, test, BASE_FEATURE_COLUMNS, fold, seed, weights
        )
        # C differs by exactly the derived absolute expected-goals gap feature.
        probabilities_d = train_for_test(
            subtrain, validation, test, GOAL_GAP_FEATURE_COLUMNS, fold, seed, weights
        )

        # A, B, C share the same 27-feature raw probabilities; D uses the
        # separate 28-feature model. All are scored on the identical test rows.
        save_one_variant(
            fold, seed, VARIANT_A, test, expected_test_keys, probabilities_a,
            fixed_params, oof_rows, summary_rows,
        )
        save_one_variant(
            fold, seed, VARIANT_B, test, expected_test_keys, probabilities_a,
            fixed_params, oof_rows, summary_rows,
        )
        save_one_variant(
            fold, seed, VARIANT_C, test, expected_test_keys, probabilities_a,
            fixed_params, oof_rows, summary_rows,
        )
        save_one_variant(
            fold, seed, VARIANT_D, test, expected_test_keys, probabilities_d,
            fixed_params, oof_rows, summary_rows,
        )

    oof_df = pd.DataFrame(oof_rows)
    seed_overall_rows = []
    for (variant, seed), group in oof_df.groupby(["variant", "seed"], sort=True):
        probabilities = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        metrics = calculate_metrics(group.actual.to_numpy(), group.predicted.to_numpy(), probabilities)
        seed_overall_rows.append({"variant": variant, "seed": seed, **metrics})
        summary_rows.append(metric_summary_row(
            "seed_overall", variant, seed, "fold3", metrics, len(group),
            cutoff=cutoff if variant in DYNAMIC_VARIANTS else np.nan,
            low_threshold=(low_threshold if variant in DYNAMIC_VARIANTS else (
                FIXED_BASELINE_THRESHOLD if variant == VARIANT_B else np.nan
            )),
            high_threshold=high_threshold if variant in DYNAMIC_VARIANTS else np.nan,
        ))

    seed_fold_df = pd.DataFrame([r for r in summary_rows if r["scope"] == "seed_fold"])
    for variant, group in seed_fold_df.groupby("variant", sort=True):
        means = {name: group[name].mean() for name in METRIC_NAMES}
        stds = {name: group[name].std(ddof=1) for name in METRIC_NAMES}
        summary_rows.append(metric_summary_row(
            "fold3_seed_summary", variant, "all", 3, means, group.kampe.mean(),
            stds=stds,
            cutoff=cutoff if variant in DYNAMIC_VARIANTS else np.nan,
            low_threshold=(low_threshold if variant in DYNAMIC_VARIANTS else (
                FIXED_BASELINE_THRESHOLD if variant == VARIANT_B else np.nan
            )),
            high_threshold=high_threshold if variant in DYNAMIC_VARIANTS else np.nan,
        ))

    # Candidate validation table is retained in the summary output for auditability.
    candidate_output = candidate_summary.copy()
    candidate_output["scope"] = "validation_candidate_mean_across_seeds"
    candidate_output["variant"] = VARIANT_C
    candidate_output["seed"] = "all"
    candidate_output["fold"] = "1+2_validation"
    candidate_output["kampe"] = np.nan
    candidate_output["selected"] = candidate_output["selected"].astype(bool)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = pd.concat([summary_df, candidate_output], ignore_index=True, sort=False)
    candidate_seed_output = candidate_seed_scores.copy()
    candidate_seed_output["scope"] = "validation_candidate_seed"
    candidate_seed_output["variant"] = VARIANT_C
    candidate_seed_output["fold"] = "1+2_validation"
    candidate_seed_output["kampe"] = np.nan
    summary_df = pd.concat([summary_df, candidate_seed_output], ignore_index=True, sort=False)

    # A/B/C share identical 27-feature probabilities; D has separate outputs.
    for seed in SEEDS:
        a = oof_df[(oof_df.seed == seed) & (oof_df.variant == VARIANT_A)].reset_index(drop=True)
        b = oof_df[(oof_df.seed == seed) & (oof_df.variant == VARIANT_B)].reset_index(drop=True)
        c = oof_df[(oof_df.seed == seed) & (oof_df.variant == VARIANT_C)].reset_index(drop=True)
        d = oof_df[(oof_df.seed == seed) & (oof_df.variant == VARIANT_D)].reset_index(drop=True)
        for candidate in (a, b, c, d):
            if not key_frame(candidate).equals(expected_test_keys):
                raise RuntimeError(f"Variant testkampene/rækkefølgen afviger for seed={seed}.")
        probabilities_a = a[["prob_H", "prob_D", "prob_A"]].to_numpy()
        for candidate in (b, c):
            if not np.array_equal(
                probabilities_a,
                candidate[["prob_H", "prob_D", "prob_A"]].to_numpy(),
            ):
                raise RuntimeError("A/B/C skal bruge identiske 27-feature sandsynligheder.")

    print("\nMetrikretning: accuracy, Macro F1, precision og recall højere er bedre; "
          "log loss og Brier score lavere er bedre. Klasseandele er deskriptive.")
    print_table(
        "Resultater pr. seed på fold 3",
        pd.DataFrame([r for r in summary_rows if r["scope"] == "seed_fold"]).sort_values(["seed", "variant"]),
        ["seed", "variant", "goal_gap_cutoff", "low_draw_threshold", "high_draw_threshold", *REPORT_METRICS],
    )
    mean_std = summary_df[summary_df.scope == "fold3_seed_summary"].copy()
    summary_metrics = [column for metric in REPORT_METRICS for column in (metric, f"{metric}_std")]
    print_table("Gennemsnit og standardafvigelse på tværs af seeds (fold 3)", mean_std, ["variant", *summary_metrics])
    overall = pd.DataFrame(seed_overall_rows)
    paired = overall.pivot(index="seed", columns="variant", values=METRIC_NAMES)
    changes = []
    for seed in paired.index:
        row = {"seed": seed}
        for variant in (VARIANT_B, VARIANT_C, VARIANT_D):
            for metric in METRIC_NAMES:
                row[f"delta_{metric}_{variant}_minus_A"] = (
                    paired.loc[seed, (metric, variant)]
                    - paired.loc[seed, (metric, VARIANT_A)]
                )
        for left, right in (
            (VARIANT_C, VARIANT_B), (VARIANT_D, VARIANT_B),
            (VARIANT_D, VARIANT_C),
        ):
            for metric in METRIC_NAMES:
                row[f"delta_{metric}_{left}_minus_{right}"] = (
                    paired.loc[seed, (metric, left)]
                    - paired.loc[seed, (metric, right)]
                )
        changes.append(row)
    change_df = pd.DataFrame(changes)
    avg = {"seed": "mean across seeds"}
    for column in change_df.columns[1:]:
        avg[column] = change_df[column].mean()
    change_df = pd.concat([change_df, pd.DataFrame([avg])], ignore_index=True)
    print_table(
        "Parrede forskelle for alle mål (fold 3; højere/lavere afhænger af målet)",
        change_df, list(change_df.columns),
    )

    # Include a compact fixed-threshold and selection record alongside metrics.
    summary_df["fixed_baseline_threshold"] = np.where(
        summary_df.get("variant", pd.Series(index=summary_df.index, dtype=object)) == VARIANT_B,
        FIXED_BASELINE_THRESHOLD, np.nan,
    )
    summary_df["selected_common_cutoff"] = cutoff
    summary_df["selected_common_low_threshold"] = low_threshold
    summary_df["selected_common_high_threshold"] = high_threshold

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

    print(f"\nOOF-fil: {OOF_PATH}")
    print(f"Summary-fil: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
