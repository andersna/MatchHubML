"""Modern, odds-covered comparison for the existing MatchHubML neural net.

The chronological date blocks are made from the full baseline dataset before
filtering for odds. The historical odds files do not contain snapshot-time
metadata, so this experiment cannot establish that odds were captured before
kickoff; treat this limitation as part of every interpretation of its results.
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
ODDS_PATH = PROJECT_ROOT / "data" / "odds_model_dataset.csv"
OOF_PATH = PROJECT_ROOT / "data" / "oof_predictions_nn_odds_modern.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "nn_odds_modern_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHA = 0.25
METRIC_NAMES = (
    "accuracy", "macro_f1", "log_loss", "brier_score",
    "recall_H", "recall_D", "recall_A",
)

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
ODDS_FEATURES = [
    "odds_prob_home", "odds_prob_draw", "odds_prob_away"
]
KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]
ODDS_MATCH_KEYS = ["date", "league", "home_team", "away_team"]
COMMON_FEATURES = [
    *BASE_FEATURES,
    "home_form_rating",
    "away_form_rating",
    "home_venue_form_rating",
    "away_venue_form_rating",
    *ATTACK_DEFENSE_FEATURES,
]
LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
INDEX_TO_LABEL = np.array(["H", "D", "A"])
VARIANTS = {
    "baseline_27": NN_FEATURES,
    "nn_with_odds": NN_FEATURES + ODDS_FEATURES,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchNet(nn.Module):
    """Existing 27 → 32 → 16 → 3 network; only input width can vary."""

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
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("Forventede tre sandsynligheder pr. kamp.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Sandsynlighederne skal være endelige og ikke-negative.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("En sandsynlighedsrække summerer ikke positivt.")
    return probabilities / row_sums


def calculate_metrics(y_true, probabilities):
    probabilities = normalize_probabilities(probabilities)
    actual_indices = np.asarray(
        [LABEL_TO_INDEX[value] for value in y_true], dtype=np.int64
    )
    predicted_indices = probabilities.argmax(axis=1)
    predicted = INDEX_TO_LABEL[predicted_indices]
    one_hot = np.eye(3, dtype=np.float64)[actual_indices]
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "macro_f1": f1_score(
            y_true, predicted, labels=["H", "D", "A"],
            average="macro", zero_division=0,
        ),
        "log_loss": log_loss(
            actual_indices, probabilities, labels=[0, 1, 2]
        ),
        "brier_score": np.mean(
            np.sum((one_hot - probabilities) ** 2, axis=1)
        ),
        "recall_H": recall_score(
            y_true, predicted, labels=["H"], average="macro", zero_division=0
        ),
        "recall_D": recall_score(
            y_true, predicted, labels=["D"], average="macro", zero_division=0
        ),
        "recall_A": recall_score(
            y_true, predicted, labels=["A"], average="macro", zero_division=0
        ),
    }


def alpha_quarter_weights(y_subtrain):
    """Inverse-frequency weights raised to alpha, from subtrain only."""
    labels = np.asarray(y_subtrain, dtype=np.int64)
    counts = np.bincount(labels, minlength=3)
    present = counts > 0
    baseline = np.zeros(3, dtype=np.float64)
    baseline[present] = len(labels) / (present.sum() * counts[present])
    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline[present] ** ALPHA
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def train_variant(subtrain, validation, test, feature_columns, fold, seed):
    # Fit imputation and scaling on subtrain only; validation/test are transforms.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_subtrain = scaler.fit_transform(
        imputer.fit_transform(subtrain[feature_columns])
    )
    x_validation = scaler.transform(
        imputer.transform(validation[feature_columns])
    )
    x_test = scaler.transform(imputer.transform(test[feature_columns]))

    y_subtrain = subtrain["result"].map(LABEL_TO_INDEX).to_numpy(np.int64)
    y_validation = validation["result"].map(LABEL_TO_INDEX).to_numpy(np.int64)
    x_train_t = torch.tensor(x_subtrain, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_subtrain, dtype=torch.long, device=DEVICE)
    x_val_t = torch.tensor(x_validation, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_validation, dtype=torch.long, device=DEVICE)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)

    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)

    network = MatchNet(len(feature_columns)).to(DEVICE)
    weights = torch.tensor(alpha_quarter_weights(y_subtrain), device=DEVICE)
    train_loss_fn = nn.CrossEntropyLoss(weight=weights)
    # The modern procedure selects checkpoints with unweighted validation CE.
    validation_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=0.001, weight_decay=0.01
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    without_improvement = 0

    for epoch in range(1, 101):
        network.train()
        order = torch.randperm(len(x_train_t), device=DEVICE)
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            optimizer.zero_grad()
            loss = train_loss_fn(network(x_train_t[indices]), y_train_t[indices])
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            validation_loss = validation_loss_fn(
                network(x_val_t), y_val_t
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in network.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= 8:
            break

    if best_state is None:
        raise RuntimeError("Early stopping gemte ikke et gyldigt checkpoint.")
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        probabilities = torch.softmax(network(x_test_t), dim=1).cpu().numpy()
    print(
        f"    seed={seed}, bedste epoke={best_epoch}, "
        f"uvægtet validation loss={best_loss:.6f}"
    )
    return normalize_probabilities(probabilities)


def load_joined_data():
    baseline = pd.read_csv(DATA_PATH, parse_dates=["date"], low_memory=False)
    elo = pd.read_csv(ELO_PATH, parse_dates=["date"], low_memory=False)
    odds = pd.read_csv(ODDS_PATH, parse_dates=["date"], low_memory=False)

    # The six attack/defense/expected-goals columns come from elo_features.csv,
    # exactly as in the modern baseline merge below; they are not in features.csv.
    features_file_columns = BASE_FEATURES + [
        "home_form_rating",
        "away_form_rating",
        "home_venue_form_rating",
        "away_venue_form_rating",
    ]
    required_base = set(KEY_COLUMNS + features_file_columns + ["result"])
    required_elo = set(KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES)
    required_odds = set(
        ODDS_MATCH_KEYS + ["result"] + ODDS_FEATURES
    )
    for label, frame, required in (
        ("features.csv", baseline, required_base),
        ("elo_features.csv", elo, required_elo),
        ("odds_model_dataset.csv", odds, required_odds),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} mangler kolonner: {missing}")

    if baseline.duplicated(KEY_COLUMNS).any():
        raise ValueError("features.csv har dubletter på baseline-kampnøgler.")
    if elo.duplicated(KEY_COLUMNS).any():
        raise ValueError("elo_features.csv har dubletter på kampnøgler.")
    if odds.duplicated(ODDS_MATCH_KEYS).any():
        raise ValueError("odds_model_dataset.csv har dubletter på odds-kampnøgler.")

    baseline = baseline.merge(
        elo[KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES],
        on=KEY_COLUMNS, how="left", validate="one_to_one",
    )
    baseline = baseline.sort_values(["date", "time"], kind="stable")
    baseline = baseline.dropna(
        subset=COMMON_FEATURES + ELO_COLUMNS
    ).reset_index(drop=True)
    baseline = baseline[baseline["result"].isin(LABEL_TO_INDEX)].copy()
    if baseline.empty:
        raise ValueError("Baseline-datasættet er tomt efter datarensning.")

    odds_prob = odds[ODDS_MATCH_KEYS + ["result"] + ODDS_FEATURES].copy()
    odds_prob[ODDS_FEATURES] = odds_prob[ODDS_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    if odds_prob[ODDS_FEATURES].isna().any().any():
        raise ValueError("Odds-sandsynligheder indeholder manglende værdier.")
    p = odds_prob[ODDS_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError("Odds-sandsynligheder skal være endelige i (0, 1].")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Odds-sandsynligheder summerer ikke til 1.")
    if not odds_prob["result"].isin(LABEL_TO_INDEX).all():
        raise ValueError("Odds-datasættet indeholder ukendte faktiske udfald.")

    # Modern fold boundaries are derived from the full valid baseline dataset.
    # Odds filtering happens only after the date blocks and 85/15 date split.
    dates = sorted(baseline["date"].dt.date.unique())
    date_blocks = [
        list(block) for block in np.array_split(dates, 4) if len(block)
    ]
    if len(date_blocks) != 4:
        raise ValueError("Kunne ikke danne de fire moderne kalenderblokke.")

    baseline_for_merge = baseline.copy()
    baseline_for_merge["_date_key"] = baseline_for_merge["date"].dt.normalize()
    odds_prob["_date_key"] = pd.to_datetime(odds_prob["date"]).dt.normalize()
    # The odds source is keyed by date/league/teams (no time field). Validate
    # one-to-one matching so a bookmaker row cannot duplicate a match.
    merged = baseline_for_merge.merge(
        odds_prob.drop(columns="date"),
        left_on=["_date_key", "league", "home_team", "away_team"],
        right_on=["_date_key", "league", "home_team", "away_team"],
        how="inner", validate="one_to_one", suffixes=("", "_odds"),
    )
    if merged.empty:
        raise ValueError("Ingen fælles odds-dækkede kampe blev fundet.")
    if not (merged["result"] == merged["result_odds"]).all():
        raise ValueError("Faktiske udfald afviger mellem baseline og oddsdata.")
    merged = merged.drop(columns=["result_odds", "_date_key"])
    return baseline, merged, date_blocks


def make_summary_row(scope, variant, seed, fold, metrics, matches, stds=None):
    row = {
        "scope": scope, "variant": variant, "seed": seed,
        "fold": fold, "matches": matches,
    }
    stds = stds or {}
    for metric in METRIC_NAMES:
        row[metric] = metrics.get(metric, np.nan)
        row[f"{metric}_std"] = stds.get(metric, np.nan)
    return row


def evaluate_and_append(summary_rows, variant, seed, fold, actual, probs, scope):
    metrics = calculate_metrics(actual, probs)
    summary_rows.append(
        make_summary_row(scope, variant, seed, fold, metrics, len(actual))
    )
    return metrics


def write_new_files(oof_df, summary_df):
    check_outputs_absent()
    handles = []
    try:
        # Exclusive creation closes the race between checking and saving.
        for path in (OOF_PATH, SUMMARY_PATH):
            handles.append(path.open("x", encoding="utf-8-sig", newline=""))
    except Exception:
        for handle in handles:
            name = Path(handle.name)
            handle.close()
            if name.exists() and name.stat().st_size == 0:
                name.unlink()
        raise
    try:
        oof_df.to_csv(handles[0], index=False)
        summary_df.to_csv(handles[1], index=False)
    except Exception:
        for handle in handles:
            name = Path(handle.name)
            handle.close()
            if name.exists():
                name.unlink()
        raise
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()


def main():
    check_outputs_absent()
    print(
        "Odds-begrænsning: datakilderne indeholder ikke odds-snapshot-tidspunkter; "
        "resultaterne dokumenterer derfor ikke, at odds var kendt før kickoff."
    )
    baseline, odds_data, date_blocks = load_joined_data()
    print(f"PyTorch: {torch.__version__}; enhed: {DEVICE}")
    print(f"Baseline-kampe efter moderne datarensning: {len(baseline)}")
    print(f"Odds-dækkede fælles kampe: {len(odds_data)}")
    print("Features A/B:", len(NN_FEATURES), len(NN_FEATURES + ODDS_FEATURES))

    predictions = []
    summary_rows = []
    evaluated_folds = []
    for fold in range(1, len(date_blocks)):
        train_dates = [date for block in date_blocks[:fold] for date in block]
        test_dates = date_blocks[fold]
        unique_train_dates = sorted(set(train_dates))
        split_index = int(len(unique_train_dates) * 0.85)
        subtrain_dates = unique_train_dates[:split_index]
        validation_dates = unique_train_dates[split_index:]

        subtrain = odds_data[
            odds_data["date"].dt.date.isin(subtrain_dates)
        ].copy()
        validation = odds_data[
            odds_data["date"].dt.date.isin(validation_dates)
        ].copy()
        test = odds_data[odds_data["date"].dt.date.isin(test_dates)].copy()

        # Fold 1 predates the first odds-covered training/validation data.
        # Keep its original boundaries and skip it, rather than moving dates
        # or reconstructing folds after filtering; doing so would change the
        # modern evaluation period and could not train a fair A/B comparison.
        if subtrain.empty or validation.empty:
            print(
                f"Fold {fold} springes over: odds-dækket subtrain="
                f"{len(subtrain)}, validation={len(validation)}, test={len(test)}."
            )
            continue
        if test.empty:
            print(f"Fold {fold} springes over: ingen odds-dækkede testkampe.")
            continue
        if not subtrain["result"].isin(LABEL_TO_INDEX).all() \
                or not validation["result"].isin(LABEL_TO_INDEX).all() \
                or not test["result"].isin(LABEL_TO_INDEX).all():
            raise ValueError(f"Fold {fold} indeholder ukendte udfald.")

        evaluated_folds.append(fold)
        print(
            f"\nFold {fold}: odds-kampe subtrain={len(subtrain)}, "
            f"validation={len(validation)}, test={len(test)}"
        )
        expected_keys = test[KEY_COLUMNS].reset_index(drop=True)
        for seed in SEEDS:
            for variant, feature_columns in VARIANTS.items():
                probabilities = train_variant(
                    subtrain, validation, test, feature_columns, fold, seed
                )
                if len(probabilities) != len(expected_keys):
                    raise RuntimeError("Testforudsigelsernes antal afviger.")
                actual = test["result"].to_numpy()
                metrics = evaluate_and_append(
                    summary_rows, variant, seed, fold, actual,
                    probabilities, "seed_fold",
                )
                for index, (_, match) in enumerate(test.iterrows()):
                    predictions.append({
                        "variant": variant,
                        "seed": seed,
                        "fold": fold,
                        "date": match["date"],
                        "time": match["time"],
                        "league": match["league"],
                        "home_team": match["home_team"],
                        "away_team": match["away_team"],
                        "actual": match["result"],
                        "predicted": INDEX_TO_LABEL[
                            int(np.argmax(probabilities[index]))
                        ],
                        "prob_H": probabilities[index, 0],
                        "prob_D": probabilities[index, 1],
                        "prob_A": probabilities[index, 2],
                    })
                candidate_keys = pd.DataFrame(
                    predictions[-len(test):]
                )[KEY_COLUMNS].reset_index(drop=True)
                if not candidate_keys.astype("string").equals(
                    expected_keys.astype("string")
                ):
                    raise RuntimeError(
                        f"Testkampenes nøgler/rækkefølge afviger: "
                        f"{variant}, seed={seed}, fold={fold}."
                    )
                print(
                    f"  {variant}, seed={seed}: log_loss="
                    f"{metrics['log_loss']:.5f}, brier="
                    f"{metrics['brier_score']:.5f}"
                )

        # Odds-only is evaluated once per match/fold, not once per seed.
        odds_probs = normalize_probabilities(
            test[ODDS_FEATURES].to_numpy(dtype=np.float64)
        )
        odds_metrics = calculate_metrics(test["result"].to_numpy(), odds_probs)
        for index, (_, match) in enumerate(test.iterrows()):
            predictions.append({
                "variant": "odds_only",
                "seed": np.nan,
                "fold": fold,
                "date": match["date"],
                "time": match["time"],
                "league": match["league"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "actual": match["result"],
                "predicted": INDEX_TO_LABEL[int(np.argmax(odds_probs[index]))],
                "prob_H": odds_probs[index, 0],
                "prob_D": odds_probs[index, 1],
                "prob_A": odds_probs[index, 2],
            })
        print(
            f"  odds_only: log_loss={odds_metrics['log_loss']:.5f}, "
            f"brier={odds_metrics['brier_score']:.5f}"
        )

    if not evaluated_folds:
        raise RuntimeError("Ingen fold havde odds i subtrain, validation og test.")

    oof_df = pd.DataFrame(predictions)
    # Each A/B seed/fold must exactly match the canonical test key sequence;
    # odds_only appears once per test match, not once per seed.
    for fold in evaluated_folds:
        fold_rows = oof_df[(oof_df.fold == fold) & (oof_df.variant != "odds_only")]
        references = {}
        for seed in SEEDS:
            for variant in VARIANTS:
                group = fold_rows[
                    (fold_rows.seed == seed) & (fold_rows.variant == variant)
                ].reset_index(drop=True)
                keys = group[KEY_COLUMNS].astype("string")
                if len(group) != len(group[KEY_COLUMNS].drop_duplicates()):
                    raise RuntimeError("Dubletter i OOF-kampnøgler.")
                key_tuple = tuple(map(tuple, keys.to_numpy()))
                references[(seed, variant)] = key_tuple
        all_keys = list(references.values())
        if any(keys != all_keys[0] for keys in all_keys[1:]):
            raise RuntimeError(f"A/B bruger ikke identiske testkampe i fold {fold}.")
        odds_group = oof_df[
            (oof_df.fold == fold) & (oof_df.variant == "odds_only")
        ].reset_index(drop=True)
        odds_keys = tuple(map(
            tuple,
            odds_group[KEY_COLUMNS].astype("string").to_numpy(),
        ))
        if odds_keys != all_keys[0]:
            raise RuntimeError(
                f"Odds-only bruger ikke de samme testkampe i fold {fold}."
            )

    # Summaries for each seed across evaluated OOF folds.
    seed_overall_metrics = []
    for variant in VARIANTS:
        for seed in SEEDS:
            group = oof_df[
                (oof_df.variant == variant) & (oof_df.seed == seed)
            ]
            metrics = calculate_metrics(
                group.actual.to_numpy(),
                group[["prob_H", "prob_D", "prob_A"]].to_numpy(),
            )
            seed_overall_metrics.append({"variant": variant, "seed": seed,
                                         **metrics, "matches": len(group)})
            summary_rows.append(make_summary_row(
                "seed_overall", variant, seed, "all", metrics, len(group)
            ))

    seed_fold_df = pd.DataFrame(
        [r for r in summary_rows if r["scope"] == "seed_fold"]
    )
    for (variant, fold), group in seed_fold_df.groupby(
        ["variant", "fold"], sort=True
    ):
        summary_rows.append(make_summary_row(
            "fold_seed_summary", variant, "all", fold,
            {m: group[m].mean() for m in METRIC_NAMES},
            group.matches.mean(),
            {m: group[m].std(ddof=1) for m in METRIC_NAMES},
        ))
    overall_seed_df = pd.DataFrame(seed_overall_metrics)
    for variant, group in overall_seed_df.groupby("variant", sort=True):
        summary_rows.append(make_summary_row(
            "overall_seed_summary", variant, "all", "all",
            {m: group[m].mean() for m in METRIC_NAMES},
            group.matches.mean(),
            {m: group[m].std(ddof=1) for m in METRIC_NAMES},
        ))

    odds_oof = oof_df[oof_df.variant == "odds_only"]
    for fold, group in odds_oof.groupby("fold", sort=True):
        summary_rows.append(make_summary_row(
            "fold", "odds_only", "single", fold,
            calculate_metrics(group.actual.to_numpy(),
                              group[["prob_H", "prob_D", "prob_A"]].to_numpy()),
            len(group),
        ))
    summary_rows.append(make_summary_row(
        "overall", "odds_only", "single", "all",
        calculate_metrics(odds_oof.actual.to_numpy(),
                          odds_oof[["prob_H", "prob_D", "prob_A"]].to_numpy()),
        len(odds_oof),
    ))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(
        0, "odds_timestamp_caveat",
        "No odds snapshot timestamps; pre-kickoff timing is unverified",
    )
    write_new_files(oof_df, summary_df)
    print("\nMetrikker: accuracy, Macro F1 og recall højere er bedre; "
          "log loss og Brier lavere er bedre.")
    print("Fold 1 blev udeladt, fordi odds-dækket subtrain og validation var tomme.")
    print("Odds-tidspunktet er ukendt; resultater kan ikke dokumentere odds før kickoff.")
    report_columns = [
        "scope", "variant", "seed", "fold", "matches", *METRIC_NAMES,
    ]
    print("\n=== Samlet resultatoversigt ===")
    print(summary_df[report_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    ))
    print(f"OOF: {OOF_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
