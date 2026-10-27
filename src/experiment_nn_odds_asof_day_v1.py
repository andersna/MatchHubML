"""Evaluate the modern neural network on the versioned day-replay odds data.

Fold dates are reconstructed from the full, cleaned baseline before odds
filtering, following experiment_nn_odds_modern.py. The versioned as-of-day
baseline is checked against those boundaries; odds rows never redefine folds.
Historical odds have no snapshot timestamps, so pre-kickoff availability is
unverified. This is an experimental backtest only.
"""

from pathlib import Path

# On this Windows environment, load PyTorch before NumPy/Pandas initialize
# their native libraries; the reverse order can prevent c10.dll from loading.
import torch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ODDS_DATA_PATH = DATA / "odds_model_dataset_asof_day_v1.csv"
OOF_PATH = DATA / "oof_predictions_nn_odds_asof_day_v1.csv"
SUMMARY_PATH = DATA / "nn_odds_asof_day_v1_summary.csv"
PREVIOUS_SUMMARY_PATH = DATA / "nn_odds_modern_summary.csv"
PREVIOUS_OOF_PATH = DATA / "oof_predictions_nn_odds_modern.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
ALPHA = 0.25
LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
INDEX_TO_LABEL = np.array(["H", "D", "A"])
KEY_COLUMNS = ["date", "time", "league", "home_team", "away_team"]
ODDS_MATCH_KEYS = ["date", "league", "home_team", "away_team"]
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
ELO_COLUMNS = [
    f"{side}_elo_k{k}" for k in (16, 32, 48) for side in ("home", "away")
] + [f"elo_difference_k{k}" for k in (16, 32, 48)]
NN_FEATURES = BASE_FEATURES + ATTACK_DEFENSE_FEATURES + ELO_COLUMNS
ODDS_FEATURES = ["odds_prob_home", "odds_prob_draw", "odds_prob_away"]
MODEL_FEATURES = NN_FEATURES + ODDS_FEATURES
COMMON_FEATURES = [
    *BASE_FEATURES,
    "home_form_rating", "away_form_rating",
    "home_venue_form_rating", "away_venue_form_rating",
    *ATTACK_DEFENSE_FEATURES,
]
METRICS = ("accuracy", "macro_f1", "log_loss", "brier_score", "recall_H", "recall_D", "recall_A")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatchNet(nn.Module):
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
        raise FileExistsError("Output findes allerede; intet overskrives: " + ", ".join(map(str, existing)))


def _require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} mangler kolonner: {missing}")


def _baseline_dates(features_path, elo_path):
    features = pd.read_csv(features_path, parse_dates=["date"], low_memory=False)
    elo = pd.read_csv(elo_path, parse_dates=["date"], low_memory=False)
    _require_columns(features, KEY_COLUMNS + BASE_FEATURES + [
        "home_form_rating", "away_form_rating", "home_venue_form_rating",
        "away_venue_form_rating", "result",
    ], str(features_path))
    _require_columns(elo, KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES, str(elo_path))
    for frame, name in ((features, str(features_path)), (elo, str(elo_path))):
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} har dublerede fulde kampnøgler.")
    full = features.merge(
        elo[KEY_COLUMNS + ELO_COLUMNS + ATTACK_DEFENSE_FEATURES],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator="_elo_merge",
    )
    if (full["_elo_merge"] != "both").any():
        raise ValueError(f"Ufuldstændigt feature/Elo-join for {features_path}.")
    full = full.drop(columns="_elo_merge").sort_values(["date", "time"], kind="stable")
    full = full.dropna(subset=COMMON_FEATURES + ELO_COLUMNS)
    full = full[full.result.isin(LABEL_TO_INDEX)].copy()
    dates = sorted(full.date.dt.date.unique())
    blocks = [list(block) for block in np.array_split(dates, 4) if len(block)]
    if len(blocks) != 4:
        raise ValueError(f"Kunne ikke danne fire kronologiske blokke for {features_path}.")
    return full, blocks


def load_data():
    # These are the exact source files used by the previous modern procedure.
    prior_full, prior_blocks = _baseline_dates(DATA / "features.csv", DATA / "elo_features.csv")
    # Validate that day-replay did not change the full-baseline calendar blocks.
    day_full, day_blocks = _baseline_dates(
        DATA / "features_asof_day_v1.csv", DATA / "elo_features_asof_day_v1.csv"
    )
    blocks_match = prior_blocks == day_blocks
    print(f"Baseline rows after modern complete-feature/result filters: prior={len(prior_full):,}; as-of-day={len(day_full):,}")
    print(f"Four calendar blocks identical to prior experiment: {blocks_match}")
    for i, (old_block, new_block) in enumerate(zip(prior_blocks, day_blocks), start=1):
        print(
            f"  Block {i}: prior {old_block[0]}..{old_block[-1]} ({len(old_block):,} dates); "
            f"as-of-day {new_block[0]}..{new_block[-1]} ({len(new_block):,} dates)"
        )
    if not blocks_match:
        raise RuntimeError("As-of-day baseline ændrede kalenderfoldgrænser; stopper før træning.")

    data = pd.read_csv(ODDS_DATA_PATH, parse_dates=["date"], low_memory=False)
    _require_columns(data, KEY_COLUMNS + ["result"] + MODEL_FEATURES, ODDS_DATA_PATH.name)
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError("Det nye datasæt har dublerede kampnøgler.")
    if len(NN_FEATURES) != 27 or len(MODEL_FEATURES) != 30:
        raise AssertionError(f"Forkert antal features: NN={len(NN_FEATURES)}, samlet={len(MODEL_FEATURES)}")
    if not data.result.isin(LABEL_TO_INDEX).all():
        raise ValueError("Det nye datasæt indeholder udfald uden H/D/A-mapping.")
    if data[MODEL_FEATURES].isna().any().any():
        raise ValueError("Det nye datasæt har manglende modelinputs.")
    probs = data[ODDS_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all() or (probs <= 0).any() or (probs > 1).any() \
            or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Odds-sandsynligheder er ugyldige.")
    _require_columns(data, KEY_COLUMNS + ["result"] + MODEL_FEATURES, "odds-asof dataset")

    known_dates = {date for block in prior_blocks for date in block}
    outside = sorted(set(data.date.dt.date.unique()) - known_dates)
    if outside:
        raise ValueError(f"Odds-datasættet indeholder datoer uden for baselinefolds: {outside[:10]}")
    previous = pd.read_csv(PREVIOUS_OOF_PATH, parse_dates=["date"], low_memory=False)
    previous_summary = pd.read_csv(PREVIOUS_SUMMARY_PATH, low_memory=False)
    return data.sort_values(["date", "time"], kind="stable").reset_index(drop=True), prior_blocks, previous, previous_summary


def alpha_quarter_weights(y_subtrain):
    counts = np.bincount(np.asarray(y_subtrain, dtype=np.int64), minlength=3)
    present = counts > 0
    baseline = np.zeros(3, dtype=np.float64)
    baseline[present] = len(y_subtrain) / (present.sum() * counts[present])
    weights = np.zeros(3, dtype=np.float64)
    weights[present] = baseline[present] ** ALPHA
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def normalize_probabilities(probs):
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3 or not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Ugyldige netværkssandsynligheder.")
    totals = probs.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Sandsynlighedsrække summerer ikke positivt.")
    return probs / totals


def calculate_metrics(actual, probs):
    probs = normalize_probabilities(probs)
    true_indices = np.asarray([LABEL_TO_INDEX[value] for value in actual], dtype=np.int64)
    predicted = INDEX_TO_LABEL[probs.argmax(axis=1)]
    one_hot = np.eye(3)[true_indices]
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=["H", "D", "A"], average="macro", zero_division=0),
        "log_loss": log_loss(true_indices, probs, labels=[0, 1, 2]),
        "brier_score": float(np.mean(np.sum((one_hot - probs) ** 2, axis=1))),
        "recall_H": recall_score(actual, predicted, labels=["H"], average="macro", zero_division=0),
        "recall_D": recall_score(actual, predicted, labels=["D"], average="macro", zero_division=0),
        "recall_A": recall_score(actual, predicted, labels=["A"], average="macro", zero_division=0),
    }


def train_variant(train, validation, test, feature_columns, fold, seed):
    # All fit operations use subtrain only; validation is used only for unweighted CE stopping.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[feature_columns]))
    x_val = scaler.transform(imputer.transform(validation[feature_columns]))
    x_test = scaler.transform(imputer.transform(test[feature_columns]))
    y_train = train.result.map(LABEL_TO_INDEX).to_numpy(np.int64)
    y_val = validation.result.map(LABEL_TO_INDEX).to_numpy(np.int64)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=DEVICE)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)

    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)
    model = MatchNet(len(feature_columns)).to(DEVICE)
    train_loss = nn.CrossEntropyLoss(weight=torch.tensor(alpha_quarter_weights(y_train), device=DEVICE))
    validation_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_value, best_state, patience_count = float("inf"), None, 0

    for _epoch in range(1, 101):
        model.train()
        order = torch.randperm(len(x_train_t), device=DEVICE)
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            optimizer.zero_grad()
            loss = train_loss(model(x_train_t[batch]), y_train_t[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = validation_loss(model(x_val_t), y_val_t).item()
        if value < best_value:
            best_value = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= 8:
            break
    if best_state is None:
        raise RuntimeError("Early stopping gemte ikke et checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x_test_t), dim=1).cpu().numpy()
    return normalize_probabilities(probs), best_value


def _keys(frame):
    result = frame[KEY_COLUMNS].copy()
    result["date"] = pd.to_datetime(result.date).dt.strftime("%Y-%m-%d")
    for col in KEY_COLUMNS[1:]:
        result[col] = result[col].astype("string").fillna("<MISSING>")
    return result


def _append_predictions(rows, test, variant, seed, fold, probs):
    for i, (_, match) in enumerate(test.iterrows()):
        rows.append({
            "variant": variant, "alpha": ALPHA, "seed": seed, "fold": fold,
            "date": match.date, "time": match.time, "league": match.league,
            "home_team": match.home_team, "away_team": match.away_team,
            "actual": match.result,
            "predicted": INDEX_TO_LABEL[int(np.argmax(probs[i]))],
            "prob_H": probs[i, 0], "prob_D": probs[i, 1], "prob_A": probs[i, 2],
        })


def _assert_same_fold_keys(test, collected, fold):
    reference = _keys(test).reset_index(drop=True)
    for variant in ("baseline_27", "nn_with_odds"):
        for seed in SEEDS:
            group = collected[(collected.fold == fold) & (collected.variant == variant) & (collected.seed == seed)]
            if not _keys(group).reset_index(drop=True).equals(reference):
                raise RuntimeError(f"Testkampene afviger for {variant}, seed={seed}, fold={fold}.")
    odds = collected[(collected.fold == fold) & (collected.variant == "odds_only")]
    if not _keys(odds).reset_index(drop=True).equals(reference):
        raise RuntimeError(f"Odds-only testkampene afviger i fold {fold}.")


def _comparison_with_previous(previous, new_oof):
    rows = []
    old = previous[previous.variant.isin(["baseline_27", "nn_with_odds"])].copy()
    old["seed"] = pd.to_numeric(old.seed).astype(int)
    for (fold, variant, seed), new_group in new_oof[
        new_oof.variant.isin(["baseline_27", "nn_with_odds"])
    ].groupby(["fold", "variant", "seed"], sort=True):
        old_group = old[(old.fold == fold) & (old.variant == variant) & (old.seed == seed)].copy()
        old_keys = _keys(old_group)
        new_keys = _keys(new_group)
        old_group = old_group.assign(**{col: old_keys[col].to_numpy() for col in KEY_COLUMNS})
        new_group = new_group.assign(**{col: new_keys[col].to_numpy() for col in KEY_COLUMNS})
        common = old_group.merge(new_group, on=KEY_COLUMNS, suffixes=("_previous", "_new"), validate="one_to_one")
        if not (common.actual_previous == common.actual_new).all():
            raise RuntimeError("Faktiske udfald afviger for fælles gamle/nye kampnøgler.")
        if common.empty:
            continue
        prev_probs = common[["prob_H_previous", "prob_D_previous", "prob_A_previous"]].to_numpy()
        new_probs = common[["prob_H_new", "prob_D_new", "prob_A_new"]].to_numpy()
        actual = common.actual_new.to_numpy()
        prev_metrics = calculate_metrics(actual, prev_probs)
        new_metrics = calculate_metrics(actual, new_probs)
        row = {"scope": "previous_common_test", "variant": variant, "seed": seed, "fold": fold, "matches": len(common)}
        for metric in METRICS:
            row[metric] = new_metrics[metric]
            row[f"previous_{metric}"] = prev_metrics[metric]
            row[f"delta_{metric}"] = new_metrics[metric] - prev_metrics[metric]
        rows.append(row)
    return rows


def write_outputs(oof, summary):
    check_outputs_absent()
    handles = []
    try:
        for path in (OOF_PATH, SUMMARY_PATH):
            handles.append(path.open("x", encoding="utf-8-sig", newline=""))
    except Exception:
        for handle in handles:
            p = Path(handle.name)
            handle.close()
            if p.exists() and p.stat().st_size == 0:
                p.unlink()
        raise
    try:
        oof.to_csv(handles[0], index=False)
        summary.to_csv(handles[1], index=False)
    except Exception:
        for handle in handles:
            p = Path(handle.name)
            handle.close()
            if p.exists():
                p.unlink()
        raise
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()


def main():
    check_outputs_absent()
    print("Odds-tidsstempler mangler: dette er en eksperimentel backtest, ikke dokumenteret før-kickoff-performance.")
    print(f"PyTorch {torch.__version__}; device={DEVICE}; alpha={ALPHA}; seeds={list(SEEDS)}")
    print(f"Features: NN={len(NN_FEATURES)}, odds={len(ODDS_FEATURES)}, total={len(MODEL_FEATURES)}")
    data, date_blocks, previous_oof, previous_summary = load_data()
    print(f"Nyt odds-covered datasæt: {len(data):,} kampe")

    prediction_rows = []
    summary_rows = []
    evaluated_folds = []
    for fold in range(1, 4):
        train_dates = [date for block in date_blocks[:fold] for date in block]
        test_dates = date_blocks[fold]
        unique_train_dates = sorted(set(train_dates))
        split = int(len(unique_train_dates) * 0.85)
        subtrain_dates, validation_dates = unique_train_dates[:split], unique_train_dates[split:]
        subtrain = data[data.date.dt.date.isin(subtrain_dates)].copy()
        validation = data[data.date.dt.date.isin(validation_dates)].copy()
        test = data[data.date.dt.date.isin(test_dates)].copy().sort_values(["date", "time"], kind="stable")
        if subtrain.empty or validation.empty or test.empty:
            print(f"Fold {fold} udeladt med uændrede grænser: subtrain={len(subtrain)}, validation={len(validation)}, test={len(test)}")
            continue
        evaluated_folds.append(fold)
        print(f"\nFold {fold}: subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}; dates={test_dates[0]}..{test_dates[-1]}")
        for seed in SEEDS:
            for variant, columns in (("baseline_27", NN_FEATURES), ("nn_with_odds", MODEL_FEATURES)):
                probs, val_loss = train_variant(subtrain, validation, test, columns, fold, seed)
                metrics = calculate_metrics(test.result.to_numpy(), probs)
                summary_rows.append({"scope": "seed_fold", "variant": variant, "seed": seed, "fold": fold, "matches": len(test), "alpha": ALPHA, "best_validation_ce": val_loss, **metrics})
                _append_predictions(prediction_rows, test, variant, seed, fold, probs)
                print(f"  {variant} seed={seed}: log_loss={metrics['log_loss']:.5f}; brier={metrics['brier_score']:.5f}")

        odds_probs = normalize_probabilities(test[ODDS_FEATURES].to_numpy(dtype=np.float64))
        _append_predictions(prediction_rows, test, "odds_only", np.nan, fold, odds_probs)
        odds_metrics = calculate_metrics(test.result.to_numpy(), odds_probs)
        print(f"  odds_only reference: log_loss={odds_metrics['log_loss']:.5f}; brier={odds_metrics['brier_score']:.5f}")

        current = pd.DataFrame(prediction_rows)
        _assert_same_fold_keys(test.reset_index(drop=True), current, fold)

    if not evaluated_folds:
        raise RuntimeError("Ingen fold havde ikke-tomme subtrain-, validation- og testmængder.")
    oof = pd.DataFrame(prediction_rows)

    # Seed/fold means and cross-fold OOF metrics, with the same metric definitions as the old experiment.
    for (variant, fold), group in pd.DataFrame([r for r in summary_rows if r["scope"] == "seed_fold"]).groupby(["variant", "fold"]):
        summary_rows.append({"scope": "fold_seed_summary", "variant": variant, "seed": "all", "fold": fold, "matches": group.matches.mean(), "alpha": ALPHA, **{m: group[m].mean() for m in METRICS}, **{f"{m}_std": group[m].std(ddof=1) for m in METRICS}})
    for variant in ("baseline_27", "nn_with_odds"):
        for seed in SEEDS:
            group = oof[(oof.variant == variant) & (oof.seed == seed)]
            metrics = calculate_metrics(group.actual.to_numpy(), group[["prob_H", "prob_D", "prob_A"]].to_numpy())
            summary_rows.append({"scope": "seed_overall", "variant": variant, "seed": seed, "fold": "all", "matches": len(group), "alpha": ALPHA, **metrics})
        seed_rows = [r for r in summary_rows if r["scope"] == "seed_overall" and r["variant"] == variant]
        summary_rows.append({"scope": "overall_seed_summary", "variant": variant, "seed": "all", "fold": "all", "matches": np.mean([r["matches"] for r in seed_rows]), "alpha": ALPHA, **{m: np.mean([r[m] for r in seed_rows]) for m in METRICS}, **{f"{m}_std": np.std([r[m] for r in seed_rows], ddof=1) for m in METRICS}})
    for fold, group in oof[oof.variant == "odds_only"].groupby("fold"):
        summary_rows.append({"scope": "fold_reference", "variant": "odds_only", "seed": "single", "fold": fold, "matches": len(group), "alpha": np.nan, **calculate_metrics(group.actual.to_numpy(), group[["prob_H", "prob_D", "prob_A"]].to_numpy())})
    odds_group = oof[oof.variant == "odds_only"]
    summary_rows.append({"scope": "overall_reference", "variant": "odds_only", "seed": "single", "fold": "all", "matches": len(odds_group), "alpha": np.nan, **calculate_metrics(odds_group.actual.to_numpy(), odds_group[["prob_H", "prob_D", "prob_A"]].to_numpy())})

    # Compare to prior OOF on the exact intersection of old/new test keys.
    comparison_rows = _comparison_with_previous(previous_oof, oof)
    summary_rows.extend(comparison_rows)
    summary_rows.append({
        "scope": "data_comparison", "variant": "all", "seed": "all", "fold": "all",
        "matches": len(data), "alpha": ALPHA,
        "previous_rows": int(previous_summary.loc[previous_summary.scope == "overall_seed_summary", "matches"].max()),
        "previous_full_test_matches": int(previous_oof[previous_oof.variant == "baseline_27"].groupby(["fold", "seed"]).size().min()),
        "shared_test_matches_per_seed_fold_min": min((r["matches"] for r in comparison_rows), default=0),
    })

    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")
    write_outputs(oof, summary)

    print("\n=== Sammenligning med tidligere odds-eksperiment ===")
    print("Tidligere samlede fem-seed gennemsnit:")
    print(previous_summary[previous_summary.scope == "overall_seed_summary"][
        ["variant", "matches", "accuracy", "macro_f1", "log_loss", "brier_score", "recall_H", "recall_D", "recall_A"]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nNy dagsvis evaluering, fem-seed gennemsnit:")
    print(summary[summary.scope == "overall_seed_summary"][
        ["variant", "matches", "accuracy", "macro_f1", "log_loss", "brier_score", "recall_H", "recall_D", "recall_A"]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nNye testfolds, gennemsnit over seeds:")
    print(summary[summary.scope == "fold_seed_summary"][
        ["variant", "fold", "matches", "accuracy", "macro_f1", "log_loss", "brier_score", "recall_H", "recall_D", "recall_A"]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if comparison_rows:
        print("\nNy minus tidligere model på fælles testkampe (pr. fold, gennemsnit over seeds):")
        paired = pd.DataFrame(comparison_rows).groupby(["variant", "fold"], as_index=False).agg(
            matches=("matches", "min"),
            delta_accuracy=("delta_accuracy", "mean"),
            delta_macro_f1=("delta_macro_f1", "mean"),
            delta_log_loss=("delta_log_loss", "mean"),
            delta_brier_score=("delta_brier_score", "mean"),
            delta_recall_D=("delta_recall_D", "mean"),
        )
        print(paired.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nParret sammenligning på fælles kampnøgler gemmes også i summary scope=previous_common_test.")
    print(f"OOF: {OOF_PATH}\nSummary: {SUMMARY_PATH}")


if __name__ == "__main__":
    check_outputs_absent()
    main()
