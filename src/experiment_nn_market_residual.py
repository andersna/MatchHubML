"""Experiment: neural network that learns corrections to the betting market.

1. Market residual: logits = log(market probabilities) + correction(features).
   The correction's last layer starts at zero, so an untrained model equals
   the market exactly, and the untouched market is the epoch-0 checkpoint for
   early stopping: validation can never select a model worse than the market.
3. Features: home-minus-away differences instead of collinear pairs, as-of-day
   shot form and form ratings, league home advantage, rest days, matches played
   this season, promoted flag and a league embedding. Every feature uses only
   information known before kickoff (as-of-day files, or fixture dates).
4. Training: no class weights, 5-seed probability ensemble, and variants with
   recency weighting and label smoothing.

Folds, subtrain/validation split and metrics are identical to
experiment_nn_odds_asof_day_v1.py, whose current model is rerun as reference.
Historical odds have no snapshot timestamps; this is a backtest only.
"""

from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from experiment_nn_odds_asof_day_v1 import (
    DEVICE,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    METRICS,
    MODEL_FEATURES,
    NN_FEATURES,
    ODDS_FEATURES,
    calculate_metrics,
    load_data,
    normalize_probabilities,
    train_variant,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OOF_PATH = DATA / "oof_predictions_nn_market_residual.csv"
SUMMARY_PATH = DATA / "nn_market_residual_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
MAX_EPOCHS = 100
PATIENCE = 8
BATCH_SIZE = 256
RECENCY_HALF_LIFE_DAYS = 3 * 365.25
REST_DAYS_CAP = 30

FORM_EXTRAS = [
    f"{side}_{name}"
    for side in ("home", "away")
    for name in (
        "last5_shots_for", "last5_shots_against",
        "last5_shots_on_target_for", "last5_shots_on_target_against",
        "form_rating", "venue_form_rating",
    )
] + ["league_home_advantage"]

DIFFERENCE_PAIRS = [
    "avg_goals_for", "avg_goals_against", "avg_points",
    "last5_goals_for", "last5_goals_against", "last5_points",
    "last5_shots_for", "last5_shots_against",
    "last5_shots_on_target_for", "last5_shots_on_target_against",
    "form_rating", "venue_form_rating", "attack_strength",
    "defense_strength", "expected_goals", "rest_days", "matches_played",
]
ENGINEERED_FEATURES = (
    [f"{name}_diff" for name in DIFFERENCE_PAIRS]
    + [
        "home_expected_goals", "away_expected_goals", "elo_difference_k32",
        "league_home_advantage", "home_rest_days", "away_rest_days",
        "home_matches_played", "away_matches_played",
        "home_promoted", "away_promoted",
    ]
    + ODDS_FEATURES  # lets the correction recalibrate the market non-linearly
)

# name -> (feature columns, residual?, league embedding?, recency?, label smoothing)
VARIANTS = {
    "current_v1": (MODEL_FEATURES, False, False, False, 0.0),
    "residual_27": (NN_FEATURES + ODDS_FEATURES, True, False, False, 0.0),
    "residual_new_features": (ENGINEERED_FEATURES, True, True, False, 0.0),
    "residual_new_recency": (ENGINEERED_FEATURES, True, True, True, 0.0),
    "residual_new_recency_ls": (ENGINEERED_FEATURES, True, True, True, 0.05),
}


def check_outputs_absent():
    existing = [path for path in (OOF_PATH, SUMMARY_PATH) if path.exists()]
    if existing:
        raise FileExistsError("Output findes allerede; intet overskrives: " + ", ".join(map(str, existing)))


def schedule_features(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Rest days, matches played this season and promoted flag per match.

    Uses only fixture dates and league membership, which are known before
    kickoff. Returned per KEY_COLUMNS row of ``fixtures``.
    """
    frame = fixtures[KEY_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    # The source season label ("2019-20"), so delayed seasons like COVID 2019-20
    # are not split by a calendar cutoff.
    frame["season"] = fixtures["season"].str[:4].astype(int).to_numpy()
    frame["row"] = np.arange(len(frame))
    long = pd.concat([
        frame.assign(side="home", team=frame["home_team"]),
        frame.assign(side="away", team=frame["away_team"]),
    ], ignore_index=True).sort_values(["date", "row"], kind="stable")

    by_team = long.groupby(["league", "team"], sort=False)
    rest = (long["date"] - by_team["date"].shift()).dt.days
    long["rest_days"] = rest.clip(upper=REST_DAYS_CAP).fillna(REST_DAYS_CAP)
    long["matches_played"] = long.groupby(["league", "team", "season"]).cumcount()

    seasons = set(zip(long["league"], long["team"], long["season"]))
    first_season = long.groupby("league")["season"].transform("min")
    long["promoted"] = [
        float(season > first and (league, team, season - 1) not in seasons)
        for league, team, season, first in zip(
            long["league"], long["team"], long["season"], first_season
        )
    ]

    result = frame[["row"]].copy()
    for side in ("home", "away"):
        part = long.loc[long["side"] == side].set_index("row")
        for name in ("rest_days", "matches_played", "promoted"):
            result[f"{side}_{name}"] = part[name].reindex(result["row"]).to_numpy()
    out = fixtures[KEY_COLUMNS].copy()
    for column in result.columns.drop("row"):
        out[column] = result[column].to_numpy()
    return out


def _key_frame(frame):
    keys = frame[KEY_COLUMNS].copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    keys["time"] = keys["time"].astype("string").fillna("<MISSING>")
    return keys


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    form = pd.read_csv(DATA / "features_asof_day_v1.csv", low_memory=False)
    fixtures = pd.read_csv(DATA / "fixtures.csv", low_memory=False)
    schedule = schedule_features(fixtures)

    result = data.copy()
    merge_keys = _key_frame(result)
    for source, columns in ((form, FORM_EXTRAS), (schedule, [
        f"{side}_{name}" for side in ("home", "away")
        for name in ("rest_days", "matches_played", "promoted")
    ])):
        right = _key_frame(source)
        right[columns] = source[columns].to_numpy()
        merged = merge_keys.merge(right, on=KEY_COLUMNS, how="left", validate="one_to_one")
        if len(merged) != len(result):
            raise RuntimeError("Featurejoin ændrede rækketallet.")
        missing_rows = merged[columns].isna().all(axis=1).sum()
        if missing_rows:
            raise RuntimeError(f"{missing_rows} kampe fandt ingen ekstra features ved join.")
        result[columns] = merged[columns].to_numpy()

    for name in DIFFERENCE_PAIRS:
        result[f"{name}_diff"] = result[f"home_{name}"] - result[f"away_{name}"]
    return result


class MarketResidualNet(nn.Module):
    """logits = log(market) + correction(features, league)."""

    def __init__(self, input_size, n_leagues, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_leagues, embedding_dim) if embedding_dim else None
        self.correction = nn.Sequential(
            nn.Linear(input_size + embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x, league, market_log):
        if self.embedding is not None:
            x = torch.cat([x, self.embedding(league)], dim=1)
        return market_log + self.correction(x)


def recency_weights(dates: pd.Series) -> np.ndarray:
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=np.float64)
    weights = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return (weights / weights.mean()).astype(np.float32)


def train_residual(train, validation, test, columns, use_embedding, use_recency,
                   label_smoothing, fold, seed, market_columns=ODDS_FEATURES):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x = {
        "train": scaler.fit_transform(imputer.fit_transform(train[columns])),
        "validation": scaler.transform(imputer.transform(validation[columns])),
        "test": scaler.transform(imputer.transform(test[columns])),
    }
    frames = {"train": train, "validation": validation, "test": test}
    leagues = {name: index for index, name in enumerate(sorted(train["league"].unique()))}

    def tensors(name):
        frame = frames[name]
        market = frame[list(market_columns)].to_numpy(dtype=np.float64)
        return (
            torch.tensor(x[name], dtype=torch.float32, device=DEVICE),
            torch.tensor(frame["league"].map(leagues).fillna(len(leagues)).to_numpy(np.int64), device=DEVICE),
            torch.tensor(np.log(market), dtype=torch.float32, device=DEVICE),
        )

    x_train, league_train, market_train = tensors("train")
    x_val, league_val, market_val = tensors("validation")
    x_test, league_test, market_test = tensors("test")
    y_train = torch.tensor(train.result.map(LABEL_TO_INDEX).to_numpy(np.int64), device=DEVICE)
    y_val = torch.tensor(validation.result.map(LABEL_TO_INDEX).to_numpy(np.int64), device=DEVICE)
    weights = torch.tensor(
        recency_weights(train["date"]) if use_recency else np.ones(len(train), np.float32),
        device=DEVICE,
    )

    effective_seed = seed + fold
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    model = MarketResidualNet(
        len(columns), len(leagues) + 1, 4 if use_embedding else 0
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    def validation_loss():
        model.eval()
        with torch.no_grad():
            return F.cross_entropy(model(x_val, league_val, market_val), y_val).item()

    # Epoch 0 is the untouched market.
    best_value = validation_loss()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, patience_count = 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_train), device=DEVICE)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            losses = F.cross_entropy(
                model(x_train[batch], league_train[batch], market_train[batch]),
                y_train[batch], reduction="none", label_smoothing=label_smoothing,
            )
            loss = (losses * weights[batch]).sum() / weights[batch].sum()
            loss.backward()
            optimizer.step()
        value = validation_loss()
        if value < best_value:
            best_value, best_epoch, patience_count = value, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x_test, league_test, market_test), dim=1).cpu().numpy()
    return normalize_probabilities(probs), best_value, best_epoch


def per_match_log_loss(actual, probs):
    index = np.asarray([LABEL_TO_INDEX[value] for value in actual])
    return -np.log(np.clip(probs[np.arange(len(index)), index], 1e-15, 1.0))


def main():
    check_outputs_absent()
    started = time.perf_counter()
    print(f"PyTorch {torch.__version__}; device={DEVICE}; seeds={list(SEEDS)}")
    data, date_blocks, _previous_oof, _previous_summary = load_data()
    data = add_features(data)
    print(f"Datasæt: {len(data):,} kampe; nye features: {len(ENGINEERED_FEATURES)}")

    prediction_rows, summary_rows = [], []
    for fold in range(1, 4):
        train_dates = sorted({date for block in date_blocks[:fold] for date in block})
        split = int(len(train_dates) * 0.85)
        subtrain = data[data.date.dt.date.isin(train_dates[:split])].copy()
        validation = data[data.date.dt.date.isin(train_dates[split:])].copy()
        test = data[data.date.dt.date.isin(date_blocks[fold])].copy().sort_values(["date", "time"], kind="stable")
        print(f"\nFold {fold}: subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}")
        if subtrain.empty or validation.empty or test.empty:
            print(f"Fold {fold} udeladt: ingen odds-dækkede træningskampe (som i v1-eksperimentet).")
            continue
        actual = test.result.to_numpy()
        keys = test[KEY_COLUMNS].reset_index(drop=True)

        def record(variant, seed, probs, extra=None):
            metrics = calculate_metrics(actual, probs)
            summary_rows.append({"scope": "seed_fold", "variant": variant, "seed": seed,
                                 "fold": fold, "matches": len(test), **(extra or {}), **metrics})
            frame = keys.assign(variant=variant, seed=seed, fold=fold, actual=actual,
                                prob_H=probs[:, 0], prob_D=probs[:, 1], prob_A=probs[:, 2])
            prediction_rows.append(frame)
            return metrics

        record("odds_only", "single", normalize_probabilities(test[ODDS_FEATURES].to_numpy(np.float64)))
        for variant, (columns, residual, embedding, recency, smoothing) in VARIANTS.items():
            seed_probs = []
            for seed in SEEDS:
                t = time.perf_counter()
                if residual:
                    probs, val_loss, epoch = train_residual(
                        subtrain, validation, test, columns, embedding, recency, smoothing, fold, seed
                    )
                else:
                    (probs, val_loss), epoch = train_variant(subtrain, validation, test, columns, fold, seed), np.nan
                seed_probs.append(probs)
                metrics = record(variant, seed, probs, {"best_validation_ce": val_loss, "best_epoch": epoch})
                print(f"  {variant:<26} seed={seed:<5} acc={metrics['accuracy']:.4f} "
                      f"ll={metrics['log_loss']:.5f} epoch={epoch} ({time.perf_counter() - t:.1f}s)", flush=True)
            record(f"{variant}_ensemble", "mean5", np.mean(seed_probs, axis=0))

    oof = pd.concat(prediction_rows, ignore_index=True)
    odds = oof[oof.variant == "odds_only"].reset_index(drop=True)
    odds_ll = per_match_log_loss(odds.actual.to_numpy(), odds[["prob_H", "prob_D", "prob_A"]].to_numpy())
    overall = []
    for (variant, seed), group in oof.groupby(["variant", "seed"], sort=False):
        group = group.reset_index(drop=True)
        if not group[KEY_COLUMNS].equals(odds[KEY_COLUMNS]):
            raise RuntimeError(f"Testkampe afviger for {variant}/{seed}.")
        probs = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        metrics = calculate_metrics(group.actual.to_numpy(), probs)
        # Paired difference against the market on identical matches.
        diff = per_match_log_loss(group.actual.to_numpy(), probs) - odds_ll
        hit = (probs.argmax(1) == group.actual.map(LABEL_TO_INDEX).to_numpy()).astype(float)
        odds_hit = (odds[["prob_H", "prob_D", "prob_A"]].to_numpy().argmax(1)
                    == odds.actual.map(LABEL_TO_INDEX).to_numpy()).astype(float)
        overall.append({
            "scope": "overall", "variant": variant, "seed": seed, "fold": "all",
            "matches": len(group), **metrics,
            "delta_log_loss_vs_odds": diff.mean(),
            "delta_log_loss_vs_odds_se": diff.std(ddof=1) / np.sqrt(len(diff)),
            "delta_accuracy_vs_odds": (hit - odds_hit).mean(),
            "delta_accuracy_vs_odds_se": (hit - odds_hit).std(ddof=1) / np.sqrt(len(diff)),
        })
    summary = pd.DataFrame(summary_rows + overall)
    summary.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")

    check_outputs_absent()
    with OOF_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        oof.to_csv(handle, index=False)
    with SUMMARY_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        summary.to_csv(handle, index=False)

    table = pd.DataFrame(overall)
    seed_mean = (table[~table.seed.isin(["single", "mean5"])]
                 .groupby("variant", sort=False)[["accuracy", "log_loss", "brier_score", "recall_D",
                                                  "delta_log_loss_vs_odds", "delta_accuracy_vs_odds"]].mean())
    columns = ["accuracy", "log_loss", "brier_score", "recall_D", "delta_accuracy_vs_odds",
               "delta_accuracy_vs_odds_se", "delta_log_loss_vs_odds", "delta_log_loss_vs_odds_se"]
    print("\n=== Alle tre testfolds samlet (OOF), gennemsnit pr. enkelt-seed ===")
    print(seed_mean.to_string(float_format=lambda v: f"{v:.5f}"))
    print("\n=== Odds alene og 5-seed ensembler ===")
    print(table[table.seed.isin(["single", "mean5"])].set_index("variant")[columns]
          .to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"\nOOF: {OOF_PATH}\nSummary: {SUMMARY_PATH}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
