"""Experiment: does margin-of-victory Elo improve the v3 market-residual model?

Elo variants (build_elo_features_asof_day_v1.MOV_ELO_COLUMNS):
- current: the 9 win/draw/loss Elo columns used by v3.
- mov: K scaled by goal difference with an autocorrelation correction, so
  big wins count more, mostly when the underdog wins.
- movr: mov plus a one-third regression to 1500 at each new season.

Every variant is the v3 setup (residual_27 in experiment_nn_market_residual.py:
no embedding, no recency weighting, no label smoothing) with only the Elo
columns changed. Folds, splits and metrics are identical to that experiment,
and each variant is compared to the current Elo on the same test matches.
Historical odds have no snapshot timestamps; this is a backtest only.
"""

from pathlib import Path
import time

# On this Windows environment, load PyTorch before NumPy/Pandas.
import torch
import numpy as np
import pandas as pd

from build_elo_features_asof_day_v1 import K_FACTORS, MOV_ELO_COLUMNS, build_mov_elo_features
from experiment_nn_market_residual import _key_frame, per_match_log_loss, train_residual
from experiment_nn_odds_asof_day_v1 import (
    BASE_FEATURES,
    ATTACK_DEFENSE_FEATURES,
    ELO_COLUMNS,
    KEY_COLUMNS,
    LABEL_TO_INDEX,
    ODDS_FEATURES,
    calculate_metrics,
    load_data,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OOF_PATH = DATA / "oof_predictions_nn_mov_elo.csv"
SUMMARY_PATH = DATA / "nn_mov_elo_summary.csv"

SEEDS = (42, 123, 2026, 3407, 9876)
BASELINE = "current"


def elo_columns(variant: str) -> list[str]:
    """The 9 Elo columns of a variant, in the same order as ELO_COLUMNS."""
    if variant == BASELINE:
        return list(ELO_COLUMNS)
    return [
        f"{side}_elo_{variant}_k{k}" for k in K_FACTORS for side in ("home", "away")
    ] + [f"elo_{variant}_difference_k{k}" for k in K_FACTORS]


VARIANTS = {
    BASELINE: BASE_FEATURES + ATTACK_DEFENSE_FEATURES + elo_columns(BASELINE) + ODDS_FEATURES,
    "mov": BASE_FEATURES + ATTACK_DEFENSE_FEATURES + elo_columns("mov") + ODDS_FEATURES,
    "movr": BASE_FEATURES + ATTACK_DEFENSE_FEATURES + elo_columns("movr") + ODDS_FEATURES,
    "current_plus_movr": (
        BASE_FEATURES + ATTACK_DEFENSE_FEATURES + elo_columns(BASELINE)
        + elo_columns("movr") + ODDS_FEATURES
    ),
}


def check_outputs_absent():
    existing = [path for path in (OOF_PATH, SUMMARY_PATH) if path.exists()]
    if existing:
        raise FileExistsError("Output findes allerede; intet overskrives: " + ", ".join(map(str, existing)))


def add_mov_elo(data: pd.DataFrame) -> pd.DataFrame:
    """Join the MoV Elo columns, computed from data/fixtures.csv, onto ``data``."""
    mov = build_mov_elo_features(pd.read_csv(DATA / "fixtures.csv", low_memory=False))
    right = _key_frame(mov)
    right[MOV_ELO_COLUMNS] = mov[MOV_ELO_COLUMNS].to_numpy()
    merged = _key_frame(data).merge(right, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if len(merged) != len(data) or merged[MOV_ELO_COLUMNS].isna().any().any():
        raise RuntimeError("MoV-Elo kunne ikke kobles entydigt på alle kampe.")
    result = data.copy()
    result[MOV_ELO_COLUMNS] = merged[MOV_ELO_COLUMNS].to_numpy()
    return result


def main():
    check_outputs_absent()
    started = time.perf_counter()
    print(f"PyTorch {torch.__version__}; seeds={list(SEEDS)}")
    data, date_blocks, _previous_oof, _previous_summary = load_data()
    data = add_mov_elo(data)
    print(f"Datasæt: {len(data):,} kampe; varianter: {list(VARIANTS)}")

    prediction_rows, summary_rows = [], []
    for fold in range(1, 4):
        train_dates = sorted({date for block in date_blocks[:fold] for date in block})
        split = int(len(train_dates) * 0.85)
        subtrain = data[data.date.dt.date.isin(train_dates[:split])].copy()
        validation = data[data.date.dt.date.isin(train_dates[split:])].copy()
        test = data[data.date.dt.date.isin(date_blocks[fold])].copy().sort_values(["date", "time"], kind="stable")
        print(f"\nFold {fold}: subtrain={len(subtrain):,}; validation={len(validation):,}; test={len(test):,}")
        if subtrain.empty or validation.empty or test.empty:
            print(f"Fold {fold} udeladt: ingen odds-dækkede træningskampe.")
            continue
        actual = test.result.to_numpy()
        keys = test[KEY_COLUMNS].reset_index(drop=True)

        def record(variant, seed, probs, extra=None):
            metrics = calculate_metrics(actual, probs)
            summary_rows.append({"scope": "seed_fold", "variant": variant, "seed": seed,
                                 "fold": fold, "matches": len(test), **(extra or {}), **metrics})
            prediction_rows.append(keys.assign(variant=variant, seed=seed, fold=fold, actual=actual,
                                               prob_H=probs[:, 0], prob_D=probs[:, 1], prob_A=probs[:, 2]))
            return metrics

        record("odds_only", "single", test[ODDS_FEATURES].to_numpy(np.float64))
        for variant, columns in VARIANTS.items():
            seed_probs = []
            for seed in SEEDS:
                t = time.perf_counter()
                probs, val_loss, epoch = train_residual(
                    subtrain, validation, test, columns, False, False, 0.0, fold, seed
                )
                seed_probs.append(probs)
                metrics = record(variant, seed, probs, {"best_validation_ce": val_loss, "best_epoch": epoch})
                print(f"  {variant:<18} seed={seed:<5} acc={metrics['accuracy']:.4f} "
                      f"ll={metrics['log_loss']:.5f} epoch={epoch} ({time.perf_counter() - t:.1f}s)", flush=True)
            record(f"{variant}_ensemble", "mean5", np.mean(seed_probs, axis=0))

    oof = pd.concat(prediction_rows, ignore_index=True)
    probs_columns = ["prob_H", "prob_D", "prob_A"]

    def losses(variant, seed):
        group = oof[(oof.variant == variant) & (oof.seed == seed)].reset_index(drop=True)
        return group, per_match_log_loss(group.actual.to_numpy(), group[probs_columns].to_numpy())

    odds, odds_ll = losses("odds_only", "single")
    baseline, baseline_ll = losses(f"{BASELINE}_ensemble", "mean5")
    overall = []
    for (variant, seed), _ in oof.groupby(["variant", "seed"], sort=False):
        group, ll = losses(variant, seed)
        if not group[KEY_COLUMNS].equals(odds[KEY_COLUMNS]):
            raise RuntimeError(f"Testkampe afviger for {variant}/{seed}.")
        row = {"scope": "overall", "variant": variant, "seed": seed, "fold": "all", "matches": len(group),
               **calculate_metrics(group.actual.to_numpy(), group[probs_columns].to_numpy())}
        # Paired differences on identical matches: vs the market and vs current Elo.
        for name, reference in (("odds", odds_ll), ("current_elo", baseline_ll)):
            diff = ll - reference
            row[f"delta_log_loss_vs_{name}"] = diff.mean()
            row[f"delta_log_loss_vs_{name}_se"] = diff.std(ddof=1) / np.sqrt(len(diff))
        overall.append(row)
    summary = pd.DataFrame(summary_rows + overall)
    summary.insert(0, "odds_timestamp_caveat", "No snapshot timestamps; odds pre-kickoff timing unverified")

    check_outputs_absent()
    with OOF_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        oof.to_csv(handle, index=False)
    with SUMMARY_PATH.open("x", encoding="utf-8-sig", newline="") as handle:
        summary.to_csv(handle, index=False)

    table = pd.DataFrame(overall)
    columns = ["accuracy", "log_loss", "brier_score", "delta_log_loss_vs_odds", "delta_log_loss_vs_odds_se",
               "delta_log_loss_vs_current_elo", "delta_log_loss_vs_current_elo_se"]
    print("\n=== Alle testfolds samlet (OOF): odds alene og 5-seed ensembler ===")
    print(table[table.seed.isin(["single", "mean5"])].set_index("variant")[columns]
          .to_string(float_format=lambda v: f"{v:.5f}"))
    print("Negativ delta = bedre. En forbedring bør være større end ca. 2 × se for at tælle.")
    print(f"\nOOF: {OOF_PATH}\nSummary: {SUMMARY_PATH}\nTid: {time.perf_counter() - started:.0f} sek.")


if __name__ == "__main__":
    main()
