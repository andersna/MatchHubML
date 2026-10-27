
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, log_loss



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.csv"
ELO_PATH = PROJECT_ROOT / "data" / "elo_features.csv"


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

KEY_COLUMNS = [
    "date", "time", "league", "home_team", "away_team"
]


COMMON_FEATURES = [
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
    "home_form_rating",
    "away_form_rating",
    "home_venue_form_rating",
    "away_venue_form_rating",
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
] + [
    f"elo_difference_k{k}" for k in [16, 32, 48]
]

NN_FEATURES = (
    BASE_FEATURES
    + ATTACK_DEFENSE_FEATURES
    + ELO_COLUMNS
)

LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}
INDEX_TO_LABEL = np.array(["H", "D", "A"])

np.random.seed(42)
print(f"LightGBM: {lgb.__version__}")

def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0,
    )


def evaluate(y_true, y_pred, probabilities=None):
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }

    if probabilities is not None:
        scores["log_loss"] = log_loss(
            y_true,
            probabilities,
            labels=["H", "D", "A"],
        )

    return scores

# Indlæs de samme data og anvend samme udvælgelse
# af kampe som i den eksisterende backtest.
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
elo = pd.read_csv(ELO_PATH, parse_dates=["date"])

df = df.merge(
    elo[KEY_COLUMNS + ELO_COLUMNS + [
        "home_attack_strength",
        "away_attack_strength",
        "home_defense_strength",
        "away_defense_strength",
        "home_expected_goals",
        "away_expected_goals",
    ]],
    on=KEY_COLUMNS,
    how="left",
    validate="one_to_one",
)

df = df.sort_values(
    ["date", "time"], kind="stable"
).reset_index(drop=True)


# Behold den eksisterende udvælgelse af kampe.
# Manglende værdier i modelinputtene håndteres senere
# ved imputering, der kun tilpasses på træningsdata.
df = df.dropna(
    subset=COMMON_FEATURES + ELO_COLUMNS
).reset_index(drop=True)

# Fjern eventuelle rækker uden et gyldigt kampresultat.
df = df[df["result"].isin(LABEL_TO_INDEX)].reset_index(drop=True)

if not df["result"].isin(LABEL_TO_INDEX).all():
    raise ValueError("Resultatkolonnen indeholder ukendte klasser.")

dates = sorted(df["date"].dt.date.unique())
date_blocks = [
    list(block)
    for block in np.array_split(dates, 4)
    if len(block) > 0
]

print(f"Kampe i backtesten: {len(df)}")


results = []
oof_predictions = []


for fold in range(1, len(date_blocks)):
    train_dates = [
        date
        for block in date_blocks[:fold]
        for date in block
    ]
    test_dates = date_blocks[fold]

    train = df[df["date"].dt.date.isin(train_dates)].copy()
    test = df[df["date"].dt.date.isin(test_dates)].copy()

    if train.empty or test.empty:
        continue

    # Kronologisk validering: de seneste 15 % af træningsdatoerne.
    unique_train_dates = sorted(train["date"].dt.date.unique())
    split_index = int(len(unique_train_dates) * 0.85)

    subtrain_dates = unique_train_dates[:split_index]
    validation_dates = unique_train_dates[split_index:]

    subtrain = train[
        train["date"].dt.date.isin(subtrain_dates)
    ].copy()
    validation = train[
        train["date"].dt.date.isin(validation_dates)
    ].copy()

    print(
        f"\nFold {fold}: "
        f"træning={len(train)} kampe, "
        f"validering={len(validation)} kampe, "
        f"test={len(test)} kampe"
    )

    # Brug samme feature-sæt som den eksisterende neural network-model.
    X_subtrain = subtrain[NN_FEATURES]
    X_validation = validation[NN_FEATURES]
    X_test = test[NN_FEATURES]

    y_subtrain = subtrain["result"].map(LABEL_TO_INDEX)
    y_validation = validation["result"].map(LABEL_TO_INDEX)
    y_test = test["result"].map(LABEL_TO_INDEX)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_subtrain,
        y_subtrain,
        eval_set=[(X_validation, y_validation)],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    probabilities_raw = model.predict_proba(
        X_test,
        num_iteration=model.best_iteration_,
    )

    # LightGBM klassificerer 0=H, 1=D, 2=A.
    probabilities = np.zeros((len(test), 3))
    for column, class_index in enumerate(model.classes_):
        probabilities[:, int(class_index)] = probabilities_raw[:, column]

    predicted_indices = probabilities.argmax(axis=1)
    predictions = INDEX_TO_LABEL[predicted_indices]
    y_test_labels = test["result"].to_numpy()

    scores = evaluate(
        y_test_labels,
        predictions,
        probabilities=probabilities,
    )

    results.append({
        "fold": fold,
        "model": "LightGBM",
        **scores,
    })

    for actual, predicted, probs in zip(
        y_test_labels, predictions, probabilities
    ):
        oof_predictions.append({
            "fold": fold,
            "model": "LightGBM",
            "actual": actual,
            "predicted": predicted,
            "prob_H": float(probs[0]),
            "prob_D": float(probs[1]),
            "prob_A": float(probs[2]),
        })

    print(
        f"  LightGBM: accuracy={scores['accuracy']:.3f}, "
        f"macro F1={scores['macro_f1']:.3f}, "
        f"log loss={scores['log_loss']:.4f}, "
        f"bedste iteration={model.best_iteration_}"
    )




# Opsummer resultaterne fra LightGBM-backtesten.
results_df = pd.DataFrame(results)

print("\n=== LIGHTGBM: GENNEMSNIT PÅ TVÆRS AF FOLDS ===")
summary = results_df.groupby("model").agg(
    mean_accuracy=("accuracy", "mean"),
    mean_macro_f1=("macro_f1", "mean"),
    mean_log_loss=("log_loss", "mean"),
    folds=("fold", "nunique"),
)

print(summary.round(4).to_string())

# Gem LightGBM's out-of-fold-forudsigelser separat.
oof_df = pd.DataFrame(oof_predictions)

predictions_path = (
    PROJECT_ROOT / "data" / "oof_predictions_lightgbm.csv"
)
oof_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

print(f"\nGemte LightGBM-forudsigelser i: {predictions_path}")

if not oof_df.empty:
    print("\n=== SAMLET LIGHTGBM-RESULTAT ===")
    for name, group in oof_df.groupby("model", sort=False):
        scores = evaluate(
            group["actual"].to_numpy(),
            group["predicted"].to_numpy(),
            group[["prob_H", "prob_D", "prob_A"]].to_numpy(),
        )
        print(
            f"{name}: accuracy={scores['accuracy']:.4f}, "
            f"macro F1={scores['macro_f1']:.4f}, "
            f"log loss={scores['log_loss']:.4f}, "
            f"kampe={len(group)}"
        )

    print("\n=== FORVEKSLINGSMATRIX ===")
    matrix = pd.crosstab(
        oof_df["actual"],
        oof_df["predicted"],
        rownames=["Faktisk resultat"],
        colnames=["Forudsagt resultat"],
    ).reindex(
        index=["H", "D", "A"],
        columns=["H", "D", "A"],
        fill_value=0,
    )
    print("Rækker = faktisk resultat, kolonner = forudsigelse")
    print(matrix.to_string())