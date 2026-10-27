
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

BASELINE_PATH = DATA_PATH / "oof_predictions_baseline.csv"
SHOTS_PATH = DATA_PATH / "oof_predictions_shots_on_target.csv"

baseline = pd.read_csv(BASELINE_PATH)
shots = pd.read_csv(SHOTS_PATH)

KEY_COLUMNS = ["fold", "actual"]

for name, df in [
    ("Baseline", baseline),
    ("Skud på mål", shots),
]:
    print(f"{name}: {len(df)} forudsigelser")

# Sammenlign hver modeltype separat.
for model_name in baseline["model"].unique():
    base = baseline[baseline["model"] == model_name].copy()
    new = shots[shots["model"] == model_name].copy()

    print(f"\n=== {model_name} ===")

    # Sammenligningsnøglen indeholder fold og faktisk resultat.
    # Den er ikke unik, så vi kontrollerer antal og rækkefølge.
    base = base.reset_index(drop=True)
    new = new.reset_index(drop=True)

    same_rows = (
        len(base) == len(new)
        and base[KEY_COLUMNS].equals(new[KEY_COLUMNS])
    )

    print("Samme fold-/resultatrækkefølge:", same_rows)

    if not same_rows:
        print(
            "ADVARSEL: Forudsigelserne kan ikke "
            "sammenlignes sikkert med denne kontrol."
        )
        continue

    for label, df in [
        ("Baseline", base),
        ("Skud på mål", new),
    ]:
        accuracy = accuracy_score(df["actual"], df["predicted"])
        macro_f1 = f1_score(
            df["actual"],
            df["predicted"],
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0,
        )

        print(
            f"{label}: accuracy={accuracy:.4f}, "
            f"macro F1={macro_f1:.4f}"
        )

    changed = (
        base["predicted"].to_numpy()
        != new["predicted"].to_numpy()
    )

    print(
        f"Ændrede forudsigelser: {changed.sum()} "
        f"af {len(changed)} ({changed.mean():.2%})"
    )

    base_correct = (
        base["actual"].to_numpy()
        == base["predicted"].to_numpy()
    )
    new_correct = (
        new["actual"].to_numpy()
        == new["predicted"].to_numpy()
    )

    improved = (~base_correct & new_correct).sum()
    worsened = (base_correct & ~new_correct).sum()

    print(f"Forudsigelser forbedret: {improved}")
    print(f"Forudsigelser forværret: {worsened}")
    print(f"Nettogevinst i korrekte svar: {improved - worsened}")