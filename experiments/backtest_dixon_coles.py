from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, f1_score, log_loss


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "oof_predictions_dixon_coles.csv"

LEAGUES = [
    "Bundesliga",
    "La Liga",
    "Ligue 1",
    "Premier League",
    "Serie A",
]

# Dixon-Coles' eksponentielle tidsvægtning.
# Parameteren holdes fast i første forsøg.
XI = 0.0018

# Antal mål i scorefordelingen.
MAX_GOALS = 10

LABELS = np.array(["H", "D", "A"])
LABEL_TO_INDEX = {"H": 0, "D": 1, "A": 2}


def tau_correction(home_goals, away_goals, lam, mu, rho):
    if home_goals == 0 and away_goals == 0:
        return 1 - lam * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + lam * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho

    return 1.0


def fit_dixon_coles(train):
    teams = sorted(set(train["home_team"]) | set(train["away_team"]))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = train["home_team"].map(team_to_idx).to_numpy()
    away_idx = train["away_team"].map(team_to_idx).to_numpy()

    home_goals = train["home_goals"].to_numpy(dtype=int)
    away_goals = train["away_goals"].to_numpy(dtype=int)

    dates = pd.to_datetime(train["date"])
    age_days = (dates.max() - dates).dt.days.to_numpy()
    weights = np.exp(-XI * age_days)
    weights = weights / weights.mean()

    def unpack(params):
        intercept = params[0]
        home_advantage = params[1]
        rho = params[2]

        attack = np.zeros(n_teams)
        defense = np.zeros(n_teams)

        attack[:-1] = params[3:3 + n_teams - 1]
        attack[-1] = -attack[:-1].sum()

        defense_start = 3 + n_teams - 1
        defense[:-1] = params[
            defense_start:defense_start + n_teams - 1
        ]
        defense[-1] = -defense[:-1].sum()

        return intercept, home_advantage, rho, attack, defense

    def objective(params):
        intercept, home_advantage, rho, attack, defense = unpack(params)

        log_lam = (
            intercept
            + home_advantage
            + attack[home_idx]
            + defense[away_idx]
        )
        log_mu = (
            intercept
            + attack[away_idx]
            + defense[home_idx]
        )

        lam = np.exp(np.clip(log_lam, -5, 5))
        mu = np.exp(np.clip(log_mu, -5, 5))

        tau = np.ones(len(train), dtype=float)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)

        tau[mask_00] = 1 - lam[mask_00] * mu[mask_00] * rho
        tau[mask_01] = 1 + lam[mask_01] * rho
        tau[mask_10] = 1 + mu[mask_10] * rho
        tau[mask_11] = 1 - rho

        if np.any(tau <= 0) or not np.all(np.isfinite(tau)):
            return 1e10

        log_likelihood = (
            home_goals * log_lam
            - lam
            - gammaln(home_goals + 1)
            + away_goals * log_mu
            - mu
            - gammaln(away_goals + 1)
            + np.log(tau)
        )

        return -np.sum(weights * log_likelihood)

    initial = np.zeros(3 + 2 * (n_teams - 1))
    initial[0] = np.log(1.15)
    initial[1] = np.log(1.45 / 1.15)
    initial[2] = -0.05

    bounds = (
        [(-1, 1), (-0.7, 0.7), (-0.2, 0.2)]
        + [(-1.5, 1.5)] * (2 * (n_teams - 1))
    )

    fit = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not fit.success:
        print(f"Advarsel: Optimering fejlede: {fit.message}")

    intercept, home_advantage, rho, attack, defense = unpack(fit.x)

    return {
        "teams": team_to_idx,
        "intercept": intercept,
        "home_advantage": home_advantage,
        "rho": rho,
        "attack": attack,
        "defense": defense,
        "success": fit.success,
        "message": fit.message,
    }
    
def predict_probabilities(model, home_team, away_team):
    teams = model["teams"]

    home_idx = teams.get(home_team)
    away_idx = teams.get(away_team)

    # Ukendte hold får neutrale angrebs- og forsvarsværdier.
    home_attack = (
        model["attack"][home_idx] if home_idx is not None else 0.0
    )
    home_defense = (
        model["defense"][home_idx] if home_idx is not None else 0.0
    )
    away_attack = (
        model["attack"][away_idx] if away_idx is not None else 0.0
    )
    away_defense = (
        model["defense"][away_idx] if away_idx is not None else 0.0
    )

    intercept = model["intercept"]
    home_advantage = model["home_advantage"]
    rho = model["rho"]

    lam = np.exp(
        np.clip(
            intercept + home_advantage + home_attack + away_defense,
            -5,
            5,
        )
    )
    mu = np.exp(
        np.clip(
            intercept + away_attack + home_defense,
            -5,
            5,
        )
    )

    score_matrix = np.outer(
        poisson.pmf(np.arange(MAX_GOALS + 1), lam),
        poisson.pmf(np.arange(MAX_GOALS + 1), mu),
    )

    for home_goals in range(2):
        for away_goals in range(2):
            tau = tau_correction(
                home_goals, away_goals, lam, mu, rho
            )
            score_matrix[home_goals, away_goals] *= tau

    total = score_matrix.sum()
    if total <= 0 or not np.isfinite(total):
        return np.array([1 / 3, 1 / 3, 1 / 3])

    score_matrix /= total

    prob_home = np.tril(score_matrix, k=-1).sum()
    prob_draw = np.trace(score_matrix)
    prob_away = np.triu(score_matrix, k=1).sum()

    probabilities = np.array([prob_home, prob_draw, prob_away])
    return probabilities / probabilities.sum()


def evaluate(actual, probabilities):
    actual_numeric = pd.Series(actual).map(
        {"H": 0, "D": 1, "A": 2}
    ).to_numpy()

    predicted_numeric = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(
        actual_numeric, predicted_numeric
    )

    macro_f1 = f1_score(
        actual_numeric,
        predicted_numeric,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    loss = log_loss(
        actual_numeric,
        probabilities,
        labels=[0, 1, 2],
    )

    return {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "log_loss": loss,
    }


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    df = df.dropna(
        subset=[
            "date", "league", "home_team", "away_team",
            "home_goals", "away_goals",
        ]
    ).copy()

    df = df[df["league"].isin(LEAGUES)].copy()
    df = df[
        (df["home_goals"] >= 0)
        & (df["away_goals"] >= 0)
    ].copy()

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    dates = np.array(sorted(df["date"].dt.date.unique()))
    date_blocks = [
        list(block)
        for block in np.array_split(dates, 4)
        if len(block) > 0
    ]

    print("Dixon-Coles-backtest")
    print(f"Kampe i datasættet: {len(df)}")
    print(f"Ligaer: {len(LEAGUES)}")
    print(f"Tidsvægtning xi: {XI}")
    print(f"Antal mål pr. hold i scorefordelingen: 0-{MAX_GOALS}")

    all_predictions = []

    for fold in range(1, len(date_blocks)):
        train_dates = [
            date
            for block in date_blocks[:fold]
            for date in block
        ]
        test_dates = date_blocks[fold]

        train = df[df["date"].dt.date.isin(train_dates)]
        test = df[df["date"].dt.date.isin(test_dates)]

        print(
            f"\n=== FOLD {fold} ==="
            f"\nTræningskampe: {len(train)}"
            f"\nTestkampe: {len(test)}"
        )

        fold_predictions = []

        for league in LEAGUES:
            league_train = train[train["league"] == league].copy()
            league_test = test[test["league"] == league].copy()

            if league_train.empty or league_test.empty:
                print(f"{league}: springes over (ingen trænings- eller testkampe)")
                continue

            model = fit_dixon_coles(league_train)

            probabilities = np.vstack([
                predict_probabilities(
                    model, row.home_team, row.away_team
                )
                for row in league_test.itertuples(index=False)
            ])

            actual = league_test["result"].to_numpy()
            predicted = LABELS[probabilities.argmax(axis=1)]

            scores = evaluate(actual, probabilities)

            print(
                f"{league}: n={len(league_test)}, "
                f"accuracy={scores['accuracy']:.4f}, "
                f"macro F1={scores['macro_f1']:.4f}, "
                f"log loss={scores['log_loss']:.4f}, "
                f"rho={model['rho']:.4f}"
            )

            for row, pred, probs in zip(
                league_test.itertuples(index=False),
                predicted,
                probabilities,
            ):
                fold_predictions.append({
                    "fold": fold,
                    "date": row.date,
                    "league": league,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "actual": row.result,
                    "predicted": pred,
                    "prob_H": float(probs[0]),
                    "prob_D": float(probs[1]),
                    "prob_A": float(probs[2]),
                    "rho": float(model["rho"]),
                })

        all_predictions.extend(fold_predictions)

    oof = pd.DataFrame(all_predictions)

    if oof.empty:
        raise RuntimeError("Backtesten gav ingen forudsigelser.")

    oof.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== SAMLET OOF-RESULTAT ===")
    probabilities = oof[["prob_H", "prob_D", "prob_A"]].to_numpy()
    actual = oof["actual"].to_numpy()
    scores = evaluate(actual, probabilities)

    print(f"Kampe: {len(oof)}")
    print(f"Accuracy: {scores['accuracy']:.4f}")
    print(f"Macro F1: {scores['macro_f1']:.4f}")
    print(f"Log loss: {scores['log_loss']:.4f}")

    print("\n=== RESULTATER PR. LIGA ===")
    for league, group in oof.groupby("league"):
        p = group[["prob_H", "prob_D", "prob_A"]].to_numpy()
        result = evaluate(group["actual"].to_numpy(), p)
        print(
            f"{league}: kampe={len(group)}, "
            f"accuracy={result['accuracy']:.4f}, "
            f"macro F1={result['macro_f1']:.4f}, "
            f"log loss={result['log_loss']:.4f}"
        )

    print("\n=== FORVEKSLINGSMATRIX ===")
    matrix = pd.crosstab(
        oof["actual"],
        oof["predicted"],
        rownames=["Faktisk resultat"],
        colnames=["Forudsagt resultat"],
    ).reindex(
        index=["H", "D", "A"],
        columns=["H", "D", "A"],
        fill_value=0,
    )
    print(matrix.to_string())

    print(f"\nGemte forudsigelser i: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
