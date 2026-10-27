
from pathlib import Path

import pandas as pd


FEATURES_FILE = Path("data/features.csv")
HISTORICAL_DIR = Path("data/raw/historical")
OUTPUT_FILE = Path("data/features_with_odds.csv")

LEAGUES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

ODDS_COLUMNS = ["B365H", "B365D", "B365A"]

features = pd.read_csv(FEATURES_FILE, low_memory=False)
features["date"] = pd.to_datetime(
    features["date"], errors="coerce"
).dt.strftime("%Y-%m-%d")

odds_frames = []

files = sorted(HISTORICAL_DIR.glob("all-euro-data-*"))

for file_path in files:
    try:
        excel = pd.ExcelFile(file_path)
    except Exception as error:
        print(f"Springer over {file_path.name}: {error}")
        continue

    for sheet, league_name in LEAGUES.items():
        if sheet not in excel.sheet_names:
            continue

        df = pd.read_excel(
            excel,
            sheet_name=sheet,
            usecols=lambda column: column in [
                "Date", "HomeTeam", "AwayTeam",
                *ODDS_COLUMNS,
            ],
        )

        required = ["Date", "HomeTeam", "AwayTeam"]
        if not all(column in df.columns for column in required):
            continue

        # Spring over rækker uden en rigtig kamp.
        df = df.dropna(subset=required).copy()

        # Spring over sæsoner uden de nødvendige odds-kolonner.
        if not all(column in df.columns for column in ODDS_COLUMNS):
            continue

        df["date"] = pd.to_datetime(
            df["Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        df = df.rename(
            columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "B365H": "odds_home",
                "B365D": "odds_draw",
                "B365A": "odds_away",
            }
        )

        df["league"] = league_name

        odds_frames.append(
            df[
                [
                    "date", "home_team", "away_team", "league",
                    "odds_home", "odds_draw", "odds_away",
                ]
            ]
        )

    print(f"Indlæst: {file_path.name}")

if not odds_frames:
    raise RuntimeError("Ingen oddsdata blev fundet.")

odds = pd.concat(odds_frames, ignore_index=True)

for column in ["odds_home", "odds_draw", "odds_away"]:
    odds[column] = pd.to_numeric(odds[column], errors="coerce")

# Kun komplette, positive odds.
complete_odds = (
    odds[["odds_home", "odds_draw", "odds_away"]]
    .gt(0)
    .all(axis=1)
)

odds = odds.loc[complete_odds].copy()

keys = ["date", "home_team", "away_team", "league"]

# Undgå at koble en kamp til flere forskellige odds-rækker.
if odds.duplicated(keys).any():
    duplicates = odds.loc[odds.duplicated(keys, keep=False), keys]
    print("\nEksempler på dublerede kampe:")
    print(duplicates.head(10).to_string(index=False))
    raise ValueError(
        "Der er dublerede kampnøgler. Datasættet er ikke gemt."
    )

# Fjern eventuelle tidligere odds-kolonner fra input for at undgå suffixes.
features = features.drop(
    columns=["odds_home", "odds_draw", "odds_away"],
    errors="ignore",
)

merged = features.merge(
    odds,
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True,
)

matched = merged["_merge"].eq("both")
complete = (
    merged[["odds_home", "odds_draw", "odds_away"]]
    .notna()
    .all(axis=1)
)

print("\n=== Samlet resultat ===")
print("Kampe i features.csv:", len(features))
print("Kampe matchet med odds:", int(matched.sum()))
print("Kampe med komplette odds:", int(complete.sum()))
print("Kampe uden komplette odds:", int((~complete).sum()))

print("\nDækning pr. liga:")
print(
    merged.assign(has_odds=complete)
    .groupby("league")["has_odds"]
    .agg(kampe="size", med_odds="sum")
    .to_string()
)

merged = merged.drop(columns="_merge")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nDatasættet er gemt i: {OUTPUT_FILE}")
