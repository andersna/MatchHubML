
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw/historical")
OUTPUT_FILE = Path("data/odds_coverage_by_season.csv")

LEAGUES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

ODDS_COLUMNS = ["B365H", "B365D", "B365A"]

results = []

files = sorted(DATA_DIR.glob("all-euro-data-*"))

for file_path in files:
    try:
        excel = pd.ExcelFile(file_path)
    except Exception as error:
        print(f"Kunne ikke åbne {file_path.name}: {error}")
        continue

    for sheet, league_name in LEAGUES.items():
        if sheet not in excel.sheet_names:
            continue

        df = pd.read_excel(excel, sheet_name=sheet)

        required_columns = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]

        if not all(column in df.columns for column in required_columns):
            continue

        valid_match = (
            df["Date"].notna()
            & df["HomeTeam"].notna()
            & df["AwayTeam"].notna()
            & df["FTHG"].notna()
            & df["FTAG"].notna()
        )

        match_count = int(valid_match.sum())

        if all(column in df.columns for column in ODDS_COLUMNS):
            odds_available = (
                df[ODDS_COLUMNS]
                .apply(pd.to_numeric, errors="coerce")
                .gt(0)
                .all(axis=1)
            )
        else:
            odds_available = pd.Series(False, index=df.index)

        odds_count = int((valid_match & odds_available).sum())

        results.append({
            "file": file_path.name,
            "league": league_name,
            "matches": match_count,
            "matches_with_odds": odds_count,
            "matches_without_odds": match_count - odds_count,
            "coverage_pct": (
                round(100 * odds_count / match_count, 2)
                if match_count
                else 0
            ),
        })

    print(f"Analyseret: {file_path.name}")

if not results:
    raise RuntimeError("Ingen sæsoner blev analyseret.")

coverage = pd.DataFrame(results)

coverage["season"] = coverage["file"].str.extract(
    r"all-euro-data-(\d{4}-\d{4})"
)

coverage = coverage.sort_values(["season", "league"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
coverage.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n=== Odds-dækning pr. sæson og liga ===")
print(
    coverage[
        ["season", "league", "matches", "matches_with_odds", "coverage_pct"]
    ].to_string(index=False)
)

print(f"\nResultatet er gemt i: {OUTPUT_FILE}")
