
import pandas as pd

features = pd.read_csv("data/features.csv", low_memory=False)

excel_path = (
    "data/raw/historical/"
    "all-euro-data-2025-2026.xlsx"
)

league_sheets = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

features["date"] = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")

features_season = features[features["season"] == "2025-26"].copy()

excel = pd.ExcelFile(excel_path)
odds_rows = []

for sheet, league_name in league_sheets.items():
    df = pd.read_excel(excel, sheet_name=sheet)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

    df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["league"] = league_name

    df = df.rename(
        columns={
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "B365H": "odds_home",
            "B365D": "odds_draw",
            "B365A": "odds_away",
        }
    )

    odds_rows.append(
        df[
            [
                "date",
                "home_team",
                "away_team",
                "league",
                "odds_home",
                "odds_draw",
                "odds_away",
            ]
        ]
    )

odds = pd.concat(odds_rows, ignore_index=True)

merged = features_season.merge(
    odds,
    on=["date", "home_team", "away_team", "league"],
    how="left",
    validate="one_to_one",
)

matched = merged["odds_home"].notna() & merged["odds_draw"].notna() & merged["odds_away"].notna()

print("Kampe i features.csv:", len(features_season))
print("Kampe matchet med Excel:", int(merged["odds_home"].notna().sum()))
print("Kampe med komplette odds:", int(matched.sum()))
print("Kampe uden komplette odds:", int((~matched).sum()))

print("\nEksempler på kampe uden komplette odds:")
print(
    merged.loc[
        ~matched,
        ["date", "home_team", "away_team", "league"],
    ].head(10).to_string(index=False)
)