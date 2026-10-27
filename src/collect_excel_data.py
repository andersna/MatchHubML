
from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
HISTORICAL_DIR = RAW_DIR / "historical"
OUTPUT_PATH = PROJECT_ROOT / "data" / "fixtures.csv"

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

COLUMNS = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
    "HxG": "home_xg",
    "AxG": "away_xg",
}


def get_season(path):
    match = re.search(
        r"all-euro-data-(\d{4})-(\d{4})",
        path.stem,
    )

    if not match:
        raise ValueError(
            f"Kunne ikke finde sæsonen i filnavnet: {path.name}"
        )

    start_year, end_year = match.groups()
    return f"{start_year}-{end_year[-2:]}"


def get_source_files():
    files = list(RAW_DIR.glob("all-euro-data-*.xls*"))
    files.extend(
        HISTORICAL_DIR.glob("all-euro-data-*.xls*")
    )

    # Sortér kronologisk efter sæsonens startår.
    return sorted(
        files,
        key=lambda path: int(
            re.search(
                r"all-euro-data-(\d{4})-(\d{4})",
                path.stem,
            ).group(1)
        ),
    )


def load_league_sheet(workbook, sheet_name, league_name, season):
    df = pd.read_excel(workbook, sheet_name=sheet_name)

    # Ensret kolonner, også når ældre sæsoner mangler
    # enkelte statistikker som xG eller skud på mål.
    df = df.reindex(columns=list(COLUMNS.keys()))
    df = df.rename(columns=COLUMNS)

    df["league"] = league_name
    df["season"] = season

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True,
    )

    for column in ["home_team", "away_team"]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    df["time"] = df["time"].astype("string").str.strip()

    # Fjern tomme rækker og rækker uden gyldig kampdato
    # eller holdnavne.
    df = df.dropna(
        subset=["date", "home_team", "away_team"]
    ).copy()

    # Sørg for numeriske målkolonner.
    for column in ["home_goals", "away_goals"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).astype("Int64")

    # Udled kampresultatet fra slutresultatet.
    # Ikke-afsluttede kampe beholder et manglende resultat.
    played = (
        df["home_goals"].notna()
        & df["away_goals"].notna()
    )

    df["result"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    home_win = (
        played
        & (df["home_goals"] > df["away_goals"])
    )
    draw = (
        played
        & (df["home_goals"] == df["away_goals"])
    )
    away_win = (
        played
        & (df["home_goals"] < df["away_goals"])
    )

    df.loc[home_win, "result"] = "H"
    df.loc[draw, "result"] = "D"
    df.loc[away_win, "result"] = "A"

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df


def main():
    source_files = get_source_files()

    if not source_files:
        raise FileNotFoundError(
            "Fandt ingen all-euro-data Excel-filer."
        )

    print(f"Fandt {len(source_files)} sæsonfiler.\n")

    all_matches = []

    for path in source_files:
        season = get_season(path)
        print(f"Indlæser sæson {season}: {path.name}")

        # ExcelFile læser både .xls og .xlsx via den
        # relevante pandas-engine.
        with pd.ExcelFile(path) as workbook:
            for sheet, league_name in LEAGUES.items():
                if sheet not in workbook.sheet_names:
                    print(
                        f"  SPRUNGET OVER: {league_name} "
                        f"(arket {sheet} findes ikke)"
                    )
                    continue

                df = load_league_sheet(
                    workbook,
                    sheet,
                    league_name,
                    season,
                )

                all_matches.append(df)

                played = df["result"].notna().sum()

                print(
                    f"  {league_name}: {len(df)} kampe, "
                    f"{played} med slutresultat"
                )

    if not all_matches:
        raise ValueError(
            "Der blev ikke indlæst nogen kampe."
        )

    combined = pd.concat(
        all_matches,
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["date", "time", "league", "home_team"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== SAMLET DATA ===")
    print(f"Antal sæsoner: {combined['season'].nunique()}")
    print(f"Antal kampe/rækker: {len(combined)}")
    print(f"Afsluttede kampe: {combined['result'].notna().sum()}")

    print("\n=== KAMPE PR. LIGA ===")
    league_summary = combined.groupby("league").agg(
        kampe=("league", "size"),
        afsluttede=("result", "count"),
    )
    print(league_summary.to_string())

    print("\n=== KAMPE PR. SÆSON ===")
    season_summary = combined.groupby("season").agg(
        kampe=("season", "size"),
        afsluttede=("result", "count"),
    )
    print(season_summary.to_string())

    print(f"\nGemte data i: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()