
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fixtures.csv"

df = pd.read_csv(DATA_PATH, dtype={"date": "string", "time": "string"})

date_text = df["date"].str.strip()
time_text = df["time"].str.strip()
missing_date = date_text.isna() | date_text.eq("")
missing_time = time_text.isna() | time_text.eq("")

# Parse the fields separately with explicit formats. In particular, never turn
# a missing time into midnight by concatenating the date with an empty string.
parsed_date = pd.to_datetime(
    date_text, format="%Y-%m-%d", errors="coerce"
)
valid_time_format = time_text.str.fullmatch(
    r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d", na=False
)
invalid_date = ~missing_date & parsed_date.isna()
invalid_time = ~missing_time & ~valid_time_format

kickoff_text = date_text.fillna("") + " " + time_text.fillna("")
parsed_kickoff = pd.to_datetime(
    kickoff_text, format="%Y-%m-%d %H:%M:%S", errors="coerce"
)
valid_kickoff = (
    ~missing_date
    & ~missing_time
    & ~invalid_date
    & ~invalid_time
    & parsed_kickoff.notna()
)
df["date"] = parsed_date
df["kickoff"] = parsed_kickoff.where(valid_kickoff)

print("=== TIDSKONTROL ===")
print(f"Antal kampe: {len(df)}")
print(f"Gyldige datoer: {int(parsed_date.notna().sum())}")
print(f"Manglende datoer: {int(missing_date.sum())}")
print(f"Ugyldige ikke-tomme datoer: {int(invalid_date.sum())}")
print(f"Gyldige kickofftider: {int(valid_time_format.sum())}")
print(f"Manglende kickofftider: {int(missing_time.sum())}")
print(f"Ugyldige ikke-tomme kickofftider: {int(invalid_time.sum())}")
print(f"Gyldige kombinerede kickoffstempler: {int(valid_kickoff.sum())}")

metadata_terms = (
    "timezone", "time_zone", "tz", "finished_at", "full_time_at",
    "result_known_at", "recorded_at", "timestamp", "snapshot",
)
temporal_metadata_columns = [
    column for column in df.columns
    if any(term in column.lower() for term in metadata_terms)
]
print(
    "Tidszone-/slut-/resultattidskolonner: "
    + (", ".join(temporal_metadata_columns) or "ingen fundet")
)

print("\n=== DUBLETTER ===")
fixture_keys = ["league", "date", "home_team", "away_team"]
duplicate_rows = df["date"].notna() & df.duplicated(
    subset=fixture_keys, keep=False
)
duplicate_count = df.loc[df["date"].notna()].duplicated(
    subset=fixture_keys
).sum()
print(f"Dubletforekomster efter første række: {int(duplicate_count)}")
print(f"Rækker involveret i dubletter: {int(duplicate_rows.sum())}")
if duplicate_rows.any():
    print(df.loc[duplicate_rows, fixture_keys].head(20).to_string(index=False))

print("\n=== HOLD MED FLERE KAMPE SAMME DAG ===")
home = df[["league", "date", "home_team"]].rename(
    columns={"home_team": "team"}
)
away = df[["league", "date", "away_team"]].rename(
    columns={"away_team": "team"}
)

appearances = pd.concat([home, away], ignore_index=True)
same_day = appearances.groupby(
    ["league", "date", "team"]
).size()

same_day = same_day[same_day > 1]
print(f"Hold med flere kampe samme dag: {len(same_day)}")
if not same_day.empty:
    print(same_day.to_string())

print("\n=== SAMTIDIGE KAMPE ===")
known_kickoffs = df.loc[df["kickoff"].notna()]
simultaneous = known_kickoffs.groupby(
    ["league", "date", "kickoff"]
).size()
simultaneous = simultaneous[simultaneous > 1]
print(f"Grupper med samme kickoff-tid i samme liga: {len(simultaneous)}")
print(f"Kampe i disse grupper: {int(simultaneous.sum())}")
if not simultaneous.empty:
    print(simultaneous.head(20).to_string())

print("\n=== TIDSRÆKKEFØLGE ===")
print(
    df.sort_values(["date", "kickoff"])[
        ["date", "time", "league", "home_team", "away_team"]
    ].head(10).to_string(index=False)
)
