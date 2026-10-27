
from pathlib import Path

import pandas as pd

file_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "all-euro-data-2026-2027.xlsx"
)

excel = pd.ExcelFile(file_path)

print("Ark i Excel-filen:")
for sheet in excel.sheet_names:
    print(f"- {sheet}")

print("\nKolonner og eksempeldata:")

for sheet in excel.sheet_names:
    df = pd.read_excel(excel, sheet_name=sheet, nrows=5)

    print(f"\n--- {sheet} ---")
    print("Kolonner:", list(df.columns))
    print(df.head(3).to_string(index=False))