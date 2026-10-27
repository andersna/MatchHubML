import pandas as pd
from pathlib import Path

folder = Path("data/raw/historical")
files = sorted(folder.glob("*.xls*"))

groups = {
    "Bet365 almindelige": ["B365H", "B365D", "B365A"],
    "Bet365 closing": ["B365CH", "B365CD", "B365CA"],
    "Pinnacle almindelige": ["PSH", "PSD", "PSA"],
    "Pinnacle closing": ["PSCH", "PSCD", "PSCA"],
}

for file in files:
    df = pd.read_excel(file, nrows=5000)
    print(f"\n{file.name}")

    for name, columns in groups.items():
        if all(col in df.columns for col in columns):
            valid = df[columns].notna().all(axis=1)
            coverage = 100 * valid.mean() if len(df) else 0
            print(
                f"  {name:<22} "
                f"{int(valid.sum()):>5}/{len(df):<5} "
                f"({coverage:.1f} %)"
            )
        else:
            print(f"  {name:<22} Ikke tilgængelige kolonner")