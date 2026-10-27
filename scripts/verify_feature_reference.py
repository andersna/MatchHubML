
import json
import math
from pathlib import Path

import pandas as pd

from src.upcoming_match_features import build_upcoming_match_features

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "data" / "feature_reference.json"
FIXTURES = ROOT / "data" / "fixtures.csv"

reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

# Keep the original reference check pinned to its historical as-of date.
request = {
    "league": reference["league"],
    "home_team": reference["home_team"],
    "away_team": reference["away_team"],
    "match_date": reference["match_date"],
}

as_of_date = reference["match_date"]
current = build_upcoming_match_features(request, as_of_date=as_of_date)
current_features = current.iloc[0].to_dict()
reference_features = reference["features"]

if set(current_features) != set(reference_features):
    raise AssertionError("Featurekolonnerne matcher ikke referencen.")

differences = []
for name, expected in reference_features.items():
    actual = float(current_features[name])
    if not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12):
        differences.append((name, float(expected), actual))

if differences:
    print("Forskelle fundet:")
    for name, expected, actual in differences:
        print(f"{name}: reference={expected}, nuværende={actual}")
    raise SystemExit(1)

print("PASS: Alle 27 features matcher referencen.")

# Batch check: requests have different match dates but share one as-of date.
batch_requests = [
    {
        "league": "La Liga",
        "home_team": "Levante UD",
        "away_team": "Athletic Club",
        "match_date": "2026-10-21",
    },
    {
        "league": "Bundesliga",
        "home_team": "VfB Stuttgart",
        "away_team": "Borussia Mönchengladbach",
        "match_date": "2026-10-23",
    },
    {
        "league": "Premier League",
        "home_team": "Aston Villa FC",
        "away_team": "Manchester City FC",
        "match_date": "2026-10-24",
    },
]

batch = build_upcoming_match_features(
    batch_requests,
    as_of_date="2026-10-21",
)

if batch.shape != (len(batch_requests), 27):
    raise AssertionError(
        f"Forventede {len(batch_requests)} rækker og 27 features; "
        f"fik {batch.shape}."
    )

if list(batch.columns) != list(current.columns):
    raise AssertionError("Batchens featurekolonner matcher ikke referencekolonnerne.")

if batch.isna().any().any():
    raise AssertionError("Batchberegningen indeholder manglende featureværdier.")

if not all(math.isfinite(float(value)) for value in batch.to_numpy().ravel()):
    raise AssertionError("Batchberegningen indeholder ikke-endelige værdier.")

# The first batch row must exactly match the pinned reference for the same
# match and same as-of date.
for name, expected in reference_features.items():
    actual = float(batch.iloc[0][name])
    if not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(
            f"Batch-reference afviger for {name}: "
            f"reference={expected}, batch={actual}"
        )

print(
    f"PASS: Batch med {len(batch_requests)} kampdatoer returnerede "
    "27 finite features pr. kamp med fælles skæringsdato."
)