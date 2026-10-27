
import json
from pathlib import Path
from datetime import date

file_path = Path(__file__).resolve().parent.parent / "data" / "fixtures.json"
matches = json.loads(file_path.read_text(encoding="utf-8"))

today = date.today()

missing_teams = []
invalid_scores = []
future_matches_with_scores = []
completed = 0

for match in matches:
    match_date = date.fromisoformat(match["date"])
    score = match.get("score")

    if not match.get("homeTeam") or not match.get("awayTeam"):
        missing_teams.append(match)

    # En kamp med et slutresultat skal have score som et objekt
    # med en ft-liste, fx {"ht": [1, 0], "ft": [2, 1]}.
    has_final_score = (
        isinstance(score, dict)
        and isinstance(score.get("ft"), list)
        and len(score["ft"]) == 2
        and all(isinstance(x, int) for x in score["ft"])
    )

    if score is not None and not has_final_score:
        invalid_scores.append(match)

    if match_date <= today and has_final_score:
        completed += 1

    if match_date > today and score is not None:
        future_matches_with_scores.append(match)

print(f"Kontrollerede kampe: {len(matches)}")
print(f"Kampe med slutresultat: {completed}")
print(f"Kampe med manglende hold: {len(missing_teams)}")
print(f"Kampe med ugyldigt scoreformat: {len(invalid_scores)}")
print(f"Fremtidige kampe med score: {len(future_matches_with_scores)}")

for match in missing_teams[:3]:
    print("Manglende hold:", match)

for match in invalid_scores[:3]:
    print("Ugyldig score:", match)
