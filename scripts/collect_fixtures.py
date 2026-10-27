
import json
from pathlib import Path

import requests

LEAGUES = {
    "Premier League": "en.1",
    "La Liga": "es.1",
    "Bundesliga": "de.1",
    "Serie A": "it.1",
    "Ligue 1": "fr.1",
}

SEASON = "2026-27"
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "openfootball/football.json/master"
)


def main():
    all_matches = []

    for league, code in LEAGUES.items():
        url = f"{BASE_URL}/{SEASON}/{code}.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        for match in data["matches"]:
            all_matches.append({
                "league": league,
                "season": SEASON,
                "date": match["date"],
                "time": match.get("time"),
                "homeTeam": match["team1"],
                "awayTeam": match["team2"],
                "score": match.get("score"),
            })

        print(f"{league}: {len(data['matches'])} kampe hentet")

    output_path = Path(__file__).resolve().parent.parent / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "fixtures.json"
    file_path.write_text(
        json.dumps(all_matches, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nI alt: {len(all_matches)} kampe")
    print(f"Gemte data i: {file_path}")


if __name__ == "__main__":
    main()
