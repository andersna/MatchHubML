
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
    for league, code in LEAGUES.items():
        url = f"{BASE_URL}/{SEASON}/{code}.json"

        try:
            response = requests.get(url, timeout=20)

            if response.status_code == 200:
                data = response.json()
                matches = data.get("matches", [])
                print(f"{league}: OK — {len(matches)} kampe")
            else:
                print(f"{league}: HTTP {response.status_code}")

        except requests.RequestException as error:
            print(f"{league}: Netværksfejl — {error}")


if __name__ == "__main__":
    main()
