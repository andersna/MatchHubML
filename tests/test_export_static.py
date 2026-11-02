from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.exceptions import InvalidTag
from fastapi import HTTPException

from src import api, export_static
from src.export_static import decrypt_admin, encrypt_admin, pack, static_name, unpack


class StaticNameTests(unittest.TestCase):
    def test_plain_endpoints(self):
        self.assertEqual(static_name("/api/health"), "health")
        self.assertEqual(static_name("/api/odds-coverage"), "odds-coverage")
        self.assertEqual(static_name("/api/sync/last"), "sync-last")

    def test_matches_by_model_and_league(self):
        # The frontend's URLSearchParams order: model_version, limit, league.
        self.assertEqual(static_name("/api/matches?model_version=v3&limit=200"), "matches-v3-all")
        self.assertEqual(
            static_name("/api/matches?model_version=v1&limit=200&league=Premier+League"),
            "matches-v1-premier-league",
        )
        self.assertEqual(static_name("/api/matches?model_version=v2&league=La%20Liga"), "matches-v2-la-liga")


class StateArchiveTests(unittest.TestCase):
    def test_round_trip_keeps_state_and_never_packs_game_accounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp, "source"), Path(tmp, "target")
            (source / "raw" / "understat").mkdir(parents=True)
            (source / "matchhub_cache.sqlite").write_bytes(b"cache")
            (source / "prediction_log.sqlite").write_bytes(b"log")
            (source / "raw" / "all-euro-data-2026-2027.xlsx").write_bytes(b"xlsx")
            (source / "raw" / "understat" / "EPL_2026.json").write_text("{}")
            (source / "game.sqlite").write_bytes(b"password hashes")
            (source / "oof_predictions_nn.csv").write_text("experiment output")

            archive = Path(tmp, "state.tar.gz")
            packed = pack(archive, source)
            unpack(archive, target)

            self.assertEqual(
                sorted(packed),
                ["matchhub_cache.sqlite", "prediction_log.sqlite",
                 "raw/all-euro-data-2026-2027.xlsx", "raw/understat/EPL_2026.json"],
            )
            self.assertEqual((target / "prediction_log.sqlite").read_bytes(), b"log")
            self.assertFalse((target / "game.sqlite").exists())
            self.assertFalse((target / "oof_predictions_nn.csv").exists())


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def _export(self, admin_password=None):
        """Export with stubbed endpoints; returns (data files by name, index.html, stale file still there)."""
        def predictions(version, league, limit):
            self.calls.append((version, league, limit))
            return {"matches": [], "skipped": [], "version": version, "league": league}

        def failing_sync():
            raise HTTPException(status_code=502, detail="football-data.org svarede ikke")

        stubs = {
            "sync_data": failing_sync,
            "health": lambda: {"available_models": {"v1": True, "v2": False, "v3": True}, "default_model": "v3"},
            "status": lambda: {"sources": "football-data.org"},
            "get_odds_coverage": lambda: {"total": 0},
            "last_sync": lambda: {"api_calls": {"5dollarfootballapi": {"calls": 3}}},
            "history_status": lambda: {"workbook": None},
            "models": lambda: {"models": []},
            "performance": lambda: {"rows": []},
            "list_predictions": predictions,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(api, **stubs):
            out = Path(tmp, "site")
            out.mkdir()
            Path(out, "stale.json").write_text("{}")  # left over from an earlier export

            export_static.export(out, now=datetime(2026, 11, 2, 20, 0, tzinfo=timezone.utc),
                                 admin_password=admin_password)

            data = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (out / "data").glob("*.json")}
            raw = "".join(p.read_text(encoding="utf-8") for p in (out / "data").glob("*.json"))
            html = (out / "index.html").read_text(encoding="utf-8")
            return data, raw, html, Path(out, "stale.json").exists()

    def test_writes_every_file_the_frontend_reads(self):
        calls = self.calls
        data, _, html, stale_exists = self._export(admin_password="correct horse battery staple")

        # A failed sync still publishes the existing data.
        self.assertEqual(data["health"]["default_model"], "v3")
        # Only available models, each for all leagues and every single league.
        self.assertEqual(len(calls), 2 * 6)
        self.assertIn(("v3", "Bundesliga", 200), calls)
        self.assertNotIn("matches-v2-all", data)
        self.assertEqual(data["matches-v1-serie-a"]["league"], "Serie A")
        self.assertEqual(data["meta"], {"generated_utc": "2026-11-02T20:00:00+00:00", "model_versions": ["v1", "v3"]})
        self.assertIn('window.MATCHHUB_STATIC = {"generated": "2026-11-02T20:00:00+00:00"}', html)
        self.assertFalse(stale_exists)

    def test_data_tab_is_only_published_encrypted(self):
        data, raw, _, _ = self._export(admin_password="correct horse battery staple")

        for name in export_static.ADMIN_FILES:
            self.assertNotIn(name, data)
        # The data providers named by the Data tab's files appear nowhere in plain text.
        self.assertNotIn("football-data", raw)
        self.assertNotIn("5dollarfootball", raw)
        admin = decrypt_admin(data["admin"], "correct horse battery staple")
        self.assertEqual(admin["status"], {"sources": "football-data.org"})
        self.assertEqual(admin["sync-last"]["api_calls"]["5dollarfootballapi"]["calls"], 3)

    def test_without_password_the_data_tab_is_left_out(self):
        data, raw, _, _ = self._export(admin_password=None)

        self.assertNotIn("admin", data)
        self.assertNotIn("status", data)
        self.assertNotIn("football-data", raw)

    def test_failed_endpoint_is_exported_as_its_error(self):
        def refused():
            raise HTTPException(status_code=403, detail="Skjult")

        self.assertEqual(export_static._call(refused), {"detail": "Skjult", "status_code": 403})


class AdminEncryptionTests(unittest.TestCase):
    FILES = {"status": '{"a": 1}', "history": '{"b": "æøå"}'}

    def test_round_trip(self):
        bundle = encrypt_admin(self.FILES, "hemmelig", iterations=1000)
        self.assertEqual(decrypt_admin(bundle, "hemmelig"), {"status": {"a": 1}, "history": {"b": "æøå"}})
        self.assertEqual(bundle["kdf"]["iterations"], 1000)

    def test_wrong_password_fails(self):
        bundle = encrypt_admin(self.FILES, "hemmelig", iterations=1000)
        with self.assertRaises(InvalidTag):
            decrypt_admin(bundle, "forkert")

    def test_a_file_cannot_be_swapped_for_another(self):
        bundle = encrypt_admin(self.FILES, "hemmelig", iterations=1000)
        bundle["files"]["status"], bundle["files"]["history"] = bundle["files"]["history"], bundle["files"]["status"]
        with self.assertRaises(InvalidTag):
            decrypt_admin(bundle, "hemmelig")

    def test_salt_and_ivs_are_fresh_every_export(self):
        first = encrypt_admin(self.FILES, "hemmelig", iterations=1000)
        second = encrypt_admin(self.FILES, "hemmelig", iterations=1000)
        self.assertNotEqual(first["kdf"]["salt"], second["kdf"]["salt"])
        self.assertNotEqual(first["files"]["status"]["data"], second["files"]["status"]["data"])


if __name__ == "__main__":
    unittest.main()
