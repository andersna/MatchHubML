from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from src import api


class HideModelTests(unittest.TestCase):
    def test_matches_endpoint_is_refused_when_the_model_is_hidden(self):
        with patch.dict(os.environ, {api.HIDE_MODEL_ENV: "1"}), patch.object(api, "predict_upcoming") as predict:
            with self.assertRaises(HTTPException) as caught:
                api.list_predictions(model_version="v3", league=None, limit=50)
        self.assertEqual(caught.exception.status_code, 403)
        predict.assert_not_called()

    def test_matches_endpoint_works_by_default(self):
        with patch.dict(os.environ, {api.HIDE_MODEL_ENV: ""}), \
                patch.object(api, "predict_upcoming", return_value={"matches": []}) as predict:
            self.assertEqual(api.list_predictions(model_version="v3", league=None, limit=50), {"matches": []})
        predict.assert_called_once_with("v3", None, 50)

    def test_health_reports_the_flag(self):
        with patch.dict(os.environ, {api.HIDE_MODEL_ENV: "true"}):
            self.assertTrue(api.health()["hide_model"])
        with patch.dict(os.environ, {api.HIDE_MODEL_ENV: "0"}):
            self.assertFalse(api.health()["hide_model"])


class GameEndpointTests(unittest.TestCase):
    def test_round_requires_login(self):
        with self.assertRaises(HTTPException) as caught:
            api.game_round(week=None, session=None)
        self.assertEqual(caught.exception.status_code, 401)

    def test_bot_refresh_runs_at_most_once_per_interval(self):
        with patch.object(api, "_bot_refreshed_at", None), patch.object(api.threading, "Thread") as thread:
            api._refresh_bot_entries()
            api._refresh_bot_entries()
        self.assertEqual(thread.call_count, 1)
        api._bot_lock.release()  # the patched thread never ran to release it


if __name__ == "__main__":
    unittest.main()
