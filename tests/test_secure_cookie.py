import unittest

from fastapi import Request, Response

from src import api


def _request(scheme: str = "http", headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({
        "type": "http", "method": "POST", "path": "/api/game/login", "query_string": b"",
        "scheme": scheme, "server": ("matchhubml.localhost", 80), "headers": raw_headers,
    })


class SecureCookieTests(unittest.TestCase):
    def _cookie(self, request: Request) -> str:
        response = Response()
        api._set_session(request, response, "token123")
        return response.headers["set-cookie"]

    def test_plain_http_cookie_is_not_secure(self):
        cookie = self._cookie(_request())
        self.assertIn("HttpOnly", cookie)
        self.assertNotIn("Secure", cookie)

    def test_cloudflare_https_cookie_is_secure(self):
        cookie = self._cookie(_request(headers={"CF-Visitor": '{"scheme":"https"}'}))
        self.assertIn("Secure", cookie)

    def test_cloudflare_http_cookie_is_not_secure(self):
        cookie = self._cookie(_request(headers={"CF-Visitor": '{"scheme":"http"}'}))
        self.assertNotIn("Secure", cookie)

    def test_direct_https_cookie_is_secure(self):
        self.assertIn("Secure", self._cookie(_request(scheme="https")))


if __name__ == "__main__":
    unittest.main()
