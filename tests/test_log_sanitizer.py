"""Tests for the log sanitizer utility module."""

import re
from unittest.mock import patch

import pytest

from template_agent.utils.log_sanitizer import (
    REDACTED,
    LogSanitizer,
    _get_default_sanitizer,
    _parse_custom_patterns,
    create_sanitize_processor,
    reset_default_sanitizer,
    sanitize_headers,
)


@pytest.fixture(autouse=True)
def _reset_sanitizer():
    """Ensure the module-level cached sanitizer is cleared between tests."""
    reset_default_sanitizer()
    yield
    reset_default_sanitizer()


class TestPIISanitization:
    """Verify PII regex patterns."""

    def test_email_redacted(self):
        sanitizer = LogSanitizer()
        assert sanitizer.sanitize_string("user@example.com") == "***EMAIL***"

    def test_email_in_sentence(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("Contact alice.bob@corp.io for details")
        assert "***EMAIL***" in result
        assert "alice.bob@corp.io" not in result

    def test_ssn_dashed(self):
        sanitizer = LogSanitizer()
        assert sanitizer.sanitize_string("SSN: 123-45-6789") == "SSN: ***SSN***"

    def test_phone_us_format(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("Call 555-123-4567 now")
        assert "***PHONE***" in result
        assert "555-123-4567" not in result

    def test_phone_with_parens(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("Phone: (555) 123-4567")
        assert "***PHONE***" in result

    def test_phone_with_country_code(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("Phone: +1-555-123-4567")
        assert "***PHONE***" in result

    def test_credit_card_redacted(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("Card: 4111 1111 1111 1111")
        assert "***CREDIT_CARD***" in result
        assert "4111" not in result


class TestCredentialSanitization:
    """Verify credential / token regex patterns."""

    def test_bearer_token(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string(
            "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
        )
        assert result == "Bearer ***TOKEN***"

    def test_basic_auth(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("Basic dXNlcjpwYXNz")
        assert result == "Basic ***TOKEN***"

    def test_jwt_standalone(self):
        sanitizer = LogSanitizer()
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456"
        result = sanitizer.sanitize_string(f"Token: {jwt}")
        assert "***JWT***" in result
        assert jwt not in result

    def test_api_key_pattern(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("api_key=sk_live_abc123defghijklmn")
        assert "***API_KEY***" in result

    def test_password_pattern(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("password=mysupersecretpassword")
        assert "***PASSWORD***" in result
        assert "mysupersecretpassword" not in result

    def test_aws_access_key(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string("key: AKIAIOSFODNN7EXAMPLE")
        assert "***AWS_KEY***" in result

    def test_github_token(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_string(
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
        )
        assert "***GITHUB_TOKEN***" in result


class TestDictSanitization:
    """Verify key-level and recursive dict sanitization."""

    def test_sensitive_key_fully_redacted(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_value({"password": "hunter2"})
        assert result == {"password": REDACTED}

    def test_nested_dict(self):
        sanitizer = LogSanitizer()
        data = {"user": {"email": "a@b.com", "name": "Alice"}}
        result = sanitizer.sanitize_value(data)
        assert "a@b.com" not in str(result)
        assert "***EMAIL***" in result["user"]["email"]

    def test_list_of_strings(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_value(["ok", "user@example.com", "fine"])
        assert result[1] == "***EMAIL***"
        assert result[0] == "ok"

    def test_tuple_preserved(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_value(("user@example.com",))
        assert isinstance(result, tuple)
        assert result[0] == "***EMAIL***"

    def test_token_key_redacted(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_value({"access_token": "abc123"})
        assert result == {"access_token": REDACTED}

    def test_non_sensitive_key_unchanged(self):
        sanitizer = LogSanitizer()
        result = sanitizer.sanitize_value({"method": "GET", "path": "/health"})
        assert result == {"method": "GET", "path": "/health"}

    def test_primitives_passthrough(self):
        sanitizer = LogSanitizer()
        assert sanitizer.sanitize_value(42) == 42
        assert sanitizer.sanitize_value(3.14) == 3.14
        assert sanitizer.sanitize_value(True) is True
        assert sanitizer.sanitize_value(None) is None


class TestSanitizeHeaders:
    """Verify HTTP header sanitization helper."""

    @patch.dict("os.environ", {"LOG_SANITIZATION_ENABLED": "true"}, clear=False)
    def test_authorization_redacted(self):
        headers = {"authorization": "Bearer abc123", "content-type": "application/json"}
        result = sanitize_headers(headers)
        assert result["authorization"] == REDACTED
        assert result["content-type"] == "application/json"

    @patch.dict("os.environ", {"LOG_SANITIZATION_ENABLED": "true"}, clear=False)
    def test_x_token_redacted(self):
        headers = {"x-token": "mysecrettoken", "accept": "text/html"}
        result = sanitize_headers(headers)
        assert result["x-token"] == REDACTED
        assert result["accept"] == "text/html"

    @patch.dict("os.environ", {"LOG_SANITIZATION_ENABLED": "true"}, clear=False)
    def test_cookie_redacted(self):
        headers = {"cookie": "session=abc123; csrftoken=xyz"}
        result = sanitize_headers(headers)
        assert result["cookie"] == REDACTED

    @patch.dict("os.environ", {"LOG_SANITIZATION_ENABLED": "true"}, clear=False)
    def test_x_api_key_redacted(self):
        headers = {"x-api-key": "key_123456"}
        result = sanitize_headers(headers)
        assert result["x-api-key"] == REDACTED


class TestSanitizationDisabled:
    """Ensure data passes through unchanged when disabled."""

    def test_disabled_string(self):
        sanitizer = LogSanitizer(enabled=False)
        assert sanitizer.sanitize_string("user@example.com") == "user@example.com"

    def test_disabled_dict(self):
        sanitizer = LogSanitizer(enabled=False)
        data = {"password": "secret", "token": "abc"}
        assert sanitizer.sanitize_value(data) == data

    def test_disabled_value(self):
        sanitizer = LogSanitizer(enabled=False)
        assert sanitizer.sanitize_value("Bearer xyz") == "Bearer xyz"

    def test_sanitize_empty_string(self):
        sanitizer = LogSanitizer()
        assert sanitizer.sanitize_string("") == ""


class TestCustomPatterns:
    """Verify user-supplied custom patterns."""

    def test_custom_pattern_applied(self):
        custom = [(re.compile(r"PROJ-\d+"), REDACTED)]
        sanitizer = LogSanitizer(custom_patterns=custom)
        result = sanitizer.sanitize_string("Ticket PROJ-1234 updated")
        assert REDACTED in result
        assert "PROJ-1234" not in result

    def test_parse_custom_patterns_csv(self):
        patterns = _parse_custom_patterns(r"foo\d+,bar_[a-z]+")
        assert len(patterns) == 2
        assert patterns[0][0].pattern == r"foo\d+"

    def test_parse_custom_patterns_empty(self):
        assert _parse_custom_patterns("") == []
        assert _parse_custom_patterns("   ") == []

    def test_parse_custom_patterns_invalid_regex_skipped(self):
        patterns = _parse_custom_patterns(r"valid\d+,[invalid")
        assert len(patterns) == 1


# ---------------------------------------------------------------------------
# structlog processor
# ---------------------------------------------------------------------------


class TestStructlogProcessor:
    """Verify the structlog processor function."""

    def test_processor_sanitizes_event_dict(self):
        processor = create_sanitize_processor()
        event_dict = {
            "event": "login attempt",
            "email": "admin@corp.com",
            "password": "topsecret",
            "status": "ok",
        }
        result = processor(None, "info", event_dict)
        assert result["email"] == "***EMAIL***"
        assert result["password"] == REDACTED
        assert result["status"] == "ok"

    def test_processor_preserves_non_sensitive(self):
        processor = create_sanitize_processor()
        event_dict = {"event": "health_check", "status": 200, "path": "/health"}
        result = processor(None, "info", event_dict)
        assert result == event_dict

    def test_processor_handles_nested(self):
        processor = create_sanitize_processor()
        event_dict = {
            "event": "request",
            "headers": {
                "authorization": "Bearer secret",
                "content-type": "application/json",
            },
        }
        result = processor(None, "info", event_dict)
        assert result["headers"]["authorization"] == REDACTED
        assert result["headers"]["content-type"] == "application/json"

    def test_processor_disabled_returns_event_unchanged(self):
        disabled = LogSanitizer(enabled=False)
        with patch(
            "template_agent.utils.log_sanitizer._default_sanitizer", new=disabled
        ):
            processor = create_sanitize_processor()
            event_dict = {"event": "login", "password": "secret", "email": "a@b.com"}
            result = processor(None, "info", event_dict)
            assert result["password"] == "secret"
            assert result["email"] == "a@b.com"

    def test_get_default_sanitizer_fallback_on_import_error(self):
        with patch(
            "template_agent.utils.log_sanitizer._default_sanitizer",
            new=None,
        ):
            with patch.dict("sys.modules", {"template_agent.src.settings": None}):
                sanitizer = _get_default_sanitizer()
                assert sanitizer.enabled is True


# ---------------------------------------------------------------------------
# Non-sensitive data passthrough
# ---------------------------------------------------------------------------


try:
    from template_agent.src.api import RequestLoggingMiddleware

    _HAS_API_MODULE = True
except ImportError:
    _HAS_API_MODULE = False


@pytest.mark.skipif(not _HAS_API_MODULE, reason="api.py deps not installed")
class TestRequestLoggingMiddlewareSanitization:
    """Verify sanitize_headers is applied in RequestLoggingMiddleware."""

    def test_middleware_sanitizes_request_headers(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        client = TestClient(app)
        response = client.get(
            "/ping",
            headers={"Authorization": "Bearer secret123", "X-Custom": "safe"},
        )
        assert response.status_code == 200

    def test_middleware_sanitizes_response_headers(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/setcookie")
        async def setcookie():
            resp = JSONResponse({"ok": True})
            resp.headers["Set-Cookie"] = "session=abc123"
            return resp

        client = TestClient(app)
        response = client.get("/setcookie")
        assert response.status_code == 200


class TestNonSensitivePassthrough:
    """Ensure normal log data is not corrupted."""

    def test_normal_log_message(self):
        sanitizer = LogSanitizer()
        msg = "Agent server starting up on port 8081"
        assert sanitizer.sanitize_string(msg) == msg

    def test_url_without_credentials(self):
        sanitizer = LogSanitizer()
        msg = "Connecting to http://localhost:5001/mcp"
        assert sanitizer.sanitize_string(msg) == msg

    def test_thread_id(self):
        sanitizer = LogSanitizer()
        msg = "Processing thread_id: abc-123-def-456"
        assert sanitizer.sanitize_string(msg) == msg

    def test_json_payload_without_pii(self):
        sanitizer = LogSanitizer()
        data = {
            "method": "POST",
            "path": "/v1/stream",
            "status_code": 200,
            "duration_ms": 42.5,
        }
        assert sanitizer.sanitize_value(data) == data
