"""Log sanitization utility for redacting sensitive data from log output.

Provides a structlog processor and helper functions that detect and redact
PII (emails, phone numbers, SSNs, credit cards) and credentials (tokens,
API keys, passwords) before they reach the log renderer.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each tuple is (compiled_regex, replacement_label).  Order matters: more
# specific patterns should come before generic ones to avoid partial matches.

PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "***EMAIL***"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "***SSN***"),
    (re.compile(r"\b\d{9}\b(?=\s|$|[^0-9])"), "***SSN***"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "***CREDIT_CARD***"),
    (
        re.compile(
            r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "***PHONE***",
    ),
]

CREDENTIAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "Bearer ***TOKEN***"),
    (re.compile(r"Basic\s+[A-Za-z0-9+/]+=*", re.IGNORECASE), "Basic ***TOKEN***"),
    (
        re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        "***JWT***",
    ),
    (
        re.compile(r"(?i)(?:api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9\-._~+/]{16,}[\"']?"),
        "***API_KEY***",
    ),
    (
        re.compile(r"(?i)(?:password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?[^\s\"',}{]+[\"']?"),
        "***PASSWORD***",
    ),
    (
        re.compile(r"(?i)(?:secret[_-]?key|client[_-]?secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9\-._~+/]{8,}[\"']?"),
        "***SECRET***",
    ),
    (
        re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
        "***AWS_KEY***",
    ),
    (
        re.compile(r"(?i)(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}"),
        "***GITHUB_TOKEN***",
    ),
]

SENSITIVE_HEADER_KEYS = frozenset(
    {
        "authorization",
        "x-token",
        "cookie",
        "set-cookie",
        "x-api-key",
        "proxy-authorization",
    }
)

SENSITIVE_DICT_KEYS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "secret_key",
        "secretkey",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "private_key",
        "privatekey",
        "credential",
        "credentials",
        "ssn",
        "credit_card",
        "creditcard",
        "card_number",
    }
)

REDACTED = "***REDACTED***"


# ---------------------------------------------------------------------------
# Sanitizer class
# ---------------------------------------------------------------------------


class LogSanitizer:
    """Configurable log sanitizer that redacts PII and credentials."""

    def __init__(
        self,
        enabled: bool = True,
        custom_patterns: Optional[List[Tuple[re.Pattern, str]]] = None,
    ):
        self.enabled = enabled
        self._patterns: List[Tuple[re.Pattern, str]] = []
        if enabled:
            self._patterns = CREDENTIAL_PATTERNS + PII_PATTERNS
            if custom_patterns:
                self._patterns.extend(custom_patterns)

    def sanitize_string(self, value: str) -> str:
        """Apply all regex patterns to a string value."""
        if not self.enabled or not value:
            return value
        for pattern, replacement in self._patterns:
            value = pattern.sub(replacement, value)
        return value

    def sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize a value (str, dict, list, or primitive)."""
        if not self.enabled:
            return value

        if isinstance(value, str):
            return self.sanitize_string(value)

        if isinstance(value, dict):
            return self._sanitize_dict(value)

        if isinstance(value, (list, tuple)):
            sanitized = [self.sanitize_value(item) for item in value]
            return type(value)(sanitized) if isinstance(value, tuple) else sanitized

        return value

    def _sanitize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary keys and values."""
        result: Dict[str, Any] = {}
        for key, val in d.items():
            lower_key = str(key).lower()
            normalised_key = lower_key.replace("-", "_")
            if (
                lower_key in SENSITIVE_HEADER_KEYS
                or normalised_key in SENSITIVE_DICT_KEYS
            ):
                result[key] = REDACTED
            else:
                result[key] = self.sanitize_value(val)
        return result


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

_default_sanitizer: Optional[LogSanitizer] = None


def _get_default_sanitizer() -> LogSanitizer:
    """Lazy-initialise the module-level sanitizer from settings.

    Uses deferred import to avoid the circular dependency chain:
    settings -> pylogger -> log_sanitizer -> settings.
    """
    global _default_sanitizer
    if _default_sanitizer is None:
        try:
            from template_agent.src.settings import settings as app_settings

            custom = _parse_custom_patterns(
                app_settings.LOG_SANITIZATION_CUSTOM_PATTERNS
            )
            _default_sanitizer = LogSanitizer(
                enabled=app_settings.LOG_SANITIZATION_ENABLED,
                custom_patterns=custom,
            )
        except Exception:
            _default_sanitizer = LogSanitizer(enabled=True)
    return _default_sanitizer


def _parse_custom_patterns(
    raw: str,
) -> List[Tuple[re.Pattern, str]]:
    """Parse comma-separated regex strings into compiled patterns."""
    if not raw:
        return []
    patterns: List[Tuple[re.Pattern, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if entry:
            try:
                patterns.append((re.compile(entry), REDACTED))
            except re.error:
                pass
    return patterns


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Redact known-sensitive HTTP header values.

    Intended for direct use in middleware *before* the data enters the
    structured-log pipeline, providing defence-in-depth.
    """
    sanitizer = _get_default_sanitizer()
    return sanitizer.sanitize_value(headers)


def reset_default_sanitizer() -> None:
    """Reset the cached default sanitizer (useful for tests)."""
    global _default_sanitizer
    _default_sanitizer = None


# ---------------------------------------------------------------------------
# structlog processor
# ---------------------------------------------------------------------------


def create_sanitize_processor():
    """Return a structlog processor that sanitizes every event dict value.

    The processor is a closure so that settings are read lazily on first
    log call rather than at import time.
    """

    def _sanitize_event(
        logger: Any, method_name: str, event_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        sanitizer = _get_default_sanitizer()
        if not sanitizer.enabled:
            return event_dict
        return sanitizer.sanitize_value(event_dict)

    return _sanitize_event
