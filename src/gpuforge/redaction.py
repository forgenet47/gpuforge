"""Centralized redaction for configuration values and log output."""

from __future__ import annotations

import logging
import re

from gpuforge.config import RuntimeSecrets

_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
    r"(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def redact_sensitive_text(text: str, secrets: RuntimeSecrets) -> str:
    """Remove configured secret values and common credential assignments."""
    redacted = text
    for value in secrets.values():
        redacted = redacted.replace(value, "<redacted>")
    redacted = _ASSIGNMENT_PATTERN.sub(r"\1\2<redacted>", redacted)
    return _BEARER_PATTERN.sub("Bearer <redacted>", redacted)


class SecretRedactingFormatter(logging.Formatter):
    """Format a complete log record, then redact secrets including exceptions."""

    def __init__(self, secrets: RuntimeSecrets, fmt: str | None = None) -> None:
        super().__init__(fmt=fmt)
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        """Return a formatted record with sensitive material removed."""
        return redact_sensitive_text(super().format(record), self._secrets)
