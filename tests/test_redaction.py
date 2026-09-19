"""Tests for log and text redaction."""

from __future__ import annotations

import io
import logging

from gpuforge.config import RuntimeSecrets
from gpuforge.redaction import SecretRedactingFormatter, redact_sensitive_text


def test_text_redaction_removes_injected_and_assigned_secrets() -> None:
    """Known values and common assignment formats are removed."""
    raw_secret = "runtime-secret-canary"
    secrets = RuntimeSecrets.from_environment({"GPUFORGE_ARTIFACT_ACCESS_TOKEN": raw_secret})
    message = f"raw={raw_secret} password=unsafe api_key:'unsafe-2' Authorization: Bearer unsafe-3"

    redacted = redact_sensitive_text(message, secrets)

    assert raw_secret not in redacted
    assert "unsafe" not in redacted
    assert redacted.count("<redacted>") == 4


def test_log_formatter_redacts_messages_and_exceptions() -> None:
    """The final formatted record cannot expose secrets from exception text."""
    raw_secret = "exception-secret-canary"
    secrets = RuntimeSecrets.from_environment({"GPUFORGE_ATTESTATION_ACCESS_TOKEN": raw_secret})
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(SecretRedactingFormatter(secrets, "%(levelname)s %(message)s"))
    logger = logging.getLogger("gpuforge.tests.redaction")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    try:
        raise RuntimeError(f"request failed with {raw_secret}")
    except RuntimeError:
        logger.exception("attestation token=%s", raw_secret)

    rendered = output.getvalue()
    assert raw_secret not in rendered
    assert rendered.count("<redacted>") >= 2
