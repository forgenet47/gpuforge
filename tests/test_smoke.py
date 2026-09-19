"""Package and command-line smoke tests."""

import json
from pathlib import Path

import pytest

from gpuforge import __version__
from gpuforge.cli import PUBLICATION_CHECK_UNAVAILABLE, main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports() -> None:
    """The package exposes a pre-release version."""
    assert __version__ == "0.0.0.dev1"


def test_cli_without_command_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    """The empty command explains the available development utilities."""
    assert main([]) == 0
    assert "GPUForge development utilities" in capsys.readouterr().out


def test_publication_check_fails_closed(capsys: pytest.CaptureFixture[str]) -> None:
    """The command stub must not give a false successful safety signal."""
    exit_code = main(["publication-check"])

    assert exit_code == PUBLICATION_CHECK_UNAVAILABLE
    assert "not implemented" in capsys.readouterr().out


@pytest.mark.parametrize("role", ("miner", "validator"))
def test_role_config_check_is_offline_and_redacted(
    role: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both role commands validate examples without exposing runtime secrets."""
    raw_secret = "cli-secret-canary"
    monkeypatch.setenv("GPUFORGE_ARTIFACT_ACCESS_TOKEN", raw_secret)

    exit_code = main(
        [
            role,
            "--config",
            str(REPOSITORY_ROOT / "config" / f"{role}.local.toml"),
            "--check-config",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["network_connection_attempted"] is False
    assert payload["settings"]["role"] == role
    assert payload["settings"]["secrets"]["artifact_access_token_configured"] is True
    assert raw_secret not in captured.out


def test_role_command_without_check_mode_fails_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No miner or validator networking behavior exists behind the scaffold."""
    exit_code = main(["miner", "--config", str(REPOSITORY_ROOT / "config" / "miner.local.toml")])

    assert exit_code == PUBLICATION_CHECK_UNAVAILABLE
    assert "not implemented" in capsys.readouterr().err


def test_role_command_rejects_mismatched_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The selected role must match the checked configuration."""
    exit_code = main(
        [
            "validator",
            "--config",
            str(REPOSITORY_ROOT / "config" / "miner.local.toml"),
            "--check-config",
        ]
    )

    assert exit_code == PUBLICATION_CHECK_UNAVAILABLE
    assert "does not match" in capsys.readouterr().err
