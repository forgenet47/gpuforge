"""Package and command-line smoke tests."""

import pytest

from gpuforge import __version__
from gpuforge.cli import PUBLICATION_CHECK_UNAVAILABLE, main


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
