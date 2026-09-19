"""Tests for repository publication boundaries."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GIT_EXECUTABLE = shutil.which("git")

PRIVATE_PATHS = (
    ".internal/example.md",
    ".private/example.md",
    "docs/development/design.md",
    "docs/internal/operations.md",
    "notes.plan.md",
    "release.private.md",
    "TODO.md",
    ".env",
    ".env.local",
    "secrets/credential.txt",
    "wallets/coldkey",
    "logs/miner.log",
    "data/private/shard.bin",
    "datasets/private/sample.bin",
    "checkpoints/model.pt",
)


@pytest.mark.parametrize("private_path", PRIVATE_PATHS)
def test_private_path_is_ignored(private_path: str) -> None:
    """Private development and runtime artifacts must stay outside Git."""
    if GIT_EXECUTABLE is None:
        pytest.skip("Git is required to verify repository ignore rules")

    result = subprocess.run(  # noqa: S603 - executable and arguments are test constants.
        [GIT_EXECUTABLE, "check-ignore", "--quiet", "--no-index", private_path],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert result.returncode == 0, f"expected Git to ignore {private_path}"


@pytest.mark.parametrize("public_path", ("README.md", "pyproject.toml", "src/gpuforge/cli.py"))
def test_public_source_is_not_ignored(public_path: str) -> None:
    """Technical source and documentation must remain publishable."""
    if GIT_EXECUTABLE is None:
        pytest.skip("Git is required to verify repository ignore rules")

    result = subprocess.run(  # noqa: S603 - executable and arguments are test constants.
        [GIT_EXECUTABLE, "check-ignore", "--quiet", "--no-index", public_path],
        cwd=REPOSITORY_ROOT,
        check=False,
    )

    assert result.returncode == 1, f"expected Git not to ignore {public_path}"
