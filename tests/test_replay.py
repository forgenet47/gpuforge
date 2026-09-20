"""Tests for bounded and persistent replay protection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuforge.replay import ReplayCache, ReplayProtectionError


def test_token_is_accepted_once_and_expired_entries_are_replaced() -> None:
    """An active token is rejected while expired storage can be reclaimed."""
    cache = ReplayCache(max_entries=1)
    cache.accept_once(scope="claim:miner", token="nonce-one", current_block=10, expires_at_block=12)

    with pytest.raises(ReplayProtectionError, match="already been accepted"):
        cache.accept_once(
            scope="claim:miner", token="nonce-one", current_block=11, expires_at_block=12
        )

    cache.accept_once(scope="claim:miner", token="nonce-two", current_block=13, expires_at_block=20)


def test_cache_capacity_fails_closed() -> None:
    """Fresh active entries are never silently evicted at the configured bound."""
    cache = ReplayCache(max_entries=1)
    cache.accept_once(scope="scope", token="one", current_block=1, expires_at_block=10)

    with pytest.raises(ReplayProtectionError, match="full"):
        cache.accept_once(scope="scope", token="two", current_block=2, expires_at_block=10)


def test_sequence_is_strictly_monotonic_across_restart(tmp_path: Path) -> None:
    """Persisted sequence state rejects repeats and regressions after restart."""
    state_path = tmp_path / "replay.json"
    ReplayCache(path=state_path).accept_sequence(scope="miner:lease", sequence=2)
    restarted = ReplayCache(path=state_path)

    with pytest.raises(ReplayProtectionError, match="strictly increasing"):
        restarted.accept_sequence(scope="miner:lease", sequence=2)
    with pytest.raises(ReplayProtectionError, match="strictly increasing"):
        restarted.accept_sequence(scope="miner:lease", sequence=1)

    restarted.accept_sequence(scope="miner:lease", sequence=3)


def test_combined_acceptance_rolls_back_on_sequence_failure() -> None:
    """A rejected evidence sequence does not consume its independent replay token."""
    cache = ReplayCache()
    cache.accept_sequence(scope="miner:lease", sequence=4)

    with pytest.raises(ReplayProtectionError, match="strictly increasing"):
        cache.accept_once_with_sequence(
            scope="evidence:miner",
            token="first-token",
            current_block=10,
            expires_at_block=20,
            sequence_scope="miner:lease",
            sequence=4,
        )

    cache.accept_once(
        scope="evidence:miner",
        token="first-token",
        current_block=10,
        expires_at_block=20,
    )


def test_persistent_state_hashes_identifiers(tmp_path: Path) -> None:
    """Disk state contains only one-way cache keys, not hotkeys or nonces."""
    state_path = tmp_path / "replay.json"
    cache = ReplayCache(path=state_path)
    cache.accept_once(
        scope="capability:5ExampleHotkey",
        token="sensitive-nonce-value",
        current_block=1,
        expires_at_block=2,
    )

    stored = state_path.read_text(encoding="utf-8")
    assert "5ExampleHotkey" not in stored
    assert "sensitive-nonce-value" not in stored
    assert json.loads(stored)["version"] == 1


@pytest.mark.parametrize(
    "content",
    (
        "not-json",
        '{"entries":{},"version":1}',
        '{"entries":{},"sequences":{},"version":2}',
        '{"entries":{"NOT-A-DIGEST":1},"sequences":{},"version":1}',
    ),
)
def test_malformed_persistent_state_fails_closed(tmp_path: Path, content: str) -> None:
    """Invalid or incompatible state is rejected rather than reset silently."""
    state_path = tmp_path / "replay.json"
    state_path.write_text(content, encoding="utf-8")

    with pytest.raises(ReplayProtectionError):
        ReplayCache(path=state_path)


def test_oversized_persistent_state_fails_closed(tmp_path: Path) -> None:
    """Replay state parsing has an explicit memory-use bound."""
    state_path = tmp_path / "replay.json"
    state_path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))

    with pytest.raises(ReplayProtectionError, match="size limit"):
        ReplayCache(path=state_path)
