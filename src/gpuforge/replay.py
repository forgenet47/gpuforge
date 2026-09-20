"""Bounded replay and monotonic-sequence protection with optional persistence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_STATE_VERSION = 1
_MAX_STATE_BYTES = 4 * 1024 * 1024
_KEY_DOMAIN = b"gpuforge-replay-key-v1\x00"


class ReplayProtectionError(ValueError):
    """Raised when replay state is invalid, full, or detects reused input."""


class ReplayCache:
    """Fail-closed bounded cache for replay tokens and monotonic sequences."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        max_entries: int = 10_000,
        max_sequences: int = 10_000,
    ) -> None:
        if not 1 <= max_entries <= 1_000_000 or not 1 <= max_sequences <= 1_000_000:
            raise ReplayProtectionError("Replay cache limits are outside the permitted range")
        self._path = path
        self._max_entries = max_entries
        self._max_sequences = max_sequences
        self._entries: dict[str, int] = {}
        self._sequences: dict[str, int] = {}
        if path is not None and path.exists():
            self._load()

    def accept_once(
        self,
        *,
        scope: str,
        token: str,
        current_block: int,
        expires_at_block: int,
    ) -> None:
        """Persist a fresh token or reject it if it is already active."""
        _block("current_block", current_block)
        _block("expires_at_block", expires_at_block)
        if expires_at_block < current_block:
            raise ReplayProtectionError("Replay token is already expired")
        self._purge(current_block)
        key = _cache_key(scope, token)
        if key in self._entries:
            raise ReplayProtectionError("Replay token has already been accepted")
        if len(self._entries) >= self._max_entries:
            raise ReplayProtectionError("Replay cache is full")
        entries_before = self._entries.copy()
        self._entries[key] = expires_at_block
        try:
            self._persist()
        except ReplayProtectionError:
            self._entries = entries_before
            raise

    def accept_sequence(self, *, scope: str, sequence: int) -> None:
        """Persist a strictly increasing sequence for one opaque scope."""
        _sequence(sequence)
        key = _cache_key("sequence", scope)
        previous = self._sequences.get(key)
        if previous is not None and sequence <= previous:
            raise ReplayProtectionError("Sequence is not strictly increasing")
        if previous is None and len(self._sequences) >= self._max_sequences:
            raise ReplayProtectionError("Sequence cache is full")
        sequences_before = self._sequences.copy()
        self._sequences[key] = sequence
        try:
            self._persist()
        except ReplayProtectionError:
            self._sequences = sequences_before
            raise

    def accept_once_with_sequence(
        self,
        *,
        scope: str,
        token: str,
        current_block: int,
        expires_at_block: int,
        sequence_scope: str,
        sequence: int,
    ) -> None:
        """Atomically persist one replay token and its monotonic sequence."""
        _block("current_block", current_block)
        _block("expires_at_block", expires_at_block)
        _sequence(sequence)
        if expires_at_block < current_block:
            raise ReplayProtectionError("Replay token is already expired")

        entries_before = self._entries.copy()
        sequences_before = self._sequences.copy()
        try:
            self._purge(current_block)
            entry_key = _cache_key(scope, token)
            sequence_key = _cache_key("sequence", sequence_scope)
            if entry_key in self._entries:
                raise ReplayProtectionError("Replay token has already been accepted")
            previous = self._sequences.get(sequence_key)
            if previous is not None and sequence <= previous:
                raise ReplayProtectionError("Sequence is not strictly increasing")
            if len(self._entries) >= self._max_entries:
                raise ReplayProtectionError("Replay cache is full")
            if previous is None and len(self._sequences) >= self._max_sequences:
                raise ReplayProtectionError("Sequence cache is full")
            self._entries[entry_key] = expires_at_block
            self._sequences[sequence_key] = sequence
            self._persist()
        except ReplayProtectionError:
            self._entries = entries_before
            self._sequences = sequences_before
            raise

    def _purge(self, current_block: int) -> None:
        self._entries = {
            key: expiry for key, expiry in self._entries.items() if expiry >= current_block
        }

    def _load(self) -> None:
        if self._path is None:  # pragma: no cover - guarded by caller
            return
        try:
            if self._path.stat().st_size > _MAX_STATE_BYTES:
                raise ReplayProtectionError("Replay state exceeds the size limit")
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except ReplayProtectionError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ReplayProtectionError("Replay state is unavailable or malformed") from None
        if not isinstance(raw, dict) or set(raw) != {"entries", "sequences", "version"}:
            raise ReplayProtectionError("Replay state has an invalid schema")
        if raw["version"] != _STATE_VERSION:
            raise ReplayProtectionError("Replay state has an unsupported version")
        entries = _state_map(raw["entries"], "entries", self._max_entries)
        sequences = _state_map(raw["sequences"], "sequences", self._max_sequences)
        self._entries = entries
        self._sequences = sequences

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        payload = json.dumps(
            {
                "entries": self._entries,
                "sequences": self._sequences,
                "version": _STATE_VERSION,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            temporary.write_text(payload, encoding="utf-8", newline="\n")
            os.replace(temporary, self._path)
        except OSError:
            raise ReplayProtectionError("Replay state could not be persisted") from None


def _cache_key(scope: str, token: str) -> str:
    if not isinstance(scope, str) or not isinstance(token, str):
        raise ReplayProtectionError("Replay scope and token must be text")
    if not 1 <= len(scope.encode("utf-8")) <= 512 or not 1 <= len(token.encode("utf-8")) <= 512:
        raise ReplayProtectionError("Replay scope or token violates its size limit")
    return hashlib.sha256(_KEY_DOMAIN + scope.encode() + b"\x00" + token.encode()).hexdigest()


def _block(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ReplayProtectionError(f"{name} is outside the permitted range")


def _sequence(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise ReplayProtectionError("Sequence is outside the permitted range")


def _state_map(value: object, name: str, limit: int) -> dict[str, int]:
    if not isinstance(value, dict) or len(value) > limit:
        raise ReplayProtectionError(f"Replay state {name} is invalid")
    result: dict[str, int] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or _is_lower_hex_digest(key) is False
            or isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item <= 2**63 - 1
        ):
            raise ReplayProtectionError(f"Replay state {name} is invalid")
        result[key] = item
    return result


def _is_lower_hex_digest(value: str) -> bool:
    """Return whether a cache key is exactly one lowercase SHA-256 hex value."""
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
