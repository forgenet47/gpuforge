"""Commit-reveal challenges bound to an accepted training lease."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HEX_256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_ITEMS = 1_000_000
_SEED_DOMAIN = b"gpuforge/challenge/seed/v1\x00"
_COMMITMENT_DOMAIN = b"gpuforge/challenge/commitment/v1\x00"
_SHARD_DOMAIN = b"gpuforge/challenge/shard/v1\x00"
_CANARY_DOMAIN = b"gpuforge/challenge/canary/v1\x00"


class ChallengeErrorCode(str, Enum):
    """Stable challenge errors that do not disclose secret material."""

    INVALID = "invalid_challenge"
    NOT_ACCEPTED = "lease_not_accepted"
    EXPIRED = "challenge_expired"
    MISMATCH = "challenge_mismatch"
    REPLAY = "challenge_replay"
    CAPACITY = "replay_guard_capacity"


class ChallengeFailure(RuntimeError):
    """Sanitized challenge failure containing only a stable code."""

    def __init__(self, code: ChallengeErrorCode) -> None:
        if not isinstance(code, ChallengeErrorCode):
            raise ValueError("Challenge failure code is invalid")
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class ChallengeCommitment:
    """Public commitment published before a miner accepts a lease."""

    challenge_id: str
    commitment_digest: str
    deadline_block: int
    shard_count: int
    canary_pool_size: int
    canary_count: int

    def __post_init__(self) -> None:
        _hex_256(self.challenge_id)
        _digest(self.commitment_digest)
        _integer(self.deadline_block, 1, 2**63 - 1)
        _integer(self.shard_count, 1, _MAX_ITEMS)
        _integer(self.canary_pool_size, 1, _MAX_ITEMS)
        _integer(self.canary_count, 1, self.canary_pool_size)


@dataclass(frozen=True, slots=True)
class ChallengeReveal:
    """Secret-derived selection disclosed only after lease acceptance."""

    challenge_id: str
    seed: str = field(repr=False)
    shard_order: tuple[int, ...]
    canary_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _hex_256(self.challenge_id)
        _hex_256(self.seed)
        _indices(self.shard_order)
        _indices(self.canary_indices)


class ValidatorChallengeGenerator:
    """Derive unpredictable challenges from validator-held entropy."""

    __slots__ = ("_id_factory", "_master_secret")

    def __init__(
        self,
        master_secret: bytes,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(master_secret, bytes) or len(master_secret) < 32:
            raise ChallengeFailure(ChallengeErrorCode.INVALID)
        self._master_secret = master_secret
        self._id_factory = id_factory or (lambda: secrets.token_hex(32))

    def __repr__(self) -> str:
        return "ValidatorChallengeGenerator(master_secret=<redacted>)"

    def commit(
        self,
        *,
        job_digest: str,
        lease_nonce: str,
        deadline_block: int,
        shard_count: int,
        canary_pool_size: int,
        canary_count: int,
        challenge_id: str | None = None,
    ) -> ChallengeCommitment:
        """Create the value published before the miner accepts the lease."""
        selected_id = self._id_factory() if challenge_id is None else challenge_id
        provisional = ChallengeCommitment(
            challenge_id=selected_id,
            commitment_digest="sha256:" + "0" * 64,
            deadline_block=deadline_block,
            shard_count=shard_count,
            canary_pool_size=canary_pool_size,
            canary_count=canary_count,
        )
        context = _context(provisional, job_digest, lease_nonce)
        seed = hmac.digest(self._master_secret, _SEED_DOMAIN + context, "sha256")
        commitment_digest = _commitment_digest(context, seed)
        return ChallengeCommitment(
            challenge_id=selected_id,
            commitment_digest=commitment_digest,
            deadline_block=deadline_block,
            shard_count=shard_count,
            canary_pool_size=canary_pool_size,
            canary_count=canary_count,
        )

    def reveal(
        self,
        commitment: ChallengeCommitment,
        *,
        job_digest: str,
        lease_nonce: str,
        accepted: bool,
        current_block: int,
    ) -> ChallengeReveal:
        """Reveal a deterministic selection after acceptance and before expiry."""
        _acceptance_window(commitment, accepted, current_block)
        context = _context(commitment, job_digest, lease_nonce)
        seed = hmac.digest(self._master_secret, _SEED_DOMAIN + context, "sha256")
        if not hmac.compare_digest(commitment.commitment_digest, _commitment_digest(context, seed)):
            raise ChallengeFailure(ChallengeErrorCode.MISMATCH)
        return _build_reveal(commitment, seed)


class ChallengeVerifier:
    """Verify and consume challenge reveals exactly once."""

    __slots__ = ("_consumed", "_max_consumed")

    def __init__(self, *, max_consumed: int = 100_000) -> None:
        _integer(max_consumed, 1, 10_000_000)
        self._max_consumed = max_consumed
        self._consumed: set[str] = set()

    def verify_and_consume(
        self,
        commitment: ChallengeCommitment,
        reveal: ChallengeReveal,
        *,
        job_digest: str,
        lease_nonce: str,
        accepted: bool,
        current_block: int,
    ) -> None:
        """Validate binding, selection, deadline, and replay state."""
        _acceptance_window(commitment, accepted, current_block)
        context = _context(commitment, job_digest, lease_nonce)
        try:
            seed = bytes.fromhex(reveal.seed)
        except ValueError:
            raise ChallengeFailure(ChallengeErrorCode.INVALID) from None
        expected_digest = _commitment_digest(context, seed)
        expected = _build_reveal(commitment, seed)
        if (
            not hmac.compare_digest(commitment.challenge_id, reveal.challenge_id)
            or not hmac.compare_digest(commitment.commitment_digest, expected_digest)
            or reveal.shard_order != expected.shard_order
            or reveal.canary_indices != expected.canary_indices
        ):
            raise ChallengeFailure(ChallengeErrorCode.MISMATCH)
        replay_key = commitment.commitment_digest
        if replay_key in self._consumed:
            raise ChallengeFailure(ChallengeErrorCode.REPLAY)
        if len(self._consumed) >= self._max_consumed:
            raise ChallengeFailure(ChallengeErrorCode.CAPACITY)
        self._consumed.add(replay_key)


def _context(
    commitment: ChallengeCommitment,
    job_digest: str,
    lease_nonce: str,
) -> bytes:
    _digest(job_digest)
    _hex_256(lease_nonce)
    value = {
        "canary_count": commitment.canary_count,
        "canary_pool_size": commitment.canary_pool_size,
        "challenge_id": commitment.challenge_id,
        "deadline_block": commitment.deadline_block,
        "job_digest": job_digest,
        "lease_nonce": lease_nonce,
        "shard_count": commitment.shard_count,
    }
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "ascii"
    )


def _commitment_digest(context: bytes, seed: bytes) -> str:
    return "sha256:" + hashlib.sha256(_COMMITMENT_DOMAIN + context + seed).hexdigest()


def _build_reveal(commitment: ChallengeCommitment, seed: bytes) -> ChallengeReveal:
    shard_order = _selection(seed, _SHARD_DOMAIN, commitment.shard_count)[: commitment.shard_count]
    canary_indices = _selection(seed, _CANARY_DOMAIN, commitment.canary_pool_size)[
        : commitment.canary_count
    ]
    return ChallengeReveal(
        challenge_id=commitment.challenge_id,
        seed=seed.hex(),
        shard_order=shard_order,
        canary_indices=canary_indices,
    )


def _selection(seed: bytes, domain: bytes, size: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(size),
            key=lambda index: (
                hmac.digest(seed, domain + index.to_bytes(8, "big"), "sha256"),
                index,
            ),
        )
    )


def _acceptance_window(commitment: ChallengeCommitment, accepted: bool, current_block: int) -> None:
    if not isinstance(accepted, bool) or not accepted:
        raise ChallengeFailure(ChallengeErrorCode.NOT_ACCEPTED)
    _integer(current_block, 0, 2**63 - 1)
    if current_block > commitment.deadline_block:
        raise ChallengeFailure(ChallengeErrorCode.EXPIRED)


def _digest(value: object) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ChallengeFailure(ChallengeErrorCode.INVALID)


def _hex_256(value: object) -> None:
    if not isinstance(value, str) or _HEX_256_PATTERN.fullmatch(value) is None:
        raise ChallengeFailure(ChallengeErrorCode.INVALID)


def _integer(value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ChallengeFailure(ChallengeErrorCode.INVALID)


def _indices(value: object) -> None:
    if not isinstance(value, tuple) or len(value) > _MAX_ITEMS:
        raise ChallengeFailure(ChallengeErrorCode.INVALID)
    for index in value:
        _integer(index, 0, _MAX_ITEMS - 1)
    if len(set(value)) != len(value):
        raise ChallengeFailure(ChallengeErrorCode.INVALID)


__all__ = [
    "ChallengeCommitment",
    "ChallengeErrorCode",
    "ChallengeFailure",
    "ChallengeReveal",
    "ChallengeVerifier",
    "ValidatorChallengeGenerator",
]
