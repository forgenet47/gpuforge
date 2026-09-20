"""Tests for signatures, freshness, nonces, and authenticated transitions."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from gpuforge.config import EvidenceTier, NetworkPolicy
from gpuforge.identity import (
    AuthenticationError,
    BittensorHotkeyAdapter,
    MessageAuthenticator,
    generate_nonce,
    sign_message,
    validate_freshness,
    verify_message_signature,
)
from gpuforge.protocol import (
    CapabilityClaim,
    CheckpointCommitment,
    ExecutionEvidence,
    JobManifest,
    Message,
    ResourcePolicy,
    SoftwareVersion,
    ValidationReceipt,
    VerificationPolicy,
    WorkLease,
)
from gpuforge.replay import ReplayCache, ReplayProtectionError

MINER = "5MinerHotkeyExample111111111111111111"
VALIDATOR = "5ValidatorHotkeyExample11111111111111"
PUBLISHER = "5PublisherHotkeyExample11111111111111"
UNSIGNED = "00" * 64
NONCE = "ab" * 32


def digest(character: str) -> str:
    """Return a valid content digest for protocol fixtures."""
    return f"sha256:{character * 64}"


@dataclass(frozen=True)
class HmacSigner:
    """Deterministic 64-byte test signer; not used by production code."""

    hotkey: str
    secret: bytes

    def sign(self, payload: bytes) -> bytes:
        """Return an HMAC-SHA512 signature."""
        return hmac.new(self.secret, payload, hashlib.sha512).digest()


@dataclass(frozen=True)
class HmacVerifier:
    """Verify test signatures for a fixed hotkey registry."""

    secrets: dict[str, bytes]

    def verify(self, hotkey: str, payload: bytes, signature: bytes) -> bool:
        """Verify without raising for unknown identities."""
        secret = self.secrets.get(hotkey)
        if secret is None:
            return False
        expected = hmac.new(secret, payload, hashlib.sha512).digest()
        return hmac.compare_digest(signature, expected)


def job() -> JobManifest:
    """Return an unsigned publisher manifest."""
    return JobManifest(
        job_id="job-identity-1",
        container_digest=digest("1"),
        entrypoint_digest=digest("2"),
        input_root=digest("3"),
        framework="pytorch",
        resource_policy=ResourcePolicy(
            gpu_count=1,
            gpu_memory_mb=81_920,
            cpu_cores=16,
            memory_mb=131_072,
            max_runtime_seconds=3_600,
            network_policy=NetworkPolicy.DENY,
        ),
        verification_policy=VerificationPolicy(
            minimum_evidence_tier=EvidenceTier.C,
            challenge_kind="gradient_slice",
            checkpoint_interval_steps=100,
        ),
        lease_seconds=3_600,
        publisher_hotkey=PUBLISHER,
        expires_at_block=1_100,
        signature=UNSIGNED,
    )


def claim(*, nonce: str = NONCE, observed_at_block: int = 1_000) -> CapabilityClaim:
    """Return an unsigned miner capability claim."""
    return CapabilityClaim(
        miner_hotkey=MINER,
        gpu_count=1,
        gpu_model="NVIDIA H100 SXM",
        gpu_memory_mb=81_920,
        runtime_versions=(SoftwareVersion("cuda", "12.8"),),
        supported_evidence_tiers=(EvidenceTier.C,),
        available_gpu_seconds=3_600,
        nonce=nonce,
        observed_at_block=observed_at_block,
        signature=UNSIGNED,
    )


def lease(*, nonce: str = NONCE) -> WorkLease:
    """Return an unsigned validator lease."""
    return WorkLease(
        manifest_digest=job().digest(),
        miner_hotkey=MINER,
        validator_hotkey=VALIDATOR,
        shard_id="shard-1",
        challenge_commitment=digest("4"),
        start_block=1_000,
        deadline_block=1_050,
        nonce=nonce,
        signature=UNSIGNED,
    )


def evidence(*, sequence: int = 1, submitted_at_block: int = 1_010) -> ExecutionEvidence:
    """Return unsigned miner execution evidence."""
    return ExecutionEvidence(
        lease_digest=lease().digest(),
        miner_hotkey=MINER,
        attestation_digest=digest("5"),
        image_digest=digest("1"),
        challenge_response_digest=digest("6"),
        checkpoints=(CheckpointCommitment(step=100, digest=digest("7")),),
        result_digest=digest("8"),
        work_units=100,
        active_seconds_ms=1_000,
        submitted_at_block=submitted_at_block,
        sequence=sequence,
        signature=UNSIGNED,
    )


def receipt() -> ValidationReceipt:
    """Return an unsigned validator receipt."""
    return ValidationReceipt(
        evidence_digest=evidence().digest(),
        validator_hotkey=VALIDATOR,
        accepted=True,
        reason_codes=(),
        verified_work_units=100,
        confidence_tier=EvidenceTier.C,
        validated_at_block=1_020,
        signature=UNSIGNED,
    )


@pytest.fixture
def signers() -> dict[str, HmacSigner]:
    """Return deterministic test signers for each protocol role."""
    return {
        PUBLISHER: HmacSigner(PUBLISHER, b"publisher-test-key"),
        MINER: HmacSigner(MINER, b"miner-test-key"),
        VALIDATOR: HmacSigner(VALIDATOR, b"validator-test-key"),
    }


@pytest.fixture
def verifier(signers: dict[str, HmacSigner]) -> HmacVerifier:
    """Return a registry verifier matching the test signers."""
    return HmacVerifier({hotkey: signer.secret for hotkey, signer in signers.items()})


@pytest.mark.parametrize("message", (job(), claim(), lease(), evidence(), receipt()))
def test_each_transition_signs_and_verifies(
    message: Message,
    signers: dict[str, HmacSigner],
    verifier: HmacVerifier,
) -> None:
    """Every transition authenticates the role identity over its exact content."""
    signer = signers[
        message.publisher_hotkey
        if isinstance(message, JobManifest)
        else message.miner_hotkey
        if isinstance(message, CapabilityClaim | ExecutionEvidence)
        else message.validator_hotkey
    ]
    signed = sign_message(message, signer)

    verify_message_signature(signed, verifier)
    assert signed.signature != UNSIGNED


def test_tampering_and_cross_message_signature_reuse_fail(
    signers: dict[str, HmacSigner], verifier: HmacVerifier
) -> None:
    """A signature cannot authenticate changed content or another message type."""
    signed_evidence = cast(ExecutionEvidence, sign_message(evidence(), signers[MINER]))
    with pytest.raises(AuthenticationError, match="verification failed"):
        verify_message_signature(replace(signed_evidence, work_units=101), verifier)

    signed_lease = sign_message(lease(), signers[VALIDATOR])
    forged_receipt = replace(receipt(), signature=signed_lease.signature)
    with pytest.raises(AuthenticationError, match="verification failed"):
        verify_message_signature(forged_receipt, verifier)


def test_wrong_role_signer_is_rejected(signers: dict[str, HmacSigner]) -> None:
    """A valid signer cannot impersonate another message role."""
    with pytest.raises(AuthenticationError, match="does not match"):
        sign_message(job(), signers[MINER])


def test_nonce_generation_is_random_and_bounded() -> None:
    """Generated nonces are unique lowercase 256-bit hexadecimal values."""
    nonces = {generate_nonce() for _ in range(64)}
    assert len(nonces) == 64
    assert all(len(value) == 64 for value in nonces)
    assert all(set(value) <= set("0123456789abcdef") for value in nonces)


@pytest.mark.parametrize(
    ("message", "current_block", "error"),
    (
        (replace(job(), expires_at_block=999), 1_000, "expired"),
        (lease(), 990, "not active"),
        (lease(), 1_051, "expired"),
        (claim(observed_at_block=900), 1_000, "stale"),
        (claim(observed_at_block=1_003), 1_000, "future"),
        (evidence(submitted_at_block=900), 1_000, "stale"),
        (replace(receipt(), validated_at_block=1_003), 1_000, "future"),
    ),
)
def test_expiry_and_clock_skew_are_rejected(
    message: Message, current_block: int, error: str
) -> None:
    """Freshness checks reject expired, stale, and implausibly future messages."""
    with pytest.raises(AuthenticationError, match=error):
        validate_freshness(message, current_block=current_block)


def test_freshness_policy_accepts_inclusive_boundaries() -> None:
    """Age and future-skew limits have stable inclusive boundary behavior."""
    assert validate_freshness(claim(observed_at_block=936), current_block=1_000) == 1_000
    assert validate_freshness(claim(observed_at_block=1_002), current_block=1_000) == 1_066


def test_authenticator_rejects_nonce_reuse_and_sequence_regression(
    signers: dict[str, HmacSigner], verifier: HmacVerifier
) -> None:
    """Authenticated transitions also enforce one-time tokens and monotonic evidence."""
    authenticator = MessageAuthenticator(verifier, ReplayCache())
    signed_claim = sign_message(claim(), signers[MINER])
    authenticator.authenticate(signed_claim, current_block=1_000)
    with pytest.raises(ReplayProtectionError, match="already been accepted"):
        authenticator.authenticate(signed_claim, current_block=1_000)

    first = sign_message(evidence(sequence=2), signers[MINER])
    authenticator.authenticate(first, current_block=1_010)
    regressed = sign_message(evidence(sequence=1), signers[MINER])
    with pytest.raises(ReplayProtectionError, match="strictly increasing"):
        authenticator.authenticate(regressed, current_block=1_010)


def test_replay_is_rejected_after_restart(
    tmp_path: Path, signers: dict[str, HmacSigner], verifier: HmacVerifier
) -> None:
    """Persisted replay state survives construction of a new authenticator."""
    state_path = tmp_path / "replay.json"
    signed = sign_message(claim(), signers[MINER])
    MessageAuthenticator(verifier, ReplayCache(path=state_path)).authenticate(
        signed, current_block=1_000
    )

    restarted = MessageAuthenticator(verifier, ReplayCache(path=state_path))
    with pytest.raises(ReplayProtectionError, match="already been accepted"):
        restarted.authenticate(signed, current_block=1_000)


class FakeKeypair:
    """Small Bittensor-shaped keypair used to test the structural adapter."""

    ss58_address = PUBLISHER

    def sign(self, payload: bytes) -> bytes:
        """Return a deterministic 64-byte test signature."""
        return hashlib.sha512(b"fake-key" + payload).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        """Verify the deterministic test signature."""
        return hmac.compare_digest(signature, self.sign(payload))


def test_bittensor_keypair_shape_is_adapted_without_runtime_dependency() -> None:
    """The adapter works with the public hotkey keypair shape via structural typing."""
    adapter = BittensorHotkeyAdapter(FakeKeypair())
    signed = sign_message(job(), adapter)

    verify_message_signature(signed, adapter)
    assert adapter.verify(MINER, signed.signing_bytes(), bytes.fromhex(signed.signature)) is False
