"""Attestation policy tests use only deterministic fake verifier facts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import pytest

from gpuforge.attestation import (
    AttestationBackendFailure,
    AttestationError,
    AttestationPolicy,
    AttestationRequest,
    CertificateChain,
    EvidenceBundle,
    PolicyAttestationVerifier,
    ReferenceMeasurement,
    RevocationResult,
    RevocationStatus,
    VerificationReason,
    VerifiedAttestationClaims,
    VerifierResult,
)
from gpuforge.config import EvidenceTier

NOW = 2_000_000_000
FORMAT = "test_evidence_v1"
MEASUREMENT = "sha256:" + "11" * 32
JOB = "sha256:" + "22" * 32
MINER = "5MinerHotkeyExample111111111111111111"
NONCE = "33" * 32


def request(**changes: object) -> AttestationRequest:
    values: dict[str, object] = {
        "nonce": NONCE,
        "miner_hotkey": MINER,
        "job_digest": JOB,
        "expected_gpu_class": "h100",
        "issued_at": NOW - 10,
        "expires_at": NOW + 30,
    }
    values.update(changes)
    return AttestationRequest(**values)  # type: ignore[arg-type]


def bundle(**changes: object) -> EvidenceBundle:
    values: dict[str, object] = {
        "evidence_format": FORMAT,
        "payload": b"sanitized-test-evidence",
        "collected_at": NOW - 5,
    }
    values.update(changes)
    return EvidenceBundle(**values)  # type: ignore[arg-type]


def policy(**changes: object) -> AttestationPolicy:
    values: dict[str, object] = {
        "accepted_formats": (FORMAT,),
        "accepted_gpu_classes": ("h100",),
        "reference_measurements": (ReferenceMeasurement(MEASUREMENT, NOW + 300),),
        "minimum_tier": EvidenceTier.B,
        "max_age_seconds": 60,
    }
    values.update(changes)
    return AttestationPolicy(**values)  # type: ignore[arg-type]


def claims(attestation_request: AttestationRequest, **changes: object) -> VerifiedAttestationClaims:
    values: dict[str, object] = {
        "evidence_format": FORMAT,
        "nonce": attestation_request.bound_nonce(),
        "gpu_class": "h100",
        "gpu_count": 1,
        "measurement_digests": (MEASUREMENT,),
        "device_chain_valid": True,
        "host_chain_valid": False,
        "revocation": RevocationResult(RevocationStatus.GOOD, NOW - 5, NOW + 120),
        "issued_at": NOW - 5,
        "collateral_expires_at": NOW + 300,
    }
    values.update(changes)
    return VerifiedAttestationClaims(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FakeBackend:
    """Return facts as if a cryptographic verifier had authenticated them."""

    result: VerifiedAttestationClaims

    def verify(self, evidence: EvidenceBundle, *, expected_nonce: str) -> VerifiedAttestationClaims:
        del evidence, expected_nonce
        return self.result


def verify(
    attestation_request: AttestationRequest,
    verified_claims: VerifiedAttestationClaims,
    *,
    trust_policy: AttestationPolicy | None = None,
) -> VerifierResult:
    return PolicyAttestationVerifier(FakeBackend(verified_claims)).verify(
        attestation_request,
        bundle(),
        trust_policy or policy(),
        now=NOW,
    )


def test_tier_is_derived_only_from_verified_chain_facts() -> None:
    """A miner cannot include a declared tier in the evidence bundle."""
    challenge = request()
    tier_b = verify(challenge, claims(challenge))
    tier_a = verify(challenge, claims(challenge, host_chain_valid=True))

    assert tier_b.accepted and tier_b.tier is EvidenceTier.B
    assert tier_a.accepted and tier_a.tier is EvidenceTier.A
    assert "tier" not in EvidenceBundle.__dataclass_fields__


def test_nonce_binds_miner_identity_and_job_digest() -> None:
    """Reusing evidence for another miner or job changes the expected hardware nonce."""
    original = request()
    original_claims = claims(original)

    for changed in (
        request(miner_hotkey="5OtherMinerHotkey222222222222222222"),
        request(job_digest="sha256:" + "44" * 32),
        request(nonce="55" * 32),
    ):
        result = verify(changed, original_claims)
        assert result.reason is VerificationReason.NONCE_MISMATCH
        assert result.tier is None


def test_stale_request_and_evidence_are_rejected_before_backend_use() -> None:
    """Expired requests and evidence outside the challenge window fail closed."""
    challenge = request(expires_at=NOW - 1)
    result = PolicyAttestationVerifier(FakeBackend(claims(request()))).verify(
        challenge,
        bundle(),
        policy(),
        now=NOW,
    )
    assert result.reason is VerificationReason.REQUEST_STALE


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ({"gpu_class": "a100"}, VerificationReason.GPU_CLASS_MISMATCH),
        ({"device_chain_valid": False}, VerificationReason.CERTIFICATE_CHAIN_INVALID),
        (
            {"revocation": RevocationResult(RevocationStatus.REVOKED, NOW - 5, NOW + 120)},
            VerificationReason.COLLATERAL_REVOKED,
        ),
        (
            {"revocation": RevocationResult(RevocationStatus.UNKNOWN, NOW - 5, NOW + 120)},
            VerificationReason.CERTIFICATE_CHAIN_INVALID,
        ),
        ({"collateral_expires_at": NOW - 1}, VerificationReason.COLLATERAL_STALE),
        (
            {"measurement_digests": ("sha256:" + "66" * 32,)},
            VerificationReason.MEASUREMENT_UNTRUSTED,
        ),
    ),
)
def test_invalid_verified_facts_fail_closed(
    change: dict[str, object], reason: VerificationReason
) -> None:
    """Hardware, chain, revocation, collateral, and RIM failures remain distinguishable."""
    challenge = request()
    result = verify(challenge, claims(challenge, **change))
    assert not result.accepted
    assert result.reason is reason
    assert result.tier is None


def test_stale_reference_measurement_and_downgrade_are_rejected() -> None:
    """Expired policy material and incomplete host chains cannot satisfy stronger policy."""
    challenge = request()
    stale_policy = policy(reference_measurements=(ReferenceMeasurement(MEASUREMENT, NOW - 1),))
    assert (
        verify(challenge, claims(challenge), trust_policy=stale_policy).reason
        is VerificationReason.MEASUREMENT_UNTRUSTED
    )
    assert (
        verify(
            challenge,
            claims(challenge, host_chain_valid=False),
            trust_policy=policy(minimum_tier=EvidenceTier.A),
        ).reason
        is VerificationReason.DOWNGRADE_REJECTED
    )


def test_format_mismatch_and_backend_failures_are_sanitized() -> None:
    """Disabled formats and unavailable dependencies do not become verified evidence."""
    challenge = request()
    unsupported = PolicyAttestationVerifier(FakeBackend(claims(challenge))).verify(
        challenge,
        bundle(evidence_format="unknown_v1"),
        policy(),
        now=NOW,
    )

    class BrokenBackend:
        def verify(
            self, evidence: EvidenceBundle, *, expected_nonce: str
        ) -> VerifiedAttestationClaims:
            del evidence, expected_nonce
            raise RuntimeError("sensitive provider failure")

    unavailable = PolicyAttestationVerifier(BrokenBackend()).verify(
        challenge, bundle(), policy(), now=NOW
    )

    class RejectedBackend:
        def verify(
            self, evidence: EvidenceBundle, *, expected_nonce: str
        ) -> VerifiedAttestationClaims:
            del evidence, expected_nonce
            raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)

    invalid = PolicyAttestationVerifier(RejectedBackend()).verify(
        challenge, bundle(), policy(), now=NOW
    )

    assert unsupported.reason is VerificationReason.FORMAT_UNSUPPORTED
    assert unavailable.reason is VerificationReason.PROVIDER_UNAVAILABLE
    assert invalid.reason is VerificationReason.EVIDENCE_INVALID
    assert "sensitive" not in repr(unavailable)


def test_result_cannot_be_rewritten_as_verified_after_rejection() -> None:
    """Rejected results cannot carry an evidence tier."""
    challenge = request()
    rejected = verify(challenge, claims(challenge, device_chain_valid=False))
    with pytest.raises(ValueError):
        replace(rejected, tier=EvidenceTier.A)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: request(expires_at=NOW - 10),
        lambda: CertificateChain(()),
        lambda: CertificateChain((b"",)),
        lambda: RevocationResult(cast(RevocationStatus, "good"), NOW, NOW + 1),
        lambda: RevocationResult(RevocationStatus.GOOD, NOW, NOW),
        lambda: bundle(payload=b""),
        lambda: claims(request(), gpu_count=True),
        lambda: claims(request(), gpu_count=9),
        lambda: claims(request(), measurement_digests=()),
        lambda: claims(request(), measurement_digests=(MEASUREMENT, MEASUREMENT)),
        lambda: claims(request(), device_chain_valid=cast(bool, 1)),
        lambda: claims(request(), revocation=cast(RevocationResult, None)),
        lambda: policy(accepted_formats=()),
        lambda: policy(accepted_gpu_classes=()),
        lambda: policy(accepted_formats=(FORMAT, FORMAT)),
        lambda: policy(reference_measurements=()),
        lambda: policy(reference_measurements=cast(tuple[ReferenceMeasurement, ...], ("bad",))),
        lambda: policy(
            reference_measurements=(
                ReferenceMeasurement(MEASUREMENT, NOW + 1),
                ReferenceMeasurement(MEASUREMENT, NOW + 2),
            )
        ),
        lambda: policy(minimum_tier=cast(EvidenceTier, "b")),
        lambda: policy(max_age_seconds=0),
        lambda: VerifierResult(
            accepted=True,
            tier=None,
            reason=VerificationReason.VERIFIED,
            evidence_digest="sha256:" + "00" * 32,
            gpu_class="h100",
            gpu_count=1,
        ),
        lambda: VerifierResult(
            accepted=False,
            tier=EvidenceTier.A,
            reason=VerificationReason.EVIDENCE_INVALID,
            evidence_digest="sha256:" + "00" * 32,
        ),
        lambda: AttestationBackendFailure(VerificationReason.VERIFIED),
    ),
)
def test_attestation_schema_rejects_malformed_security_values(
    factory: Callable[[], object],
) -> None:
    """Malformed trust inputs cannot reach backend or scoring decisions."""
    with pytest.raises(AttestationError):
        factory()
