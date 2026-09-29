"""Validator gate order and receipt behavior for independently checked work."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, replace
from typing import cast

from gpuforge.attestation import VerificationReason, VerifierResult
from gpuforge.baseline import BaselineDecision
from gpuforge.config import EvidenceTier, NetworkPolicy
from gpuforge.evidence import BlobReference, EvidenceBundle, result_binding_digest
from gpuforge.identity import sign_message
from gpuforge.performance import PerformanceResult, Precision, WorkUnit
from gpuforge.protocol import (
    UNSIGNED_SIGNATURE,
    CheckpointCommitment,
    ExecutionEvidence,
    JobManifest,
    ResourcePolicy,
    VerificationPolicy,
    WorkLease,
    decode_message,
)
from gpuforge.validation import ValidationInputs, ValidationReason, validate_execution


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def ref(data: bytes) -> BlobReference:
    return BlobReference(digest(data), len(data))


@dataclass(frozen=True)
class Signer:
    hotkey: str
    key: bytes

    def sign(self, payload: bytes) -> bytes:
        return hmac.digest(self.key, payload, "sha512")


@dataclass(frozen=True)
class Verifier:
    keys: dict[str, bytes]

    def verify(self, hotkey: str, payload: bytes, signature: bytes) -> bool:
        key = self.keys.get(hotkey)
        return key is not None and hmac.compare_digest(
            hmac.digest(key, payload, "sha512"), signature
        )


def setup() -> tuple[ValidationInputs, Verifier]:
    publisher = Signer("PublisherTest", b"publisher")
    validator = Signer("ValidatorTest", b"validator")
    miner = Signer("MinerTest", b"miner")
    verifier = Verifier({item.hotkey: item.key for item in (publisher, validator, miner)})
    manifest = JobManifest(
        job_id="job-1",
        container_digest=digest(b"image"),
        entrypoint_digest=digest(b"script"),
        input_root=digest(b"input"),
        framework="pytorch",
        resource_policy=ResourcePolicy(1, 81920, 16, 131072, 3600, NetworkPolicy.DENY),
        verification_policy=VerificationPolicy(EvidenceTier.B, "canary", 10),
        lease_seconds=3600,
        publisher_hotkey=publisher.hotkey,
        expires_at_block=1100,
        signature=UNSIGNED_SIGNATURE,
    )
    manifest = cast(JobManifest, sign_message(manifest, publisher))
    lease = WorkLease(
        manifest.digest(),
        miner.hotkey,
        validator.hotkey,
        "shard-1",
        digest(b"challenge"),
        1000,
        1050,
        "ab" * 32,
        UNSIGNED_SIGNATURE,
    )
    lease = cast(WorkLease, sign_message(lease, validator))
    bundle = EvidenceBundle(
        manifest.digest(),
        lease.digest(),
        lease.challenge_commitment,
        ref(b"response"),
        ref(b"attestation"),
        manifest.container_digest,
        (CheckpointCommitment(10, digest(b"checkpoint")),),
        ref(b"result"),
        ref(b"measurement"),
        ref(b"[]"),
    )
    evidence = ExecutionEvidence(
        lease.digest(),
        miner.hotkey,
        bundle.attestation.digest if bundle.attestation else None,
        manifest.container_digest,
        bundle.challenge_response.digest,
        bundle.checkpoints,
        result_binding_digest(bundle.result, bundle.measurement, bundle.failure_events),
        100,
        1000,
        1010,
        1,
        UNSIGNED_SIGNATURE,
        bundle_root=bundle.root(),
    )
    evidence = cast(ExecutionEvidence, sign_message(evidence, miner))
    performance = PerformanceResult(
        WorkUnit.SAMPLES,
        100,
        1_000_000_000,
        0,
        100_000,
        10,
        10,
        Precision.BF16,
        digest(b"model"),
        1,
        1_200_000_000,
    )
    attestation = VerifierResult(
        True,
        EvidenceTier.B,
        VerificationReason.VERIFIED,
        ref(b"attestation").digest,
        "h100_sxm",
        1,
    )
    inputs = ValidationInputs(
        manifest,
        lease,
        evidence,
        bundle,
        attestation,
        performance,
        1020,
        EvidenceTier.B,
        lambda: True,
        lambda: True,
        BaselineDecision.PLAUSIBLE,
    )
    return inputs, verifier


def test_valid_work_gets_unsigned_receipt_for_validator_signing() -> None:
    inputs, verifier = setup()
    assert decode_message(inputs.evidence.canonical_bytes()) == inputs.evidence
    assert inputs.evidence.bundle_root == inputs.bundle.root()
    result = validate_execution(
        inputs, validator_hotkey="ValidatorTest", signature_verifier=verifier
    )
    assert result.accepted
    assert result.verified_work_units == 100
    assert result.confidence_tier is EvidenceTier.B
    assert result.signature == UNSIGNED_SIGNATURE


def test_signature_and_failure_precedence() -> None:
    inputs, verifier = setup()
    bad = replace(inputs.evidence, work_units=200)
    changed = replace(inputs, evidence=bad, challenge_check=lambda: False)
    result = validate_execution(
        changed, validator_hotkey="ValidatorTest", signature_verifier=verifier
    )
    assert result.reason_codes == (ValidationReason.SIGNATURE.value,)


def test_failed_challenge_never_reaches_scoring() -> None:
    inputs, verifier = setup()
    changed = replace(inputs, challenge_check=lambda: False)
    result = validate_execution(
        changed, validator_hotkey="ValidatorTest", signature_verifier=verifier
    )
    assert not result.accepted
    assert result.verified_work_units == 0
    assert result.confidence_tier is None
    assert result.reason_codes == (ValidationReason.CHALLENGE.value,)


def test_tampered_measurement_binding_and_outlier_fail() -> None:
    inputs, verifier = setup()
    changed = replace(inputs, bundle=replace(inputs.bundle, measurement=ref(b"forged")))
    result = validate_execution(
        changed, validator_hotkey="ValidatorTest", signature_verifier=verifier
    )
    assert result.reason_codes == (ValidationReason.LEASE.value,)
    changed = replace(inputs, plausibility=BaselineDecision.IMPOSSIBLE)
    result = validate_execution(
        changed, validator_hotkey="ValidatorTest", signature_verifier=verifier
    )
    assert result.reason_codes == (ValidationReason.PLAUSIBILITY.value,)


def test_slow_correct_work_is_not_rejected() -> None:
    inputs, verifier = setup()
    result = validate_execution(
        replace(inputs, plausibility=BaselineDecision.SLOW_REVIEW),
        validator_hotkey="ValidatorTest",
        signature_verifier=verifier,
    )
    assert result.accepted


def test_freshness_attestation_and_measurement_gates() -> None:
    inputs, verifier = setup()
    cases = (
        (replace(inputs, current_block=1101), ValidationReason.FRESHNESS),
        (
            replace(
                inputs,
                attestation=replace(
                    inputs.attestation,
                    accepted=False,
                    tier=None,
                    reason=VerificationReason.EVIDENCE_INVALID,
                ),
            ),
            ValidationReason.ATTESTATION,
        ),
        (
            replace(inputs, performance=replace(inputs.performance, accepted_units=101)),
            ValidationReason.MEASUREMENT,
        ),
        (replace(inputs, plausibility=BaselineDecision.UNPROFILED), ValidationReason.DEPENDENCY),
        (replace(inputs, correctness_check=lambda: False), ValidationReason.CORRECTNESS),
    )
    for changed, expected in cases:
        result = validate_execution(
            changed, validator_hotkey="ValidatorTest", signature_verifier=verifier
        )
        assert result.reason_codes == (expected.value,)


def test_verifier_dependency_error_is_sanitized() -> None:
    inputs, verifier = setup()

    def unavailable() -> bool:
        raise RuntimeError("private backend details")

    result = validate_execution(
        replace(inputs, challenge_check=unavailable),
        validator_hotkey="ValidatorTest",
        signature_verifier=verifier,
    )
    assert result.reason_codes == (ValidationReason.DEPENDENCY.value,)
    assert "private backend details" not in repr(result)
