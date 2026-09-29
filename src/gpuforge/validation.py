"""Ordered fail-closed checks before issuing a signed validation receipt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from gpuforge.attestation import VerifierResult
from gpuforge.baseline import BaselineDecision
from gpuforge.config import EvidenceTier
from gpuforge.evidence import EvidenceBundle, EvidenceFailure
from gpuforge.identity import AuthenticationError, SignatureVerifier, verify_message_signature
from gpuforge.performance import PerformanceResult
from gpuforge.protocol import (
    UNSIGNED_SIGNATURE,
    ExecutionEvidence,
    JobManifest,
    ValidationReceipt,
    WorkLease,
)


class ValidationReason(str, Enum):
    """Stable public reasons in precedence order."""

    SCHEMA = "schema_invalid"
    SIGNATURE = "signature_invalid"
    FRESHNESS = "evidence_stale"
    LEASE = "lease_mismatch"
    ATTESTATION = "attestation_rejected"
    IMAGE = "image_mismatch"
    CHALLENGE = "challenge_rejected"
    CORRECTNESS = "training_incorrect"
    MEASUREMENT = "measurement_invalid"
    PLAUSIBILITY = "throughput_implausible"
    DEPENDENCY = "verifier_unavailable"


@dataclass(frozen=True, slots=True)
class ValidationInputs:
    """Validator-owned references and independently checked outcomes."""

    manifest: JobManifest
    lease: WorkLease
    evidence: ExecutionEvidence
    bundle: EvidenceBundle
    attestation: VerifierResult
    performance: PerformanceResult
    current_block: int
    minimum_tier: EvidenceTier
    challenge_check: Callable[[], bool]
    correctness_check: Callable[[], bool]
    plausibility: BaselineDecision


def validate_execution(
    inputs: ValidationInputs,
    *,
    validator_hotkey: str,
    signature_verifier: SignatureVerifier,
    maximum_age_blocks: int = 64,
) -> ValidationReceipt:
    """Apply gates in order and issue an unsigned receipt for validator signing."""
    evidence = inputs.evidence

    def receipt(reason: ValidationReason | None) -> ValidationReceipt:
        accepted = reason is None
        return ValidationReceipt(
            evidence_digest=evidence.digest(),
            validator_hotkey=validator_hotkey,
            accepted=accepted,
            reason_codes=() if reason is None else (reason.value,),
            verified_work_units=inputs.performance.accepted_units if accepted else 0,
            confidence_tier=inputs.attestation.tier if accepted else None,
            validated_at_block=inputs.current_block,
            signature=UNSIGNED_SIGNATURE,
        )

    try:
        evidence.canonical_bytes()
        inputs.bundle.canonical_bytes()
    except (ValueError, TypeError):
        return receipt(ValidationReason.SCHEMA)
    try:
        verify_message_signature(inputs.manifest, signature_verifier)
        verify_message_signature(inputs.lease, signature_verifier)
        verify_message_signature(evidence, signature_verifier)
    except AuthenticationError:
        return receipt(ValidationReason.SIGNATURE)
    if (
        isinstance(maximum_age_blocks, bool)
        or not isinstance(maximum_age_blocks, int)
        or maximum_age_blocks < 1
        or inputs.current_block < evidence.submitted_at_block
        or inputs.current_block - evidence.submitted_at_block > maximum_age_blocks
        or inputs.current_block > inputs.manifest.expires_at_block
        or not inputs.lease.start_block <= inputs.current_block <= inputs.lease.deadline_block
    ):
        return receipt(ValidationReason.FRESHNESS)
    if (
        inputs.lease.manifest_digest != inputs.manifest.digest()
        or inputs.lease.miner_hotkey != evidence.miner_hotkey
    ):
        return receipt(ValidationReason.LEASE)
    try:
        inputs.bundle.verify_binding(inputs.lease, evidence)
    except EvidenceFailure:
        return receipt(ValidationReason.LEASE)
    tier_rank = {EvidenceTier.A: 3, EvidenceTier.B: 2, EvidenceTier.C: 1}
    if (
        not inputs.attestation.accepted
        or inputs.attestation.tier is None
        or tier_rank[inputs.attestation.tier] < tier_rank[inputs.minimum_tier]
        or inputs.attestation.evidence_digest != evidence.attestation_digest
        or inputs.attestation.gpu_count != inputs.performance.gpu_count
    ):
        return receipt(ValidationReason.ATTESTATION)
    if evidence.image_digest != inputs.manifest.container_digest:
        return receipt(ValidationReason.IMAGE)
    for check, reason in (
        (inputs.challenge_check, ValidationReason.CHALLENGE),
        (inputs.correctness_check, ValidationReason.CORRECTNESS),
    ):
        try:
            if not check():
                return receipt(reason)
        except Exception:
            return receipt(ValidationReason.DEPENDENCY)
    if (
        inputs.performance.accepted_units != evidence.work_units
        or (inputs.performance.active_ns + 500_000) // 1_000_000 != evidence.active_seconds_ms
        or inputs.performance.accepted_units <= 0
    ):
        return receipt(ValidationReason.MEASUREMENT)
    if inputs.plausibility is BaselineDecision.IMPOSSIBLE:
        return receipt(ValidationReason.PLAUSIBILITY)
    if inputs.plausibility is BaselineDecision.UNPROFILED:
        return receipt(ValidationReason.DEPENDENCY)
    return receipt(None)
