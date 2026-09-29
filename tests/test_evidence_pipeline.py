"""Evidence linkage, baseline policy, and validator gate regressions."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from gpuforge.baseline import BaselineDecision, BaselineProfile, assess_baseline
from gpuforge.evidence import BlobReference, EvidenceBundle, EvidenceFailure, result_binding_digest
from gpuforge.performance import PerformanceResult, Precision, WorkUnit
from gpuforge.protocol import UNSIGNED_SIGNATURE, CheckpointCommitment, ExecutionEvidence, WorkLease


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def ref(value: bytes) -> BlobReference:
    return BlobReference(digest(value), len(value))


def bundle() -> EvidenceBundle:
    return EvidenceBundle(
        manifest_digest=digest(b"manifest"),
        lease_digest=digest(b"lease"),
        challenge_commitment=digest(b"commitment"),
        challenge_response=ref(b"challenge"),
        attestation=ref(b"attestation"),
        runtime_image_digest=digest(b"image"),
        checkpoints=(CheckpointCommitment(1, digest(b"checkpoint")),),
        result=ref(b"result"),
        measurement=ref(b"measurement"),
        failure_events=ref(b"[]"),
    )


def test_external_blob_has_exact_content_and_length() -> None:
    reference = ref(b"known")
    reference.verify(b"known")
    for payload in (b"other", b"knownextra", b"know"):
        with pytest.raises(EvidenceFailure, match="blob_mismatch"):
            reference.verify(payload)


@pytest.mark.parametrize(
    "value,size",
    [("bad", 1), ("sha256:" + "a" * 64, 0), ("sha256:" + "a" * 64, True)],
)
def test_invalid_blob_references_fail(value: str, size: int) -> None:
    with pytest.raises(EvidenceFailure, match="invalid_blob_reference"):
        BlobReference(value, size)


def test_bundle_root_binds_every_component() -> None:
    original = bundle()
    assert original.root() == replace(original, checkpoints=original.checkpoints).root()
    for changed in (
        replace(original, measurement=ref(b"changed")),
        replace(original, failure_events=ref(b"[1]")),
        replace(original, challenge_response=ref(b"other")),
        replace(original, checkpoints=(CheckpointCommitment(1, digest(b"other")),)),
    ):
        assert changed.root() != original.root()
        if (
            changed.measurement != original.measurement
            or changed.failure_events != original.failure_events
        ):
            assert result_binding_digest(
                changed.result, changed.measurement, changed.failure_events
            ) != result_binding_digest(
                original.result, original.measurement, original.failure_events
            )


def test_invalid_bundle_shape_fails_closed() -> None:
    original = bundle()
    for changes in (
        {"manifest_digest": "bad"},
        {"checkpoints": ()},
        {"checkpoints": (original.checkpoints[0], original.checkpoints[0])},
    ):
        with pytest.raises(EvidenceFailure):
            replace(original, **changes)


def test_bundle_binding_rejects_swapped_components() -> None:
    original = bundle()
    lease = WorkLease(
        manifest_digest=original.manifest_digest,
        miner_hotkey="MinerExample",
        validator_hotkey="ValidatorExample",
        shard_id="shard-1",
        challenge_commitment=original.challenge_commitment,
        start_block=1,
        deadline_block=10,
        nonce="ab" * 32,
        signature=UNSIGNED_SIGNATURE,
    )
    original = replace(original, lease_digest=lease.digest())
    evidence = ExecutionEvidence(
        lease_digest=lease.digest(),
        miner_hotkey=lease.miner_hotkey,
        attestation_digest=original.attestation.digest if original.attestation else None,
        image_digest=original.runtime_image_digest,
        challenge_response_digest=original.challenge_response.digest,
        checkpoints=original.checkpoints,
        result_digest=result_binding_digest(
            original.result, original.measurement, original.failure_events
        ),
        work_units=1,
        active_seconds_ms=1,
        submitted_at_block=2,
        sequence=0,
        signature=UNSIGNED_SIGNATURE,
        bundle_root=original.root(),
    )
    original.verify_binding(lease, evidence)
    with pytest.raises(EvidenceFailure, match="evidence_binding_mismatch"):
        replace(original, failure_events=ref(b"failure")).verify_binding(lease, evidence)
    with pytest.raises(EvidenceFailure, match="evidence_binding_mismatch"):
        replace(original, checkpoints=(CheckpointCommitment(1, digest(b"fake")),)).verify_binding(
            lease, evidence
        )


def performance(rate: int) -> PerformanceResult:
    return PerformanceResult(
        unit=WorkUnit.SAMPLES,
        accepted_units=100,
        active_ns=1_000_000_000,
        excluded_ns=0,
        rate_milli_units_per_second=rate,
        active_steps=10,
        global_batch_size=10,
        precision=Precision.BF16,
        model_shape_digest=digest(b"model"),
        gpu_count=1,
        validator_observed_ns=1_000_000_000,
    )


def profile() -> BaselineProfile:
    return BaselineProfile(
        version=1,
        workload_family="training",
        model_shape_digest=digest(b"model"),
        gpu_class="h100_sxm",
        gpu_count=1,
        precision=Precision.BF16,
        unit=WorkUnit.SAMPLES,
        lower_milli_units_per_second=50_000,
        upper_milli_units_per_second=200_000,
        reviewer_signature="ab" * 64,
    )


def test_baseline_flags_impossible_but_does_not_reject_slow() -> None:
    def assess(rate: int, **changes: object) -> BaselineDecision:
        arguments: dict[str, object] = {
            "workload_family": "training",
            "gpu_class": "h100_sxm",
            "required_version": 1,
            "review_verified": True,
        }
        arguments.update(changes)
        return assess_baseline(profile(), performance(rate), **arguments)  # type: ignore[arg-type]

    assert assess(100_000) is BaselineDecision.PLAUSIBLE
    assert assess(10_000) is BaselineDecision.SLOW_REVIEW
    assert assess(300_000) is BaselineDecision.IMPOSSIBLE
    assert assess(100_000, mig_enabled=True) is BaselineDecision.UNPROFILED
    assert assess(100_000, gpu_class="a100") is BaselineDecision.UNPROFILED
    assert assess(100_000, review_verified=False) is BaselineDecision.UNPROFILED
    assert assess(100_000, required_version=2) is BaselineDecision.UNPROFILED
    assert assess(100_000, workload_family="other") is BaselineDecision.UNPROFILED


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 0},
        {"workload_family": "BAD"},
        {"model_shape_digest": "bad"},
        {"gpu_class": "a100"},
        {"gpu_count": 0},
        {"precision": "bf16"},
        {"lower_milli_units_per_second": 0},
        {"lower_milli_units_per_second": 300_000},
        {"reviewer_signature": "bad"},
    ],
)
def test_invalid_baseline_profiles_fail(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid_baseline_profile"):
        replace(profile(), **changes)  # type: ignore[arg-type]


def test_baseline_identity_is_deterministic() -> None:
    item = profile()
    assert item.signing_bytes() == profile().signing_bytes()
    assert item.digest() == profile().digest()
