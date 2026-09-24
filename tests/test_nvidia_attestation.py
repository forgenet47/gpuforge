"""NVAT integration tests use sanitized JSON and an injected command runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from gpuforge.attestation import (
    AttestationBackendFailure,
    AttestationError,
    AttestationPolicy,
    AttestationRequest,
    EvidenceBundle,
    PolicyAttestationVerifier,
    ReferenceMeasurement,
    VerificationReason,
    VerifierResult,
)
from gpuforge.config import EvidenceTier
from gpuforge.nvidia_attestation import (
    NVAT_EVIDENCE_FORMAT,
    NVAT_REFERENCE_POLICY_DIGEST,
    NvatCliProvider,
    NvatCliVerifierBackend,
    NvatCommandResult,
    PublicCollateral,
    PublicCollateralCache,
    SubprocessNvatRunner,
)

NOW = 2_000_000_000
MINER = "5MinerHotkeyExample111111111111111111"


def request() -> AttestationRequest:
    return AttestationRequest(
        nonce="11" * 32,
        miner_hotkey=MINER,
        job_digest="sha256:" + "22" * 32,
        expected_gpu_class="h100",
        issued_at=NOW - 10,
        expires_at=NOW + 30,
    )


def iso_time(value: int) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def certificate(expiry: int = NOW + 3_600) -> dict[str, object]:
    return {
        "x-nvidia-cert-expiration-date": iso_time(expiry),
        "x-nvidia-cert-ocsp-status": "good",
        "x-nvidia-cert-revocation-reason": None,
        "x-nvidia-cert-status": "valid",
    }


def gpu_claim(nonce: str, **changes: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "dbgstat": "disabled",
        "eat_nonce": nonce,
        "hwmodel": "GH100 A01 GSP BROM",
        "measres": "success",
        "secboot": True,
        "x-nvidia-device-type": "gpu",
        "x-nvidia-gpu-arch-check": True,
        "x-nvidia-gpu-attestation-report-cert-chain": certificate(),
        "x-nvidia-gpu-attestation-report-cert-chain-fwid-match": True,
        "x-nvidia-gpu-attestation-report-nonce-match": True,
        "x-nvidia-gpu-attestation-report-parsed": True,
        "x-nvidia-gpu-attestation-report-signature-verified": True,
        "x-nvidia-gpu-claims-version": "3.0",
        "x-nvidia-gpu-driver-rim-cert-chain": certificate(),
        "x-nvidia-gpu-driver-rim-fetched": True,
        "x-nvidia-gpu-driver-rim-measurements-available": True,
        "x-nvidia-gpu-driver-rim-schema-validated": True,
        "x-nvidia-gpu-driver-rim-signature-verified": True,
        "x-nvidia-gpu-driver-rim-version-match": True,
        "x-nvidia-gpu-driver-version": "590.12",
        "x-nvidia-gpu-vbios-index-no-conflict": True,
        "x-nvidia-gpu-vbios-rim-cert-chain": certificate(),
        "x-nvidia-gpu-vbios-rim-fetched": True,
        "x-nvidia-gpu-vbios-rim-measurements-available": True,
        "x-nvidia-gpu-vbios-rim-schema-validated": True,
        "x-nvidia-gpu-vbios-rim-signature-verified": True,
        "x-nvidia-gpu-vbios-rim-version-match": True,
        "x-nvidia-gpu-vbios-version": "96.00.A5.00.01",
        "x-nvidia-mismatch-measurement-records": None,
    }
    claim.update(changes)
    return claim


def output(nonce: str, **changes: object) -> bytes:
    value: dict[str, object] = {
        "claims": [gpu_claim(nonce)],
        "detached_eat": [["JWT", "sanitized-token"], {"GPU-0": "sanitized-token"}],
        "result_code": 0,
        "result_message": "Ok",
    }
    value.update(changes)
    return json.dumps(value).encode("utf-8")


@dataclass
class FakeRunner:
    """Return fixed CLI output and record argv without executing software."""

    result: NvatCommandResult | None = None
    failure: Exception | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, argv: Sequence[str], *, timeout_seconds: int) -> NvatCommandResult:
        assert 1 <= timeout_seconds <= 300
        self.calls.append(tuple(argv))
        if self.failure is not None:
            raise self.failure
        assert self.result is not None
        return self.result


def evidence(attestation_request: AttestationRequest) -> EvidenceBundle:
    return EvidenceBundle(
        evidence_format=NVAT_EVIDENCE_FORMAT,
        payload=b'{"evidences":[{"sanitized":true}]}',
        collected_at=NOW - 5,
    )


def policy() -> AttestationPolicy:
    return AttestationPolicy(
        accepted_formats=(NVAT_EVIDENCE_FORMAT,),
        accepted_gpu_classes=("h100",),
        reference_measurements=(ReferenceMeasurement(NVAT_REFERENCE_POLICY_DIGEST, NOW + 7_200),),
        minimum_tier=EvidenceTier.B,
        max_age_seconds=60,
    )


def verify_with(stdout: bytes) -> tuple[VerifierResult, FakeRunner]:
    challenge = request()
    runner = FakeRunner(NvatCommandResult(0, stdout))
    verifier = PolicyAttestationVerifier(NvatCliVerifierBackend(runner=runner, enabled=True))
    result = verifier.verify(challenge, evidence(challenge), policy(), now=NOW)
    return result, runner


def test_valid_sanitized_nvat_claims_produce_tier_b() -> None:
    """GPU-only NVAT evidence cannot claim the complete host chain required for Tier A."""
    challenge = request()
    result, runner = verify_with(output(challenge.bound_nonce()))

    assert result.accepted
    assert result.tier is EvidenceTier.B
    assert result.gpu_class == "h100"
    assert result.gpu_count == 1
    assert runner.calls[0][0] == "nvattest"
    assert "--gpu-evidence-source" in runner.calls[0]
    assert "--nonce" in runner.calls[0]
    assert "--format" in runner.calls[0]


def test_provider_binds_nonce_and_accepts_only_hopper_collection() -> None:
    """Collection uses the bound nonce and returns canonical transported evidence."""
    challenge = request()
    collected = {
        "evidences": [
            {
                "arch": "HOPPER",
                "nonce": challenge.bound_nonce(),
                "evidence": "sanitized-evidence",
                "certificate": "sanitized-certificate",
            }
        ],
        "result_code": 0,
        "result_message": "Ok",
    }
    runner = FakeRunner(NvatCommandResult(0, json.dumps(collected).encode()))
    provider = NvatCliProvider(runner=runner, enabled=True, clock=lambda: NOW - 5)

    result = provider.collect(challenge)

    assert result.evidence_format == NVAT_EVIDENCE_FORMAT
    assert result.collected_at == NOW - 5
    nonce_index = runner.calls[0].index("--nonce")
    assert runner.calls[0][nonce_index + 1] == challenge.bound_nonce()
    assert b"sanitized" in result.payload


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ({"eat_nonce": "ff" * 32}, VerificationReason.NONCE_MISMATCH),
        ({"hwmodel": "GA100"}, VerificationReason.GPU_CLASS_MISMATCH),
        (
            {"x-nvidia-gpu-attestation-report-signature-verified": False},
            VerificationReason.EVIDENCE_INVALID,
        ),
        ({"measres": "failure"}, VerificationReason.MEASUREMENT_UNTRUSTED),
    ),
)
def test_tampered_or_incomplete_gpu_claims_fail_closed(
    change: dict[str, object], reason: VerificationReason
) -> None:
    """Nonce, hardware, signature, and measurement failures retain stable reasons."""
    challenge = request()
    result, _ = verify_with(
        output(challenge.bound_nonce(), claims=[gpu_claim(challenge.bound_nonce(), **change)])
    )
    assert not result.accepted
    assert result.reason is reason


def test_revoked_and_stale_collateral_are_rejected() -> None:
    """OCSP revocation and expired certificate material never silently fall back."""
    challenge = request()
    revoked = certificate()
    revoked["x-nvidia-cert-ocsp-status"] = "revoked"
    revoked_result, _ = verify_with(
        output(
            challenge.bound_nonce(),
            claims=[
                gpu_claim(
                    challenge.bound_nonce(),
                    **{"x-nvidia-gpu-driver-rim-cert-chain": revoked},
                )
            ],
        )
    )
    stale_result, _ = verify_with(
        output(
            challenge.bound_nonce(),
            claims=[
                gpu_claim(
                    challenge.bound_nonce(),
                    **{
                        "x-nvidia-gpu-driver-rim-cert-chain": certificate(NOW - 1),
                    },
                )
            ],
        )
    )

    assert revoked_result.reason is VerificationReason.COLLATERAL_REVOKED
    assert stale_result.reason is VerificationReason.COLLATERAL_STALE


@pytest.mark.parametrize(
    "stdout",
    (
        b'{"claims":[],"detached_eat":[],"result_code":0}',
        b'{"claims":[],"detached_eat":["value"],"result_code":0}',
        b"not-json",
    ),
)
def test_missing_claims_tampered_token_and_invalid_json_are_rejected(stdout: bytes) -> None:
    result, _ = verify_with(stdout)
    assert result.reason is VerificationReason.EVIDENCE_INVALID


def test_feature_gate_and_verifier_outage_fail_closed_without_raw_error() -> None:
    """Missing integration never downgrades to software-only verification."""
    challenge = request()
    with pytest.raises(AttestationBackendFailure) as disabled:
        NvatCliVerifierBackend(runner=FakeRunner(), enabled=False).verify(
            evidence(challenge), expected_nonce=challenge.bound_nonce()
        )
    assert disabled.value.reason is VerificationReason.PROVIDER_UNAVAILABLE

    runner = FakeRunner(failure=OSError("private runtime detail"))
    verifier = PolicyAttestationVerifier(NvatCliVerifierBackend(runner=runner, enabled=True))
    result = verifier.verify(challenge, evidence(challenge), policy(), now=NOW)
    assert result.reason is VerificationReason.PROVIDER_UNAVAILABLE
    assert "private runtime detail" not in repr(result)


def test_public_collateral_cache_enforces_digest_expiry_and_capacity() -> None:
    """Only digest-verified, unexpired public material is returned."""
    payload = b"sanitized-public-collateral"
    item = PublicCollateral(
        digest=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        payload=payload,
        expires_at=NOW + 10,
    )
    cache = PublicCollateralCache(max_entries=1)
    cache.put(item, now=NOW)

    assert cache.get(item.digest, now=NOW) == item
    assert cache.get(item.digest, now=NOW + 10) is None
    with pytest.raises(ValueError, match="Expired"):
        cache.put(item, now=NOW + 10)


def test_subprocess_runner_bounds_output_and_sanitizes_os_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real adapter wrapper uses argv without a shell and hides launch details."""

    def completed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(["nvattest"], 0, b'{"result_code":0}', b"")

    monkeypatch.setattr(subprocess, "run", completed)
    result = SubprocessNvatRunner().run(("nvattest", "version"), timeout_seconds=1)
    assert result.returncode == 0

    def unavailable(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        raise OSError("private executable path")

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(AttestationBackendFailure) as error:
        SubprocessNvatRunner().run(("nvattest", "version"), timeout_seconds=1)
    assert error.value.reason is VerificationReason.PROVIDER_UNAVAILABLE


@pytest.mark.parametrize(
    ("collected", "reason"),
    (
        ({"evidences": [], "result_code": 0}, VerificationReason.EVIDENCE_INVALID),
        (
            {
                "evidences": [
                    {"arch": "HOPPER", "nonce": "bad", "evidence": "x", "certificate": "y"}
                ],
                "result_code": 0,
            },
            VerificationReason.NONCE_MISMATCH,
        ),
        (
            {
                "evidences": [
                    {
                        "arch": "BLACKWELL",
                        "nonce": request().bound_nonce(),
                        "evidence": "x",
                        "certificate": "y",
                    }
                ],
                "result_code": 0,
            },
            VerificationReason.GPU_CLASS_MISMATCH,
        ),
    ),
)
def test_provider_rejects_invalid_collection_shapes(
    collected: dict[str, object], reason: VerificationReason
) -> None:
    runner = FakeRunner(NvatCommandResult(0, json.dumps(collected).encode()))
    with pytest.raises(AttestationBackendFailure) as error:
        NvatCliProvider(runner=runner, enabled=True).collect(request())
    assert error.value.reason is reason


def test_verifier_rejects_wrong_transport_format_and_claims_version() -> None:
    challenge = request()
    backend = NvatCliVerifierBackend(runner=FakeRunner(), enabled=True)
    with pytest.raises(AttestationBackendFailure) as wrong_format:
        backend.verify(
            EvidenceBundle("other_v1", b"value", NOW - 5),
            expected_nonce=challenge.bound_nonce(),
        )
    assert wrong_format.value.reason is VerificationReason.FORMAT_UNSUPPORTED

    result, _ = verify_with(
        output(
            challenge.bound_nonce(),
            claims=[
                gpu_claim(
                    challenge.bound_nonce(),
                    **{"x-nvidia-gpu-claims-version": "2.0"},
                )
            ],
        )
    )
    assert result.reason is VerificationReason.FORMAT_UNSUPPORTED


def test_collateral_values_and_cache_capacity_are_fail_closed() -> None:
    payload = b"public-collateral"
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    with pytest.raises(AttestationError, match="digest"):
        PublicCollateral("invalid", payload, NOW + 10)
    with pytest.raises(AttestationError, match="payload"):
        PublicCollateral(digest, b"different", NOW + 10)
    with pytest.raises(AttestationError, match="bound"):
        PublicCollateralCache(max_entries=0)

    first = PublicCollateral(digest, payload, NOW + 10)
    other_payload = b"other-public-collateral"
    second = PublicCollateral(
        f"sha256:{hashlib.sha256(other_payload).hexdigest()}",
        other_payload,
        NOW + 10,
    )
    cache = PublicCollateralCache(max_entries=1)
    cache.put(first, now=NOW)
    with pytest.raises(AttestationError, match="full"):
        cache.put(second, now=NOW)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: NvatCliProvider(FakeRunner(), enabled=True, executable="shell"),
        lambda: NvatCliProvider(FakeRunner(), enabled=True, timeout_seconds=0),
        lambda: NvatCliVerifierBackend(FakeRunner(), enabled=True, collateral_max_age_seconds=1),
    ),
)
def test_nvat_adapter_configuration_is_bounded(factory: Callable[[], object]) -> None:
    with pytest.raises(AttestationError):
        factory()
