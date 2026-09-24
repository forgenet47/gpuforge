"""Feature-gated NVIDIA NVAT CLI attestation adapter.

The adapter targets the NVAT 1.x command-line JSON contract and GPU claims
schema 3.0. It never treats decoded claims as verified unless ``nvattest``
returns a successful appraisal for the exact validator-bound nonce.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

from gpuforge.attestation import (
    AttestationBackendFailure,
    AttestationError,
    AttestationRequest,
    EvidenceBundle,
    RevocationResult,
    RevocationStatus,
    VerificationReason,
    VerifiedAttestationClaims,
)

NVAT_EVIDENCE_FORMAT = "nvidia_nvat_evidence_v1"
NVAT_CLAIMS_VERSION = "3.0"
NVAT_REFERENCE_POLICY_DIGEST = (
    "sha256:" + hashlib.sha256(b"nvidia-nvat-v1-claims-v3-rim-and-certificate-policy").hexdigest()
)
_MAX_CLI_OUTPUT_BYTES = 4 * 1024 * 1024
_MAX_COLLATERAL_BYTES = 2 * 1024 * 1024
_REQUIRED_TRUE_CLAIMS = (
    "x-nvidia-gpu-arch-check",
    "x-nvidia-gpu-attestation-report-cert-chain-fwid-match",
    "x-nvidia-gpu-attestation-report-parsed",
    "x-nvidia-gpu-attestation-report-nonce-match",
    "x-nvidia-gpu-attestation-report-signature-verified",
    "x-nvidia-gpu-driver-rim-fetched",
    "x-nvidia-gpu-driver-rim-schema-validated",
    "x-nvidia-gpu-driver-rim-signature-verified",
    "x-nvidia-gpu-driver-rim-version-match",
    "x-nvidia-gpu-driver-rim-measurements-available",
    "x-nvidia-gpu-vbios-rim-fetched",
    "x-nvidia-gpu-vbios-rim-schema-validated",
    "x-nvidia-gpu-vbios-rim-signature-verified",
    "x-nvidia-gpu-vbios-rim-version-match",
    "x-nvidia-gpu-vbios-rim-measurements-available",
    "x-nvidia-gpu-vbios-index-no-conflict",
)
_CERTIFICATE_CLAIMS = (
    "x-nvidia-gpu-attestation-report-cert-chain",
    "x-nvidia-gpu-driver-rim-cert-chain",
    "x-nvidia-gpu-vbios-rim-cert-chain",
)


@dataclass(frozen=True, slots=True)
class NvatCommandResult:
    """Bounded output from one NVAT invocation."""

    returncode: int
    stdout: bytes
    stderr: bytes = b""


class NvatCommandRunner(Protocol):
    """Run an argv-only command without a shell."""

    def run(self, argv: Sequence[str], *, timeout_seconds: int) -> NvatCommandResult: ...


@dataclass(frozen=True, slots=True)
class SubprocessNvatRunner:
    """Invoke a pre-installed NVAT binary with a minimal environment."""

    def run(self, argv: Sequence[str], *, timeout_seconds: int) -> NvatCommandResult:
        try:
            result = subprocess.run(  # noqa: S603 - argv is constructed and validated here
                tuple(argv),
                check=False,
                capture_output=True,
                env={"PATH": os.defpath},
                shell=False,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            raise AttestationBackendFailure(VerificationReason.PROVIDER_UNAVAILABLE) from None
        if len(result.stdout) > _MAX_CLI_OUTPUT_BYTES or len(result.stderr) > _MAX_CLI_OUTPUT_BYTES:
            raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
        return NvatCommandResult(result.returncode, result.stdout, result.stderr)


@dataclass(frozen=True, slots=True)
class NvatCliProvider:
    """Collect H100 evidence with a pre-installed NVAT 1.x CLI."""

    runner: NvatCommandRunner
    enabled: bool = False
    executable: str = "nvattest"
    timeout_seconds: int = 60
    clock: Callable[[], int] = lambda: int(time.time())

    def __post_init__(self) -> None:
        _validate_cli_settings(self.enabled, self.executable, self.timeout_seconds)

    def collect(self, request: AttestationRequest) -> EvidenceBundle:
        """Collect evidence for the nonce bound to the miner and job."""
        if not self.enabled:
            raise AttestationBackendFailure(VerificationReason.PROVIDER_UNAVAILABLE)
        nonce = request.bound_nonce()
        outcome = self.runner.run(
            (
                self.executable,
                "collect-evidence",
                "--device",
                "gpu",
                "--nonce",
                nonce,
                "--format",
                "json",
                "--log-level",
                "off",
            ),
            timeout_seconds=self.timeout_seconds,
        )
        data = _successful_json(outcome)
        evidences = _list(data.get("evidences"), "evidences")
        if not 1 <= len(evidences) <= 8:
            raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
        for item in evidences:
            evidence = _mapping(item, "evidence")
            if evidence.get("nonce") != nonce or not isinstance(evidence.get("evidence"), str):
                raise AttestationBackendFailure(VerificationReason.NONCE_MISMATCH)
            if not isinstance(evidence.get("certificate"), str):
                raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
            if str(evidence.get("arch", "")).casefold() != "hopper":
                raise AttestationBackendFailure(VerificationReason.GPU_CLASS_MISMATCH)
        payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "ascii"
        )
        return EvidenceBundle(
            evidence_format=NVAT_EVIDENCE_FORMAT,
            payload=payload,
            collected_at=self.clock(),
        )


@dataclass(frozen=True, slots=True)
class NvatCliVerifierBackend:
    """Verify transported GPU evidence with NVAT local appraisal."""

    runner: NvatCommandRunner
    enabled: bool = False
    executable: str = "nvattest"
    timeout_seconds: int = 60
    collateral_max_age_seconds: int = 3_600

    def __post_init__(self) -> None:
        _validate_cli_settings(self.enabled, self.executable, self.timeout_seconds)
        if (
            isinstance(self.collateral_max_age_seconds, bool)
            or not isinstance(self.collateral_max_age_seconds, int)
            or not 60 <= self.collateral_max_age_seconds <= 86_400
        ):
            raise AttestationError("NVAT collateral maximum age is invalid")

    def verify(
        self,
        bundle: EvidenceBundle,
        *,
        expected_nonce: str,
    ) -> VerifiedAttestationClaims:
        """Run NVAT and convert its authenticated claims into neutral facts."""
        if not self.enabled:
            raise AttestationBackendFailure(VerificationReason.PROVIDER_UNAVAILABLE)
        if bundle.evidence_format != NVAT_EVIDENCE_FORMAT:
            raise AttestationBackendFailure(VerificationReason.FORMAT_UNSUPPORTED)
        with tempfile.TemporaryDirectory(prefix="gpuforge-nvat-") as temporary:
            evidence_path = Path(temporary) / "evidence.json"
            evidence_path.write_bytes(bundle.payload)
            try:
                evidence_path.chmod(0o600)
            except OSError:
                raise AttestationBackendFailure(VerificationReason.PROVIDER_UNAVAILABLE) from None
            outcome = self.runner.run(
                (
                    self.executable,
                    "attest",
                    "--device",
                    "gpu",
                    "--verifier",
                    "local",
                    "--gpu-evidence-source",
                    "file",
                    "--gpu-evidence-file",
                    str(evidence_path),
                    "--nonce",
                    expected_nonce,
                    "--format",
                    "json",
                    "--log-level",
                    "off",
                ),
                timeout_seconds=self.timeout_seconds,
            )
        data = _successful_json(outcome)
        return _verified_claims(
            data,
            expected_nonce=expected_nonce,
            collected_at=bundle.collected_at,
            collateral_max_age_seconds=self.collateral_max_age_seconds,
        )


@dataclass(frozen=True, slots=True)
class PublicCollateral:
    """Bounded public verifier collateral with an explicit expiry."""

    digest: str
    payload: bytes
    expires_at: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.digest, str)
            or not self.digest.startswith("sha256:")
            or len(self.digest) != 71
        ):
            raise AttestationError("Collateral digest is invalid")
        if (
            not isinstance(self.payload, bytes)
            or not 1 <= len(self.payload) <= _MAX_COLLATERAL_BYTES
        ):
            raise AttestationError("Collateral payload size is invalid")
        if (
            isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, int)
            or self.expires_at < 0
        ):
            raise AttestationError("Collateral expiry is invalid")
        if f"sha256:{hashlib.sha256(self.payload).hexdigest()}" != self.digest:
            raise AttestationError("Collateral digest does not match its payload")


class PublicCollateralCache:
    """Small fail-closed in-memory cache for non-secret verifier collateral."""

    def __init__(self, max_entries: int = 32) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or not 1 <= max_entries <= 256
        ):
            raise AttestationError("Collateral cache bound is invalid")
        self._max_entries = max_entries
        self._values: dict[str, PublicCollateral] = {}

    def put(self, collateral: PublicCollateral, *, now: int) -> None:
        """Cache only unexpired digest-verified public material."""
        if collateral.expires_at <= now:
            raise AttestationError("Expired collateral cannot be cached")
        self._purge(now)
        if collateral.digest not in self._values and len(self._values) >= self._max_entries:
            raise AttestationError("Collateral cache is full")
        self._values[collateral.digest] = collateral

    def get(self, digest: str, *, now: int) -> PublicCollateral | None:
        """Return fresh collateral or fail closed with a cache miss."""
        self._purge(now)
        return self._values.get(digest)

    def _purge(self, now: int) -> None:
        expired = [key for key, value in self._values.items() if value.expires_at <= now]
        for key in expired:
            del self._values[key]


def _verified_claims(
    data: Mapping[str, object],
    *,
    expected_nonce: str,
    collected_at: int,
    collateral_max_age_seconds: int,
) -> VerifiedAttestationClaims:
    raw_claims = _list(data.get("claims"), "claims")
    if not 1 <= len(raw_claims) <= 8:
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
    detached = data.get("detached_eat")
    if not isinstance(detached, list) or not detached:
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)

    expiries: list[int] = []
    for raw_claim in raw_claims:
        claim = _mapping(raw_claim, "claim")
        if claim.get("x-nvidia-device-type") != "gpu":
            raise AttestationBackendFailure(VerificationReason.GPU_CLASS_MISMATCH)
        if claim.get("x-nvidia-gpu-claims-version") != NVAT_CLAIMS_VERSION:
            raise AttestationBackendFailure(VerificationReason.FORMAT_UNSUPPORTED)
        if claim.get("eat_nonce") != expected_nonce:
            raise AttestationBackendFailure(VerificationReason.NONCE_MISMATCH)
        if claim.get("measres") not in {"success", "Success"}:
            raise AttestationBackendFailure(VerificationReason.MEASUREMENT_UNTRUSTED)
        mismatches = claim.get("x-nvidia-mismatch-measurement-records")
        if mismatches is not None and mismatches != []:
            raise AttestationBackendFailure(VerificationReason.MEASUREMENT_UNTRUSTED)
        if any(claim.get(name) is not True for name in _REQUIRED_TRUE_CLAIMS):
            raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
        if claim.get("secboot") is not True or claim.get("dbgstat") != "disabled":
            raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
        model = claim.get("hwmodel")
        if not isinstance(model, str) or not any(
            marker in model.casefold() for marker in ("gh100", "h100")
        ):
            raise AttestationBackendFailure(VerificationReason.GPU_CLASS_MISMATCH)
        for name in _CERTIFICATE_CLAIMS:
            certificate = _mapping(claim.get(name), name)
            status = certificate.get("x-nvidia-cert-status")
            ocsp = certificate.get("x-nvidia-cert-ocsp-status")
            if status == "revoked" or ocsp == "revoked":
                raise AttestationBackendFailure(VerificationReason.COLLATERAL_REVOKED)
            if status != "valid" or ocsp != "good":
                raise AttestationBackendFailure(VerificationReason.CERTIFICATE_CHAIN_INVALID)
            expiry = certificate.get("x-nvidia-cert-expiration-date")
            if not isinstance(expiry, str):
                raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
            expiries.append(_iso8601_epoch(expiry))

    collateral_expiry = min(expiries)
    return VerifiedAttestationClaims(
        evidence_format=NVAT_EVIDENCE_FORMAT,
        nonce=expected_nonce,
        gpu_class="h100",
        gpu_count=len(raw_claims),
        measurement_digests=(NVAT_REFERENCE_POLICY_DIGEST,),
        device_chain_valid=True,
        host_chain_valid=False,
        revocation=RevocationResult(
            status=RevocationStatus.GOOD,
            checked_at=collected_at,
            next_update=min(collected_at + collateral_max_age_seconds, collateral_expiry),
        ),
        issued_at=collected_at,
        collateral_expires_at=collateral_expiry,
    )


def _successful_json(outcome: NvatCommandResult) -> Mapping[str, object]:
    if outcome.returncode != 0 or not outcome.stdout:
        raise AttestationBackendFailure(VerificationReason.PROVIDER_UNAVAILABLE)
    if len(outcome.stdout) > _MAX_CLI_OUTPUT_BYTES:
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
    try:
        decoded = json.loads(outcome.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID) from None
    data = _mapping(decoded, "NVAT output")
    result_code = data.get("result_code")
    if isinstance(result_code, bool) or result_code != 0:
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
    return data


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
    return cast(Mapping[str, object], value)


def _list(value: object, label: str) -> list[object]:
    del label
    if not isinstance(value, list):
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
    return cast(list[object], value)


def _iso8601_epoch(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID) from None
    if parsed.tzinfo is None:
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = int((parsed.astimezone(timezone.utc) - epoch).total_seconds())
    if seconds < 0:
        raise AttestationBackendFailure(VerificationReason.EVIDENCE_INVALID)
    return seconds


def _validate_cli_settings(enabled: object, executable: object, timeout_seconds: object) -> None:
    if not isinstance(enabled, bool):
        raise AttestationError("NVAT feature gate is invalid")
    if not isinstance(executable, str) or Path(executable).name not in {"nvattest", "nvattest.exe"}:
        raise AttestationError("NVAT executable is invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= 300
    ):
        raise AttestationError("NVAT timeout is invalid")


__all__ = [
    "NVAT_CLAIMS_VERSION",
    "NVAT_EVIDENCE_FORMAT",
    "NVAT_REFERENCE_POLICY_DIGEST",
    "NvatCliProvider",
    "NvatCliVerifierBackend",
    "NvatCommandResult",
    "NvatCommandRunner",
    "PublicCollateral",
    "PublicCollateralCache",
    "SubprocessNvatRunner",
]
