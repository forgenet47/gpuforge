"""Fail-closed attestation contracts and policy evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from gpuforge.config import EvidenceTier

MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
MAX_CERTIFICATE_BYTES = 64 * 1024
MAX_CERTIFICATES = 16
_NONCE_PATTERN = re.compile(r"[0-9a-f]{64}")
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HOTKEY_PATTERN = re.compile(r"[A-Za-z0-9]{3,128}")
_CODE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_BINDING_DOMAIN = b"gpuforge-attestation-binding-v1\x00"


class AttestationError(ValueError):
    """Raised when an attestation value or operation is unsafe."""


class VerificationReason(str, Enum):
    """Stable, non-sensitive attestation decision reasons."""

    VERIFIED = "verified"
    FORMAT_UNSUPPORTED = "format_unsupported"
    REQUEST_STALE = "request_stale"
    EVIDENCE_INVALID = "evidence_invalid"
    NONCE_MISMATCH = "nonce_mismatch"
    GPU_CLASS_MISMATCH = "gpu_class_mismatch"
    CERTIFICATE_CHAIN_INVALID = "certificate_chain_invalid"
    COLLATERAL_REVOKED = "collateral_revoked"
    COLLATERAL_STALE = "collateral_stale"
    MEASUREMENT_UNTRUSTED = "measurement_untrusted"
    DOWNGRADE_REJECTED = "downgrade_rejected"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


class RevocationStatus(str, Enum):
    """Normalized revocation result from a trusted verification backend."""

    GOOD = "good"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AttestationRequest:
    """Validator challenge bound to a miner and immutable job digest."""

    nonce: str
    miner_hotkey: str
    job_digest: str
    expected_gpu_class: str
    issued_at: int
    expires_at: int

    def __post_init__(self) -> None:
        _match("nonce", self.nonce, _NONCE_PATTERN)
        _match("miner hotkey", self.miner_hotkey, _HOTKEY_PATTERN)
        _match("job digest", self.job_digest, _DIGEST_PATTERN)
        _match("GPU class", self.expected_gpu_class, _CODE_PATTERN)
        _timestamp("issued_at", self.issued_at)
        _timestamp("expires_at", self.expires_at)
        if self.expires_at <= self.issued_at:
            raise AttestationError("Attestation request expiry must follow issuance")

    def bound_nonce(self) -> str:
        """Derive the 32-byte nonce passed to the hardware verifier."""
        value = json.dumps(
            {
                "job_digest": self.job_digest,
                "miner_hotkey": self.miner_hotkey,
                "nonce": self.nonce,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_BINDING_DOMAIN + value).hexdigest()


@dataclass(frozen=True, slots=True)
class CertificateChain:
    """Bounded opaque certificate chain for a trusted verifier adapter."""

    certificates: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.certificates, tuple)
            or not 1 <= len(self.certificates) <= MAX_CERTIFICATES
        ):
            raise AttestationError("Certificate chain length is invalid")
        if any(
            not isinstance(item, bytes) or not 1 <= len(item) <= MAX_CERTIFICATE_BYTES
            for item in self.certificates
        ):
            raise AttestationError("Certificate chain contains an invalid certificate")


@dataclass(frozen=True, slots=True)
class ReferenceMeasurement:
    """Operator-approved measurement identity with an explicit expiry."""

    digest: str
    expires_at: int

    def __post_init__(self) -> None:
        _match("reference measurement", self.digest, _DIGEST_PATTERN)
        _timestamp("reference measurement expiry", self.expires_at)


@dataclass(frozen=True, slots=True)
class RevocationResult:
    """Revocation status and freshness window returned by a checker."""

    status: RevocationStatus
    checked_at: int
    next_update: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, RevocationStatus):
            raise AttestationError("Revocation status is invalid")
        _timestamp("revocation check time", self.checked_at)
        _timestamp("revocation next update", self.next_update)
        if self.next_update <= self.checked_at:
            raise AttestationError("Revocation freshness window is invalid")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Bounded opaque evidence passed to an enabled verifier backend."""

    evidence_format: str
    payload: bytes
    collected_at: int

    def __post_init__(self) -> None:
        _match("evidence format", self.evidence_format, _CODE_PATTERN)
        if not isinstance(self.payload, bytes) or not 1 <= len(self.payload) <= MAX_EVIDENCE_BYTES:
            raise AttestationError("Attestation evidence size is invalid")
        _timestamp("evidence collection time", self.collected_at)

    @property
    def digest(self) -> str:
        """Return a safe identity without exposing evidence bytes."""
        return f"sha256:{hashlib.sha256(self.payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class VerifiedAttestationClaims:
    """Facts authenticated by a verifier backend, before local policy."""

    evidence_format: str
    nonce: str
    gpu_class: str
    gpu_count: int
    measurement_digests: tuple[str, ...]
    device_chain_valid: bool
    host_chain_valid: bool
    revocation: RevocationResult
    issued_at: int
    collateral_expires_at: int

    def __post_init__(self) -> None:
        _match("evidence format", self.evidence_format, _CODE_PATTERN)
        _match("verified nonce", self.nonce, _NONCE_PATTERN)
        _match("verified GPU class", self.gpu_class, _CODE_PATTERN)
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int):
            raise AttestationError("Verified GPU count is invalid")
        if not 1 <= self.gpu_count <= 8:
            raise AttestationError("Verified GPU count is outside protocol bounds")
        if not isinstance(self.measurement_digests, tuple) or not self.measurement_digests:
            raise AttestationError("Verified measurements are required")
        for digest in self.measurement_digests:
            _match("verified measurement", digest, _DIGEST_PATTERN)
        if len(self.measurement_digests) != len(set(self.measurement_digests)):
            raise AttestationError("Verified measurements must be unique")
        if not isinstance(self.device_chain_valid, bool) or not isinstance(
            self.host_chain_valid, bool
        ):
            raise AttestationError("Verified chain state is invalid")
        if not isinstance(self.revocation, RevocationResult):
            raise AttestationError("Verified revocation state is invalid")
        _timestamp("verified evidence issuance", self.issued_at)
        _timestamp("collateral expiry", self.collateral_expires_at)


@dataclass(frozen=True, slots=True)
class AttestationPolicy:
    """Local trust roots, reference measurements, and minimum assurance."""

    accepted_formats: tuple[str, ...]
    accepted_gpu_classes: tuple[str, ...]
    reference_measurements: tuple[ReferenceMeasurement, ...]
    minimum_tier: EvidenceTier
    max_age_seconds: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.accepted_formats, tuple) or not self.accepted_formats:
            raise AttestationError("At least one attestation format is required")
        if not isinstance(self.accepted_gpu_classes, tuple) or not self.accepted_gpu_classes:
            raise AttestationError("At least one GPU class is required")
        for value in self.accepted_formats:
            _match("accepted evidence format", value, _CODE_PATTERN)
        for value in self.accepted_gpu_classes:
            _match("accepted GPU class", value, _CODE_PATTERN)
        if len(self.accepted_formats) != len(set(self.accepted_formats)) or len(
            self.accepted_gpu_classes
        ) != len(set(self.accepted_gpu_classes)):
            raise AttestationError("Attestation policy entries must be unique")
        if not isinstance(self.reference_measurements, tuple) or not self.reference_measurements:
            raise AttestationError("Reference measurements are required")
        if not all(isinstance(item, ReferenceMeasurement) for item in self.reference_measurements):
            raise AttestationError("Reference measurement policy is invalid")
        digests = [item.digest for item in self.reference_measurements]
        if len(digests) != len(set(digests)):
            raise AttestationError("Reference measurements must be unique")
        if not isinstance(self.minimum_tier, EvidenceTier):
            raise AttestationError("Minimum evidence tier is invalid")
        if (
            isinstance(self.max_age_seconds, bool)
            or not isinstance(self.max_age_seconds, int)
            or not 1 <= self.max_age_seconds <= 3_600
        ):
            raise AttestationError("Attestation maximum age is invalid")


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """Only attestation value permitted to enter later scoring gates."""

    accepted: bool
    tier: EvidenceTier | None
    reason: VerificationReason
    evidence_digest: str
    gpu_class: str | None = None
    gpu_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise AttestationError("Attestation decision is invalid")
        _match("evidence digest", self.evidence_digest, _DIGEST_PATTERN)
        if self.accepted:
            if self.reason is not VerificationReason.VERIFIED or not isinstance(
                self.tier, EvidenceTier
            ):
                raise AttestationError("Accepted attestation result is incomplete")
            if self.gpu_class is None:
                raise AttestationError("Accepted attestation has no GPU class")
            _match("result GPU class", self.gpu_class, _CODE_PATTERN)
            if not 1 <= self.gpu_count <= 8:
                raise AttestationError("Accepted attestation has an invalid GPU count")
        elif self.tier is not None or self.reason is VerificationReason.VERIFIED:
            raise AttestationError("Rejected attestation must not carry an evidence tier")


class AttestationProvider(Protocol):
    """Collect evidence for a validator-created request."""

    def collect(self, request: AttestationRequest) -> EvidenceBundle: ...


class AttestationBackend(Protocol):
    """Authenticate evidence and return facts without selecting a tier."""

    def verify(
        self,
        bundle: EvidenceBundle,
        *,
        expected_nonce: str,
    ) -> VerifiedAttestationClaims: ...


class EvidenceVerifier(Protocol):
    """Evaluate evidence against local trust policy."""

    def verify(
        self,
        request: AttestationRequest,
        bundle: EvidenceBundle,
        policy: AttestationPolicy,
        *,
        now: int,
    ) -> VerifierResult: ...


class CertificateChainVerifier(Protocol):
    """Validate a chain against configured trust roots."""

    def verify(self, chain: CertificateChain, *, now: int) -> bool: ...


class RevocationChecker(Protocol):
    """Check certificate collateral without accepting unknown state."""

    def check(self, chain: CertificateChain, *, now: int) -> RevocationResult: ...


class ReferenceMeasurementStore(Protocol):
    """Resolve an approved reference measurement by digest."""

    def resolve(self, digest: str, *, now: int) -> ReferenceMeasurement | None: ...


class AttestationBackendFailure(RuntimeError):
    """Sanitized backend failure mapped to a stable decision reason."""

    def __init__(self, reason: VerificationReason = VerificationReason.EVIDENCE_INVALID) -> None:
        if reason is VerificationReason.VERIFIED:
            raise AttestationError("Backend failure cannot be verified")
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class PolicyAttestationVerifier:
    """Derive assurance from authenticated claims and fail closed."""

    backend: AttestationBackend

    def verify(
        self,
        request: AttestationRequest,
        bundle: EvidenceBundle,
        policy: AttestationPolicy,
        *,
        now: int,
    ) -> VerifierResult:
        _timestamp("verification time", now)
        if bundle.evidence_format not in policy.accepted_formats:
            return _rejected(bundle, VerificationReason.FORMAT_UNSUPPORTED)
        if (
            now < request.issued_at
            or now > request.expires_at
            or now - request.issued_at > policy.max_age_seconds
            or bundle.collected_at < request.issued_at
            or bundle.collected_at > now
        ):
            return _rejected(bundle, VerificationReason.REQUEST_STALE)
        try:
            claims = self.backend.verify(bundle, expected_nonce=request.bound_nonce())
        except AttestationBackendFailure as error:
            return _rejected(bundle, error.reason)
        except Exception:
            return _rejected(bundle, VerificationReason.PROVIDER_UNAVAILABLE)
        if claims.evidence_format != bundle.evidence_format:
            return _rejected(bundle, VerificationReason.FORMAT_UNSUPPORTED)
        if claims.nonce != request.bound_nonce():
            return _rejected(bundle, VerificationReason.NONCE_MISMATCH)
        if (
            claims.gpu_class != request.expected_gpu_class
            or claims.gpu_class not in policy.accepted_gpu_classes
        ):
            return _rejected(bundle, VerificationReason.GPU_CLASS_MISMATCH)
        if (
            claims.issued_at < request.issued_at
            or claims.issued_at > now
            or now - claims.issued_at > policy.max_age_seconds
        ):
            return _rejected(bundle, VerificationReason.REQUEST_STALE)
        if not claims.device_chain_valid:
            return _rejected(bundle, VerificationReason.CERTIFICATE_CHAIN_INVALID)
        if claims.revocation.status is RevocationStatus.REVOKED:
            return _rejected(bundle, VerificationReason.COLLATERAL_REVOKED)
        if claims.revocation.status is not RevocationStatus.GOOD:
            return _rejected(bundle, VerificationReason.CERTIFICATE_CHAIN_INVALID)
        if (
            claims.revocation.checked_at > now
            or claims.revocation.next_update < now
            or claims.collateral_expires_at < now
        ):
            return _rejected(bundle, VerificationReason.COLLATERAL_STALE)

        trusted = {item.digest: item for item in policy.reference_measurements}
        for digest in claims.measurement_digests:
            reference = trusted.get(digest)
            if reference is None or reference.expires_at < now:
                return _rejected(bundle, VerificationReason.MEASUREMENT_UNTRUSTED)

        derived_tier = EvidenceTier.A if claims.host_chain_valid else EvidenceTier.B
        if _tier_rank(derived_tier) < _tier_rank(policy.minimum_tier):
            return _rejected(bundle, VerificationReason.DOWNGRADE_REJECTED)
        return VerifierResult(
            accepted=True,
            tier=derived_tier,
            reason=VerificationReason.VERIFIED,
            evidence_digest=bundle.digest,
            gpu_class=claims.gpu_class,
            gpu_count=claims.gpu_count,
        )


def _rejected(bundle: EvidenceBundle, reason: VerificationReason) -> VerifierResult:
    return VerifierResult(
        accepted=False,
        tier=None,
        reason=reason,
        evidence_digest=bundle.digest,
    )


def _tier_rank(tier: EvidenceTier) -> int:
    return {EvidenceTier.C: 1, EvidenceTier.B: 2, EvidenceTier.A: 3}[tier]


def _match(label: str, value: object, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise AttestationError(f"{label} is invalid")


def _timestamp(label: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise AttestationError(f"{label} is invalid")


__all__ = [
    "AttestationBackend",
    "AttestationBackendFailure",
    "AttestationError",
    "AttestationPolicy",
    "AttestationProvider",
    "AttestationRequest",
    "CertificateChain",
    "CertificateChainVerifier",
    "EvidenceBundle",
    "EvidenceVerifier",
    "PolicyAttestationVerifier",
    "ReferenceMeasurement",
    "ReferenceMeasurementStore",
    "RevocationChecker",
    "RevocationResult",
    "RevocationStatus",
    "VerificationReason",
    "VerifiedAttestationClaims",
    "VerifierResult",
]
