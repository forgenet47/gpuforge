"""Versioned protocol messages with deterministic canonical encoding."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeVar, cast

from gpuforge.config import EvidenceTier, NetworkPolicy

PROTOCOL_VERSION = 1
MAX_WIRE_BYTES = 64 * 1024
MAX_BLOCK_NUMBER = 2**63 - 1
MAX_WORK_UNITS = 2**63 - 1
_DIGEST_DOMAIN = b"gpuforge-protocol-digest-v1\x00"
_CONTENT_DOMAIN = b"gpuforge-protocol-content-v1\x00"
_SIGNATURE_DOMAIN = b"gpuforge-protocol-signature-v1\x00"
UNSIGNED_SIGNATURE = "00" * 64

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HEX_PATTERN = re.compile(r"[0-9a-f]+")
_HOTKEY_PATTERN = re.compile(r"[A-Za-z0-9]{3,128}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_CODE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]{0,63}")
_SOFTWARE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9._-]{0,31}")


class ProtocolError(ValueError):
    """Base error for protocol encoding, decoding, and validation."""


class ProtocolValidationError(ProtocolError):
    """Raised when a constructed protocol value violates its schema."""


class ProtocolDecodeError(ProtocolError):
    """Raised when wire bytes are malformed, ambiguous, or non-canonical."""


class CapabilityTrust(str, Enum):
    """Trust attached to software-discovered hardware capability fields."""

    SELF_REPORTED = "self_reported"


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Resource limits declared by a training job."""

    gpu_count: int
    gpu_memory_mb: int
    cpu_cores: int
    memory_mb: int
    max_runtime_seconds: int
    network_policy: NetworkPolicy

    def __post_init__(self) -> None:
        _bounded_int("resource_policy.gpu_count", self.gpu_count, 1, 8)
        _bounded_int("resource_policy.gpu_memory_mb", self.gpu_memory_mb, 1_024, 1_048_576)
        _bounded_int("resource_policy.cpu_cores", self.cpu_cores, 1, 256)
        _bounded_int("resource_policy.memory_mb", self.memory_mb, 1_024, 2_097_152)
        _bounded_int("resource_policy.max_runtime_seconds", self.max_runtime_seconds, 60, 86_400)
        if not isinstance(self.network_policy, NetworkPolicy):
            raise ProtocolValidationError("Invalid resource network policy")

    def to_primitive(self) -> dict[str, object]:
        """Return the canonical primitive representation."""
        return {
            "cpu_cores": self.cpu_cores,
            "gpu_count": self.gpu_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "max_runtime_seconds": self.max_runtime_seconds,
            "memory_mb": self.memory_mb,
            "network_policy": self.network_policy.value,
        }

    @classmethod
    def from_primitive(cls, value: object) -> ResourcePolicy:
        """Parse a strict primitive resource policy."""
        data = _mapping(value, "resource_policy")
        _exact_fields(
            data,
            {
                "cpu_cores",
                "gpu_count",
                "gpu_memory_mb",
                "max_runtime_seconds",
                "memory_mb",
                "network_policy",
            },
            "resource_policy",
        )
        return cls(
            gpu_count=_integer(data["gpu_count"], "resource_policy.gpu_count"),
            gpu_memory_mb=_integer(data["gpu_memory_mb"], "resource_policy.gpu_memory_mb"),
            cpu_cores=_integer(data["cpu_cores"], "resource_policy.cpu_cores"),
            memory_mb=_integer(data["memory_mb"], "resource_policy.memory_mb"),
            max_runtime_seconds=_integer(
                data["max_runtime_seconds"], "resource_policy.max_runtime_seconds"
            ),
            network_policy=_enum_value(
                NetworkPolicy, data["network_policy"], "resource_policy.network_policy"
            ),
        )


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Evidence and challenge requirements declared by a training job."""

    minimum_evidence_tier: EvidenceTier
    challenge_kind: str
    checkpoint_interval_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_evidence_tier, EvidenceTier):
            raise ProtocolValidationError("Invalid minimum evidence tier")
        _identifier("verification_policy.challenge_kind", self.challenge_kind)
        _bounded_int(
            "verification_policy.checkpoint_interval_steps",
            self.checkpoint_interval_steps,
            1,
            1_000_000_000,
        )

    def to_primitive(self) -> dict[str, object]:
        """Return the canonical primitive representation."""
        return {
            "challenge_kind": self.challenge_kind,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "minimum_evidence_tier": self.minimum_evidence_tier.value,
        }

    @classmethod
    def from_primitive(cls, value: object) -> VerificationPolicy:
        """Parse a strict primitive verification policy."""
        data = _mapping(value, "verification_policy")
        _exact_fields(
            data,
            {"challenge_kind", "checkpoint_interval_steps", "minimum_evidence_tier"},
            "verification_policy",
        )
        return cls(
            minimum_evidence_tier=_enum_value(
                EvidenceTier,
                data["minimum_evidence_tier"],
                "verification_policy.minimum_evidence_tier",
            ),
            challenge_kind=_text(data["challenge_kind"], "verification_policy.challenge_kind"),
            checkpoint_interval_steps=_integer(
                data["checkpoint_interval_steps"],
                "verification_policy.checkpoint_interval_steps",
            ),
        )


@dataclass(frozen=True, slots=True)
class SoftwareVersion:
    """A normalized runtime component name and version."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _bounded_text("software.name", self.name, 1, 32, _SOFTWARE_NAME_PATTERN)
        _bounded_text("software.version", self.version, 1, 64, ascii_only=True)

    def to_primitive(self) -> dict[str, object]:
        """Return the canonical primitive representation."""
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_primitive(cls, value: object) -> SoftwareVersion:
        """Parse a strict primitive software version."""
        data = _mapping(value, "runtime_version")
        _exact_fields(data, {"name", "version"}, "runtime_version")
        return cls(
            name=_text(data["name"], "runtime_version.name"),
            version=_text(data["version"], "runtime_version.version"),
        )


@dataclass(frozen=True, slots=True)
class CheckpointCommitment:
    """A content commitment for one completed training step."""

    step: int
    digest: str

    def __post_init__(self) -> None:
        _bounded_int("checkpoint.step", self.step, 1, MAX_WORK_UNITS)
        _digest("checkpoint.digest", self.digest)

    def to_primitive(self) -> dict[str, object]:
        """Return the canonical primitive representation."""
        return {"digest": self.digest, "step": self.step}

    @classmethod
    def from_primitive(cls, value: object) -> CheckpointCommitment:
        """Parse a strict primitive checkpoint commitment."""
        data = _mapping(value, "checkpoint")
        _exact_fields(data, {"digest", "step"}, "checkpoint")
        return cls(
            step=_integer(data["step"], "checkpoint.step"),
            digest=_text(data["digest"], "checkpoint.digest"),
        )


class ProtocolMessage:
    """Shared deterministic encoding and digest behavior."""

    message_type: ClassVar[str]
    protocol_version: int

    def _payload(self) -> dict[str, object]:  # pragma: no cover - abstract contract
        raise NotImplementedError

    def to_primitive(self) -> dict[str, object]:
        """Return the complete versioned wire object."""
        return {
            "message_type": self.message_type,
            "payload": self._payload(),
            "protocol_version": self.protocol_version,
        }

    def canonical_bytes(self) -> bytes:
        """Encode this message as canonical UTF-8 JSON."""
        encoded = _canonical_json(self.to_primitive())
        if len(encoded) > MAX_WIRE_BYTES:
            raise ProtocolValidationError("Encoded message exceeds the wire size limit")
        return encoded

    def _unsigned_primitive(self) -> dict[str, object]:
        value = self.to_primitive()
        payload = cast(dict[str, object], value["payload"])
        payload.pop("signature", None)
        return value

    def signing_bytes(self) -> bytes:
        """Return domain-separated bytes signed by a role hotkey."""
        return _SIGNATURE_DOMAIN + _canonical_json(self._unsigned_primitive())

    def content_digest(self) -> str:
        """Return a signature-independent identity for semantic content."""
        value = hashlib.sha256(
            _CONTENT_DOMAIN + _canonical_json(self._unsigned_primitive())
        ).hexdigest()
        return f"sha256:{value}"

    def digest(self) -> str:
        """Return a domain-separated SHA-256 digest of canonical bytes."""
        value = hashlib.sha256(_DIGEST_DOMAIN + self.canonical_bytes()).hexdigest()
        return f"sha256:{value}"


@dataclass(frozen=True, slots=True)
class JobManifest(ProtocolMessage):
    """Publisher-signed identity and constraints for a training job."""

    message_type: ClassVar[str] = "job_manifest"

    job_id: str
    container_digest: str
    entrypoint_digest: str
    input_root: str
    framework: str
    resource_policy: ResourcePolicy
    verification_policy: VerificationPolicy
    lease_seconds: int
    publisher_hotkey: str
    expires_at_block: int
    signature: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _identifier("job_id", self.job_id)
        _digest("container_digest", self.container_digest)
        _digest("entrypoint_digest", self.entrypoint_digest)
        _digest("input_root", self.input_root)
        _bounded_text("framework", self.framework, 1, 32, _SOFTWARE_NAME_PATTERN)
        if not isinstance(self.resource_policy, ResourcePolicy):
            raise ProtocolValidationError("Invalid resource policy")
        if not isinstance(self.verification_policy, VerificationPolicy):
            raise ProtocolValidationError("Invalid verification policy")
        _bounded_int("lease_seconds", self.lease_seconds, 60, 86_400)
        _hotkey("publisher_hotkey", self.publisher_hotkey)
        _bounded_int("expires_at_block", self.expires_at_block, 1, MAX_BLOCK_NUMBER)
        _signature(self.signature)

    def _payload(self) -> dict[str, object]:
        return {
            "container_digest": self.container_digest,
            "entrypoint_digest": self.entrypoint_digest,
            "expires_at_block": self.expires_at_block,
            "framework": self.framework,
            "input_root": self.input_root,
            "job_id": self.job_id,
            "lease_seconds": self.lease_seconds,
            "publisher_hotkey": self.publisher_hotkey,
            "resource_policy": self.resource_policy.to_primitive(),
            "signature": self.signature,
            "verification_policy": self.verification_policy.to_primitive(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], version: int) -> JobManifest:
        """Construct a manifest from a strict decoded payload."""
        _exact_fields(
            payload,
            {
                "container_digest",
                "entrypoint_digest",
                "expires_at_block",
                "framework",
                "input_root",
                "job_id",
                "lease_seconds",
                "publisher_hotkey",
                "resource_policy",
                "signature",
                "verification_policy",
            },
            "job_manifest",
        )
        return cls(
            job_id=_text(payload["job_id"], "job_id"),
            container_digest=_text(payload["container_digest"], "container_digest"),
            entrypoint_digest=_text(payload["entrypoint_digest"], "entrypoint_digest"),
            input_root=_text(payload["input_root"], "input_root"),
            framework=_text(payload["framework"], "framework"),
            resource_policy=ResourcePolicy.from_primitive(payload["resource_policy"]),
            verification_policy=VerificationPolicy.from_primitive(payload["verification_policy"]),
            lease_seconds=_integer(payload["lease_seconds"], "lease_seconds"),
            publisher_hotkey=_text(payload["publisher_hotkey"], "publisher_hotkey"),
            expires_at_block=_integer(payload["expires_at_block"], "expires_at_block"),
            signature=_text(payload["signature"], "signature"),
            protocol_version=version,
        )


@dataclass(frozen=True, slots=True)
class CapabilityClaim(ProtocolMessage):
    """Miner-signed bounded capability advertisement."""

    message_type: ClassVar[str] = "capability_claim"

    miner_hotkey: str
    gpu_count: int
    gpu_model: str
    gpu_memory_mb: int
    gpu_interconnect: str
    discovery_trust: CapabilityTrust
    runtime_versions: tuple[SoftwareVersion, ...]
    supported_evidence_tiers: tuple[EvidenceTier, ...]
    available_gpu_seconds: int
    nonce: str
    observed_at_block: int
    signature: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _hotkey("miner_hotkey", self.miner_hotkey)
        _bounded_int("gpu_count", self.gpu_count, 1, 8)
        _bounded_text("gpu_model", self.gpu_model, 1, 64, ascii_only=True)
        _bounded_int("gpu_memory_mb", self.gpu_memory_mb, 1_024, 1_048_576)
        _bounded_text("gpu_interconnect", self.gpu_interconnect, 1, 32, _CODE_PATTERN)
        if self.discovery_trust is not CapabilityTrust.SELF_REPORTED:
            raise ProtocolValidationError("Capability discovery trust must be self-reported")
        if (
            not isinstance(self.runtime_versions, tuple)
            or not 1 <= len(self.runtime_versions) <= 32
        ):
            raise ProtocolValidationError("Invalid runtime version list")
        if not all(isinstance(item, SoftwareVersion) for item in self.runtime_versions):
            raise ProtocolValidationError("Invalid runtime version entry")
        names = [item.name for item in self.runtime_versions]
        if len(names) != len(set(names)):
            raise ProtocolValidationError("Runtime component names must be unique")
        object.__setattr__(
            self,
            "runtime_versions",
            tuple(sorted(self.runtime_versions, key=lambda item: item.name)),
        )
        if not isinstance(self.supported_evidence_tiers, tuple) or not 1 <= len(
            self.supported_evidence_tiers
        ) <= len(EvidenceTier):
            raise ProtocolValidationError("Invalid supported evidence tier list")
        if not all(isinstance(item, EvidenceTier) for item in self.supported_evidence_tiers):
            raise ProtocolValidationError("Invalid supported evidence tier entry")
        if len(self.supported_evidence_tiers) != len(set(self.supported_evidence_tiers)):
            raise ProtocolValidationError("Supported evidence tiers must be unique")
        object.__setattr__(
            self,
            "supported_evidence_tiers",
            tuple(sorted(self.supported_evidence_tiers, key=lambda item: item.value)),
        )
        _bounded_int("available_gpu_seconds", self.available_gpu_seconds, 0, MAX_WORK_UNITS)
        _nonce(self.nonce)
        _bounded_int("observed_at_block", self.observed_at_block, 1, MAX_BLOCK_NUMBER)
        _signature(self.signature)

    def _payload(self) -> dict[str, object]:
        return {
            "available_gpu_seconds": self.available_gpu_seconds,
            "gpu_count": self.gpu_count,
            "gpu_interconnect": self.gpu_interconnect,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_model": self.gpu_model,
            "discovery_trust": self.discovery_trust.value,
            "miner_hotkey": self.miner_hotkey,
            "nonce": self.nonce,
            "observed_at_block": self.observed_at_block,
            "runtime_versions": [item.to_primitive() for item in self.runtime_versions],
            "signature": self.signature,
            "supported_evidence_tiers": [item.value for item in self.supported_evidence_tiers],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], version: int) -> CapabilityClaim:
        """Construct a capability claim from a strict decoded payload."""
        _exact_fields(
            payload,
            {
                "available_gpu_seconds",
                "gpu_count",
                "gpu_interconnect",
                "gpu_memory_mb",
                "gpu_model",
                "discovery_trust",
                "miner_hotkey",
                "nonce",
                "observed_at_block",
                "runtime_versions",
                "signature",
                "supported_evidence_tiers",
            },
            "capability_claim",
        )
        runtime_values = _list(payload["runtime_versions"], "runtime_versions", 1, 32)
        tier_values = _list(
            payload["supported_evidence_tiers"],
            "supported_evidence_tiers",
            1,
            len(EvidenceTier),
        )
        return cls(
            miner_hotkey=_text(payload["miner_hotkey"], "miner_hotkey"),
            gpu_count=_integer(payload["gpu_count"], "gpu_count"),
            gpu_model=_text(payload["gpu_model"], "gpu_model"),
            gpu_memory_mb=_integer(payload["gpu_memory_mb"], "gpu_memory_mb"),
            gpu_interconnect=_text(payload["gpu_interconnect"], "gpu_interconnect"),
            discovery_trust=_enum_value(
                CapabilityTrust, payload["discovery_trust"], "discovery_trust"
            ),
            runtime_versions=tuple(SoftwareVersion.from_primitive(item) for item in runtime_values),
            supported_evidence_tiers=tuple(
                _enum_value(EvidenceTier, item, "supported_evidence_tiers") for item in tier_values
            ),
            available_gpu_seconds=_integer(
                payload["available_gpu_seconds"], "available_gpu_seconds"
            ),
            nonce=_text(payload["nonce"], "nonce"),
            observed_at_block=_integer(payload["observed_at_block"], "observed_at_block"),
            signature=_text(payload["signature"], "signature"),
            protocol_version=version,
        )


@dataclass(frozen=True, slots=True)
class WorkLease(ProtocolMessage):
    """Validator-signed assignment of one job shard to one miner."""

    message_type: ClassVar[str] = "work_lease"

    manifest_digest: str
    miner_hotkey: str
    validator_hotkey: str
    shard_id: str
    challenge_commitment: str
    start_block: int
    deadline_block: int
    nonce: str
    signature: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _digest("manifest_digest", self.manifest_digest)
        _hotkey("miner_hotkey", self.miner_hotkey)
        _hotkey("validator_hotkey", self.validator_hotkey)
        _identifier("shard_id", self.shard_id)
        _digest("challenge_commitment", self.challenge_commitment)
        _bounded_int("start_block", self.start_block, 1, MAX_BLOCK_NUMBER)
        _bounded_int("deadline_block", self.deadline_block, 1, MAX_BLOCK_NUMBER)
        if self.deadline_block <= self.start_block:
            raise ProtocolValidationError("Lease deadline must be after its start block")
        _nonce(self.nonce)
        _signature(self.signature)

    def _payload(self) -> dict[str, object]:
        return {
            "challenge_commitment": self.challenge_commitment,
            "deadline_block": self.deadline_block,
            "manifest_digest": self.manifest_digest,
            "miner_hotkey": self.miner_hotkey,
            "nonce": self.nonce,
            "shard_id": self.shard_id,
            "signature": self.signature,
            "start_block": self.start_block,
            "validator_hotkey": self.validator_hotkey,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], version: int) -> WorkLease:
        """Construct a work lease from a strict decoded payload."""
        _exact_fields(
            payload,
            {
                "challenge_commitment",
                "deadline_block",
                "manifest_digest",
                "miner_hotkey",
                "nonce",
                "shard_id",
                "signature",
                "start_block",
                "validator_hotkey",
            },
            "work_lease",
        )
        return cls(
            manifest_digest=_text(payload["manifest_digest"], "manifest_digest"),
            miner_hotkey=_text(payload["miner_hotkey"], "miner_hotkey"),
            validator_hotkey=_text(payload["validator_hotkey"], "validator_hotkey"),
            shard_id=_text(payload["shard_id"], "shard_id"),
            challenge_commitment=_text(payload["challenge_commitment"], "challenge_commitment"),
            start_block=_integer(payload["start_block"], "start_block"),
            deadline_block=_integer(payload["deadline_block"], "deadline_block"),
            nonce=_text(payload["nonce"], "nonce"),
            signature=_text(payload["signature"], "signature"),
            protocol_version=version,
        )


@dataclass(frozen=True, slots=True)
class ExecutionEvidence(ProtocolMessage):
    """Miner-signed commitments and measurements for one completed lease."""

    message_type: ClassVar[str] = "execution_evidence"

    lease_digest: str
    miner_hotkey: str
    attestation_digest: str | None
    image_digest: str
    challenge_response_digest: str
    checkpoints: tuple[CheckpointCommitment, ...]
    result_digest: str
    work_units: int
    active_seconds_ms: int
    submitted_at_block: int
    sequence: int
    signature: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _digest("lease_digest", self.lease_digest)
        _hotkey("miner_hotkey", self.miner_hotkey)
        if self.attestation_digest is not None:
            _digest("attestation_digest", self.attestation_digest)
        _digest("image_digest", self.image_digest)
        _digest("challenge_response_digest", self.challenge_response_digest)
        if not isinstance(self.checkpoints, tuple) or not 1 <= len(self.checkpoints) <= 4_096:
            raise ProtocolValidationError("Invalid checkpoint list")
        if not all(isinstance(item, CheckpointCommitment) for item in self.checkpoints):
            raise ProtocolValidationError("Invalid checkpoint entry")
        steps = [item.step for item in self.checkpoints]
        if len(steps) != len(set(steps)):
            raise ProtocolValidationError("Checkpoint steps must be unique")
        object.__setattr__(
            self, "checkpoints", tuple(sorted(self.checkpoints, key=lambda item: item.step))
        )
        _digest("result_digest", self.result_digest)
        _bounded_int("work_units", self.work_units, 1, MAX_WORK_UNITS)
        _bounded_int("active_seconds_ms", self.active_seconds_ms, 1, 86_400_000)
        _bounded_int("submitted_at_block", self.submitted_at_block, 1, MAX_BLOCK_NUMBER)
        _bounded_int("sequence", self.sequence, 0, MAX_WORK_UNITS)
        _signature(self.signature)

    def _payload(self) -> dict[str, object]:
        return {
            "active_seconds_ms": self.active_seconds_ms,
            "attestation_digest": self.attestation_digest,
            "challenge_response_digest": self.challenge_response_digest,
            "checkpoints": [item.to_primitive() for item in self.checkpoints],
            "image_digest": self.image_digest,
            "lease_digest": self.lease_digest,
            "miner_hotkey": self.miner_hotkey,
            "result_digest": self.result_digest,
            "sequence": self.sequence,
            "signature": self.signature,
            "submitted_at_block": self.submitted_at_block,
            "work_units": self.work_units,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], version: int) -> ExecutionEvidence:
        """Construct execution evidence from a strict decoded payload."""
        _exact_fields(
            payload,
            {
                "active_seconds_ms",
                "attestation_digest",
                "challenge_response_digest",
                "checkpoints",
                "image_digest",
                "lease_digest",
                "miner_hotkey",
                "result_digest",
                "sequence",
                "signature",
                "submitted_at_block",
                "work_units",
            },
            "execution_evidence",
        )
        checkpoint_values = _list(payload["checkpoints"], "checkpoints", 1, 4_096)
        attestation_value = payload["attestation_digest"]
        if attestation_value is not None and not isinstance(attestation_value, str):
            raise ProtocolValidationError("Invalid attestation digest type")
        return cls(
            lease_digest=_text(payload["lease_digest"], "lease_digest"),
            miner_hotkey=_text(payload["miner_hotkey"], "miner_hotkey"),
            attestation_digest=attestation_value,
            image_digest=_text(payload["image_digest"], "image_digest"),
            challenge_response_digest=_text(
                payload["challenge_response_digest"], "challenge_response_digest"
            ),
            checkpoints=tuple(
                CheckpointCommitment.from_primitive(item) for item in checkpoint_values
            ),
            result_digest=_text(payload["result_digest"], "result_digest"),
            work_units=_integer(payload["work_units"], "work_units"),
            active_seconds_ms=_integer(payload["active_seconds_ms"], "active_seconds_ms"),
            submitted_at_block=_integer(payload["submitted_at_block"], "submitted_at_block"),
            sequence=_integer(payload["sequence"], "sequence"),
            signature=_text(payload["signature"], "signature"),
            protocol_version=version,
        )


@dataclass(frozen=True, slots=True)
class ValidationReceipt(ProtocolMessage):
    """Validator-signed decision over one execution-evidence object."""

    message_type: ClassVar[str] = "validation_receipt"

    evidence_digest: str
    validator_hotkey: str
    accepted: bool
    reason_codes: tuple[str, ...]
    verified_work_units: int
    confidence_tier: EvidenceTier | None
    validated_at_block: int
    signature: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _version(self.protocol_version)
        _digest("evidence_digest", self.evidence_digest)
        _hotkey("validator_hotkey", self.validator_hotkey)
        if not isinstance(self.accepted, bool):
            raise ProtocolValidationError("Invalid receipt acceptance flag")
        if not isinstance(self.reason_codes, tuple) or len(self.reason_codes) > 32:
            raise ProtocolValidationError("Invalid receipt reason-code list")
        for code in self.reason_codes:
            _bounded_text("reason_code", code, 1, 64, _CODE_PATTERN)
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ProtocolValidationError("Receipt reason codes must be unique")
        object.__setattr__(self, "reason_codes", tuple(sorted(self.reason_codes)))
        _bounded_int("verified_work_units", self.verified_work_units, 0, MAX_WORK_UNITS)
        if self.confidence_tier is not None and not isinstance(self.confidence_tier, EvidenceTier):
            raise ProtocolValidationError("Invalid receipt confidence tier")
        if self.accepted:
            if self.reason_codes or self.verified_work_units == 0 or self.confidence_tier is None:
                raise ProtocolValidationError("Accepted receipt has inconsistent result fields")
        elif (
            not self.reason_codes
            or self.verified_work_units != 0
            or self.confidence_tier is not None
        ):
            raise ProtocolValidationError("Rejected receipt has inconsistent result fields")
        _bounded_int("validated_at_block", self.validated_at_block, 1, MAX_BLOCK_NUMBER)
        _signature(self.signature)

    def _payload(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "confidence_tier": (
                self.confidence_tier.value if self.confidence_tier is not None else None
            ),
            "evidence_digest": self.evidence_digest,
            "reason_codes": list(self.reason_codes),
            "signature": self.signature,
            "validated_at_block": self.validated_at_block,
            "validator_hotkey": self.validator_hotkey,
            "verified_work_units": self.verified_work_units,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], version: int) -> ValidationReceipt:
        """Construct a validation receipt from a strict decoded payload."""
        _exact_fields(
            payload,
            {
                "accepted",
                "confidence_tier",
                "evidence_digest",
                "reason_codes",
                "signature",
                "validated_at_block",
                "validator_hotkey",
                "verified_work_units",
            },
            "validation_receipt",
        )
        reason_values = _list(payload["reason_codes"], "reason_codes", 0, 32)
        confidence_value = payload["confidence_tier"]
        confidence = (
            None
            if confidence_value is None
            else _enum_value(EvidenceTier, confidence_value, "confidence_tier")
        )
        return cls(
            evidence_digest=_text(payload["evidence_digest"], "evidence_digest"),
            validator_hotkey=_text(payload["validator_hotkey"], "validator_hotkey"),
            accepted=_boolean(payload["accepted"], "accepted"),
            reason_codes=tuple(_text(item, "reason_code") for item in reason_values),
            verified_work_units=_integer(payload["verified_work_units"], "verified_work_units"),
            confidence_tier=confidence,
            validated_at_block=_integer(payload["validated_at_block"], "validated_at_block"),
            signature=_text(payload["signature"], "signature"),
            protocol_version=version,
        )


Message = JobManifest | CapabilityClaim | WorkLease | ExecutionEvidence | ValidationReceipt
_Decoder = Callable[[Mapping[str, object], int], Message]
_DECODERS: dict[str, _Decoder] = {
    JobManifest.message_type: JobManifest.from_payload,
    CapabilityClaim.message_type: CapabilityClaim.from_payload,
    WorkLease.message_type: WorkLease.from_payload,
    ExecutionEvidence.message_type: ExecutionEvidence.from_payload,
    ValidationReceipt.message_type: ValidationReceipt.from_payload,
}


def decode_message(data: bytes) -> Message:
    """Decode only canonical, version-supported wire bytes into a typed message."""
    if not isinstance(data, bytes):
        raise ProtocolDecodeError("Protocol input must be bytes")
    if not data or len(data) > MAX_WIRE_BYTES:
        raise ProtocolDecodeError("Protocol input violates the wire size limit")

    try:
        decoded = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except ProtocolDecodeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ProtocolDecodeError("Protocol input is not valid canonical JSON") from None

    try:
        root = _mapping(decoded, "message")
        _exact_fields(root, {"message_type", "payload", "protocol_version"}, "message")
        message_type = _text(root["message_type"], "message_type")
        version = _integer(root["protocol_version"], "protocol_version")
        _version(version)
        payload = _mapping(root["payload"], "payload")
        decoder = _DECODERS.get(message_type)
        if decoder is None:
            raise ProtocolDecodeError("Unsupported protocol message type")
        message = decoder(payload, version)
    except ProtocolValidationError as error:
        raise ProtocolDecodeError(str(error)) from None

    if message.canonical_bytes() != data:
        raise ProtocolDecodeError("Protocol input is valid JSON but not canonical encoding")
    return message


def _canonical_json(value: object) -> bytes:
    _validate_canonical_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError):
        raise ProtocolValidationError("Protocol value is not canonically encodable") from None


def _validate_canonical_value(value: object) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if not -(2**63) <= value <= MAX_WORK_UNITS:
            raise ProtocolValidationError("Canonical integer exceeds the signed 64-bit range")
        return
    if isinstance(value, str):
        _bounded_text("canonical text", value, 0, MAX_WIRE_BYTES)
        return
    if isinstance(value, list):
        for item in value:
            _validate_canonical_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolValidationError("Canonical object keys must be text")
            _bounded_text("canonical object key", key, 1, 128)
            _validate_canonical_value(item)
        return
    raise ProtocolValidationError("Protocol value is not part of the canonical JSON profile")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolDecodeError("Protocol input contains a duplicate field")
        result[key] = value
    return result


def _reject_float(_value: str) -> object:
    raise ProtocolDecodeError("Protocol input cannot contain floating-point numbers")


def _reject_constant(_value: str) -> object:
    raise ProtocolDecodeError("Protocol input cannot contain non-finite numbers")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProtocolValidationError(f"Invalid object for '{name}'")
    return cast(Mapping[str, object], value)


def _exact_fields(data: Mapping[str, object], fields: set[str], name: str) -> None:
    keys = set(data)
    if keys - fields:
        raise ProtocolValidationError(f"'{name}' contains unknown fields")
    missing = fields - keys
    if missing:
        raise ProtocolValidationError(f"'{name}' is missing required fields")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ProtocolValidationError(f"Invalid text type for '{name}'")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(f"Invalid integer type for '{name}'")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolValidationError(f"Invalid boolean type for '{name}'")
    return value


def _list(value: object, name: str, minimum: int, maximum: int) -> list[object]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ProtocolValidationError(f"Invalid list for '{name}'")
    return value


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    integer = _integer(value, name)
    if not minimum <= integer <= maximum:
        raise ProtocolValidationError(f"Integer '{name}' is outside the permitted range")


def _bounded_text(
    name: str,
    value: object,
    minimum_bytes: int,
    maximum_bytes: int,
    pattern: re.Pattern[str] | None = None,
    *,
    ascii_only: bool = False,
) -> None:
    text = _text(value, name)
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        raise ProtocolValidationError(f"Text '{name}' is not valid UTF-8") from None
    if not minimum_bytes <= len(encoded) <= maximum_bytes:
        raise ProtocolValidationError(f"Text '{name}' violates its byte-length limit")
    if not unicodedata.is_normalized("NFC", text):
        raise ProtocolValidationError(f"Text '{name}' must use NFC normalization")
    if any(unicodedata.category(character).startswith("C") for character in text):
        raise ProtocolValidationError(f"Text '{name}' contains a prohibited character")
    if ascii_only and not text.isascii():
        raise ProtocolValidationError(f"Text '{name}' must be ASCII")
    if pattern is not None and pattern.fullmatch(text) is None:
        raise ProtocolValidationError(f"Text '{name}' violates its required format")


def _identifier(name: str, value: object) -> None:
    _bounded_text(name, value, 1, 64, _IDENTIFIER_PATTERN, ascii_only=True)


def _digest(name: str, value: object) -> None:
    _bounded_text(name, value, 71, 71, _DIGEST_PATTERN, ascii_only=True)


def _hotkey(name: str, value: object) -> None:
    _bounded_text(name, value, 3, 128, _HOTKEY_PATTERN, ascii_only=True)


def _nonce(value: object) -> None:
    _bounded_text("nonce", value, 64, 64, _HEX_PATTERN, ascii_only=True)


def _signature(value: object) -> None:
    _bounded_text("signature", value, 128, 128, _HEX_PATTERN, ascii_only=True)


def _version(value: object) -> None:
    if _integer(value, "protocol_version") != PROTOCOL_VERSION:
        raise ProtocolValidationError("Unsupported protocol version")


EnumValue = TypeVar("EnumValue", CapabilityTrust, EvidenceTier, NetworkPolicy)


def _enum_value(enum_type: type[EnumValue], value: object, name: str) -> EnumValue:
    text = _text(value, name)
    try:
        return enum_type(text)
    except ValueError:
        raise ProtocolValidationError(f"Invalid enum value for '{name}'") from None
