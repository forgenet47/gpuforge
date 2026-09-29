"""Bounded, hash-linked references behind a signed execution message."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from gpuforge.protocol import CheckpointCommitment, ExecutionEvidence, WorkLease

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_BLOB_BYTES = 4 * 1024 * 1024
_MAX_BUNDLE_BYTES = 64 * 1024
_ROOT_DOMAIN = b"gpuforge-evidence-bundle-v1\x00"


class EvidenceFailure(ValueError):
    """Evidence binding failed without exposing the source payload."""


@dataclass(frozen=True, slots=True)
class BlobReference:
    """Content-addressed external data; never a URL or access credential."""

    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or _DIGEST.fullmatch(self.digest) is None:
            raise EvidenceFailure("invalid_blob_reference")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise EvidenceFailure("invalid_blob_reference")
        if not 1 <= self.size_bytes <= _MAX_BLOB_BYTES:
            raise EvidenceFailure("invalid_blob_reference")

    def verify(self, payload: bytes) -> None:
        """Verify complete external bytes before their contents are interpreted."""
        if not isinstance(payload, bytes) or len(payload) != self.size_bytes:
            raise EvidenceFailure("blob_mismatch")
        if "sha256:" + hashlib.sha256(payload).hexdigest() != self.digest:
            raise EvidenceFailure("blob_mismatch")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """All independently fetched proof references for one signed evidence object."""

    manifest_digest: str
    lease_digest: str
    challenge_commitment: str
    challenge_response: BlobReference
    attestation: BlobReference | None
    runtime_image_digest: str
    checkpoints: tuple[CheckpointCommitment, ...]
    result: BlobReference
    measurement: BlobReference
    failure_events: BlobReference

    def __post_init__(self) -> None:
        for value in (
            self.manifest_digest,
            self.lease_digest,
            self.challenge_commitment,
            self.runtime_image_digest,
        ):
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise EvidenceFailure("invalid_evidence_bundle")
        references = (
            self.challenge_response,
            self.result,
            self.measurement,
            self.failure_events,
        )
        if not all(isinstance(ref, BlobReference) for ref in references):
            raise EvidenceFailure("invalid_evidence_bundle")
        if self.attestation is not None and not isinstance(self.attestation, BlobReference):
            raise EvidenceFailure("invalid_evidence_bundle")
        if not isinstance(self.checkpoints, tuple) or not 1 <= len(self.checkpoints) <= 4_096:
            raise EvidenceFailure("invalid_evidence_bundle")
        if not all(isinstance(item, CheckpointCommitment) for item in self.checkpoints):
            raise EvidenceFailure("invalid_evidence_bundle")
        steps = [item.step for item in self.checkpoints]
        if steps != sorted(set(steps)):
            raise EvidenceFailure("invalid_evidence_bundle")
        self.canonical_bytes()

    def canonical_bytes(self) -> bytes:
        """Return bounded canonical bytes for storage beside the evidence message."""

        def ref(value: BlobReference) -> dict[str, object]:
            return {"digest": value.digest, "size_bytes": value.size_bytes}

        value = {
            "attestation": None if self.attestation is None else ref(self.attestation),
            "challenge_commitment": self.challenge_commitment,
            "challenge_response": ref(self.challenge_response),
            "checkpoints": [item.to_primitive() for item in self.checkpoints],
            "failure_events": ref(self.failure_events),
            "lease_digest": self.lease_digest,
            "manifest_digest": self.manifest_digest,
            "measurement": ref(self.measurement),
            "result": ref(self.result),
            "runtime_image_digest": self.runtime_image_digest,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        if len(encoded) > _MAX_BUNDLE_BYTES:
            raise EvidenceFailure("evidence_bundle_oversized")
        return encoded

    def root(self) -> str:
        """Bind every component, including the failure-event record, in one root."""
        return "sha256:" + hashlib.sha256(_ROOT_DOMAIN + self.canonical_bytes()).hexdigest()

    def verify_binding(self, lease: WorkLease, evidence: ExecutionEvidence) -> None:
        """Check the signed message against every bundle field it can carry."""
        if (
            self.manifest_digest != lease.manifest_digest
            or self.lease_digest != lease.digest()
            or self.lease_digest != evidence.lease_digest
            or self.challenge_commitment != lease.challenge_commitment
            or self.challenge_response.digest != evidence.challenge_response_digest
            or (None if self.attestation is None else self.attestation.digest)
            != evidence.attestation_digest
            or self.runtime_image_digest != evidence.image_digest
            or self.checkpoints != evidence.checkpoints
            or self.root() != evidence.bundle_root
            or result_binding_digest(self.result, self.measurement, self.failure_events)
            != evidence.result_digest
        ):
            raise EvidenceFailure("evidence_binding_mismatch")


def result_binding_digest(
    result: BlobReference, measurement: BlobReference, failure_events: BlobReference
) -> str:
    """Bind result and telemetry references for producers that create result blobs."""
    value = b"gpuforge-result-binding-v1\x00"
    for reference in (result, measurement, failure_events):
        value += bytes.fromhex(reference.digest[7:]) + reference.size_bytes.to_bytes(8, "big")
    return "sha256:" + hashlib.sha256(value).hexdigest()
