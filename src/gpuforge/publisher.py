"""Deterministic publisher-side construction of signed training job manifests."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gpuforge.identity import (
    BittensorHotkeyAdapter,
    BittensorKeypair,
    MessageSigner,
    sign_message,
)
from gpuforge.protocol import (
    UNSIGNED_SIGNATURE,
    JobManifest,
    ResourcePolicy,
    VerificationPolicy,
)

_INPUT_ROOT_DOMAIN = b"gpuforge-input-root-v1\x00"
_CONTAINER_REFERENCE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}@sha256:(?P<digest>[0-9a-f]{64})"
)
_MAX_INPUTS = 4_096
_MAX_INPUT_BYTES = 1024**4


class PackagingError(ValueError):
    """Raised when a job cannot be packaged without weakening its identity."""


@dataclass(frozen=True, slots=True)
class InputRecord:
    """Deterministic identity for one declared regular input file."""

    path: str
    digest: str
    size: int

    def to_primitive(self) -> dict[str, object]:
        """Return the canonical input-record representation."""
        return {"digest": self.digest, "path": self.path, "size": self.size}


@dataclass(frozen=True, slots=True)
class PackagedJob:
    """Signed manifest plus the ordered records used for its input root."""

    manifest: JobManifest
    inputs: tuple[InputRecord, ...]

    @property
    def package_digest(self) -> str:
        """Return the signature-independent package identity."""
        return self.manifest.content_digest()


@dataclass(frozen=True, slots=True)
class PackageRequest:
    """Validated inputs required to construct one publisher manifest."""

    root: Path
    entrypoint: Path
    inputs: tuple[Path, ...]
    container_image: str
    job_id: str
    framework: str
    resource_policy: ResourcePolicy
    verification_policy: VerificationPolicy
    lease_seconds: int
    current_block: int
    expires_at_block: int


def package_job(request: PackageRequest, signer: MessageSigner) -> PackagedJob:
    """Hash declared files, validate immutable policy, and sign a job manifest."""
    if (
        isinstance(request.current_block, bool)
        or not isinstance(request.current_block, int)
        or request.current_block < 0
    ):
        raise PackagingError("Current block is outside the permitted range")
    if request.expires_at_block <= request.current_block:
        raise PackagingError("Job expiry must be after the current block")
    if len(request.inputs) == 0 or len(request.inputs) > _MAX_INPUTS:
        raise PackagingError("Declared input count is outside the permitted range")

    root = _safe_root(request.root)
    entrypoint_path, _ = _safe_regular_file(root, request.entrypoint, "entrypoint")
    records = tuple(sorted((_input_record(root, path) for path in request.inputs), key=_by_path))
    paths = [record.path for record in records]
    if len(paths) != len(set(paths)):
        raise PackagingError("Declared input paths must be unique")

    unsigned = JobManifest(
        job_id=request.job_id,
        container_digest=container_digest(request.container_image),
        entrypoint_digest=_hash_file(entrypoint_path),
        input_root=input_root(records),
        framework=request.framework,
        resource_policy=request.resource_policy,
        verification_policy=request.verification_policy,
        lease_seconds=request.lease_seconds,
        publisher_hotkey=signer.hotkey,
        expires_at_block=request.expires_at_block,
        signature=UNSIGNED_SIGNATURE,
    )
    signed = sign_message(unsigned, signer)
    if not isinstance(signed, JobManifest):  # pragma: no cover - type-preserving contract
        raise PackagingError("Manifest signer returned an unexpected message type")
    return PackagedJob(manifest=signed, inputs=records)


def container_digest(reference: str) -> str:
    """Extract a digest only from an OCI reference pinned by SHA-256."""
    if not isinstance(reference, str):
        raise PackagingError("Container image reference must be text")
    match = _CONTAINER_REFERENCE.fullmatch(reference)
    if match is None:
        raise PackagingError("Container image must be pinned with @sha256:<64 lowercase hex>")
    return f"sha256:{match.group('digest')}"


def input_root(records: tuple[InputRecord, ...]) -> str:
    """Hash the ordered canonical declaration of input paths and file identities."""
    encoded = json.dumps(
        [record.to_primitive() for record in records],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(_INPUT_ROOT_DOMAIN + encoded).hexdigest()}"


def write_manifest(packaged: PackagedJob, destination: Path) -> None:
    """Atomically write a signed canonical manifest to a new or existing path."""
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(packaged.manifest.canonical_bytes())
        os.replace(temporary, destination)
    except OSError:
        raise PackagingError("Signed manifest could not be written") from None


def load_bittensor_hotkey(wallet_name: str, hotkey_name: str) -> BittensorHotkeyAdapter:
    """Load a local Bittensor hotkey without accepting secret material as arguments."""
    if not wallet_name or not hotkey_name:
        raise PackagingError("Wallet and hotkey names are required")
    try:
        bittensor = importlib.import_module("bittensor")
        try:
            wallet_factory = bittensor.Wallet
        except AttributeError:
            wallet_factory = bittensor.wallet
        wallet = wallet_factory(name=wallet_name, hotkey=hotkey_name)
        keypair = cast(BittensorKeypair, wallet.hotkey)
        return BittensorHotkeyAdapter(keypair)
    except Exception:
        raise PackagingError("Bittensor hotkey could not be loaded") from None


def _safe_root(path: Path) -> Path:
    if path.is_symlink():
        raise PackagingError("Package root must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise PackagingError("Package root is unavailable") from None
    if not resolved.is_dir():
        raise PackagingError("Package root must be a directory")
    return resolved


def _safe_regular_file(root: Path, relative: Path, label: str) -> tuple[Path, str]:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PackagingError(f"Declared {label} path must be a normalized relative path")
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise PackagingError(f"Declared {label} path must not traverse a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
        normalized = resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        raise PackagingError(
            f"Declared {label} file is unavailable or outside the package root"
        ) from None
    if not resolved.is_file():
        raise PackagingError(f"Declared {label} must be a regular file")
    return resolved, normalized


def _input_record(root: Path, relative: Path) -> InputRecord:
    path, normalized = _safe_regular_file(root, relative, "input")
    try:
        size = path.stat().st_size
    except OSError:
        raise PackagingError("Declared input file is unavailable") from None
    if not 0 <= size <= _MAX_INPUT_BYTES:
        raise PackagingError("Declared input file exceeds the size limit")
    return InputRecord(path=normalized, digest=_hash_file(path), size=size)


def _hash_file(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
    except OSError:
        raise PackagingError("Declared file could not be hashed") from None
    return f"sha256:{value.hexdigest()}"


def _by_path(record: InputRecord) -> str:
    return record.path


__all__ = [
    "InputRecord",
    "PackageRequest",
    "PackagedJob",
    "PackagingError",
    "container_digest",
    "input_root",
    "load_bittensor_hotkey",
    "package_job",
    "write_manifest",
]
