"""Content-addressed artifact storage with verified streaming transfers."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Protocol, cast

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_ARTIFACT_BYTES = 1024**4
_MAX_ATTEMPTS = 5
_DEFAULT_CHUNK_BYTES = 1024 * 1024


class ArtifactError(ValueError):
    """Base error for artifact references, storage, and transfers."""


class ArtifactAuthorizationError(ArtifactError):
    """Raised when an artifact operation is not authorized."""


class ArtifactTransferError(ArtifactError):
    """Raised when an artifact cannot be completely verified."""


class ArtifactKind(str, Enum):
    """Content classes stored behind the artifact interface."""

    IMAGE = "image"
    SCRIPT = "script"
    DATASET_SHARD = "dataset_shard"
    CHECKPOINT = "checkpoint"


class ArtifactOperation(str, Enum):
    """Operations that may be authorized independently."""

    FETCH = "fetch"
    STORE = "store"


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Public content identity with no locator or credential material."""

    kind: ArtifactKind
    digest: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            raise ArtifactError("Artifact kind is invalid")
        if not isinstance(self.digest, str) or _DIGEST_PATTERN.fullmatch(self.digest) is None:
            raise ArtifactError("Artifact digest must be lowercase SHA-256")
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise ArtifactError("Artifact size must be an integer")
        if not 0 <= self.size <= _MAX_ARTIFACT_BYTES:
            raise ArtifactError("Artifact size is outside the permitted range")


@dataclass(frozen=True, slots=True, repr=False)
class AccessGrant:
    """Short-lived runtime credential that redacts its bearer value."""

    token: str = field(repr=False)
    expires_at_epoch_s: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token, str)
            or not 1 <= len(self.token.encode("utf-8")) <= 4_096
            or any(ord(character) < 32 for character in self.token)
        ):
            raise ArtifactAuthorizationError("Artifact access token is invalid")
        if (
            isinstance(self.expires_at_epoch_s, bool)
            or not isinstance(self.expires_at_epoch_s, int)
            or self.expires_at_epoch_s < 0
        ):
            raise ArtifactAuthorizationError("Artifact access expiry is invalid")

    def __repr__(self) -> str:
        """Return a representation that never exposes the bearer value."""
        return f"AccessGrant(token=<redacted>, expires_at_epoch_s={self.expires_at_epoch_s})"


class AccessTokenProvider(Protocol):
    """Issue a scoped grant immediately before an artifact operation."""

    def grant_for(
        self, operation: ArtifactOperation, reference: ArtifactReference
    ) -> AccessGrant: ...


class ArtifactAuthorizer(Protocol):
    """Validate a grant against one exact operation and content reference."""

    def authorize(
        self,
        grant: AccessGrant,
        operation: ArtifactOperation,
        reference: ArtifactReference,
    ) -> bool: ...


class ArtifactReader(Protocol):
    """Open an artifact stream at a verified resume offset."""

    def open_read(
        self,
        reference: ArtifactReference,
        *,
        offset: int,
        grant: AccessGrant | None = None,
    ) -> BinaryIO: ...


class ArtifactStore(Protocol):
    """Store a stream under its expected content identity."""

    def store(
        self,
        reference: ArtifactReference,
        source: BinaryIO,
        *,
        grant: AccessGrant | None = None,
    ) -> ArtifactReference: ...


class LocalArtifactAdapter:
    """Local content-addressed adapter for tests and offline development."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        authorizer: ArtifactAuthorizer | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if isinstance(max_bytes, bool) or not 1 <= max_bytes <= _MAX_ARTIFACT_BYTES:
            raise ArtifactError("Artifact storage byte limit is invalid")
        if root.exists() and root.is_symlink():
            raise ArtifactError("Artifact storage root must not be a symbolic link")
        try:
            root.mkdir(parents=True, exist_ok=True)
            resolved = root.resolve(strict=True)
        except OSError:
            raise ArtifactError("Artifact storage root is unavailable") from None
        if not resolved.is_dir():
            raise ArtifactError("Artifact storage root must be a directory")
        self._root = resolved
        self._max_bytes = max_bytes
        self._authorizer = authorizer
        self._clock = clock or (lambda: int(time.time()))

    def store(
        self,
        reference: ArtifactReference,
        source: BinaryIO,
        *,
        grant: AccessGrant | None = None,
    ) -> ArtifactReference:
        """Stream, hash, and atomically store only the expected complete content."""
        self._validate_reference(reference)
        self._authorize(grant, ArtifactOperation.STORE, reference)
        directory = self._kind_directory(reference.kind)
        destination = self._artifact_path(reference)
        if destination.exists():
            self._verify_existing(destination, reference)
            return reference

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".upload-", delete=False
            ) as target:
                temporary_path = Path(target.name)
                digest, size = _copy_and_hash(source, cast(BinaryIO, target), self._max_bytes)
            if size != reference.size:
                raise ArtifactTransferError("Stored artifact size does not match its reference")
            if digest != reference.digest:
                raise ArtifactTransferError("Stored artifact digest does not match its reference")
            os.replace(temporary_path, destination)
        except ArtifactError:
            _unlink(temporary_path)
            raise
        except (OSError, ValueError):
            _unlink(temporary_path)
            raise ArtifactTransferError("Artifact store operation failed") from None
        return reference

    def open_read(
        self,
        reference: ArtifactReference,
        *,
        offset: int,
        grant: AccessGrant | None = None,
    ) -> BinaryIO:
        """Open stored bytes at an exact bounded offset after authorization."""
        self._validate_reference(reference)
        self._authorize(grant, ArtifactOperation.FETCH, reference)
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset <= reference.size
        ):
            raise ArtifactTransferError("Artifact resume offset is invalid")
        path = self._artifact_path(reference)
        if path.is_symlink() or not path.is_file():
            raise ArtifactTransferError("Artifact reference is unavailable")
        try:
            if path.stat().st_size != reference.size:
                raise ArtifactTransferError("Stored artifact is truncated or oversized")
            stream = path.open("rb")
            stream.seek(offset)
            return stream
        except ArtifactError:
            raise
        except OSError:
            raise ArtifactTransferError("Artifact could not be opened") from None

    def _validate_reference(self, reference: ArtifactReference) -> None:
        if not isinstance(reference, ArtifactReference):
            raise ArtifactError("Artifact reference is invalid")
        if reference.size > self._max_bytes:
            raise ArtifactTransferError("Artifact exceeds the configured byte limit")

    def _authorize(
        self,
        grant: AccessGrant | None,
        operation: ArtifactOperation,
        reference: ArtifactReference,
    ) -> None:
        if self._authorizer is None:
            return
        if grant is None or grant.expires_at_epoch_s < self._clock():
            raise ArtifactAuthorizationError("Artifact reference is not authorized")
        try:
            permitted = self._authorizer.authorize(grant, operation, reference)
        except Exception:
            permitted = False
        if permitted is not True:
            raise ArtifactAuthorizationError("Artifact reference is not authorized")

    def _kind_directory(self, kind: ArtifactKind) -> Path:
        directory = self._root / kind.value
        if directory.exists() and directory.is_symlink():
            raise ArtifactError("Artifact kind directory must not be a symbolic link")
        try:
            directory.mkdir(exist_ok=True)
        except OSError:
            raise ArtifactError("Artifact kind directory is unavailable") from None
        return directory

    def _artifact_path(self, reference: ArtifactReference) -> Path:
        return self._kind_directory(reference.kind) / reference.digest.removeprefix("sha256:")

    def _verify_existing(self, path: Path, reference: ArtifactReference) -> None:
        if path.is_symlink() or not path.is_file():
            raise ArtifactTransferError("Existing artifact path is unsafe")
        try:
            with path.open("rb") as source:
                digest, size = stream_sha256(source, max_bytes=self._max_bytes)
        except OSError:
            raise ArtifactTransferError("Existing artifact could not be verified") from None
        if digest != reference.digest or size != reference.size:
            raise ArtifactTransferError("Existing artifact does not match its reference")


def stream_sha256(source: BinaryIO, *, max_bytes: int) -> tuple[str, int]:
    """Hash a stream while enforcing a strict byte limit."""
    digest = hashlib.sha256()
    size = 0
    while True:
        try:
            chunk = source.read(_DEFAULT_CHUNK_BYTES)
        except OSError:
            raise ArtifactTransferError("Artifact stream was interrupted") from None
        if not isinstance(chunk, bytes):
            raise ArtifactTransferError("Artifact stream returned non-byte content")
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise ArtifactTransferError("Artifact exceeds the configured byte limit")
        digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def fetch_verified(
    reader: ArtifactReader,
    reference: ArtifactReference,
    destination: Path,
    *,
    max_bytes: int,
    max_attempts: int = 3,
    token_provider: AccessTokenProvider | None = None,
) -> Path:
    """Resume a bounded transfer and publish it only after complete digest verification."""
    if reference.size > max_bytes:
        raise ArtifactTransferError("Artifact exceeds the configured byte limit")
    if not 1 <= max_attempts <= _MAX_ATTEMPTS:
        raise ArtifactTransferError("Artifact retry limit is invalid")
    if destination.is_symlink():
        raise ArtifactTransferError("Artifact destination must not be a symbolic link")
    partial = destination.with_name(f".{destination.name}.part")
    if partial.exists() and (partial.is_symlink() or not partial.is_file()):
        raise ArtifactTransferError("Artifact partial path is unsafe")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ArtifactTransferError("Artifact destination is unavailable") from None

    last_error = "Artifact transfer did not complete"
    for _attempt in range(max_attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > reference.size or offset > max_bytes:
                _unlink(partial)
                raise ArtifactTransferError("Artifact partial file exceeds its expected size")
            grant = _grant(token_provider, ArtifactOperation.FETCH, reference)
            with reader.open_read(reference, offset=offset, grant=grant) as source:
                with partial.open("ab") as target:
                    _copy_remaining(source, target, offset, reference.size, max_bytes)
            digest, size = _hash_path(partial, max_bytes)
            if size < reference.size:
                raise ArtifactTransferError("Artifact transfer was truncated")
            if size > reference.size:
                _unlink(partial)
                raise ArtifactTransferError("Artifact transfer exceeded its expected size")
            if digest != reference.digest:
                _unlink(partial)
                raise ArtifactTransferError("Artifact digest verification failed")
            os.replace(partial, destination)
            return destination
        except ArtifactAuthorizationError:
            raise
        except ArtifactTransferError as error:
            last_error = str(error)
        except OSError:
            last_error = "Artifact transfer was interrupted"
    raise ArtifactTransferError(f"{last_error}; retry limit reached")


def _copy_and_hash(source: BinaryIO, target: BinaryIO, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        try:
            chunk = source.read(_DEFAULT_CHUNK_BYTES)
        except OSError:
            raise ArtifactTransferError("Artifact stream was interrupted") from None
        if not isinstance(chunk, bytes):
            raise ArtifactTransferError("Artifact stream returned non-byte content")
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise ArtifactTransferError("Artifact exceeds the configured byte limit")
        digest.update(chunk)
        target.write(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _copy_remaining(
    source: BinaryIO,
    target: BinaryIO,
    offset: int,
    expected_size: int,
    max_bytes: int,
) -> None:
    total = offset
    while True:
        chunk = source.read(_DEFAULT_CHUNK_BYTES)
        if not isinstance(chunk, bytes):
            raise ArtifactTransferError("Artifact stream returned non-byte content")
        if not chunk:
            return
        total += len(chunk)
        if total > expected_size or total > max_bytes:
            raise ArtifactTransferError("Artifact transfer exceeded its expected size")
        target.write(chunk)


def _hash_path(path: Path, max_bytes: int) -> tuple[str, int]:
    try:
        with path.open("rb") as source:
            return stream_sha256(source, max_bytes=max_bytes)
    except OSError:
        raise ArtifactTransferError("Artifact partial file could not be verified") from None


def _grant(
    provider: AccessTokenProvider | None,
    operation: ArtifactOperation,
    reference: ArtifactReference,
) -> AccessGrant | None:
    if provider is None:
        return None
    try:
        grant = provider.grant_for(operation, reference)
    except Exception:
        raise ArtifactAuthorizationError("Artifact access token is unavailable") from None
    if not isinstance(grant, AccessGrant):
        raise ArtifactAuthorizationError("Artifact access token is unavailable")
    return grant


def _unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
