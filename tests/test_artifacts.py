"""Tests for content-addressed artifact storage and verified transfers."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from gpuforge.artifacts import (
    AccessGrant,
    ArtifactAuthorizationError,
    ArtifactKind,
    ArtifactOperation,
    ArtifactReference,
    ArtifactTransferError,
    LocalArtifactAdapter,
    fetch_verified,
    stream_sha256,
)


def reference(data: bytes, kind: ArtifactKind = ArtifactKind.SCRIPT) -> ArtifactReference:
    """Return the exact reference for test bytes."""
    return ArtifactReference(kind, f"sha256:{hashlib.sha256(data).hexdigest()}", len(data))


def test_each_artifact_kind_round_trips_after_digest_verification(tmp_path: Path) -> None:
    """Images, scripts, shards, and checkpoints share one verified interface."""
    adapter = LocalArtifactAdapter(tmp_path / "store", max_bytes=1_024)
    for kind in ArtifactKind:
        data = f"safe-{kind.value}".encode()
        item = reference(data, kind)
        assert adapter.store(item, io.BytesIO(data)) == item
        destination = tmp_path / f"fetched-{kind.value}"
        assert fetch_verified(adapter, item, destination, max_bytes=1_024) == destination
        assert destination.read_bytes() == data


@pytest.mark.parametrize(
    "item",
    (
        lambda: ArtifactReference(cast(ArtifactKind, "unknown"), "sha256:" + "0" * 64, 1),
        lambda: ArtifactReference(ArtifactKind.SCRIPT, "sha256:" + "A" * 64, 1),
        lambda: ArtifactReference(ArtifactKind.SCRIPT, "sha256:" + "0" * 64, cast(int, True)),
        lambda: ArtifactReference(ArtifactKind.SCRIPT, "sha256:" + "0" * 64, -1),
    ),
)
def test_artifact_reference_schema_is_bounded(item: Callable[[], object]) -> None:
    """References reject unknown kinds, malformed digests, and invalid sizes."""
    with pytest.raises(ValueError):
        item()


@pytest.mark.parametrize(
    "grant",
    (
        lambda: AccessGrant("", 1),
        lambda: AccessGrant("token", cast(int, True)),
    ),
)
def test_access_grant_schema_is_bounded(grant: Callable[[], object]) -> None:
    """Runtime grants reject empty bearer values and invalid expiries."""
    with pytest.raises(ArtifactAuthorizationError):
        grant()


def test_adapter_rejects_invalid_limits_and_roots(tmp_path: Path) -> None:
    """Storage setup fails closed for invalid limits and nondirectory roots."""
    with pytest.raises(ValueError, match="byte limit"):
        LocalArtifactAdapter(tmp_path / "store", max_bytes=0)
    root_file = tmp_path / "root-file"
    root_file.write_bytes(b"not-a-directory")
    with pytest.raises(ValueError, match="unavailable|directory"):
        LocalArtifactAdapter(root_file, max_bytes=1)


def test_store_rejects_digest_mismatch_and_leaves_no_artifact(tmp_path: Path) -> None:
    """Bytes are never trusted under a claimed digest that they do not match."""
    adapter = LocalArtifactAdapter(tmp_path / "store", max_bytes=1_024)
    claimed = reference(b"expected")

    with pytest.raises(ArtifactTransferError, match="digest"):
        adapter.store(claimed, io.BytesIO(b"tampered"))

    with pytest.raises(ArtifactTransferError, match="unavailable"):
        adapter.open_read(claimed, offset=0)


def test_existing_content_is_reverified_and_corruption_is_rejected(tmp_path: Path) -> None:
    """Deduplicated content is rehashed instead of trusted by its filename."""
    root = tmp_path / "store"
    adapter = LocalArtifactAdapter(root, max_bytes=1_024)
    data = b"existing"
    item = reference(data)
    adapter.store(item, io.BytesIO(data))
    assert adapter.store(item, io.BytesIO(data)) == item

    stored = root / item.kind.value / item.digest.removeprefix("sha256:")
    stored.write_bytes(b"corrupt!")
    with pytest.raises(ArtifactTransferError, match="does not match"):
        adapter.store(item, io.BytesIO(data))

    stored.unlink()
    stored.mkdir()
    with pytest.raises(ArtifactTransferError, match="unsafe"):
        adapter.store(item, io.BytesIO(data))


def test_open_rejects_invalid_offset_and_truncated_storage(tmp_path: Path) -> None:
    """Range reads cannot escape declared content length or use damaged storage."""
    root = tmp_path / "store"
    adapter = LocalArtifactAdapter(root, max_bytes=1_024)
    data = b"stored"
    item = reference(data)
    adapter.store(item, io.BytesIO(data))

    with pytest.raises(ArtifactTransferError, match="offset"):
        adapter.open_read(item, offset=-1)
    stored = root / item.kind.value / item.digest.removeprefix("sha256:")
    stored.write_bytes(b"short")
    with pytest.raises(ArtifactTransferError, match="truncated"):
        adapter.open_read(item, offset=0)

    with pytest.raises(ValueError, match="reference"):
        adapter.open_read(cast(ArtifactReference, object()), offset=0)


def test_store_rejects_truncation_and_oversized_stream(tmp_path: Path) -> None:
    """Declared length and configured byte bounds are both enforced while streaming."""
    adapter = LocalArtifactAdapter(tmp_path / "store", max_bytes=8)
    with pytest.raises(ArtifactTransferError, match="size"):
        adapter.store(reference(b"12345678"), io.BytesIO(b"1234"))
    with pytest.raises(ArtifactTransferError, match="byte limit"):
        adapter.store(reference(b"123456789"), io.BytesIO(b"123456789"))
    with pytest.raises(ArtifactTransferError, match="byte limit"):
        adapter.store(reference(b"12345678"), io.BytesIO(b"123456789"))


def test_store_rejects_interrupted_and_nonbyte_sources(tmp_path: Path) -> None:
    """Broken source implementations cannot create content-addressed files."""
    adapter = LocalArtifactAdapter(tmp_path / "store", max_bytes=1_024)
    item = reference(b"expected")
    with pytest.raises(ArtifactTransferError, match="interrupted"):
        adapter.store(item, InterruptingStream(b"expected", 2))

    class TextSource:
        def read(self, _size: int = -1) -> bytes:
            return "text"  # type: ignore[return-value]

    with pytest.raises(ArtifactTransferError, match="non-byte"):
        adapter.store(item, TextSource())  # type: ignore[arg-type]


class InterruptingStream(io.BytesIO):
    """Raise once after returning a bounded prefix."""

    def __init__(self, data: bytes, first_chunk: int) -> None:
        super().__init__(data)
        self._first_chunk = first_chunk
        self._reads = 0

    def read(self, size: int | None = -1) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return super().read(self._first_chunk)
        raise OSError("simulated interruption")


@dataclass
class InterruptOnceReader:
    """Interrupt one transfer, then honor the resumed offset."""

    data: bytes
    calls: list[int]

    def open_read(
        self,
        _reference: ArtifactReference,
        *,
        offset: int,
        grant: AccessGrant | None = None,
    ) -> BinaryIO:
        del grant
        self.calls.append(offset)
        if len(self.calls) == 1:
            return InterruptingStream(self.data[offset:], 4)
        return io.BytesIO(self.data[offset:])


def test_interrupted_transfer_resumes_and_verifies(tmp_path: Path) -> None:
    """A retry begins at the persisted partial length and publishes only complete bytes."""
    data = b"resumable-artifact"
    reader = InterruptOnceReader(data, [])
    destination = tmp_path / "artifact.bin"

    fetch_verified(reader, reference(data), destination, max_bytes=1_024, max_attempts=2)

    assert reader.calls == [0, 4]
    assert destination.read_bytes() == data
    assert not (tmp_path / ".artifact.bin.part").exists()


@dataclass
class TruncatedReader:
    """Return the same truncated content and record bounded retries."""

    data: bytes
    calls: int = 0

    def open_read(
        self,
        _reference: ArtifactReference,
        *,
        offset: int,
        grant: AccessGrant | None = None,
    ) -> BinaryIO:
        del grant
        self.calls += 1
        return io.BytesIO(self.data[offset:])


def test_truncation_stops_at_retry_bound_without_final_file(tmp_path: Path) -> None:
    """Repeated short reads cannot create a trusted destination or retry forever."""
    expected = b"complete-content"
    reader = TruncatedReader(expected[:5])
    destination = tmp_path / "artifact.bin"

    with pytest.raises(ArtifactTransferError, match="retry limit"):
        fetch_verified(reader, reference(expected), destination, max_bytes=1_024, max_attempts=3)

    assert reader.calls == 3
    assert not destination.exists()


@dataclass(frozen=True)
class ScopedAuthorizer:
    """Permit one token, operation, and digest combination."""

    token: str
    operation: ArtifactOperation
    digest: str

    def authorize(
        self,
        grant: AccessGrant,
        operation: ArtifactOperation,
        item: ArtifactReference,
    ) -> bool:
        return (
            grant.token == self.token and operation is self.operation and item.digest == self.digest
        )


@dataclass(frozen=True)
class FixedTokenProvider:
    """Supply one short-lived token to the transfer helper."""

    grant: AccessGrant

    def grant_for(
        self, _operation: ArtifactOperation, _reference: ArtifactReference
    ) -> AccessGrant:
        return self.grant


class BrokenAuthorizer:
    """Simulate an unavailable authorization dependency."""

    def authorize(
        self,
        _grant: AccessGrant,
        _operation: ArtifactOperation,
        _item: ArtifactReference,
    ) -> bool:
        raise RuntimeError("dependency unavailable")


def test_unauthorized_and_expired_references_fail_without_token_leak(tmp_path: Path) -> None:
    """Authorization is scoped, expiry-checked, and bearer values remain redacted."""
    data = b"protected"
    item = reference(data)
    token = "test-bearer-canary"
    authorizer = ScopedAuthorizer(token, ArtifactOperation.STORE, item.digest)
    adapter = LocalArtifactAdapter(
        tmp_path / "store", max_bytes=1_024, authorizer=authorizer, clock=lambda: 100
    )
    valid = AccessGrant(token, expires_at_epoch_s=101)
    adapter.store(item, io.BytesIO(data), grant=valid)

    with pytest.raises(ArtifactAuthorizationError) as denied:
        fetch_verified(
            adapter,
            item,
            tmp_path / "denied.bin",
            max_bytes=1_024,
            token_provider=FixedTokenProvider(valid),
        )
    assert token not in str(denied.value)
    assert token not in repr(valid)

    expired_adapter = LocalArtifactAdapter(
        tmp_path / "expired-store", max_bytes=1_024, authorizer=authorizer, clock=lambda: 102
    )
    with pytest.raises(ArtifactAuthorizationError, match="not authorized"):
        expired_adapter.store(item, io.BytesIO(data), grant=valid)

    broken_adapter = LocalArtifactAdapter(
        tmp_path / "broken-store",
        max_bytes=1_024,
        authorizer=BrokenAuthorizer(),
        clock=lambda: 100,
    )
    with pytest.raises(ArtifactAuthorizationError, match="not authorized"):
        broken_adapter.store(item, io.BytesIO(data), grant=valid)


def test_fetch_validates_limits_partial_state_and_digest(tmp_path: Path) -> None:
    """Fetch setup and final verification reject unsafe or inconsistent state."""
    data = b"expected"
    item = reference(data)
    reader = TruncatedReader(b"different")
    with pytest.raises(ArtifactTransferError, match="byte limit"):
        fetch_verified(reader, item, tmp_path / "too-large", max_bytes=1)
    with pytest.raises(ArtifactTransferError, match="retry limit"):
        fetch_verified(reader, item, tmp_path / "bad-retries", max_bytes=100, max_attempts=0)

    unsafe_destination = tmp_path / "unsafe.bin"
    (tmp_path / ".unsafe.bin.part").mkdir()
    with pytest.raises(ArtifactTransferError, match="partial path"):
        fetch_verified(reader, item, unsafe_destination, max_bytes=100)

    oversized_destination = tmp_path / "oversized.bin"
    (tmp_path / ".oversized.bin.part").write_bytes(b"x" * (item.size + 1))
    with pytest.raises(ArtifactTransferError, match="partial file"):
        fetch_verified(reader, item, oversized_destination, max_bytes=100, max_attempts=1)

    mismatch_reader = TruncatedReader(b"tampered")
    with pytest.raises(ArtifactTransferError, match="digest verification"):
        fetch_verified(
            mismatch_reader, item, tmp_path / "mismatch.bin", max_bytes=100, max_attempts=1
        )

    class TextReader:
        def open_read(
            self,
            _reference: ArtifactReference,
            *,
            offset: int,
            grant: AccessGrant | None = None,
        ) -> BinaryIO:
            del offset, grant

            class TextStream:
                def __enter__(self) -> TextStream:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

                def read(self, _size: int = -1) -> bytes:
                    return "text"  # type: ignore[return-value]

            return cast(BinaryIO, TextStream())

    with pytest.raises(ArtifactTransferError, match="non-byte"):
        fetch_verified(TextReader(), item, tmp_path / "nonbytes.bin", max_bytes=100, max_attempts=1)


def test_token_provider_failures_are_sanitized(tmp_path: Path) -> None:
    """Provider exceptions and invalid returns do not expose dependency details."""
    data = b"protected"
    item = reference(data)
    adapter = LocalArtifactAdapter(tmp_path / "store", max_bytes=100)
    adapter.store(item, io.BytesIO(data))

    class RaisingProvider:
        def grant_for(
            self, _operation: ArtifactOperation, _reference: ArtifactReference
        ) -> AccessGrant:
            raise RuntimeError("sensitive provider details")

    class InvalidProvider:
        def grant_for(
            self, _operation: ArtifactOperation, _reference: ArtifactReference
        ) -> AccessGrant:
            return cast(AccessGrant, object())

    for provider in (RaisingProvider(), InvalidProvider()):
        with pytest.raises(ArtifactAuthorizationError, match="unavailable") as error:
            fetch_verified(
                adapter,
                item,
                tmp_path / "denied.bin",
                max_bytes=100,
                token_provider=provider,
            )
        assert "sensitive provider details" not in str(error.value)


def test_stream_hashing_rejects_nonbytes_and_size_overflow() -> None:
    """Untrusted stream implementations cannot bypass byte and type bounds."""

    class TextStream:
        def read(self, _size: int = -1) -> bytes:
            return "not-bytes"  # type: ignore[return-value]

    with pytest.raises(ArtifactTransferError, match="non-byte"):
        stream_sha256(TextStream(), max_bytes=100)  # type: ignore[arg-type]
    with pytest.raises(ArtifactTransferError, match="byte limit"):
        stream_sha256(io.BytesIO(b"too large"), max_bytes=2)
    with pytest.raises(ArtifactTransferError, match="interrupted"):
        stream_sha256(InterruptingStream(b"content", 2), max_bytes=100)
