"""Tests for deterministic and fail-closed publisher packaging."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import gpuforge.cli as cli_module
from gpuforge.cli import main
from gpuforge.config import EvidenceTier, NetworkPolicy
from gpuforge.protocol import (
    JobManifest,
    ProtocolValidationError,
    ResourcePolicy,
    VerificationPolicy,
    decode_message,
)
from gpuforge.publisher import (
    PackageRequest,
    PackagingError,
    container_digest,
    load_bittensor_hotkey,
    package_job,
)

PUBLISHER = "5PublisherHotkeyExample11111111111111"
IMAGE = "registry.example/gpuforge/trainer@sha256:" + "1" * 64


@dataclass(frozen=True)
class DemoSigner:
    """Deterministic signer for publisher tests."""

    hotkey: str = PUBLISHER

    def sign(self, payload: bytes) -> bytes:
        """Return a deterministic 64-byte signature."""
        return hmac.new(bytes(range(64)), payload, hashlib.sha512).digest()


def resource_policy() -> ResourcePolicy:
    """Return the H100-oriented default package limits."""
    return ResourcePolicy(
        gpu_count=1,
        gpu_memory_mb=81_920,
        cpu_cores=16,
        memory_mb=131_072,
        max_runtime_seconds=3_600,
        network_policy=NetworkPolicy.DENY,
    )


def verification_policy() -> VerificationPolicy:
    """Return deterministic verification requirements."""
    return VerificationPolicy(
        minimum_evidence_tier=EvidenceTier.C,
        challenge_kind="gradient_slice",
        checkpoint_interval_steps=100,
    )


def sample_root(tmp_path: Path) -> Path:
    """Create a harmless package fixture."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "train.py").write_text("print('sample training')\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample.bin").write_bytes(b"sample-input\n")
    return tmp_path


def request(root: Path, *, inputs: tuple[Path, ...] | None = None) -> PackageRequest:
    """Return a complete valid package request."""
    return PackageRequest(
        root=root,
        entrypoint=Path("train.py"),
        inputs=inputs or (Path("data/sample.bin"),),
        container_image=IMAGE,
        job_id="sample-training-1",
        framework="pytorch",
        resource_policy=resource_policy(),
        verification_policy=verification_policy(),
        lease_seconds=3_600,
        current_block=1_000,
        expires_at_block=1_100,
    )


def test_same_package_is_reproducible(tmp_path: Path) -> None:
    """Packaging identical harmless inputs twice produces the same identity and bytes."""
    root = sample_root(tmp_path)

    first = package_job(request(root), DemoSigner())
    second = package_job(request(root), DemoSigner())

    assert first.package_digest == second.package_digest
    assert first.manifest.canonical_bytes() == second.manifest.canonical_bytes()
    assert first.inputs == second.inputs
    assert first.manifest.publisher_hotkey == PUBLISHER
    assert first.manifest.resource_policy.network_policy is NetworkPolicy.DENY


@pytest.mark.parametrize("filename", ("train.py", "data/sample.bin"))
def test_changed_declared_bytes_change_package_identity(tmp_path: Path, filename: str) -> None:
    """Changing either the entrypoint or an input changes the signed package identity."""
    root = sample_root(tmp_path)
    before = package_job(request(root), DemoSigner())
    (root / filename).write_bytes(b"changed-bytes\n")
    after = package_job(request(root), DemoSigner())

    assert before.package_digest != after.package_digest


def test_input_order_does_not_change_root(tmp_path: Path) -> None:
    """The input root sorts normalized paths before hashing records."""
    root = sample_root(tmp_path)
    (root / "data" / "second.bin").write_bytes(b"second")
    paths = (Path("data/sample.bin"), Path("data/second.bin"))

    first = package_job(request(root, inputs=paths), DemoSigner())
    second = package_job(request(root, inputs=tuple(reversed(paths))), DemoSigner())

    assert first.package_digest == second.package_digest
    assert first.manifest.input_root == second.manifest.input_root


@pytest.mark.parametrize(
    "unsafe_path",
    (Path("../outside.bin"), Path("data/../train.py"), Path("C:/absolute.bin")),
)
def test_path_traversal_and_absolute_inputs_are_rejected(tmp_path: Path, unsafe_path: Path) -> None:
    """Declared inputs cannot escape or ambiguously traverse the package root."""
    root = sample_root(tmp_path)

    with pytest.raises(PackagingError, match="normalized relative"):
        package_job(request(root, inputs=(unsafe_path,)), DemoSigner())


def test_symlinked_input_is_rejected(tmp_path: Path) -> None:
    """Input hashing never follows a symbolic link."""
    root = sample_root(tmp_path)
    link = root / "data" / "linked.bin"
    try:
        os.symlink(root / "data" / "sample.bin", link)
    except OSError:
        pytest.skip("Symbolic-link creation is unavailable")

    with pytest.raises(PackagingError, match="symbolic link"):
        package_job(request(root, inputs=(Path("data/linked.bin"),)), DemoSigner())


@pytest.mark.parametrize(
    "reference",
    (
        "registry.example/trainer:latest",
        "registry.example/trainer:1.0",
        "registry.example/trainer@sha256:" + "A" * 64,
        "sha256:" + "1" * 64,
    ),
)
def test_mutable_or_malformed_container_references_are_rejected(reference: str) -> None:
    """Only a named OCI image pinned by a lowercase SHA-256 digest is accepted."""
    with pytest.raises(PackagingError, match="pinned"):
        container_digest(reference)


def test_expired_manifest_is_rejected_before_signing(tmp_path: Path) -> None:
    """A publisher cannot package an already expired manifest."""
    root = sample_root(tmp_path)
    expired = request(root)
    expired = PackageRequest(
        root=expired.root,
        entrypoint=expired.entrypoint,
        inputs=expired.inputs,
        container_image=expired.container_image,
        job_id=expired.job_id,
        framework=expired.framework,
        resource_policy=expired.resource_policy,
        verification_policy=expired.verification_policy,
        lease_seconds=expired.lease_seconds,
        current_block=1_000,
        expires_at_block=1_000,
    )

    with pytest.raises(PackagingError, match="after the current"):
        package_job(expired, DemoSigner())


@pytest.mark.parametrize("current_block", (-1, True))
def test_invalid_current_block_is_rejected(tmp_path: Path, current_block: int) -> None:
    """Library callers cannot bypass integer block bounds."""
    root = sample_root(tmp_path)
    original = request(root)
    invalid = PackageRequest(
        root=original.root,
        entrypoint=original.entrypoint,
        inputs=original.inputs,
        container_image=original.container_image,
        job_id=original.job_id,
        framework=original.framework,
        resource_policy=original.resource_policy,
        verification_policy=original.verification_policy,
        lease_seconds=original.lease_seconds,
        current_block=current_block,
        expires_at_block=original.expires_at_block,
    )

    with pytest.raises(PackagingError, match="Current block"):
        package_job(invalid, DemoSigner())


def test_empty_and_duplicate_input_declarations_are_rejected(tmp_path: Path) -> None:
    """The input commitment requires a nonempty set of unique normalized paths."""
    root = sample_root(tmp_path)
    empty = request(root)
    empty = PackageRequest(
        root=empty.root,
        entrypoint=empty.entrypoint,
        inputs=(),
        container_image=empty.container_image,
        job_id=empty.job_id,
        framework=empty.framework,
        resource_policy=empty.resource_policy,
        verification_policy=empty.verification_policy,
        lease_seconds=empty.lease_seconds,
        current_block=empty.current_block,
        expires_at_block=empty.expires_at_block,
    )
    with pytest.raises(PackagingError, match="input count"):
        package_job(empty, DemoSigner())

    duplicate = request(root, inputs=(Path("data/sample.bin"), Path("data/sample.bin")))
    with pytest.raises(PackagingError, match="unique"):
        package_job(duplicate, DemoSigner())


def test_unavailable_or_nondirectory_root_is_rejected(tmp_path: Path) -> None:
    """A package root must exist and be a directory before any file is hashed."""
    missing = tmp_path / "missing"
    with pytest.raises(PackagingError, match="unavailable"):
        package_job(request(missing), DemoSigner())

    root_file = tmp_path / "root-file"
    root_file.write_bytes(b"not-a-directory")
    with pytest.raises(PackagingError, match="directory"):
        package_job(request(root_file), DemoSigner())


def test_nontext_container_reference_is_rejected() -> None:
    """Container identity parsing does not coerce arbitrary values."""
    with pytest.raises(PackagingError, match="text"):
        container_digest(cast(str, 123))


def test_current_bittensor_wallet_shape_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional adapter loads the current capitalized Bittensor wallet API."""

    class FakeKeypair:
        ss58_address = PUBLISHER

        def sign(self, payload: bytes) -> bytes:
            return hashlib.sha512(payload).digest()

        def verify(self, payload: bytes, signature: bytes) -> bool:
            return hmac.compare_digest(signature, self.sign(payload))

    def wallet_factory(*, name: str, hotkey: str) -> SimpleNamespace:
        assert (name, hotkey) == ("publisher-wallet", "publisher-hotkey")
        return SimpleNamespace(hotkey=FakeKeypair())

    module = SimpleNamespace(Wallet=wallet_factory)
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)

    adapter = load_bittensor_hotkey("publisher-wallet", "publisher-hotkey")
    assert adapter.hotkey == PUBLISHER


def test_invalid_network_policy_is_rejected_by_typed_policy() -> None:
    """Publisher requests cannot construct an unrecognized outbound-network policy."""
    with pytest.raises(ProtocolValidationError, match="network policy"):
        ResourcePolicy(
            gpu_count=1,
            gpu_memory_mb=81_920,
            cpu_cores=16,
            memory_mb=131_072,
            max_runtime_seconds=3_600,
            network_policy=cast(NetworkPolicy, "open"),
        )


def test_cli_writes_signed_canonical_manifest_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The publisher CLI writes only canonical protocol bytes and reports no local paths."""
    root = sample_root(tmp_path / "package")
    destination = tmp_path / "signed-manifest.json"
    monkeypatch.setattr(cli_module, "load_bittensor_hotkey", lambda _wallet, _hotkey: DemoSigner())

    exit_code = main(
        [
            "publisher",
            "package",
            "--root",
            str(root),
            "--entrypoint",
            "train.py",
            "--input",
            "data/sample.bin",
            "--container",
            IMAGE,
            "--job-id",
            "sample-training-1",
            "--current-block",
            "1000",
            "--expires-at-block",
            "1100",
            "--wallet-name",
            "test-wallet",
            "--hotkey-name",
            "test-hotkey",
            "--output",
            str(destination),
        ]
    )

    captured = capsys.readouterr()
    decoded = decode_message(destination.read_bytes())
    assert exit_code == 0
    assert captured.err == ""
    assert isinstance(decoded, JobManifest)
    assert str(tmp_path) not in captured.out
    assert "network_connection_attempted" in captured.out
