"""Tests for minimal, self-reported GPU capability discovery."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import cast

import pytest

from gpuforge.capability import (
    CapabilityDiscoveryError,
    CapabilityPolicy,
    GpuClass,
    GpuInterconnect,
    GpuObservation,
    RuntimeCompatibilityRule,
    build_capability_claim,
    discover_inventory,
    normalize_gpu,
)
from gpuforge.config import EvidenceTier
from gpuforge.identity import verify_message_signature
from gpuforge.protocol import CapabilityTrust

MINER = "5MinerHotkeyExample111111111111111111"
NONCE = "ab" * 32


@dataclass(frozen=True)
class DemoSigner:
    """Deterministic signer and verifier for capability tests."""

    hotkey: str = MINER

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(b"capability-test-key", payload, hashlib.sha512).digest()

    def verify(self, hotkey: str, payload: bytes, signature: bytes) -> bool:
        return hotkey == self.hotkey and hmac.compare_digest(signature, self.sign(payload))


@dataclass(frozen=True)
class FixedProvider:
    """Return an injected set of minimal device observations."""

    devices: Sequence[GpuObservation]

    def discover(self) -> Sequence[GpuObservation]:
        return self.devices


def policy() -> CapabilityPolicy:
    """Return a deterministic compatibility policy for tests."""
    return CapabilityPolicy((RuntimeCompatibilityRule(12, (535, 54)),))


def sxm(**changes: object) -> GpuObservation:
    """Return one valid H100 SXM provider observation."""
    values: dict[str, object] = {
        "name": "NVIDIA H100 80GB HBM3",
        "memory_mb": 81_920,
        "driver_version": "550.054.015",
        "runtime_version": "12.08",
        "interconnect": "NVLink-NVSwitch",
    }
    values.update(changes)
    return GpuObservation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("observation", "gpu_class", "model", "interconnect"),
    (
        (sxm(), GpuClass.H100_SXM, "NVIDIA H100 SXM", GpuInterconnect.NVLINK),
        (
            sxm(name="NVIDIA H100 PCIe", interconnect="PCI Express"),
            GpuClass.H100_PCIE,
            "NVIDIA H100 PCIe",
            GpuInterconnect.PCIE,
        ),
        (
            sxm(name="NVIDIA H100 NVL", memory_mb=95_000, interconnect="NVLink"),
            GpuClass.H100_NVL,
            "NVIDIA H100 NVL",
            GpuInterconnect.NVLINK,
        ),
    ),
)
def test_h100_variants_are_normalized_deterministically(
    observation: GpuObservation,
    gpu_class: GpuClass,
    model: str,
    interconnect: GpuInterconnect,
) -> None:
    """Known provider spellings map to stable public capability values."""
    normalized = normalize_gpu(observation, policy())

    assert normalized.gpu_class is gpu_class
    assert normalized.model == model
    assert normalized.interconnect is interconnect
    assert normalized.driver_version == "550.54.15"
    assert normalized.runtime_version == "12.8"


def test_multiple_h100s_build_a_signed_self_reported_claim() -> None:
    """Count is derived from observations and the resulting claim authenticates the miner."""
    provider = FixedProvider((sxm(), sxm()))
    signer = DemoSigner()

    claim = build_capability_claim(
        provider,
        policy(),
        signer,
        observed_at_block=1_000,
        available_gpu_seconds=7_200,
        supported_evidence_tiers=(EvidenceTier.C,),
        nonce=NONCE,
    )

    verify_message_signature(claim, signer)
    assert claim.gpu_count == 2
    assert claim.gpu_model == "NVIDIA H100 SXM"
    assert claim.gpu_memory_mb == 81_920
    assert claim.gpu_interconnect == "nvlink"
    assert claim.discovery_trust is CapabilityTrust.SELF_REPORTED
    assert tuple((item.name, item.version) for item in claim.runtime_versions) == (
        ("cuda", "12.8"),
        ("driver", "550.54.15"),
    )


def test_equivalent_provider_formatting_produces_identical_claim_content() -> None:
    """Case, spacing, and harmless numeric padding do not alter normalized content."""
    first = FixedProvider((sxm(),))
    second = FixedProvider(
        (
            sxm(
                name="  nvidia   h100-sxm5  ",
                driver_version="550.54.15",
                runtime_version="12.8",
                interconnect="nvlink",
            ),
        )
    )
    one = build_capability_claim(
        first,
        policy(),
        DemoSigner(),
        observed_at_block=1_000,
        available_gpu_seconds=3_600,
        supported_evidence_tiers=(EvidenceTier.C,),
        nonce=NONCE,
    )
    two = build_capability_claim(
        second,
        policy(),
        DemoSigner(),
        observed_at_block=1_000,
        available_gpu_seconds=3_600,
        supported_evidence_tiers=(EvidenceTier.C,),
        nonce=NONCE,
    )

    assert one.content_digest() == two.content_digest()


def test_non_h100_and_ambiguous_h100_are_rejected() -> None:
    """Software names outside the explicit H100 variant map cannot become eligible claims."""
    with pytest.raises(CapabilityDiscoveryError, match="not an approved H100"):
        discover_inventory(FixedProvider((sxm(name="NVIDIA A100", memory_mb=81_920),)), policy())
    with pytest.raises(CapabilityDiscoveryError, match="cannot be normalized"):
        discover_inventory(FixedProvider((sxm(name="NVIDIA H100"),)), policy())


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"runtime_version": None}, "versions are required"),
        ({"driver_version": None}, "versions are required"),
        ({"interconnect": None}, "interconnect observation"),
        ({"driver_version": "not-a-version"}, "driver version"),
        ({"runtime_version": "13.0"}, "not approved"),
        ({"driver_version": "525.1"}, "incompatible"),
        ({"memory_mb": 70_000}, "below"),
        ({"interconnect": "PCIe"}, "inconsistent"),
        ({"interconnect": "custom-fabric"}, "unsupported"),
    ),
)
def test_missing_or_incompatible_h100_fields_are_rejected(
    changes: dict[str, object], error: str
) -> None:
    """Runtime, memory, and topology policy checks fail closed."""
    with pytest.raises(CapabilityDiscoveryError, match=error):
        discover_inventory(FixedProvider((sxm(**changes),)), policy())


def test_mixed_or_excessive_inventory_is_rejected() -> None:
    """A single claim cannot hide heterogeneous devices or exceed its protocol bound."""
    mixed = FixedProvider((sxm(), sxm(name="NVIDIA H100 PCIe", interconnect="PCIe")))
    with pytest.raises(CapabilityDiscoveryError, match="homogeneous"):
        discover_inventory(mixed, policy())
    with pytest.raises(CapabilityDiscoveryError, match="GPU count"):
        discover_inventory(FixedProvider(()), policy())
    with pytest.raises(CapabilityDiscoveryError, match="GPU count"):
        discover_inventory(FixedProvider(tuple(sxm() for _ in range(9))), policy())


def test_spoofed_or_failed_provider_data_is_sanitized() -> None:
    """Provider type violations and exceptions do not enter claims or leak raw details."""

    class SpoofedProvider:
        def discover(self) -> Sequence[GpuObservation]:
            return cast(Sequence[GpuObservation], ({"name": "NVIDIA H100 SXM"},))

    class BrokenProvider:
        def discover(self) -> Sequence[GpuObservation]:
            raise RuntimeError("host-specific sensitive details")

    with pytest.raises(CapabilityDiscoveryError, match="invalid observation"):
        discover_inventory(SpoofedProvider(), policy())
    with pytest.raises(CapabilityDiscoveryError, match="provider failed") as error:
        discover_inventory(BrokenProvider(), policy())
    assert "host-specific" not in str(error.value)


def test_observation_schema_excludes_identity_and_rejects_unsafe_text() -> None:
    """The provider contract has no serial/hostname fields and enforces bounded ASCII."""
    assert {item.name for item in fields(GpuObservation)} == {
        "name",
        "memory_mb",
        "driver_version",
        "runtime_version",
        "interconnect",
    }
    with pytest.raises(CapabilityDiscoveryError, match="ASCII"):
        sxm(name="NVIDIA H100-é")
    with pytest.raises(CapabilityDiscoveryError, match="text bounds"):
        sxm(name="NVIDIA H100 SXM\nprivate-host")


@pytest.mark.parametrize(
    "factory",
    (
        lambda: RuntimeCompatibilityRule(cast(int, "12"), (535, 54)),
        lambda: RuntimeCompatibilityRule(12, cast(tuple[int, ...], (535, "54"))),
        lambda: CapabilityPolicy(cast(tuple[RuntimeCompatibilityRule, ...], ())),
        lambda: CapabilityPolicy(
            (RuntimeCompatibilityRule(12, (535, 54)), RuntimeCompatibilityRule(12, (550, 1)))
        ),
        lambda: CapabilityPolicy((RuntimeCompatibilityRule(12, (535, 54)),), minimum_memory_mb=1),
        lambda: CapabilityPolicy((RuntimeCompatibilityRule(12, (535, 54)),), max_devices=9),
    ),
)
def test_compatibility_policy_schema_is_bounded(factory: Callable[[], object]) -> None:
    """Malformed compatibility policy cannot reach provider evaluation."""
    with pytest.raises(CapabilityDiscoveryError):
        factory()
