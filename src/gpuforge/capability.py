"""Minimal, explicitly self-reported GPU capability discovery."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, cast

from gpuforge.config import EvidenceTier
from gpuforge.identity import MessageSigner, generate_nonce, sign_message
from gpuforge.protocol import (
    UNSIGNED_SIGNATURE,
    CapabilityClaim,
    CapabilityTrust,
    SoftwareVersion,
)

_VERSION_PATTERN = re.compile(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,2}")
_MAX_DEVICE_NAME_BYTES = 128
_MAX_DEVICES = 8
_MAX_MEMORY_MB = 1_048_576


class CapabilityDiscoveryError(ValueError):
    """Raised when software inventory is missing, unsafe, or incompatible."""


class GpuClass(str, Enum):
    """Normalized GPU classes recognized by capability discovery."""

    H100_SXM = "h100_sxm"
    H100_PCIE = "h100_pcie"
    H100_NVL = "h100_nvl"
    UNSUPPORTED = "unsupported"


class GpuInterconnect(str, Enum):
    """Normalized miner-reported GPU interconnect classes."""

    NVLINK = "nvlink"
    PCIE = "pcie"


@dataclass(frozen=True, slots=True)
class GpuObservation:
    """Minimal provider output; deliberately excludes serial and host identity data."""

    name: str
    memory_mb: int
    driver_version: str | None
    runtime_version: str | None
    interconnect: str | None

    def __post_init__(self) -> None:
        _bounded_ascii("GPU name", self.name, _MAX_DEVICE_NAME_BYTES)
        if (
            isinstance(self.memory_mb, bool)
            or not isinstance(self.memory_mb, int)
            or not 0 <= self.memory_mb <= _MAX_MEMORY_MB
        ):
            raise CapabilityDiscoveryError("GPU memory observation is invalid")
        for label, value in (
            ("driver version", self.driver_version),
            ("runtime version", self.runtime_version),
            ("interconnect", self.interconnect),
        ):
            if value is not None:
                _bounded_ascii(label, value, 64)


class GpuInventoryProvider(Protocol):
    """Return only the minimal GPU observations required by this protocol."""

    def discover(self) -> Sequence[GpuObservation]: ...


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityRule:
    """Minimum driver version approved for one runtime major version."""

    runtime_major: int
    minimum_driver: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.runtime_major, bool)
            or not isinstance(self.runtime_major, int)
            or not 1 <= self.runtime_major <= 99
        ):
            raise CapabilityDiscoveryError("Runtime major version rule is invalid")
        if (
            not isinstance(self.minimum_driver, tuple)
            or not 2 <= len(self.minimum_driver) <= 3
            or any(
                isinstance(part, bool) or not isinstance(part, int) or not 0 <= part <= 9_999
                for part in self.minimum_driver
            )
        ):
            raise CapabilityDiscoveryError("Minimum driver version rule is invalid")


@dataclass(frozen=True, slots=True)
class CapabilityPolicy:
    """Operator-supplied H100 and driver/runtime compatibility policy."""

    runtime_rules: tuple[RuntimeCompatibilityRule, ...]
    minimum_memory_mb: int = 75_000
    max_devices: int = _MAX_DEVICES

    def __post_init__(self) -> None:
        if (
            not isinstance(self.runtime_rules, tuple)
            or not self.runtime_rules
            or not all(isinstance(rule, RuntimeCompatibilityRule) for rule in self.runtime_rules)
        ):
            raise CapabilityDiscoveryError("At least one runtime compatibility rule is required")
        majors = [rule.runtime_major for rule in self.runtime_rules]
        if len(majors) != len(set(majors)):
            raise CapabilityDiscoveryError("Runtime compatibility majors must be unique")
        if (
            isinstance(self.minimum_memory_mb, bool)
            or not isinstance(self.minimum_memory_mb, int)
            or not 1_024 <= self.minimum_memory_mb <= _MAX_MEMORY_MB
        ):
            raise CapabilityDiscoveryError("Minimum GPU memory policy is invalid")
        if (
            isinstance(self.max_devices, bool)
            or not isinstance(self.max_devices, int)
            or not 1 <= self.max_devices <= _MAX_DEVICES
        ):
            raise CapabilityDiscoveryError("Maximum GPU count policy is invalid")

    def minimum_driver_for(self, runtime_major: int) -> tuple[int, ...]:
        """Return the configured minimum or reject an unapproved runtime major."""
        for rule in self.runtime_rules:
            if rule.runtime_major == runtime_major:
                return rule.minimum_driver
        raise CapabilityDiscoveryError("GPU runtime is not approved by compatibility policy")


@dataclass(frozen=True, slots=True)
class NormalizedGpu:
    """Deterministic normalization of one untrusted software observation."""

    gpu_class: GpuClass
    model: str
    memory_mb: int
    driver_version: str
    runtime_version: str
    interconnect: GpuInterconnect


@dataclass(frozen=True, slots=True)
class NormalizedInventory:
    """Homogeneous inventory ready for a bounded capability claim."""

    gpu_class: GpuClass
    model: str
    count: int
    memory_mb: int
    driver_version: str
    runtime_version: str
    interconnect: GpuInterconnect


def discover_inventory(
    provider: GpuInventoryProvider,
    policy: CapabilityPolicy,
) -> NormalizedInventory:
    """Collect, normalize, and policy-check an explicitly untrusted inventory."""
    try:
        observed_value: object = provider.discover()
    except Exception:
        raise CapabilityDiscoveryError("GPU inventory provider failed") from None
    if isinstance(observed_value, (str, bytes)) or not isinstance(observed_value, Sequence):
        raise CapabilityDiscoveryError("GPU inventory provider returned an invalid collection")
    observed = cast(Sequence[object], observed_value)
    if not 1 <= len(observed) <= policy.max_devices:
        raise CapabilityDiscoveryError("GPU count is outside the configured policy")
    if not all(isinstance(device, GpuObservation) for device in observed):
        raise CapabilityDiscoveryError("GPU inventory contains an invalid observation")

    normalized = tuple(normalize_gpu(cast(GpuObservation, device), policy) for device in observed)
    first = normalized[0]
    if first.gpu_class is GpuClass.UNSUPPORTED:
        raise CapabilityDiscoveryError("GPU inventory is not an approved H100 class")
    for device in normalized[1:]:
        if (
            device.gpu_class is not first.gpu_class
            or device.memory_mb != first.memory_mb
            or device.driver_version != first.driver_version
            or device.runtime_version != first.runtime_version
            or device.interconnect is not first.interconnect
        ):
            raise CapabilityDiscoveryError("GPU inventory must be homogeneous")
    return NormalizedInventory(
        gpu_class=first.gpu_class,
        model=first.model,
        count=len(normalized),
        memory_mb=first.memory_mb,
        driver_version=first.driver_version,
        runtime_version=first.runtime_version,
        interconnect=first.interconnect,
    )


def normalize_gpu(observation: GpuObservation, policy: CapabilityPolicy) -> NormalizedGpu:
    """Normalize one provider observation without treating it as verified evidence."""
    if not isinstance(observation, GpuObservation):
        raise CapabilityDiscoveryError("GPU observation is invalid")
    gpu_class, model = _normalize_model(observation.name)
    if gpu_class is GpuClass.UNSUPPORTED:
        return NormalizedGpu(
            gpu_class=gpu_class,
            model="Unsupported GPU",
            memory_mb=observation.memory_mb,
            driver_version="0.0",
            runtime_version="0.0",
            interconnect=GpuInterconnect.PCIE,
        )
    if observation.memory_mb < policy.minimum_memory_mb:
        raise CapabilityDiscoveryError("GPU memory is below the configured H100 minimum")
    if observation.driver_version is None or observation.runtime_version is None:
        raise CapabilityDiscoveryError("GPU driver and runtime versions are required")
    if observation.interconnect is None:
        raise CapabilityDiscoveryError("GPU interconnect observation is required")

    driver_parts, driver = _normalize_version(observation.driver_version, "driver")
    runtime_parts, runtime = _normalize_version(observation.runtime_version, "runtime")
    minimum = policy.minimum_driver_for(runtime_parts[0])
    if _padded(driver_parts) < _padded(minimum):
        raise CapabilityDiscoveryError("GPU driver is incompatible with the runtime policy")
    interconnect = _normalize_interconnect(observation.interconnect)
    if gpu_class in {GpuClass.H100_SXM, GpuClass.H100_NVL}:
        expected = GpuInterconnect.NVLINK
    else:
        expected = GpuInterconnect.PCIE
    if interconnect is not expected:
        raise CapabilityDiscoveryError("GPU interconnect is inconsistent with the H100 variant")
    return NormalizedGpu(
        gpu_class=gpu_class,
        model=model,
        memory_mb=observation.memory_mb,
        driver_version=driver,
        runtime_version=runtime,
        interconnect=interconnect,
    )


def build_capability_claim(
    provider: GpuInventoryProvider,
    policy: CapabilityPolicy,
    signer: MessageSigner,
    *,
    observed_at_block: int,
    available_gpu_seconds: int,
    supported_evidence_tiers: tuple[EvidenceTier, ...],
    nonce: str | None = None,
) -> CapabilityClaim:
    """Build and sign a bounded self-reported capability claim."""
    inventory = discover_inventory(provider, policy)
    unsigned = CapabilityClaim(
        miner_hotkey=signer.hotkey,
        gpu_count=inventory.count,
        gpu_model=inventory.model,
        gpu_memory_mb=inventory.memory_mb,
        gpu_interconnect=inventory.interconnect.value,
        discovery_trust=CapabilityTrust.SELF_REPORTED,
        runtime_versions=(
            SoftwareVersion("cuda", inventory.runtime_version),
            SoftwareVersion("driver", inventory.driver_version),
        ),
        supported_evidence_tiers=supported_evidence_tiers,
        available_gpu_seconds=available_gpu_seconds,
        nonce=nonce or generate_nonce(),
        observed_at_block=observed_at_block,
        signature=UNSIGNED_SIGNATURE,
    )
    signed = sign_message(unsigned, signer)
    if not isinstance(signed, CapabilityClaim):  # pragma: no cover - type-preserving contract
        raise CapabilityDiscoveryError("Capability signer returned an unexpected message type")
    return signed


def _normalize_model(name: str) -> tuple[GpuClass, str]:
    normalized = " ".join(name.casefold().replace("-", " ").split())
    if "nvidia" not in normalized or "h100" not in normalized:
        return GpuClass.UNSUPPORTED, "Unsupported GPU"
    if "nvl" in normalized:
        return GpuClass.H100_NVL, "NVIDIA H100 NVL"
    if "pcie" in normalized or "pci express" in normalized:
        return GpuClass.H100_PCIE, "NVIDIA H100 PCIe"
    if "sxm" in normalized or "80gb hbm3" in normalized:
        return GpuClass.H100_SXM, "NVIDIA H100 SXM"
    raise CapabilityDiscoveryError("H100 variant cannot be normalized safely")


def _normalize_interconnect(value: str) -> GpuInterconnect:
    normalized = " ".join(value.casefold().replace("-", " ").split())
    if normalized in {"nvlink", "nvswitch", "nvlink nvswitch"}:
        return GpuInterconnect.NVLINK
    if normalized in {"pcie", "pci express"}:
        return GpuInterconnect.PCIE
    raise CapabilityDiscoveryError("GPU interconnect is unsupported")


def _normalize_version(value: str, label: str) -> tuple[tuple[int, ...], str]:
    if _VERSION_PATTERN.fullmatch(value.strip()) is None:
        raise CapabilityDiscoveryError(f"GPU {label} version is invalid")
    parts = tuple(int(part) for part in value.strip().split("."))
    return parts, ".".join(str(part) for part in parts)


def _padded(value: tuple[int, ...]) -> tuple[int, int, int]:
    return cast(tuple[int, int, int], (value + (0, 0, 0))[:3])


def _bounded_ascii(label: str, value: object, maximum_bytes: int) -> None:
    if not isinstance(value, str):
        raise CapabilityDiscoveryError(f"{label} must be text")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise CapabilityDiscoveryError(f"{label} must contain only ASCII") from None
    if not 1 <= len(encoded) <= maximum_bytes or any(byte < 32 or byte == 127 for byte in encoded):
        raise CapabilityDiscoveryError(f"{label} violates its text bounds")
