"""Typed runtime configuration with secure, explicit defaults."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypeVar, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

MAX_CONFIG_BYTES = 64 * 1024
MAX_NETUID = 65_535
MAX_ARTIFACT_BYTES = 1024**4

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "accesstoken",
        "accesskey",
        "apikey",
        "apisecret",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "mnemonic",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secret",
        "secrets",
        "seedphrase",
        "token",
    }
)
_SENSITIVE_FIELD_MARKERS = (
    "accesskey",
    "accesstoken",
    "apikey",
    "apisecret",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "credential",
    "mnemonic",
    "password",
    "passwd",
    "privatekey",
    "refreshtoken",
    "secret",
    "seedphrase",
)


class ConfigurationError(ValueError):
    """Raised when configuration violates the runtime safety policy."""


class Role(str, Enum):
    """Process role selected for this runtime."""

    MINER = "miner"
    VALIDATOR = "validator"
    PUBLISHER = "publisher"


class Network(str, Enum):
    """Supported network selections."""

    LOCAL = "local"
    TEST = "test"
    FINNEY = "finney"


class EvidenceTier(str, Enum):
    """Minimum accepted execution-evidence tier."""

    A = "a"
    B = "b"
    C = "c"


class NetworkPolicy(str, Enum):
    """Outbound network policy for publisher-supplied workloads."""

    DENY = "deny"
    ALLOWLIST = "allowlist"


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeSecrets:
    """Secrets injected from the process environment and never serialized."""

    artifact_access_token: str | None = field(default=None, repr=False)
    attestation_access_token: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> RuntimeSecrets:
        """Load the supported secret values from runtime environment injection."""
        source = os.environ if environ is None else environ
        return cls(
            artifact_access_token=source.get("GPUFORGE_ARTIFACT_ACCESS_TOKEN") or None,
            attestation_access_token=source.get("GPUFORGE_ATTESTATION_ACCESS_TOKEN") or None,
        )

    def values(self) -> tuple[str, ...]:
        """Return configured values for use by the log-redaction layer."""
        return tuple(
            value for value in (self.artifact_access_token, self.attestation_access_token) if value
        )

    def to_safe_dict(self) -> dict[str, bool]:
        """Report only whether each secret is configured."""
        return {
            "artifact_access_token_configured": self.artifact_access_token is not None,
            "attestation_access_token_configured": self.attestation_access_token is not None,
        }

    def __repr__(self) -> str:
        """Prevent interactive output and logs from exposing secret values."""
        artifact = "<redacted>" if self.artifact_access_token else "<unset>"
        attestation = "<redacted>" if self.attestation_access_token else "<unset>"
        return (
            "RuntimeSecrets("
            f"artifact_access_token={artifact}, "
            f"attestation_access_token={attestation})"
        )


@dataclass(frozen=True, slots=True)
class TimeoutSettings:
    """Bounded operation timeouts, in seconds."""

    request_seconds: int = 30
    job_seconds: int = 3_600
    shutdown_seconds: int = 30

    def __post_init__(self) -> None:
        _validate_bounded_int("timeouts.request_seconds", self.request_seconds, 1, 300)
        _validate_bounded_int("timeouts.job_seconds", self.job_seconds, 60, 86_400)
        _validate_bounded_int("timeouts.shutdown_seconds", self.shutdown_seconds, 1, 300)


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Local storage boundaries for untrusted job artifacts."""

    root: Path = Path(".gpuforge/runtime")
    max_artifact_bytes: int = 20 * 1024**3
    max_checkpoint_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise ConfigurationError("Invalid type for 'storage.root'")
        if self.root.anchor or ".." in self.root.parts or str(self.root) in {"", "."}:
            raise ConfigurationError("'storage.root' must be a non-root relative path")
        _validate_bounded_int(
            "storage.max_artifact_bytes", self.max_artifact_bytes, 1, MAX_ARTIFACT_BYTES
        )
        _validate_bounded_int(
            "storage.max_checkpoint_bytes", self.max_checkpoint_bytes, 1, MAX_ARTIFACT_BYTES
        )
        if self.max_checkpoint_bytes > self.max_artifact_bytes:
            raise ConfigurationError(
                "'storage.max_checkpoint_bytes' must not exceed 'storage.max_artifact_bytes'"
            )


@dataclass(frozen=True, slots=True)
class EvidenceSettings:
    """Evidence verification policy."""

    minimum_tier: EvidenceTier = EvidenceTier.C
    fail_closed: bool = True
    max_age_seconds: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_tier, EvidenceTier):
            raise ConfigurationError("Invalid type for 'evidence.minimum_tier'")
        if not isinstance(self.fail_closed, bool):
            raise ConfigurationError("Invalid type for 'evidence.fail_closed'")
        _validate_bounded_int("evidence.max_age_seconds", self.max_age_seconds, 1, 3_600)


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Resource ceilings for the feature-gated isolated workload runner."""

    cpu_cores: int = 8
    memory_mb: int = 32_768
    gpu_count: int = 1
    pids_limit: int = 512
    writable_bytes: int = 20 * 1024**3
    network_policy: NetworkPolicy = NetworkPolicy.DENY

    def __post_init__(self) -> None:
        _validate_bounded_int("sandbox.cpu_cores", self.cpu_cores, 1, 256)
        _validate_bounded_int("sandbox.memory_mb", self.memory_mb, 1_024, 2_097_152)
        _validate_bounded_int("sandbox.gpu_count", self.gpu_count, 0, 8)
        _validate_bounded_int("sandbox.pids_limit", self.pids_limit, 64, 65_536)
        _validate_bounded_int("sandbox.writable_bytes", self.writable_bytes, 1, MAX_ARTIFACT_BYTES)
        if not isinstance(self.network_policy, NetworkPolicy):
            raise ConfigurationError("Invalid type for 'sandbox.network_policy'")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Complete typed settings for a single GPUForge process."""

    role: Role
    network: Network
    netuid: int
    timeouts: TimeoutSettings = field(default_factory=TimeoutSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    evidence: EvidenceSettings = field(default_factory=EvidenceSettings)
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    secrets: RuntimeSecrets = field(default_factory=RuntimeSecrets.from_environment, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            raise ConfigurationError("Invalid type for 'role'")
        if not isinstance(self.network, Network):
            raise ConfigurationError("Invalid type for 'network'")
        _validate_bounded_int("netuid", self.netuid, 1, MAX_NETUID)
        if self.role is Role.MINER and self.sandbox.gpu_count < 1:
            raise ConfigurationError("A miner requires at least one sandbox GPU")
        if self.network is not Network.LOCAL:
            if self.evidence.minimum_tier is EvidenceTier.C:
                raise ConfigurationError("Evidence tier C is restricted to the local network")
            if not self.evidence.fail_closed:
                raise ConfigurationError("Non-local evidence verification must fail closed")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeSettings:
        """Validate an untrusted mapping and construct runtime settings."""
        reject_secret_like_fields(data, context="configuration")
        _reject_unknown_keys(
            data,
            {"role", "network", "netuid", "timeouts", "storage", "evidence", "sandbox"},
            "configuration",
        )

        role = _parse_enum(Role, _required_value(data, "role"), "role")
        network = _parse_enum(Network, _required_value(data, "network"), "network")
        netuid = _parse_int(_required_value(data, "netuid"), "netuid")
        timeouts_data = _optional_section(data, "timeouts")
        storage_data = _optional_section(data, "storage")
        evidence_data = _optional_section(data, "evidence")
        sandbox_data = _optional_section(data, "sandbox")

        _reject_unknown_keys(
            timeouts_data,
            {"request_seconds", "job_seconds", "shutdown_seconds"},
            "timeouts",
        )
        _reject_unknown_keys(
            storage_data,
            {"root", "max_artifact_bytes", "max_checkpoint_bytes"},
            "storage",
        )
        _reject_unknown_keys(
            evidence_data,
            {"minimum_tier", "fail_closed", "max_age_seconds"},
            "evidence",
        )
        _reject_unknown_keys(
            sandbox_data,
            {
                "cpu_cores",
                "memory_mb",
                "gpu_count",
                "pids_limit",
                "writable_bytes",
                "network_policy",
            },
            "sandbox",
        )

        timeouts = TimeoutSettings(
            request_seconds=_optional_int(timeouts_data, "request_seconds", 30),
            job_seconds=_optional_int(timeouts_data, "job_seconds", 3_600),
            shutdown_seconds=_optional_int(timeouts_data, "shutdown_seconds", 30),
        )
        storage = StorageSettings(
            root=_optional_relative_path(storage_data, "root", Path(".gpuforge/runtime")),
            max_artifact_bytes=_optional_int(storage_data, "max_artifact_bytes", 20 * 1024**3),
            max_checkpoint_bytes=_optional_int(storage_data, "max_checkpoint_bytes", 8 * 1024**3),
        )
        evidence = EvidenceSettings(
            minimum_tier=_optional_enum(
                EvidenceTier, evidence_data, "minimum_tier", EvidenceTier.C
            ),
            fail_closed=_optional_bool(evidence_data, "fail_closed", True),
            max_age_seconds=_optional_int(evidence_data, "max_age_seconds", 300),
        )
        sandbox = SandboxSettings(
            cpu_cores=_optional_int(sandbox_data, "cpu_cores", 8),
            memory_mb=_optional_int(sandbox_data, "memory_mb", 32_768),
            gpu_count=_optional_int(sandbox_data, "gpu_count", 1),
            pids_limit=_optional_int(sandbox_data, "pids_limit", 512),
            writable_bytes=_optional_int(sandbox_data, "writable_bytes", 20 * 1024**3),
            network_policy=_optional_enum(
                NetworkPolicy, sandbox_data, "network_policy", NetworkPolicy.DENY
            ),
        )

        return cls(
            role=role,
            network=network,
            netuid=netuid,
            timeouts=timeouts,
            storage=storage,
            evidence=evidence,
            sandbox=sandbox,
            secrets=RuntimeSecrets.from_environment(environ),
        )

    @classmethod
    def from_toml(
        cls,
        path: Path,
        *,
        expected_role: Role | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeSettings:
        """Load a bounded TOML file without exposing its contents in failures."""
        try:
            if path.stat().st_size > MAX_CONFIG_BYTES:
                raise ConfigurationError("Configuration file exceeds the size limit")
            raw = path.read_bytes()
        except ConfigurationError:
            raise
        except OSError:
            raise ConfigurationError("Configuration file is unavailable") from None

        try:
            parsed = tomllib.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            raise ConfigurationError("Configuration file is not valid UTF-8 TOML") from None

        settings = cls.from_mapping(parsed, environ=environ)
        if expected_role is not None and settings.role is not expected_role:
            raise ConfigurationError("Configuration role does not match the selected command")
        return settings

    def to_safe_dict(self) -> dict[str, object]:
        """Serialize settings without including any secret value."""
        return {
            "role": self.role.value,
            "network": self.network.value,
            "netuid": self.netuid,
            "timeouts": {
                "request_seconds": self.timeouts.request_seconds,
                "job_seconds": self.timeouts.job_seconds,
                "shutdown_seconds": self.timeouts.shutdown_seconds,
            },
            "storage": {
                "root": self.storage.root.as_posix(),
                "max_artifact_bytes": self.storage.max_artifact_bytes,
                "max_checkpoint_bytes": self.storage.max_checkpoint_bytes,
            },
            "evidence": {
                "minimum_tier": self.evidence.minimum_tier.value,
                "fail_closed": self.evidence.fail_closed,
                "max_age_seconds": self.evidence.max_age_seconds,
            },
            "sandbox": {
                "cpu_cores": self.sandbox.cpu_cores,
                "memory_mb": self.sandbox.memory_mb,
                "gpu_count": self.sandbox.gpu_count,
                "pids_limit": self.sandbox.pids_limit,
                "writable_bytes": self.sandbox.writable_bytes,
                "network_policy": self.sandbox.network_policy.value,
            },
            "secrets": self.secrets.to_safe_dict(),
        }


def reject_secret_like_fields(value: object, *, context: str) -> None:
    """Reject fields that could place a credential in persisted protocol data."""
    if isinstance(value, Mapping):
        for raw_key, nested_value in value.items():
            if not isinstance(raw_key, str):
                raise ConfigurationError(f"{context} contains a non-string field name")
            compact_key = re.sub(r"[^a-z0-9]", "", raw_key.casefold())
            if compact_key in _SENSITIVE_FIELD_NAMES or any(
                marker in compact_key for marker in _SENSITIVE_FIELD_MARKERS
            ):
                raise ConfigurationError(f"{context} contains a prohibited sensitive field")
            reject_secret_like_fields(nested_value, context=context)
    elif isinstance(value, list | tuple):
        for item in value:
            reject_secret_like_fields(item, context=context)


def _validate_bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Invalid type for '{name}'")
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"'{name}' is outside the permitted range")


def _required_value(data: Mapping[str, object], name: str) -> object:
    if name not in data:
        raise ConfigurationError(f"Missing required setting '{name}'")
    return data[name]


def _reject_unknown_keys(data: Mapping[str, object], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown setting in '{section}': '{unknown[0]}'")


def _optional_section(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Invalid type for '{name}'")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"'{name}' contains a non-string field name")
    return cast(Mapping[str, object], value)


EnumType = TypeVar("EnumType", bound=Enum)


def _parse_enum(enum_type: type[EnumType], value: object, name: str) -> EnumType:
    if not isinstance(value, str):
        raise ConfigurationError(f"Invalid type for '{name}'")
    try:
        return enum_type(value)
    except ValueError:
        raise ConfigurationError(f"Invalid value for '{name}'") from None


def _optional_enum(
    enum_type: type[EnumType],
    data: Mapping[str, object],
    name: str,
    default: EnumType,
) -> EnumType:
    if name not in data:
        return default
    return _parse_enum(enum_type, data[name], name)


def _parse_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"Invalid type for '{name}'")
    return value


def _optional_int(data: Mapping[str, object], name: str, default: int) -> int:
    if name not in data:
        return default
    return _parse_int(data[name], name)


def _optional_bool(data: Mapping[str, object], name: str, default: bool) -> bool:
    if name not in data:
        return default
    value = data[name]
    if not isinstance(value, bool):
        raise ConfigurationError(f"Invalid type for '{name}'")
    return value


def _optional_relative_path(data: Mapping[str, object], name: str, default: Path) -> Path:
    if name not in data:
        return default
    value = data[name]
    if not isinstance(value, str):
        raise ConfigurationError(f"Invalid type for '{name}'")
    if not value or len(value) > 200:
        raise ConfigurationError(f"Invalid value for '{name}'")
    return Path(value)
