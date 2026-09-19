"""Tests for typed configuration and secret-handling policy."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from gpuforge.config import (
    ConfigurationError,
    EvidenceSettings,
    EvidenceTier,
    Network,
    NetworkPolicy,
    Role,
    RuntimeSecrets,
    RuntimeSettings,
    SandboxSettings,
    StorageSettings,
    TimeoutSettings,
    reject_secret_like_fields,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def valid_config(*, role: str = "miner") -> dict[str, object]:
    """Return a complete local-only configuration mapping."""
    return {
        "role": role,
        "network": "local",
        "netuid": 1,
        "timeouts": {
            "request_seconds": 30,
            "job_seconds": 3_600,
            "shutdown_seconds": 30,
        },
        "storage": {
            "root": f".gpuforge/{role}",
            "max_artifact_bytes": 20 * 1024**3,
            "max_checkpoint_bytes": 8 * 1024**3,
        },
        "evidence": {
            "minimum_tier": "c",
            "fail_closed": True,
            "max_age_seconds": 300,
        },
        "sandbox": {
            "cpu_cores": 8,
            "memory_mb": 32_768,
            "gpu_count": 1 if role == "miner" else 0,
            "pids_limit": 512,
            "writable_bytes": 20 * 1024**3,
            "network_policy": "deny",
        },
    }


@pytest.mark.parametrize(
    ("filename", "expected_role"),
    (("miner.local.toml", Role.MINER), ("validator.local.toml", Role.VALIDATOR)),
)
def test_checked_in_local_config_is_valid(filename: str, expected_role: Role) -> None:
    """Both role examples load without environment secrets."""
    settings = RuntimeSettings.from_toml(
        REPOSITORY_ROOT / "config" / filename,
        expected_role=expected_role,
        environ={},
    )

    assert settings.role is expected_role
    assert settings.network is Network.LOCAL
    assert settings.secrets.to_safe_dict() == {
        "artifact_access_token_configured": False,
        "attestation_access_token_configured": False,
    }


def test_omitted_sections_use_secure_defaults() -> None:
    """Optional sections default to bounded, fail-closed local values."""
    settings = RuntimeSettings.from_mapping(
        {"role": "miner", "network": "local", "netuid": 1}, environ={}
    )

    assert settings.timeouts == TimeoutSettings()
    assert settings.storage == StorageSettings()
    assert settings.evidence == EvidenceSettings()
    assert settings.sandbox.network_policy is NetworkPolicy.DENY
    assert settings.evidence.fail_closed is True


@pytest.mark.parametrize("missing", ("role", "network", "netuid"))
def test_required_setting_cannot_be_omitted(missing: str) -> None:
    """Role, network, and subnet identity require explicit selection."""
    config = valid_config()
    del config[missing]

    with pytest.raises(ConfigurationError, match="Missing required setting"):
        RuntimeSettings.from_mapping(config, environ={})


@pytest.mark.parametrize(
    "factory",
    (
        lambda: TimeoutSettings(request_seconds=0),
        lambda: TimeoutSettings(job_seconds=30),
        lambda: TimeoutSettings(shutdown_seconds=301),
        lambda: StorageSettings(max_artifact_bytes=0),
        lambda: StorageSettings(max_artifact_bytes=1, max_checkpoint_bytes=2),
        lambda: SandboxSettings(cpu_cores=0),
        lambda: SandboxSettings(memory_mb=512),
        lambda: SandboxSettings(gpu_count=9),
        lambda: SandboxSettings(pids_limit=10),
        lambda: EvidenceSettings(max_age_seconds=0),
    ),
)
def test_invalid_ranges_are_rejected(factory: Callable[[], object]) -> None:
    """Resource and timeout values cannot escape their safety bounds."""
    with pytest.raises(ConfigurationError):
        factory()


@pytest.mark.parametrize(
    "storage_root",
    (Path("."), Path("../outside"), REPOSITORY_ROOT / "private"),
)
def test_storage_root_must_be_scoped_and_relative(storage_root: Path) -> None:
    """Configuration cannot target a root, parent, or absolute host path."""
    with pytest.raises(ConfigurationError, match="relative path"):
        StorageSettings(root=storage_root)


def test_miner_requires_a_gpu_allocation() -> None:
    """A miner configuration cannot advertise zero runnable GPUs."""
    with pytest.raises(ConfigurationError, match="at least one"):
        RuntimeSettings(
            role=Role.MINER,
            network=Network.LOCAL,
            netuid=1,
            sandbox=SandboxSettings(gpu_count=0),
            secrets=RuntimeSecrets.from_environment({}),
        )


def test_non_local_policy_rejects_weak_or_fail_open_evidence() -> None:
    """Remote networks require stronger and fail-closed evidence settings."""
    with pytest.raises(ConfigurationError, match="tier C"):
        RuntimeSettings(
            role=Role.VALIDATOR,
            network=Network.TEST,
            netuid=1,
            evidence=EvidenceSettings(minimum_tier=EvidenceTier.C),
            secrets=RuntimeSecrets.from_environment({}),
        )

    with pytest.raises(ConfigurationError, match="fail closed"):
        RuntimeSettings(
            role=Role.VALIDATOR,
            network=Network.FINNEY,
            netuid=1,
            evidence=EvidenceSettings(minimum_tier=EvidenceTier.B, fail_closed=False),
            secrets=RuntimeSecrets.from_environment({}),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "password",
        "api_key",
        "access-token",
        "privateKey",
        "seed_phrase",
        "mnemonic",
        "credentials",
        "secrets",
        "artifact_access_token",
        "database_password",
    ),
)
def test_secret_like_manifest_fields_are_rejected(field_name: str) -> None:
    """Persisted job-like data cannot contain credential-shaped fields."""
    sensitive_value = "must-never-appear"
    manifest = {"job_id": "example", "inputs": {field_name: sensitive_value}}

    with pytest.raises(ConfigurationError) as error:
        reject_secret_like_fields(manifest, context="job manifest")

    assert sensitive_value not in str(error.value)


def test_non_secret_training_token_count_is_allowed() -> None:
    """Training metrics named in tokens are not mistaken for credentials."""
    reject_secret_like_fields({"token_count": 512}, context="job manifest")


def test_config_cannot_contain_a_secret_section() -> None:
    """Secrets must arrive from the runtime environment, never persisted config."""
    config = valid_config()
    config["secrets"] = {"artifact_access_token": "must-never-appear"}

    with pytest.raises(ConfigurationError) as error:
        RuntimeSettings.from_mapping(config, environ={})

    assert "must-never-appear" not in str(error.value)


def test_environment_secrets_are_redacted_from_repr_and_serialization() -> None:
    """Safe output reports presence without copying secret values."""
    artifact_secret = "artifact-secret-canary"
    attestation_secret = "attestation-secret-canary"
    settings = RuntimeSettings.from_mapping(
        valid_config(),
        environ={
            "GPUFORGE_ARTIFACT_ACCESS_TOKEN": artifact_secret,
            "GPUFORGE_ATTESTATION_ACCESS_TOKEN": attestation_secret,
        },
    )

    safe_output = json.dumps(settings.to_safe_dict(), sort_keys=True)
    representation = repr(settings.secrets)
    assert artifact_secret not in safe_output
    assert attestation_secret not in safe_output
    assert artifact_secret not in representation
    assert attestation_secret not in representation
    assert settings.to_safe_dict()["secrets"] == {
        "artifact_access_token_configured": True,
        "attestation_access_token_configured": True,
    }


def test_unknown_and_invalid_values_do_not_echo_input() -> None:
    """Errors identify the field without echoing an unsafe value."""
    unsafe_value = "unsafe-value-canary"
    config = valid_config()
    config["network"] = unsafe_value

    with pytest.raises(ConfigurationError) as error:
        RuntimeSettings.from_mapping(config, environ={})

    assert unsafe_value not in str(error.value)


def test_wrong_role_file_is_rejected() -> None:
    """A validator cannot accidentally start from a miner configuration."""
    with pytest.raises(ConfigurationError, match="does not match"):
        RuntimeSettings.from_toml(
            REPOSITORY_ROOT / "config" / "miner.local.toml",
            expected_role=Role.VALIDATOR,
            environ={},
        )


def test_invalid_toml_does_not_echo_contents(tmp_path: Path) -> None:
    """Parse failures never include raw configuration text."""
    unsafe_value = "parse-secret-canary"
    config_file = tmp_path / "invalid.toml"
    config_file.write_text(f'broken = "{unsafe_value}', encoding="utf-8")

    with pytest.raises(ConfigurationError) as error:
        RuntimeSettings.from_toml(config_file, environ={})

    assert unsafe_value not in str(error.value)


def test_unavailable_config_has_generic_error(tmp_path: Path) -> None:
    """Missing configuration paths produce a stable, non-leaking error."""
    with pytest.raises(ConfigurationError, match="unavailable"):
        RuntimeSettings.from_toml(tmp_path / "missing.toml", environ={})


def test_oversized_config_is_rejected(tmp_path: Path) -> None:
    """Configuration input is bounded before parsing."""
    config_file = tmp_path / "oversized.toml"
    config_file.write_bytes(b"x" * (64 * 1024 + 1))

    with pytest.raises(ConfigurationError, match="size limit"):
        RuntimeSettings.from_toml(config_file, environ={})


@pytest.mark.parametrize(
    "factory",
    (
        lambda: StorageSettings(root=cast(Path, "relative")),
        lambda: EvidenceSettings(minimum_tier=cast(EvidenceTier, "c")),
        lambda: EvidenceSettings(fail_closed=cast(bool, 1)),
        lambda: SandboxSettings(network_policy=cast(NetworkPolicy, "deny")),
        lambda: RuntimeSettings(
            role=cast(Role, "miner"),
            network=Network.LOCAL,
            netuid=1,
            secrets=RuntimeSecrets.from_environment({}),
        ),
        lambda: RuntimeSettings(
            role=Role.MINER,
            network=cast(Network, "local"),
            netuid=1,
            secrets=RuntimeSecrets.from_environment({}),
        ),
        lambda: TimeoutSettings(request_seconds=cast(int, True)),
    ),
)
def test_direct_construction_rejects_invalid_runtime_types(
    factory: Callable[[], object],
) -> None:
    """Runtime validation remains active even if callers bypass type checking."""
    with pytest.raises(ConfigurationError):
        factory()


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    (
        (None, "role", 1),
        (None, "netuid", "1"),
        ("evidence", "fail_closed", "true"),
        ("storage", "root", 1),
        ("storage", "root", ""),
    ),
)
def test_mapping_rejects_invalid_field_types(
    section: str | None, field: str, invalid_value: object
) -> None:
    """Untrusted mappings cannot rely on implicit conversions."""
    config = valid_config()
    if section is None:
        config[field] = invalid_value
    else:
        section_data = config[section]
        assert isinstance(section_data, dict)
        section_data[field] = invalid_value

    with pytest.raises(ConfigurationError):
        RuntimeSettings.from_mapping(config, environ={})


def test_mapping_rejects_unknown_or_malformed_sections() -> None:
    """Unknown keys and non-mapping sections fail closed."""
    unknown = valid_config()
    unknown["debug_mode"] = True
    with pytest.raises(ConfigurationError, match="Unknown setting"):
        RuntimeSettings.from_mapping(unknown, environ={})

    malformed = valid_config()
    malformed["timeouts"] = "not-a-table"
    with pytest.raises(ConfigurationError, match="Invalid type"):
        RuntimeSettings.from_mapping(malformed, environ={})


def test_nested_sequences_and_non_string_keys_are_checked() -> None:
    """Secret-field traversal covers nested sequences and rejects invalid keys."""
    with pytest.raises(ConfigurationError, match="sensitive field"):
        reject_secret_like_fields(
            [{"safe": ({"private_key": "must-never-appear"},)}],
            context="job manifest",
        )

    invalid_mapping = cast(dict[str, object], {1: "invalid"})
    with pytest.raises(ConfigurationError, match="non-string"):
        reject_secret_like_fields(invalid_mapping, context="job manifest")
