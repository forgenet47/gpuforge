"""Sandbox tests validate commands through an injected runner only."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest

from gpuforge.config import NetworkPolicy
from gpuforge.sandbox import (
    ContainerJob,
    ContainerSandboxPolicy,
    LinuxContainerBackend,
    SandboxCommandResult,
    SandboxCommandTimeout,
    SandboxPolicyError,
    SandboxUnavailable,
    SubprocessSandboxRunner,
)

IMAGE = "registry.example/trainer@sha256:" + "11" * 32
NAME = "gpuforge-0123456789abcdef"


def job(**changes: object) -> ContainerJob:
    values: dict[str, object] = {
        "image": IMAGE,
        "argv": ("python", "/opt/job/train.py"),
    }
    values.update(changes)
    return ContainerJob(**values)  # type: ignore[arg-type]


def policy(**changes: object) -> ContainerSandboxPolicy:
    values: dict[str, object] = {
        "cpu_cores": 8,
        "memory_mb": 32_768,
        "gpu_count": 1,
        "pids_limit": 256,
        "writable_bytes": 1024**3,
        "max_runtime_seconds": 600,
        "network_policy": NetworkPolicy.DENY,
    }
    values.update(changes)
    return ContainerSandboxPolicy(**values)  # type: ignore[arg-type]


@dataclass
class FakeRunner:
    """Record runtime commands and optionally simulate a deadline."""

    outcome: SandboxCommandResult = SandboxCommandResult(0, b"ok", b"")
    time_out_first: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> SandboxCommandResult:
        assert timeout_seconds > 0
        assert max_output_bytes > 0
        self.calls.append(tuple(argv))
        if self.time_out_first and len(self.calls) == 1:
            raise SandboxCommandTimeout
        return self.outcome


def backend(runner: FakeRunner, **changes: object) -> LinuxContainerBackend:
    values: dict[str, object] = {
        "runner": runner,
        "enabled": True,
        "executable": "/usr/bin/docker",
        "platform": "linux",
        "name_factory": lambda: NAME,
        "clock": iter((10.0, 10.25)).__next__,
    }
    values.update(changes)
    return LinuxContainerBackend(**values)  # type: ignore[arg-type]


def option_value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def test_command_enforces_least_privilege_and_resource_bounds() -> None:
    """The launcher emits only its closed, security-reviewed option set."""
    runner = FakeRunner()
    command = backend(runner).build_command(job(), policy(), container_name=NAME)

    assert command[:2] == ("/usr/bin/docker", "run")
    assert "--read-only" in command
    assert option_value(command, "--user") == "65532:65532"
    assert option_value(command, "--cap-drop") == "ALL"
    assert "no-new-privileges=true" in command
    assert option_value(command, "--pids-limit") == "256"
    assert option_value(command, "--cpus") == "8"
    assert option_value(command, "--memory") == "32768m"
    assert option_value(command, "--memory-swap") == "32768m"
    assert option_value(command, "--network") == "none"
    assert option_value(command, "--ipc") == "none"
    assert "count=1" in command
    assert "--pull" in command and "never" in command
    assert IMAGE in command


def test_writable_storage_is_bounded_tmpfs_with_no_host_mount() -> None:
    command = backend(FakeRunner()).build_command(job(), policy(), container_name=NAME)
    tmpfs_values = [command[index + 1] for index, value in enumerate(command) if value == "--tmpfs"]

    assert any(
        value.startswith("/job/scratch:") and "size=1073741824" in value for value in tmpfs_values
    )
    assert all(
        "nosuid" in value and "nodev" in value and "noexec" in value for value in tmpfs_values
    )
    assert "--mount" not in command
    assert "--volume" not in command
    assert "/var/run/docker.sock" not in command


def test_host_mount_device_and_mutable_image_requests_are_rejected() -> None:
    """Publisher values cannot add host filesystem or raw device access."""
    with pytest.raises(SandboxPolicyError, match="host mounts"):
        job(host_mounts=(Path("/"),))
    with pytest.raises(SandboxPolicyError, match="device access"):
        job(device_requests=("/dev/nvidiactl",))
    with pytest.raises(SandboxPolicyError, match="immutable"):
        job(image="registry.example/trainer:latest")


def test_network_allowlist_fails_closed_until_mediated_backend_exists() -> None:
    """No raw outbound network mode can be enabled by a publisher workload."""
    with pytest.raises(SandboxPolicyError, match="mediated"):
        backend(FakeRunner()).build_command(
            job(),
            policy(network_policy=NetworkPolicy.ALLOWLIST),
            container_name=NAME,
        )


def test_seccomp_and_apparmor_hooks_cannot_be_unconfined() -> None:
    command = backend(FakeRunner()).build_command(
        job(),
        policy(
            seccomp_profile=PurePosixPath("/etc/gpuforge/seccomp.json"),
            apparmor_profile="gpuforge-training",
        ),
        container_name=NAME,
    )
    assert "seccomp=/etc/gpuforge/seccomp.json" in command
    assert "apparmor=gpuforge-training" in command
    with pytest.raises(SandboxPolicyError, match="AppArmor"):
        policy(apparmor_profile="unconfined")


def test_forced_termination_kills_and_removes_timed_out_container() -> None:
    """A deadline triggers explicit engine-side kill and forced cleanup."""
    runner = FakeRunner(time_out_first=True)
    result = backend(runner).run(job(), policy(max_runtime_seconds=1))

    assert result.timed_out
    assert result.exit_code is None
    assert runner.calls[1] == ("/usr/bin/docker", "kill", NAME)
    assert runner.calls[2] == ("/usr/bin/docker", "rm", "--force", NAME)


def test_result_contains_only_digests_and_bounded_output_flags() -> None:
    """Host or workload output is not retained in the execution result."""
    runner = FakeRunner(
        outcome=SandboxCommandResult(
            7,
            b"publisher-output",
            b"sanitized-error",
            stdout_truncated=True,
        )
    )
    result = backend(runner).run(job(), policy())

    assert result.exit_code == 7
    assert not result.timed_out
    assert result.duration_ms == 250
    assert result.stdout_digest == f"sha256:{hashlib.sha256(b'publisher-output').hexdigest()}"
    assert result.stderr_digest == f"sha256:{hashlib.sha256(b'sanitized-error').hexdigest()}"
    assert result.stdout_truncated
    assert "publisher-output" not in repr(result)


def test_backend_is_feature_gated_and_linux_only() -> None:
    """Constructing the backend never implies permission to execute a container."""
    with pytest.raises(SandboxUnavailable):
        backend(FakeRunner(), enabled=False).run(job(), policy())
    with pytest.raises(SandboxUnavailable):
        backend(FakeRunner(), platform="win32").run(job(), policy())


@pytest.mark.parametrize(
    "argument",
    ("", "line\nbreak", "nul\x00byte"),
)
def test_unsafe_container_arguments_are_rejected(argument: str) -> None:
    with pytest.raises(SandboxPolicyError, match="argument"):
        job(argv=(argument,))


def test_subprocess_runner_captures_bounded_output_without_a_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime wrapper captures output in bounded temporary files."""

    class FakeProcess:
        pid = 123

        def __init__(self, argv: Sequence[str], **kwargs: object) -> None:
            assert tuple(argv) == ("/usr/bin/docker", "version")
            assert kwargs["shell"] is False
            stdout = kwargs["stdout"]
            stderr = kwargs["stderr"]
            stdout.write(b"ok")  # type: ignore[attr-defined]
            stderr.write(b"warning")  # type: ignore[attr-defined]

        def wait(self, timeout: int | None = None) -> int:
            assert timeout == 1
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    result = SubprocessSandboxRunner().run(
        ("/usr/bin/docker", "version"), timeout_seconds=1, max_output_bytes=4
    )
    assert result.stdout == b"ok"
    assert result.stderr == b"warn"
    assert result.stderr_truncated

    def unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("private runtime detail")

    monkeypatch.setattr(subprocess, "Popen", unavailable)
    with pytest.raises(SandboxUnavailable, match="unavailable"):
        SubprocessSandboxRunner().run(
            ("/usr/bin/docker", "version"), timeout_seconds=1, max_output_bytes=4
        )


def test_sandbox_schema_rejects_unsafe_runtime_configuration() -> None:
    with pytest.raises(SandboxPolicyError, match="count"):
        job(argv=tuple("x" for _ in range(257)))
    with pytest.raises(SandboxPolicyError, match="Network"):
        policy(network_policy="deny")
    with pytest.raises(SandboxPolicyError, match="absolute"):
        policy(seccomp_profile=PurePosixPath("relative.json"))
    with pytest.raises(SandboxPolicyError, match="seccomp"):
        policy(seccomp_profile=PurePosixPath("/unconfined"))
    with pytest.raises(SandboxPolicyError, match="feature gate"):
        backend(FakeRunner(), enabled=1)
    with pytest.raises(SandboxPolicyError, match="executable"):
        backend(FakeRunner(), executable="/bin/sh")
    with pytest.raises(SandboxPolicyError, match="name"):
        backend(FakeRunner()).build_command(job(), policy(), container_name="unsafe")
    with pytest.raises(SandboxPolicyError, match="CPU"):
        policy(cpu_cores=0)


def test_subprocess_runner_rejects_unbounded_capture() -> None:
    with pytest.raises(SandboxPolicyError, match="output bound"):
        SubprocessSandboxRunner().run(
            ("/usr/bin/docker", "version"), timeout_seconds=1, max_output_bytes=0
        )
