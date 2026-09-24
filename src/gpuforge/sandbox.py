"""Fail-closed Linux OCI container policy and execution backend."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from gpuforge.config import NetworkPolicy

_IMAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}@sha256:[0-9a-f]{64}")
_NAME_PATTERN = re.compile(r"gpuforge-[0-9a-f]{16}")
_PROFILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MAX_ARGUMENTS = 256
_MAX_ARGUMENT_BYTES = 4_096
_MAX_CAPTURE_BYTES = 1024 * 1024


class SandboxError(RuntimeError):
    """Base error for sandbox policy and execution failures."""


class SandboxPolicyError(SandboxError):
    """Raised when a workload requests an unsafe capability."""


class SandboxUnavailable(SandboxError):
    """Raised when the feature-gated Linux backend is unavailable."""


class SandboxCommandTimeout(TimeoutError):
    """Raised when the container client exceeds its deadline."""


@dataclass(frozen=True, slots=True)
class ContainerJob:
    """Minimal publisher-controlled values accepted by the launcher."""

    image: str
    argv: tuple[str, ...]
    host_mounts: tuple[Path, ...] = ()
    device_requests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.image, str) or _IMAGE_PATTERN.fullmatch(self.image) is None:
            raise SandboxPolicyError("Container image must use an immutable SHA-256 digest")
        if not isinstance(self.argv, tuple) or not 1 <= len(self.argv) <= _MAX_ARGUMENTS:
            raise SandboxPolicyError("Container command argument count is invalid")
        for argument in self.argv:
            _safe_argument(argument)
        if self.host_mounts:
            raise SandboxPolicyError("Publisher-requested host mounts are prohibited")
        if self.device_requests:
            raise SandboxPolicyError("Publisher-requested device access is prohibited")


@dataclass(frozen=True, slots=True)
class ContainerSandboxPolicy:
    """Explicit resource ceilings and Linux isolation hooks."""

    cpu_cores: int
    memory_mb: int
    gpu_count: int
    pids_limit: int
    writable_bytes: int
    max_runtime_seconds: int
    network_policy: NetworkPolicy = NetworkPolicy.DENY
    seccomp_profile: PurePosixPath | None = None
    apparmor_profile: str | None = None

    def __post_init__(self) -> None:
        _bounded_int("CPU limit", self.cpu_cores, 1, 256)
        _bounded_int("memory limit", self.memory_mb, 1_024, 2_097_152)
        _bounded_int("GPU limit", self.gpu_count, 0, 8)
        _bounded_int("PID limit", self.pids_limit, 64, 65_536)
        _bounded_int("writable byte limit", self.writable_bytes, 1, 1024**4)
        _bounded_int("runtime limit", self.max_runtime_seconds, 1, 86_400)
        if not isinstance(self.network_policy, NetworkPolicy):
            raise SandboxPolicyError("Network policy is invalid")
        if self.seccomp_profile is not None:
            if (
                not isinstance(self.seccomp_profile, PurePosixPath)
                or not self.seccomp_profile.is_absolute()
            ):
                raise SandboxPolicyError("Seccomp profile must use an absolute operator path")
            if self.seccomp_profile.name.casefold() == "unconfined":
                raise SandboxPolicyError("Unconfined seccomp is prohibited")
        if self.apparmor_profile is not None:
            if (
                not isinstance(self.apparmor_profile, str)
                or _PROFILE_PATTERN.fullmatch(self.apparmor_profile) is None
                or self.apparmor_profile.casefold() == "unconfined"
            ):
                raise SandboxPolicyError("AppArmor profile is invalid")


@dataclass(frozen=True, slots=True)
class SandboxCommandResult:
    """Bounded output returned by the local container client."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Non-sensitive execution summary; raw workload output is not retained."""

    exit_code: int | None
    timed_out: bool
    duration_ms: int
    stdout_digest: str
    stderr_digest: str
    stdout_truncated: bool
    stderr_truncated: bool


class SandboxCommandRunner(Protocol):
    """Execute a validated container-client argv without a shell."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> SandboxCommandResult: ...


class ExecutionBackend(Protocol):
    """Run one immutable container job under explicit policy."""

    def run(self, job: ContainerJob, policy: ContainerSandboxPolicy) -> SandboxResult: ...


@dataclass(frozen=True, slots=True)
class SubprocessSandboxRunner:
    """Bound output while invoking an already-installed container engine."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> SandboxCommandResult:
        if not 1 <= max_output_bytes <= _MAX_CAPTURE_BYTES:
            raise SandboxPolicyError("Command output bound is invalid")
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(  # noqa: S603 - validated argv, no shell
                    tuple(argv),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env={"PATH": os.defpath},
                    shell=False,
                    start_new_session=True,
                )
            except OSError:
                raise SandboxUnavailable("Container runtime is unavailable") from None
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                kill_process_group = getattr(os, "killpg", None)
                if os.name == "posix" and callable(kill_process_group):
                    kill_process_group(process.pid, 9)
                else:  # pragma: no cover - backend is Linux-only
                    process.kill()
                process.wait()
                raise SandboxCommandTimeout("Container client timed out") from None
            stdout, stdout_truncated = _read_bounded(stdout_file, max_output_bytes)
            stderr, stderr_truncated = _read_bounded(stderr_file, max_output_bytes)
        return SandboxCommandResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )


@dataclass(frozen=True, slots=True)
class LinuxContainerBackend:
    """Launch a digest-pinned container with least privilege and no network."""

    runner: SandboxCommandRunner
    enabled: bool = False
    executable: str = "/usr/bin/docker"
    platform: str = sys.platform
    name_factory: Callable[[], str] = lambda: f"gpuforge-{secrets.token_hex(8)}"
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise SandboxPolicyError("Sandbox feature gate is invalid")
        if not isinstance(self.executable, str) or Path(self.executable).name not in {
            "docker",
            "podman",
        }:
            raise SandboxPolicyError("Container runtime executable is invalid")

    def build_command(
        self,
        job: ContainerJob,
        policy: ContainerSandboxPolicy,
        *,
        container_name: str,
    ) -> tuple[str, ...]:
        """Build a closed set of runtime flags; publisher flags are never appended."""
        if _NAME_PATTERN.fullmatch(container_name) is None:
            raise SandboxPolicyError("Container name is invalid")
        if policy.network_policy is not NetworkPolicy.DENY:
            raise SandboxPolicyError("Network allowlists require a mediated backend")
        command = [
            self.executable,
            "run",
            "--rm",
            "--name",
            container_name,
            "--pull",
            "never",
            "--user",
            "65532:65532",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            str(policy.pids_limit),
            "--cpus",
            str(policy.cpu_cores),
            "--memory",
            f"{policy.memory_mb}m",
            "--memory-swap",
            f"{policy.memory_mb}m",
            "--network",
            "none",
            "--ipc",
            "none",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=67108864",  # noqa: S108 - container tmpfs
            "--tmpfs",
            f"/job/scratch:rw,noexec,nosuid,nodev,size={policy.writable_bytes}",
            "--workdir",
            "/job/scratch",
        ]
        if policy.seccomp_profile is not None:
            command.extend(("--security-opt", f"seccomp={policy.seccomp_profile}"))
        if policy.apparmor_profile is not None:
            command.extend(("--security-opt", f"apparmor={policy.apparmor_profile}"))
        if policy.gpu_count:
            command.extend(("--gpus", f"count={policy.gpu_count}"))
        command.append(job.image)
        command.extend(job.argv)
        return tuple(command)

    def run(self, job: ContainerJob, policy: ContainerSandboxPolicy) -> SandboxResult:
        """Execute or return a timeout result after forcibly removing the container."""
        if not self.enabled or self.platform != "linux":
            raise SandboxUnavailable("Linux container sandbox is not enabled")
        container_name = self.name_factory()
        command = self.build_command(job, policy, container_name=container_name)
        started = self.clock()
        try:
            outcome = self.runner.run(
                command,
                timeout_seconds=policy.max_runtime_seconds,
                max_output_bytes=_MAX_CAPTURE_BYTES,
            )
        except SandboxCommandTimeout:
            self._force_remove(container_name)
            return SandboxResult(
                exit_code=None,
                timed_out=True,
                duration_ms=_duration_ms(started, self.clock()),
                stdout_digest=_digest(b""),
                stderr_digest=_digest(b""),
                stdout_truncated=False,
                stderr_truncated=False,
            )
        return SandboxResult(
            exit_code=outcome.returncode,
            timed_out=False,
            duration_ms=_duration_ms(started, self.clock()),
            stdout_digest=_digest(outcome.stdout),
            stderr_digest=_digest(outcome.stderr),
            stdout_truncated=outcome.stdout_truncated,
            stderr_truncated=outcome.stderr_truncated,
        )

    def _force_remove(self, container_name: str) -> None:
        for action in (("kill", container_name), ("rm", "--force", container_name)):
            try:
                self.runner.run(
                    (self.executable, *action),
                    timeout_seconds=15,
                    max_output_bytes=4_096,
                )
            except (SandboxError, OSError, TimeoutError):
                continue


def _safe_argument(value: object) -> None:
    if not isinstance(value, str):
        raise SandboxPolicyError("Container command argument must be text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise SandboxPolicyError("Container command argument is invalid") from None
    if not 1 <= len(encoded) <= _MAX_ARGUMENT_BYTES or any(
        character in value for character in ("\x00", "\r", "\n")
    ):
        raise SandboxPolicyError("Container command argument violates its bounds")


def _bounded_int(label: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SandboxPolicyError(f"{label} is invalid")


def _read_bounded(handle: object, maximum: int) -> tuple[bytes, bool]:
    file_handle = handle
    file_handle.seek(0)  # type: ignore[attr-defined]
    value = file_handle.read(maximum + 1)  # type: ignore[attr-defined]
    return value[:maximum], len(value) > maximum


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _duration_ms(started: float, finished: float) -> int:
    return max(0, int((finished - started) * 1_000))


__all__ = [
    "ContainerJob",
    "ContainerSandboxPolicy",
    "ExecutionBackend",
    "LinuxContainerBackend",
    "SandboxCommandResult",
    "SandboxCommandRunner",
    "SandboxCommandTimeout",
    "SandboxError",
    "SandboxPolicyError",
    "SandboxResult",
    "SandboxUnavailable",
    "SubprocessSandboxRunner",
]
