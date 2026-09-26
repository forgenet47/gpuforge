"""Trusted training-runner contract and untrusted process supervision."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, cast

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_CODE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]{0,63}")
_MAX_EVENT_BYTES = 16 * 1024
_MAX_FRAME_BYTES = 64 * 1024 * 1024


class RunnerErrorCode(str, Enum):
    """Stable failure categories that never include publisher output."""

    INVALID_EVENT = "invalid_event"
    EVENT_LIMIT = "event_limit"
    LOG_LIMIT = "log_limit"
    CHECKPOINT_LIMIT = "checkpoint_limit"
    CHECKPOINT_FAILED = "checkpoint_failed"
    PROCESS_FAILED = "process_failed"
    CANCELLATION_IGNORED = "cancellation_ignored"
    TIMED_OUT = "timed_out"


class RunnerFailure(RuntimeError):
    """Sanitized supervisor failure containing only a stable code."""

    def __init__(self, code: RunnerErrorCode) -> None:
        if not isinstance(code, RunnerErrorCode):
            raise ValueError("Runner failure code is invalid")
        self.code = code
        super().__init__(code.value)


class EventKind(str, Enum):
    """Structured events emitted by the trusted runner shim."""

    STARTED = "started"
    PROGRESS = "progress"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FrameKind(str, Enum):
    """Separate structured control data from arbitrary workload logs."""

    EVENT = "event"
    LOG = "log"
    CHECKPOINT = "checkpoint"


class RunnerStatus(str, Enum):
    """Terminal supervised outcome."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunnerContract:
    """Bounded input passed from the trusted shim to a workload."""

    job_digest: str
    lease_digest: str
    deterministic_seed: int
    max_steps: int
    checkpoint_interval_steps: int
    max_runtime_ms: int
    max_log_bytes: int = 1024 * 1024
    max_checkpoint_bytes: int = 8 * 1024**3
    max_events: int = 100_000
    cancellation_grace_frames: int = 8

    def __post_init__(self) -> None:
        _digest("job digest", self.job_digest)
        _digest("lease digest", self.lease_digest)
        _integer("deterministic seed", self.deterministic_seed, 0, 2**63 - 1)
        _integer("maximum steps", self.max_steps, 1, 2**63 - 1)
        _integer("checkpoint interval", self.checkpoint_interval_steps, 1, self.max_steps)
        _integer("maximum runtime", self.max_runtime_ms, 1_000, 86_400_000)
        _integer("maximum log bytes", self.max_log_bytes, 0, 64 * 1024 * 1024)
        _integer(
            "maximum checkpoint bytes",
            self.max_checkpoint_bytes,
            1,
            1024**4,
        )
        _integer("maximum events", self.max_events, 2, 1_000_000)
        _integer("cancellation grace frames", self.cancellation_grace_frames, 1, 1_000)

    def trusted_environment(self) -> tuple[tuple[str, str], ...]:
        """Return only public identifiers and the validator-selected seed."""
        return (
            ("GPUFORGE_JOB_DIGEST", self.job_digest),
            ("GPUFORGE_LEASE_DIGEST", self.lease_digest),
            ("GPUFORGE_RUNNER_SEED", str(self.deterministic_seed)),
        )


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Canonical event that is never inferred from arbitrary log text."""

    sequence: int
    kind: EventKind
    step: int
    elapsed_ms: int
    output_digest: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        _integer("event sequence", self.sequence, 0, 2**63 - 1)
        if not isinstance(self.kind, EventKind):
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        _integer("event step", self.step, 0, 2**63 - 1)
        _integer("event elapsed time", self.elapsed_ms, 0, 86_400_000)
        if self.output_digest is not None:
            _digest("event output digest", self.output_digest)
        if self.error_code is not None:
            _code("event error code", self.error_code)
        if self.kind in {EventKind.CHECKPOINTED, EventKind.COMPLETED}:
            if self.output_digest is None or self.error_code is not None:
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        elif self.kind is EventKind.FAILED:
            if self.error_code is None or self.output_digest is not None:
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        elif self.output_digest is not None or self.error_code is not None:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        if self.kind is EventKind.STARTED and (self.sequence != 0 or self.step != 0):
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)

    def to_primitive(self) -> dict[str, object]:
        """Return the strict event wire object."""
        return {
            "elapsed_ms": self.elapsed_ms,
            "error_code": self.error_code,
            "kind": self.kind.value,
            "output_digest": self.output_digest,
            "sequence": self.sequence,
            "step": self.step,
        }

    def canonical_bytes(self) -> bytes:
        """Encode a canonical bounded event."""
        value = json.dumps(
            self.to_primitive(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(value) > _MAX_EVENT_BYTES:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        return value

    @classmethod
    def from_bytes(cls, value: bytes) -> ProgressEvent:
        """Decode only canonical JSON with exact fields and no duplicate keys."""
        if not isinstance(value, bytes) or not 1 <= len(value) <= _MAX_EVENT_BYTES:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        try:
            decoded = json.loads(value, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, RunnerFailure):
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT) from None
        if not isinstance(decoded, dict) or set(decoded) != {
            "elapsed_ms",
            "error_code",
            "kind",
            "output_digest",
            "sequence",
            "step",
        }:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        data = cast(dict[str, object], decoded)
        try:
            event = cls(
                sequence=_raw_integer(data["sequence"]),
                kind=EventKind(data["kind"]),
                step=_raw_integer(data["step"]),
                elapsed_ms=_raw_integer(data["elapsed_ms"]),
                output_digest=_optional_text(data["output_digest"]),
                error_code=_optional_text(data["error_code"]),
            )
        except (TypeError, ValueError):
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT) from None
        if event.canonical_bytes() != value:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        return event


@dataclass(frozen=True, slots=True)
class RunnerFrame:
    """One bounded frame from the process transport."""

    kind: FrameKind
    payload: bytes
    step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FrameKind):
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        if not isinstance(self.payload, bytes) or len(self.payload) > _MAX_FRAME_BYTES:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        if self.kind is FrameKind.CHECKPOINT:
            if self.step is None:
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
            _integer("checkpoint step", self.step, 1, 2**63 - 1)
        elif self.step is not None:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Digest and size returned by trusted checkpoint storage."""

    step: int
    digest: str
    size: int

    def __post_init__(self) -> None:
        _integer("checkpoint step", self.step, 1, 2**63 - 1)
        _digest("checkpoint digest", self.digest)
        _integer("checkpoint size", self.size, 0, 1024**4)


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Sanitized structured result of one supervised process."""

    status: RunnerStatus
    result_digest: str | None
    checkpoints: tuple[CheckpointRecord, ...]
    event_count: int
    log_bytes: int
    log_digest: str
    exit_code: int
    terminal_error: RunnerErrorCode | None = None


class RunnerProcess(Protocol):
    """Transport implemented by the trusted shim around untrusted code."""

    def next_frame(self, *, timeout_seconds: int) -> RunnerFrame | None: ...

    def request_checkpoint(self) -> None: ...

    def request_cancel(self) -> None: ...

    def terminate(self) -> None: ...

    def wait(self, *, timeout_seconds: int) -> int | None: ...


class CheckpointSink(Protocol):
    """Store checkpoint bytes and return their verified digest."""

    def commit(self, *, step: int, payload: bytes) -> str: ...


class CancellationToken(Protocol):
    """Expose cooperative cancellation without sharing mutable host state."""

    def is_cancelled(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class TrainingSupervisor:
    """Supervise structured events, bounded logs, checkpoints, and cancellation."""

    clock_ms: Callable[[], int] = lambda: time.monotonic_ns() // 1_000_000

    def run(
        self,
        contract: RunnerContract,
        process: RunnerProcess,
        checkpoint_sink: CheckpointSink,
        cancellation: CancellationToken,
    ) -> TrainingOutcome:
        started_at = self.clock_ms()
        expected_sequence = 0
        last_step = 0
        event_count = 0
        log_bytes = 0
        log_hash = hashlib.sha256()
        checkpoints: dict[int, CheckpointRecord] = {}
        terminal: ProgressEvent | None = None
        cancel_requested = False
        timed_out = False
        grace_frames = 0

        while terminal is None:
            elapsed = self.clock_ms() - started_at
            if elapsed >= contract.max_runtime_ms and not cancel_requested:
                process.request_checkpoint()
                process.request_cancel()
                cancel_requested = True
                timed_out = True
            elif cancellation.is_cancelled() and not cancel_requested:
                process.request_checkpoint()
                process.request_cancel()
                cancel_requested = True

            frame = process.next_frame(timeout_seconds=1)
            if frame is None:
                break
            if cancel_requested:
                grace_frames += 1
                if grace_frames > contract.cancellation_grace_frames:
                    process.terminate()
                    code = (
                        RunnerErrorCode.TIMED_OUT
                        if timed_out
                        else RunnerErrorCode.CANCELLATION_IGNORED
                    )
                    raise RunnerFailure(code)

            if frame.kind is FrameKind.LOG:
                log_bytes += len(frame.payload)
                if log_bytes > contract.max_log_bytes:
                    process.terminate()
                    raise RunnerFailure(RunnerErrorCode.LOG_LIMIT)
                log_hash.update(frame.payload)
                continue
            if frame.kind is FrameKind.CHECKPOINT:
                if expected_sequence == 0:
                    process.terminate()
                    raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
                try:
                    checkpoint = self._checkpoint(contract, checkpoint_sink, frame)
                except RunnerFailure:
                    process.terminate()
                    raise
                if checkpoint.step in checkpoints:
                    process.terminate()
                    raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
                checkpoints[checkpoint.step] = checkpoint
                continue

            event_count += 1
            if event_count > contract.max_events:
                process.terminate()
                raise RunnerFailure(RunnerErrorCode.EVENT_LIMIT)
            try:
                event = ProgressEvent.from_bytes(frame.payload)
            except RunnerFailure:
                process.terminate()
                raise
            if event.sequence != expected_sequence:
                process.terminate()
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
            if expected_sequence == 0 and event.kind is not EventKind.STARTED:
                process.terminate()
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
            if expected_sequence > 0 and event.kind is EventKind.STARTED:
                process.terminate()
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
            if event.step < last_step or event.step > contract.max_steps:
                process.terminate()
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
            if event.elapsed_ms > contract.max_runtime_ms:
                process.terminate()
                raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
            if event.kind is EventKind.CHECKPOINTED:
                recorded_checkpoint = checkpoints.get(event.step)
                if recorded_checkpoint is None or recorded_checkpoint.digest != event.output_digest:
                    process.terminate()
                    raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
            expected_sequence += 1
            last_step = event.step
            if event.kind in {EventKind.COMPLETED, EventKind.FAILED, EventKind.CANCELLED}:
                terminal = event

        exit_code = process.wait(timeout_seconds=5)
        if exit_code is None:
            process.terminate()
            raise RunnerFailure(
                RunnerErrorCode.TIMED_OUT if timed_out else RunnerErrorCode.PROCESS_FAILED
            )
        if terminal is None:
            raise RunnerFailure(
                RunnerErrorCode.TIMED_OUT if timed_out else RunnerErrorCode.PROCESS_FAILED
            )
        if timed_out and terminal.kind is not EventKind.CANCELLED:
            raise RunnerFailure(RunnerErrorCode.TIMED_OUT)
        ordered_checkpoints = tuple(checkpoints[step] for step in sorted(checkpoints))
        log_digest = f"sha256:{log_hash.hexdigest()}"
        if terminal.kind is EventKind.CANCELLED and cancel_requested:
            return TrainingOutcome(
                status=RunnerStatus.CANCELLED,
                result_digest=None,
                checkpoints=ordered_checkpoints,
                event_count=event_count,
                log_bytes=log_bytes,
                log_digest=log_digest,
                exit_code=exit_code,
                terminal_error=(RunnerErrorCode.TIMED_OUT if timed_out else None),
            )
        if terminal.kind is not EventKind.COMPLETED or exit_code != 0:
            raise RunnerFailure(RunnerErrorCode.PROCESS_FAILED)
        return TrainingOutcome(
            status=RunnerStatus.COMPLETED,
            result_digest=terminal.output_digest,
            checkpoints=ordered_checkpoints,
            event_count=event_count,
            log_bytes=log_bytes,
            log_digest=log_digest,
            exit_code=exit_code,
        )

    @staticmethod
    def _checkpoint(
        contract: RunnerContract,
        sink: CheckpointSink,
        frame: RunnerFrame,
    ) -> CheckpointRecord:
        if frame.step is None:  # pragma: no cover - guaranteed by frame validation
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        if (
            len(frame.payload) > contract.max_checkpoint_bytes
            or frame.step > contract.max_steps
            or (
                frame.step % contract.checkpoint_interval_steps != 0
                and frame.step != contract.max_steps
            )
        ):
            raise RunnerFailure(RunnerErrorCode.CHECKPOINT_LIMIT)
        try:
            digest = sink.commit(step=frame.step, payload=frame.payload)
            _digest("checkpoint digest", digest)
        except RunnerFailure:
            raise
        except Exception:
            raise RunnerFailure(RunnerErrorCode.CHECKPOINT_FAILED) from None
        return CheckpointRecord(frame.step, digest, len(frame.payload))


def _unique_object(pairs: list[tuple[str, object]]) -> Mapping[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
        result[key] = value
    return result


def _raw_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
    return value


def _optional_text(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise RunnerFailure(RunnerErrorCode.INVALID_EVENT)
    return value


def _integer(label: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} is invalid")


def _digest(label: str, value: object) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")


def _code(label: str, value: object) -> None:
    if not isinstance(value, str) or _CODE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")


__all__ = [
    "CancellationToken",
    "CheckpointRecord",
    "CheckpointSink",
    "EventKind",
    "FrameKind",
    "ProgressEvent",
    "RunnerContract",
    "RunnerErrorCode",
    "RunnerFailure",
    "RunnerFrame",
    "RunnerProcess",
    "RunnerStatus",
    "TrainingOutcome",
    "TrainingSupervisor",
]
