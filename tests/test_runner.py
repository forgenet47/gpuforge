"""Trusted runner contract tests with an injected untrusted process."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from gpuforge.runner import (
    EventKind,
    FrameKind,
    ProgressEvent,
    RunnerContract,
    RunnerErrorCode,
    RunnerFailure,
    RunnerFrame,
    RunnerStatus,
    TrainingSupervisor,
)

JOB = "sha256:" + "11" * 32
LEASE = "sha256:" + "22" * 32
RESULT = "sha256:" + "33" * 32


def contract(**changes: object) -> RunnerContract:
    values: dict[str, object] = {
        "job_digest": JOB,
        "lease_digest": LEASE,
        "deterministic_seed": 42,
        "max_steps": 100,
        "checkpoint_interval_steps": 10,
        "max_runtime_ms": 10_000,
        "max_log_bytes": 1_024,
        "max_checkpoint_bytes": 1_024,
        "max_events": 100,
        "cancellation_grace_frames": 3,
    }
    values.update(changes)
    return RunnerContract(**values)  # type: ignore[arg-type]


def event(
    sequence: int,
    kind: EventKind,
    *,
    step: int,
    digest: str | None = None,
    error_code: str | None = None,
    elapsed_ms: int = 1,
) -> RunnerFrame:
    return RunnerFrame(
        FrameKind.EVENT,
        ProgressEvent(
            sequence=sequence,
            kind=kind,
            step=step,
            elapsed_ms=elapsed_ms,
            output_digest=digest,
            error_code=error_code,
        ).canonical_bytes(),
    )


@dataclass
class FakeProcess:
    """Expose predetermined frames while recording supervisor controls."""

    frames: list[RunnerFrame]
    exit_code: int | None = 0
    checkpoint_requested: bool = False
    cancel_requested: bool = False
    terminated: bool = False

    def next_frame(self, *, timeout_seconds: int) -> RunnerFrame | None:
        assert timeout_seconds == 1
        return self.frames.pop(0) if self.frames else None

    def request_checkpoint(self) -> None:
        self.checkpoint_requested = True

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, *, timeout_seconds: int) -> int | None:
        assert timeout_seconds == 5
        return self.exit_code


@dataclass
class MemoryCheckpointSink:
    """Hash checkpoint payloads exactly as a content-addressed adapter would."""

    values: dict[int, bytes] = field(default_factory=dict)
    fail: bool = False

    def commit(self, *, step: int, payload: bytes) -> str:
        if self.fail:
            raise OSError("private storage detail")
        self.values[step] = payload
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class StaticCancellation:
    cancelled: bool = False

    def is_cancelled(self) -> bool:
        return self.cancelled


def test_honest_training_events_checkpoint_and_result_are_supervised() -> None:
    checkpoint = b"checkpoint-bytes"
    checkpoint_digest = f"sha256:{hashlib.sha256(checkpoint).hexdigest()}"
    process = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0, elapsed_ms=0),
            event(1, EventKind.PROGRESS, step=5),
            RunnerFrame(FrameKind.LOG, b"epoch complete"),
            RunnerFrame(FrameKind.CHECKPOINT, checkpoint, step=10),
            event(2, EventKind.CHECKPOINTED, step=10, digest=checkpoint_digest),
            event(3, EventKind.COMPLETED, step=100, digest=RESULT),
        ]
    )
    sink = MemoryCheckpointSink()

    outcome = TrainingSupervisor().run(contract(), process, sink, StaticCancellation())

    assert outcome.status is RunnerStatus.COMPLETED
    assert outcome.result_digest == RESULT
    assert outcome.event_count == 4
    assert outcome.log_bytes == len(b"epoch complete")
    assert outcome.checkpoints[0].digest == checkpoint_digest
    assert sink.values == {10: checkpoint}


def test_seed_handoff_contains_no_secret_or_host_environment() -> None:
    environment = dict(contract(deterministic_seed=123).trusted_environment())
    assert environment == {
        "GPUFORGE_JOB_DIGEST": JOB,
        "GPUFORGE_LEASE_DIGEST": LEASE,
        "GPUFORGE_RUNNER_SEED": "123",
    }
    assert not any("TOKEN" in key or "PASSWORD" in key for key in environment)


def test_progress_event_round_trip_is_canonical() -> None:
    original = ProgressEvent(0, EventKind.STARTED, 0, 0)
    assert ProgressEvent.from_bytes(original.canonical_bytes()) == original
    with pytest.raises(RunnerFailure) as noncanonical:
        ProgressEvent.from_bytes(b'{"sequence":0}')
    assert noncanonical.value.code is RunnerErrorCode.INVALID_EVENT


@pytest.mark.parametrize(
    "payload",
    (
        b"not-json",
        b'{"elapsed_ms":0,"error_code":null,"kind":"started","output_digest":null,"sequence":0,"sequence":0,"step":0}',
        b'{"elapsed_ms":0, "error_code":null,"kind":"started","output_digest":null,"sequence":0,"step":0}',
    ),
)
def test_malformed_duplicate_and_noncanonical_events_are_rejected(payload: bytes) -> None:
    process = FakeProcess([RunnerFrame(FrameKind.EVENT, payload)])
    with pytest.raises(RunnerFailure) as error:
        TrainingSupervisor().run(contract(), process, MemoryCheckpointSink(), StaticCancellation())
    assert error.value.code is RunnerErrorCode.INVALID_EVENT


def test_arbitrary_log_text_is_never_interpreted_as_completion() -> None:
    fake_proof = ProgressEvent(9, EventKind.COMPLETED, 100, 1, RESULT).canonical_bytes()
    process = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0, elapsed_ms=0),
            RunnerFrame(FrameKind.LOG, fake_proof),
            event(1, EventKind.COMPLETED, step=100, digest=RESULT),
        ]
    )
    outcome = TrainingSupervisor().run(
        contract(), process, MemoryCheckpointSink(), StaticCancellation()
    )
    assert outcome.event_count == 2
    assert outcome.log_digest == f"sha256:{hashlib.sha256(fake_proof).hexdigest()}"


def test_excessive_output_terminates_process() -> None:
    process = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            RunnerFrame(FrameKind.LOG, b"x" * 5),
        ]
    )
    with pytest.raises(RunnerFailure) as error:
        TrainingSupervisor().run(
            contract(max_log_bytes=4), process, MemoryCheckpointSink(), StaticCancellation()
        )
    assert error.value.code is RunnerErrorCode.LOG_LIMIT
    assert process.terminated


def test_graceful_cancellation_accepts_only_structured_cancel_event() -> None:
    process = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.CANCELLED, step=4),
        ],
        exit_code=130,
    )
    outcome = TrainingSupervisor().run(
        contract(), process, MemoryCheckpointSink(), StaticCancellation(True)
    )
    assert process.cancel_requested
    assert process.checkpoint_requested
    assert not process.terminated
    assert outcome.status is RunnerStatus.CANCELLED
    assert outcome.terminal_error is None


def test_ignored_cancellation_is_forcibly_terminated() -> None:
    process = FakeProcess([RunnerFrame(FrameKind.LOG, b"one"), RunnerFrame(FrameKind.LOG, b"two")])
    with pytest.raises(RunnerFailure) as error:
        TrainingSupervisor().run(
            contract(cancellation_grace_frames=1),
            process,
            MemoryCheckpointSink(),
            StaticCancellation(True),
        )
    assert error.value.code is RunnerErrorCode.CANCELLATION_IGNORED
    assert process.terminated


def test_timeout_requests_checkpoint_then_cancellation() -> None:
    moments = iter((0, 10_001, 10_002, 10_003))
    process = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.CANCELLED, step=7),
        ],
        exit_code=130,
    )
    outcome = TrainingSupervisor(clock_ms=moments.__next__).run(
        contract(), process, MemoryCheckpointSink(), StaticCancellation()
    )
    assert process.checkpoint_requested
    assert process.cancel_requested
    assert outcome.status is RunnerStatus.CANCELLED
    assert outcome.terminal_error is RunnerErrorCode.TIMED_OUT


def test_timeout_never_accepts_late_completion() -> None:
    moments = iter((0, 10_001, 10_002))
    process = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.COMPLETED, step=100, digest=RESULT),
        ]
    )
    with pytest.raises(RunnerFailure) as error:
        TrainingSupervisor(clock_ms=moments.__next__).run(
            contract(), process, MemoryCheckpointSink(), StaticCancellation()
        )
    assert error.value.code is RunnerErrorCode.TIMED_OUT


@pytest.mark.parametrize(
    "frames",
    [
        [event(0, EventKind.PROGRESS, step=1)],
        [event(0, EventKind.STARTED, step=0), event(0, EventKind.STARTED, step=0)],
        [RunnerFrame(FrameKind.CHECKPOINT, b"early", step=10)],
        [event(0, EventKind.STARTED, step=0), event(2, EventKind.PROGRESS, step=1)],
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.PROGRESS, step=2),
            event(2, EventKind.PROGRESS, step=1),
        ],
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.PROGRESS, step=1, elapsed_ms=10_001),
        ],
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.CHECKPOINTED, step=10, digest=RESULT),
        ],
    ],
)
def test_out_of_order_or_unsubstantiated_events_fail_closed(
    frames: list[RunnerFrame],
) -> None:
    with pytest.raises(RunnerFailure) as error:
        TrainingSupervisor().run(
            contract(), FakeProcess(frames), MemoryCheckpointSink(), StaticCancellation()
        )
    assert error.value.code is RunnerErrorCode.INVALID_EVENT


def test_duplicate_checkpoint_and_event_flood_are_terminated() -> None:
    duplicate = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            RunnerFrame(FrameKind.CHECKPOINT, b"first", step=10),
            RunnerFrame(FrameKind.CHECKPOINT, b"second", step=10),
        ]
    )
    with pytest.raises(RunnerFailure) as duplicate_error:
        TrainingSupervisor().run(
            contract(), duplicate, MemoryCheckpointSink(), StaticCancellation()
        )
    assert duplicate_error.value.code is RunnerErrorCode.INVALID_EVENT
    assert duplicate.terminated

    flooded = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.PROGRESS, step=1),
            event(2, EventKind.COMPLETED, step=100, digest=RESULT),
        ]
    )
    with pytest.raises(RunnerFailure) as flood_error:
        TrainingSupervisor().run(
            contract(max_events=2), flooded, MemoryCheckpointSink(), StaticCancellation()
        )
    assert flood_error.value.code is RunnerErrorCode.EVENT_LIMIT
    assert flooded.terminated


def test_unresponsive_process_is_terminated() -> None:
    process = FakeProcess([], exit_code=None)
    with pytest.raises(RunnerFailure) as error:
        TrainingSupervisor().run(contract(), process, MemoryCheckpointSink(), StaticCancellation())
    assert error.value.code is RunnerErrorCode.PROCESS_FAILED
    assert process.terminated


def test_checkpoint_cadence_size_and_storage_failures_are_stable() -> None:
    bad_step = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            RunnerFrame(FrameKind.CHECKPOINT, b"value", step=7),
        ]
    )
    with pytest.raises(RunnerFailure) as cadence:
        TrainingSupervisor().run(contract(), bad_step, MemoryCheckpointSink(), StaticCancellation())
    assert cadence.value.code is RunnerErrorCode.CHECKPOINT_LIMIT

    storage = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            RunnerFrame(FrameKind.CHECKPOINT, b"value", step=10),
        ]
    )
    with pytest.raises(RunnerFailure) as failed:
        TrainingSupervisor().run(
            contract(), storage, MemoryCheckpointSink(fail=True), StaticCancellation()
        )
    assert failed.value.code is RunnerErrorCode.CHECKPOINT_FAILED
    assert "private storage detail" not in str(failed.value)
    assert storage.terminated


def test_nonzero_exit_and_missing_terminal_event_fail_closed() -> None:
    nonzero = FakeProcess(
        [
            event(0, EventKind.STARTED, step=0),
            event(1, EventKind.COMPLETED, step=100, digest=RESULT),
        ],
        exit_code=7,
    )
    with pytest.raises(RunnerFailure) as process_error:
        TrainingSupervisor().run(contract(), nonzero, MemoryCheckpointSink(), StaticCancellation())
    assert process_error.value.code is RunnerErrorCode.PROCESS_FAILED

    missing = FakeProcess([event(0, EventKind.STARTED, step=0)])
    with pytest.raises(RunnerFailure) as missing_error:
        TrainingSupervisor().run(contract(), missing, MemoryCheckpointSink(), StaticCancellation())
    assert missing_error.value.code is RunnerErrorCode.PROCESS_FAILED
