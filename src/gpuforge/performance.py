"""Synchronized useful-work measurement with explicit excluded phases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_INTERVALS = 100_000
_MAX_INTEGER = 2**63 - 1
_NANOSECONDS_PER_SECOND = 1_000_000_000


class WorkUnit(str, Enum):
    """Canonical unit derived from accepted batches."""

    SAMPLES = "samples"
    TOKENS = "tokens"
    FLOP_PROXY = "flop_proxy"


class Precision(str, Enum):
    """Bounded precision modes recorded with a measurement."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    TF32 = "tf32"


class IntervalPhase(str, Enum):
    """Trusted timing phases; only active work contributes units."""

    WARMUP = "warmup"
    ACTIVE = "active"
    CHECKPOINT = "checkpoint"


class PerformanceErrorCode(str, Enum):
    """Stable measurement failures that contain no host telemetry."""

    INVALID = "invalid_measurement"
    IDENTITY_MISMATCH = "measurement_identity_mismatch"
    METADATA_MISMATCH = "measurement_metadata_mismatch"
    TIMER_INVALID = "measurement_timer_invalid"
    SYNCHRONIZATION_MISSING = "measurement_sync_missing"
    PARTIAL_RESULT = "measurement_partial_result"
    BATCH_INFLATION = "measurement_batch_inflation"
    WARMUP_INSUFFICIENT = "measurement_warmup_insufficient"
    WORK_INSUFFICIENT = "measurement_work_insufficient"
    OBSERVER_BOUND = "measurement_observer_bound"


class PerformanceFailure(RuntimeError):
    """Sanitized performance failure containing only a stable code."""

    def __init__(self, code: PerformanceErrorCode) -> None:
        if not isinstance(code, PerformanceErrorCode):
            raise ValueError("Performance failure code is invalid")
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class PerformancePolicy:
    """Immutable workload metadata and validator timing ceilings."""

    manifest_digest: str
    lease_digest: str
    unit: WorkUnit
    global_batch_size: int
    tokens_per_sample: int
    flop_proxy_per_sample: int
    batches_per_step: int
    precision: Precision
    model_shape_digest: str
    gpu_count: int
    minimum_warmup_steps: int
    minimum_active_steps: int
    maximum_interval_ns: int
    maximum_active_ns: int
    maximum_checkpoint_ns: int
    maximum_setup_ns: int
    maximum_validator_observed_ns: int

    def __post_init__(self) -> None:
        _digest(self.manifest_digest)
        _digest(self.lease_digest)
        if not isinstance(self.unit, WorkUnit) or not isinstance(self.precision, Precision):
            raise PerformanceFailure(PerformanceErrorCode.INVALID)
        _integer(self.global_batch_size, 1, 10_000_000)
        _integer(self.tokens_per_sample, 1, 10_000_000)
        _integer(self.flop_proxy_per_sample, 1, _MAX_INTEGER)
        _integer(self.batches_per_step, 1, 1_000_000)
        _digest(self.model_shape_digest)
        _integer(self.gpu_count, 1, 16_384)
        _integer(self.minimum_warmup_steps, 0, 1_000_000)
        _integer(self.minimum_active_steps, 1, 1_000_000_000)
        _integer(self.maximum_interval_ns, 1, _MAX_INTEGER)
        _integer(self.maximum_active_ns, 1, _MAX_INTEGER)
        _integer(self.maximum_checkpoint_ns, 0, _MAX_INTEGER)
        _integer(self.maximum_setup_ns, 0, _MAX_INTEGER)
        _integer(self.maximum_validator_observed_ns, 1, _MAX_INTEGER)
        if self.maximum_interval_ns > self.maximum_active_ns:
            raise PerformanceFailure(PerformanceErrorCode.INVALID)


@dataclass(frozen=True, slots=True)
class WorkInterval:
    """One ordered interval emitted after device synchronization."""

    sequence: int
    phase: IntervalPhase
    start_ns: int
    end_ns: int
    first_step: int
    last_step: int
    completed_batches: int
    synchronized_before: bool
    synchronized_after: bool
    complete: bool = True

    def __post_init__(self) -> None:
        _integer(self.sequence, 0, _MAX_INTEGER)
        if not isinstance(self.phase, IntervalPhase):
            raise PerformanceFailure(PerformanceErrorCode.INVALID)
        _integer(self.start_ns, 0, _MAX_INTEGER)
        _integer(self.end_ns, 1, _MAX_INTEGER)
        if self.end_ns <= self.start_ns:
            raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)
        for value in (self.synchronized_before, self.synchronized_after, self.complete):
            if not isinstance(value, bool):
                raise PerformanceFailure(PerformanceErrorCode.INVALID)
        if self.phase is IntervalPhase.CHECKPOINT:
            if self.first_step != 0 or self.last_step != 0 or self.completed_batches != 0:
                raise PerformanceFailure(PerformanceErrorCode.INVALID)
        else:
            _integer(self.first_step, 1, _MAX_INTEGER)
            _integer(self.last_step, self.first_step, _MAX_INTEGER)
            _integer(self.completed_batches, 1, _MAX_INTEGER)

    @property
    def duration_ns(self) -> int:
        """Return the monotonic duration after constructor validation."""
        return self.end_ns - self.start_ns

    @property
    def completed_steps(self) -> int:
        """Return zero for excluded checkpoints and the inclusive step count otherwise."""
        if self.phase is IntervalPhase.CHECKPOINT:
            return 0
        return self.last_step - self.first_step + 1


@dataclass(frozen=True, slots=True)
class PerformanceRun:
    """Bounded metadata and intervals observed for one completed lease."""

    manifest_digest: str
    lease_digest: str
    precision: Precision
    model_shape_digest: str
    gpu_count: int
    global_batch_size: int
    fetch_ns: int
    setup_ns: int
    validator_observed_ns: int
    intervals: tuple[WorkInterval, ...]

    def __post_init__(self) -> None:
        _digest(self.manifest_digest)
        _digest(self.lease_digest)
        if not isinstance(self.precision, Precision):
            raise PerformanceFailure(PerformanceErrorCode.INVALID)
        _digest(self.model_shape_digest)
        _integer(self.gpu_count, 1, 16_384)
        _integer(self.global_batch_size, 1, 10_000_000)
        _integer(self.fetch_ns, 0, _MAX_INTEGER)
        _integer(self.setup_ns, 0, _MAX_INTEGER)
        _integer(self.validator_observed_ns, 1, _MAX_INTEGER)
        if not isinstance(self.intervals, tuple) or not 1 <= len(self.intervals) <= _MAX_INTERVALS:
            raise PerformanceFailure(PerformanceErrorCode.INVALID)
        if not all(isinstance(interval, WorkInterval) for interval in self.intervals):
            raise PerformanceFailure(PerformanceErrorCode.INVALID)


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    """Deterministic accepted work and rate with excluded-time accounting."""

    unit: WorkUnit
    accepted_units: int
    active_ns: int
    excluded_ns: int
    rate_milli_units_per_second: int
    active_steps: int
    global_batch_size: int
    precision: Precision
    model_shape_digest: str
    gpu_count: int
    validator_observed_ns: int


@dataclass(frozen=True, slots=True)
class PerformanceVerifier:
    """Derive useful throughput from ordered synchronized intervals."""

    policy: PerformancePolicy

    def measure(self, run: PerformanceRun) -> PerformanceResult:
        """Validate a complete run and derive units without trusting claimed rates."""
        if (
            run.manifest_digest != self.policy.manifest_digest
            or run.lease_digest != self.policy.lease_digest
        ):
            raise PerformanceFailure(PerformanceErrorCode.IDENTITY_MISMATCH)
        if (
            run.precision is not self.policy.precision
            or run.model_shape_digest != self.policy.model_shape_digest
            or run.gpu_count != self.policy.gpu_count
            or run.global_batch_size != self.policy.global_batch_size
        ):
            raise PerformanceFailure(PerformanceErrorCode.METADATA_MISMATCH)
        if run.fetch_ns + run.setup_ns > self.policy.maximum_setup_ns:
            raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)

        previous_end: int | None = None
        previous_work_step: int | None = None
        active_started = False
        first_active_ns: int | None = None
        last_active_ns: int | None = None
        warmup_steps = 0
        active_steps = 0
        active_batches = 0
        warmup_ns = 0
        checkpoint_intervals: list[WorkInterval] = []
        for expected_sequence, interval in enumerate(run.intervals):
            if interval.sequence != expected_sequence:
                raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)
            if interval.duration_ns > self.policy.maximum_interval_ns:
                raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)
            if previous_end is not None and interval.start_ns < previous_end:
                raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)
            previous_end = interval.end_ns
            if not interval.complete:
                raise PerformanceFailure(PerformanceErrorCode.PARTIAL_RESULT)
            if not interval.synchronized_before or not interval.synchronized_after:
                raise PerformanceFailure(PerformanceErrorCode.SYNCHRONIZATION_MISSING)

            if interval.phase is IntervalPhase.CHECKPOINT:
                if not active_started:
                    raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)
                checkpoint_intervals.append(interval)
                continue
            if previous_work_step is not None and interval.first_step != previous_work_step + 1:
                raise PerformanceFailure(PerformanceErrorCode.PARTIAL_RESULT)
            previous_work_step = interval.last_step
            expected_batches = interval.completed_steps * self.policy.batches_per_step
            if interval.completed_batches != expected_batches:
                raise PerformanceFailure(PerformanceErrorCode.BATCH_INFLATION)
            if interval.phase is IntervalPhase.WARMUP:
                if active_started:
                    raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)
                warmup_steps += interval.completed_steps
                warmup_ns += interval.duration_ns
                continue

            active_started = True
            active_steps += interval.completed_steps
            active_batches += interval.completed_batches
            if first_active_ns is None:
                first_active_ns = interval.start_ns
            last_active_ns = interval.end_ns

        if warmup_steps < self.policy.minimum_warmup_steps:
            raise PerformanceFailure(PerformanceErrorCode.WARMUP_INSUFFICIENT)
        if active_steps < self.policy.minimum_active_steps:
            raise PerformanceFailure(PerformanceErrorCode.WORK_INSUFFICIENT)
        if first_active_ns is None or last_active_ns is None:
            raise PerformanceFailure(PerformanceErrorCode.WORK_INSUFFICIENT)
        checkpoint_ns = sum(interval.duration_ns for interval in checkpoint_intervals)
        if checkpoint_ns > self.policy.maximum_checkpoint_ns:
            raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)

        active_checkpoint_ns = sum(
            interval.duration_ns
            for interval in checkpoint_intervals
            if interval.start_ns >= first_active_ns and interval.end_ns <= last_active_ns
        )
        active_ns = last_active_ns - first_active_ns - active_checkpoint_ns
        if active_ns <= 0 or active_ns > self.policy.maximum_active_ns:
            raise PerformanceFailure(PerformanceErrorCode.TIMER_INVALID)
        measured_span_ns = run.intervals[-1].end_ns - run.intervals[0].start_ns
        required_observer_ns = run.fetch_ns + run.setup_ns + measured_span_ns
        if (
            run.validator_observed_ns < required_observer_ns
            or run.validator_observed_ns > self.policy.maximum_validator_observed_ns
        ):
            raise PerformanceFailure(PerformanceErrorCode.OBSERVER_BOUND)

        accepted_samples = active_batches * self.policy.global_batch_size
        accepted_units = self._units(accepted_samples)
        if accepted_units <= 0 or accepted_units > _MAX_INTEGER:
            raise PerformanceFailure(PerformanceErrorCode.INVALID)
        rate = (accepted_units * 1_000 * _NANOSECONDS_PER_SECOND) // active_ns
        excluded_ns = run.fetch_ns + run.setup_ns + warmup_ns + checkpoint_ns
        return PerformanceResult(
            unit=self.policy.unit,
            accepted_units=accepted_units,
            active_ns=active_ns,
            excluded_ns=excluded_ns,
            rate_milli_units_per_second=rate,
            active_steps=active_steps,
            global_batch_size=self.policy.global_batch_size,
            precision=self.policy.precision,
            model_shape_digest=self.policy.model_shape_digest,
            gpu_count=self.policy.gpu_count,
            validator_observed_ns=run.validator_observed_ns,
        )

    def _units(self, accepted_samples: int) -> int:
        if self.policy.unit is WorkUnit.SAMPLES:
            return accepted_samples
        if self.policy.unit is WorkUnit.TOKENS:
            return accepted_samples * self.policy.tokens_per_sample
        return accepted_samples * self.policy.flop_proxy_per_sample


def _digest(value: object) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise PerformanceFailure(PerformanceErrorCode.INVALID)


def _integer(value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise PerformanceFailure(PerformanceErrorCode.INVALID)


__all__ = [
    "IntervalPhase",
    "PerformanceErrorCode",
    "PerformanceFailure",
    "PerformancePolicy",
    "PerformanceResult",
    "PerformanceRun",
    "PerformanceVerifier",
    "Precision",
    "WorkInterval",
    "WorkUnit",
]
