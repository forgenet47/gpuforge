"""Synchronized useful-work measurement and manipulation tests."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from gpuforge.performance import (
    IntervalPhase,
    PerformanceErrorCode,
    PerformanceFailure,
    PerformancePolicy,
    PerformanceRun,
    PerformanceVerifier,
    Precision,
    WorkInterval,
    WorkUnit,
)

MANIFEST = "sha256:" + "1" * 64
LEASE = "sha256:" + "2" * 64
MODEL_SHAPE = "sha256:" + "3" * 64
MILLISECOND = 1_000_000


def policy(unit: WorkUnit = WorkUnit.SAMPLES, **changes: object) -> PerformancePolicy:
    values: dict[str, object] = {
        "manifest_digest": MANIFEST,
        "lease_digest": LEASE,
        "unit": unit,
        "global_batch_size": 8,
        "tokens_per_sample": 512,
        "flop_proxy_per_sample": 10_000,
        "batches_per_step": 1,
        "precision": Precision.BF16,
        "model_shape_digest": MODEL_SHAPE,
        "gpu_count": 2,
        "minimum_warmup_steps": 2,
        "minimum_active_steps": 5,
        "maximum_interval_ns": 60 * MILLISECOND,
        "maximum_active_ns": 1_000 * MILLISECOND,
        "maximum_checkpoint_ns": 20 * MILLISECOND,
        "maximum_setup_ns": 20 * MILLISECOND,
        "maximum_validator_observed_ns": 2_000 * MILLISECOND,
    }
    values.update(changes)
    return PerformancePolicy(**values)  # type: ignore[arg-type]


def intervals() -> tuple[WorkInterval, ...]:
    return (
        WorkInterval(0, IntervalPhase.WARMUP, 0, 10 * MILLISECOND, 1, 2, 2, True, True),
        WorkInterval(
            1,
            IntervalPhase.ACTIVE,
            10 * MILLISECOND,
            40 * MILLISECOND,
            3,
            5,
            3,
            True,
            True,
        ),
        WorkInterval(
            2,
            IntervalPhase.CHECKPOINT,
            40 * MILLISECOND,
            50 * MILLISECOND,
            0,
            0,
            0,
            True,
            True,
        ),
        WorkInterval(
            3,
            IntervalPhase.ACTIVE,
            50 * MILLISECOND,
            70 * MILLISECOND,
            6,
            7,
            2,
            True,
            True,
        ),
    )


def run(**changes: object) -> PerformanceRun:
    values: dict[str, object] = {
        "manifest_digest": MANIFEST,
        "lease_digest": LEASE,
        "precision": Precision.BF16,
        "model_shape_digest": MODEL_SHAPE,
        "gpu_count": 2,
        "global_batch_size": 8,
        "fetch_ns": 5 * MILLISECOND,
        "setup_ns": 5 * MILLISECOND,
        "validator_observed_ns": 80 * MILLISECOND,
        "intervals": intervals(),
    }
    values.update(changes)
    return PerformanceRun(**values)  # type: ignore[arg-type]


def test_stable_measurement_excludes_setup_warmup_and_checkpoint_time() -> None:
    result = PerformanceVerifier(policy()).measure(run())

    assert result.accepted_units == 40
    assert result.active_steps == 5
    assert result.active_ns == 50 * MILLISECOND
    assert result.excluded_ns == 30 * MILLISECOND
    assert result.rate_milli_units_per_second == 800_000
    assert result.global_batch_size == 8
    assert result.gpu_count == 2


@pytest.mark.parametrize(
    ("unit", "expected"),
    (
        (WorkUnit.SAMPLES, 40),
        (WorkUnit.TOKENS, 40 * 512),
        (WorkUnit.FLOP_PROXY, 40 * 10_000),
    ),
)
def test_units_are_derived_from_accepted_batches(unit: WorkUnit, expected: int) -> None:
    assert PerformanceVerifier(policy(unit)).measure(run()).accepted_units == expected


def test_multi_gpu_count_is_metadata_not_a_second_work_multiplier() -> None:
    result = PerformanceVerifier(policy(gpu_count=4)).measure(run(gpu_count=4))
    assert result.accepted_units == 5 * 8
    assert result.gpu_count == 4


def test_unclassified_sleep_reduces_measured_rate() -> None:
    compact = PerformanceVerifier(policy()).measure(run())
    delayed = list(intervals())
    delayed[3] = replace(
        delayed[3],
        start_ns=70 * MILLISECOND,
        end_ns=90 * MILLISECOND,
    )
    slow = PerformanceVerifier(policy()).measure(
        run(intervals=tuple(delayed), validator_observed_ns=100 * MILLISECOND)
    )

    assert slow.accepted_units == compact.accepted_units
    assert slow.active_ns == 70 * MILLISECOND
    assert slow.rate_milli_units_per_second < compact.rate_milli_units_per_second


@pytest.mark.parametrize(
    "changes",
    (
        {"precision": Precision.FP16},
        {"model_shape_digest": "sha256:" + "4" * 64},
        {"gpu_count": 1},
        {"global_batch_size": 16},
    ),
)
def test_precision_shape_gpu_and_batch_metadata_are_bound(changes: dict[str, object]) -> None:
    with pytest.raises(PerformanceFailure) as error:
        PerformanceVerifier(policy()).measure(run(**changes))
    assert error.value.code is PerformanceErrorCode.METADATA_MISMATCH


def test_timer_overlap_and_validator_undercount_are_rejected() -> None:
    overlapping = list(intervals())
    overlapping[1] = replace(overlapping[1], start_ns=5 * MILLISECOND)
    with pytest.raises(PerformanceFailure) as timer:
        PerformanceVerifier(policy()).measure(run(intervals=tuple(overlapping)))
    assert timer.value.code is PerformanceErrorCode.TIMER_INVALID

    with pytest.raises(PerformanceFailure) as observer:
        PerformanceVerifier(policy()).measure(run(validator_observed_ns=79 * MILLISECOND))
    assert observer.value.code is PerformanceErrorCode.OBSERVER_BOUND


def test_asynchronous_cuda_interval_is_never_accepted() -> None:
    unsynchronized = list(intervals())
    unsynchronized[1] = replace(unsynchronized[1], synchronized_after=False)
    with pytest.raises(PerformanceFailure) as error:
        PerformanceVerifier(policy()).measure(run(intervals=tuple(unsynchronized)))
    assert error.value.code is PerformanceErrorCode.SYNCHRONIZATION_MISSING


def test_batch_inflation_partial_result_and_step_gap_fail_closed() -> None:
    inflated = list(intervals())
    inflated[1] = replace(inflated[1], completed_batches=30)
    with pytest.raises(PerformanceFailure) as batch:
        PerformanceVerifier(policy()).measure(run(intervals=tuple(inflated)))
    assert batch.value.code is PerformanceErrorCode.BATCH_INFLATION

    partial = list(intervals())
    partial[1] = replace(partial[1], complete=False)
    with pytest.raises(PerformanceFailure) as incomplete:
        PerformanceVerifier(policy()).measure(run(intervals=tuple(partial)))
    assert incomplete.value.code is PerformanceErrorCode.PARTIAL_RESULT

    gap = list(intervals())
    gap[3] = replace(gap[3], first_step=7, last_step=8)
    with pytest.raises(PerformanceFailure) as missing:
        PerformanceVerifier(policy()).measure(run(intervals=tuple(gap)))
    assert missing.value.code is PerformanceErrorCode.PARTIAL_RESULT


def test_warmup_and_active_work_minimums_are_enforced() -> None:
    no_warmup = tuple(replace(item, sequence=index) for index, item in enumerate(intervals()[1:]))
    with pytest.raises(PerformanceFailure) as warmup:
        PerformanceVerifier(policy()).measure(run(intervals=no_warmup))
    assert warmup.value.code is PerformanceErrorCode.WARMUP_INSUFFICIENT

    with pytest.raises(PerformanceFailure) as active:
        PerformanceVerifier(policy(minimum_active_steps=6)).measure(run())
    assert active.value.code is PerformanceErrorCode.WORK_INSUFFICIENT


def test_checkpoint_budget_and_post_run_checkpoint_accounting() -> None:
    with pytest.raises(PerformanceFailure) as excessive:
        PerformanceVerifier(policy(maximum_checkpoint_ns=9 * MILLISECOND)).measure(run())
    assert excessive.value.code is PerformanceErrorCode.TIMER_INVALID

    completed = intervals() + (
        WorkInterval(
            4,
            IntervalPhase.CHECKPOINT,
            70 * MILLISECOND,
            75 * MILLISECOND,
            0,
            0,
            0,
            True,
            True,
        ),
    )
    result = PerformanceVerifier(policy()).measure(
        run(intervals=completed, validator_observed_ns=85 * MILLISECOND)
    )
    assert result.active_ns == 50 * MILLISECOND
    assert result.excluded_ns == 35 * MILLISECOND


def test_identity_mismatch_is_separate_from_metadata_failure() -> None:
    with pytest.raises(PerformanceFailure) as error:
        PerformanceVerifier(policy()).measure(run(lease_digest="sha256:" + "9" * 64))
    assert error.value.code is PerformanceErrorCode.IDENTITY_MISMATCH


@pytest.mark.parametrize(
    "factory",
    (
        lambda: WorkInterval(0, IntervalPhase.ACTIVE, 1, 1, 1, 1, 1, True, True),
        lambda: WorkInterval(
            0,
            IntervalPhase.ACTIVE,
            1,
            2,
            1,
            1,
            cast(int, float("nan")),
            True,
            True,
        ),
        lambda: PerformancePolicy(
            MANIFEST,
            LEASE,
            WorkUnit.SAMPLES,
            0,
            1,
            1,
            1,
            Precision.BF16,
            MODEL_SHAPE,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
        ),
    ),
)
def test_zero_nan_and_invalid_durations_are_rejected(factory: object) -> None:
    with pytest.raises(PerformanceFailure):
        factory()  # type: ignore[operator]


def test_sequence_phase_order_and_setup_ceiling_fail_closed() -> None:
    bad_sequence = list(intervals())
    bad_sequence[1] = replace(bad_sequence[1], sequence=4)
    with pytest.raises(PerformanceFailure) as sequence:
        PerformanceVerifier(policy()).measure(run(intervals=tuple(bad_sequence)))
    assert sequence.value.code is PerformanceErrorCode.TIMER_INVALID

    checkpoint_first = (replace(intervals()[2], sequence=0),)
    with pytest.raises(PerformanceFailure) as phase:
        PerformanceVerifier(policy()).measure(run(intervals=checkpoint_first))
    assert phase.value.code is PerformanceErrorCode.TIMER_INVALID

    with pytest.raises(PerformanceFailure) as setup:
        PerformanceVerifier(policy()).measure(
            run(fetch_ns=11 * MILLISECOND, setup_ns=10 * MILLISECOND)
        )
    assert setup.value.code is PerformanceErrorCode.TIMER_INVALID


def test_warmup_after_active_and_missing_active_interval_are_rejected() -> None:
    active_then_warmup = (
        WorkInterval(0, IntervalPhase.ACTIVE, 0, 10, 1, 1, 1, True, True),
        WorkInterval(1, IntervalPhase.WARMUP, 10, 20, 2, 3, 2, True, True),
    )
    with pytest.raises(PerformanceFailure) as order:
        PerformanceVerifier(policy(minimum_active_steps=1)).measure(
            run(
                intervals=active_then_warmup,
                fetch_ns=0,
                setup_ns=0,
                validator_observed_ns=20,
            )
        )
    assert order.value.code is PerformanceErrorCode.TIMER_INVALID

    warmup_only = (intervals()[0],)
    with pytest.raises(PerformanceFailure) as absent:
        PerformanceVerifier(policy()).measure(
            run(
                intervals=warmup_only,
                fetch_ns=0,
                setup_ns=0,
                validator_observed_ns=10 * MILLISECOND,
            )
        )
    assert absent.value.code is PerformanceErrorCode.WORK_INSUFFICIENT
