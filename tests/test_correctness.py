"""Deterministic training-correctness and checkpoint commitment tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from gpuforge.correctness import (
    CheckpointLeaf,
    CorrectnessChallenge,
    CorrectnessErrorCode,
    CorrectnessFailure,
    CorrectnessPolicy,
    NumericTolerance,
    ProbeRequest,
    ProbeResult,
    ProbeTarget,
    ProofKind,
    TrainingCorrectnessVerifier,
    TrainingProofBundle,
    checkpoint_merkle_root,
    generate_correctness_challenge,
)

SCRIPT = "sha256:" + "1" * 64
HYPERPARAMETERS = "sha256:" + "2" * 64
OTHER_DIGEST = "sha256:" + "3" * 64
SEED = "4" * 64


def targets() -> tuple[ProbeTarget, ...]:
    return (
        ProbeTarget(ProofKind.LOSS, "loss.total", 1),
        ProbeTarget(ProofKind.GRADIENT, "encoder.weight.grad", 32),
        ProbeTarget(ProofKind.PARAMETER_DELTA, "encoder.weight.delta", 32),
        ProbeTarget(ProofKind.OPTIMIZER_STATE, "adam.exp_avg", 32),
        ProbeTarget(ProofKind.CANARY_OUTPUT, "canary.logits", 16),
    )


def challenge(seed: str = SEED) -> CorrectnessChallenge:
    return generate_correctness_challenge(
        seed=seed,
        script_digest=SCRIPT,
        hyperparameters_digest=HYPERPARAMETERS,
        checkpoint_step=100,
        candidate_steps=(10, 20, 30, 40, 50, 60, 70, 80, 90),
        targets=targets(),
        selected_step_count=3,
        values_per_target=4,
    )


def checkpoint_root() -> str:
    return checkpoint_merkle_root(
        (
            CheckpointLeaf("model/encoder", "sha256:" + "a" * 64),
            CheckpointLeaf("optimizer/adam", "sha256:" + "b" * 64),
            CheckpointLeaf("scheduler/state", "sha256:" + "c" * 64),
        )
    )


def probe_values(request: ProbeRequest) -> tuple[Decimal, ...]:
    return tuple(
        Decimal(request.step) / Decimal(100) + Decimal(index) / Decimal(1_000)
        for index in request.indices
    )


def bundle(**changes: object) -> TrainingProofBundle:
    selected = challenge()
    values: dict[str, object] = {
        "script_digest": SCRIPT,
        "hyperparameters_digest": HYPERPARAMETERS,
        "checkpoint_step": 100,
        "checkpoint_root": checkpoint_root(),
        "probes": tuple(
            ProbeResult(request, probe_values(request)) for request in selected.requests
        ),
    }
    values.update(changes)
    return TrainingProofBundle(**values)  # type: ignore[arg-type]


def policy() -> CorrectnessPolicy:
    return CorrectnessPolicy(
        tuple((kind, NumericTolerance(Decimal("0.001"), Decimal("0.0001"))) for kind in ProofKind)
    )


def test_hidden_step_and_slice_selection_is_deterministic_and_seed_bound() -> None:
    first = challenge()
    assert first == challenge()
    assert first != challenge("5" * 64)
    assert SEED not in repr(first)
    assert {request.kind for request in first.requests} == set(ProofKind)
    assert len({request.step for request in first.requests}) == 3
    assert all(len(request.indices) <= 4 for request in first.requests)


def test_large_tensor_sampling_is_bounded_by_requested_slice_size() -> None:
    selected = generate_correctness_challenge(
        seed=SEED,
        script_digest=SCRIPT,
        hyperparameters_digest=HYPERPARAMETERS,
        checkpoint_step=10,
        candidate_steps=(10,),
        targets=(
            ProbeTarget(ProofKind.LOSS, "loss.total", 1),
            ProbeTarget(ProofKind.GRADIENT, "large.weight", 2**62),
            ProbeTarget(ProofKind.PARAMETER_DELTA, "weight.delta", 1),
            ProbeTarget(ProofKind.OPTIMIZER_STATE, "adam.state", 1),
            ProbeTarget(ProofKind.CANARY_OUTPUT, "canary.output", 1),
        ),
        selected_step_count=1,
        values_per_target=4,
    )
    sampled = next(request for request in selected.requests if request.kind is ProofKind.GRADIENT)
    assert len(sampled.indices) == 4
    assert max(sampled.indices) < 2**62


def test_honest_reference_and_observation_are_accepted() -> None:
    selected = challenge()
    reference = bundle()
    result = TrainingCorrectnessVerifier(policy()).verify(selected, reference, reference)

    assert result.checked_probes == len(selected.requests)
    assert result.checked_values == sum(len(request.indices) for request in selected.requests)
    assert result.checkpoint_root == checkpoint_root()


def test_skipped_training_step_and_extra_probe_fail_closed() -> None:
    selected = challenge()
    reference = bundle()
    with pytest.raises(CorrectnessFailure) as missing:
        TrainingCorrectnessVerifier(policy()).verify(
            selected,
            reference,
            replace(reference, probes=reference.probes[1:]),
        )
    assert missing.value.code is CorrectnessErrorCode.MISSING_PROBE

    extra_request = replace(selected.requests[0], step=selected.requests[0].step + 1)
    extra = ProbeResult(extra_request, probe_values(extra_request))
    with pytest.raises(CorrectnessFailure) as unexpected:
        TrainingCorrectnessVerifier(policy()).verify(
            selected,
            reference,
            replace(reference, probes=reference.probes + (extra,)),
        )
    assert unexpected.value.code is CorrectnessErrorCode.UNEXPECTED_PROBE


@pytest.mark.parametrize("kind", [ProofKind.LOSS, ProofKind.GRADIENT, ProofKind.CANARY_OUTPUT])
def test_fabricated_training_signals_are_rejected(kind: ProofKind) -> None:
    selected = challenge()
    reference = bundle()
    index = next(
        position for position, probe in enumerate(reference.probes) if probe.request.kind is kind
    )
    forged = replace(
        reference.probes[index], values=(Decimal("999"),) * len(reference.probes[index].values)
    )
    probes = list(reference.probes)
    probes[index] = forged

    with pytest.raises(CorrectnessFailure) as error:
        TrainingCorrectnessVerifier(policy()).verify(
            selected,
            reference,
            replace(reference, probes=tuple(probes)),
        )
    assert error.value.code is CorrectnessErrorCode.NUMERIC_MISMATCH
    assert "999" not in str(error.value)


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        ({"checkpoint_step": 90}, CorrectnessErrorCode.STALE_CHECKPOINT),
        ({"checkpoint_root": OTHER_DIGEST}, CorrectnessErrorCode.CHECKPOINT_MISMATCH),
        ({"script_digest": OTHER_DIGEST}, CorrectnessErrorCode.IDENTITY_MISMATCH),
        ({"hyperparameters_digest": OTHER_DIGEST}, CorrectnessErrorCode.IDENTITY_MISMATCH),
    ),
)
def test_stale_checkpoint_modified_script_and_hyperparameters_are_rejected(
    changes: dict[str, object], code: CorrectnessErrorCode
) -> None:
    selected = challenge()
    reference = bundle()
    with pytest.raises(CorrectnessFailure) as error:
        TrainingCorrectnessVerifier(policy()).verify(selected, reference, bundle(**changes))
    assert error.value.code is code


def test_absolute_and_relative_numeric_tolerance_edges_are_inclusive() -> None:
    tolerance = NumericTolerance(Decimal("0.1"), Decimal("0.01"))
    assert tolerance.accepts(Decimal("0"), Decimal("0.1"))
    assert not tolerance.accepts(Decimal("0"), Decimal("0.1001"))
    assert tolerance.accepts(Decimal("100"), Decimal("101"))
    assert not tolerance.accepts(Decimal("100"), Decimal("101.001"))


def test_checkpoint_merkle_root_is_canonical_and_detects_changes() -> None:
    leaves = (
        CheckpointLeaf("model/encoder", "sha256:" + "a" * 64),
        CheckpointLeaf("optimizer/adam", "sha256:" + "b" * 64),
        CheckpointLeaf("scheduler/state", "sha256:" + "c" * 64),
    )
    root = checkpoint_merkle_root(leaves)
    assert checkpoint_merkle_root(tuple(reversed(leaves))) == root
    assert checkpoint_merkle_root((replace(leaves[0], digest=OTHER_DIGEST), *leaves[1:])) != root


@pytest.mark.parametrize(
    "factory",
    (
        lambda: NumericTolerance(Decimal("NaN"), Decimal("0")),
        lambda: NumericTolerance(Decimal("1e1001"), Decimal("0")),
        lambda: NumericTolerance(Decimal("-0.1"), Decimal("0")),
        lambda: ProbeTarget(ProofKind.GRADIENT, "../host", 1),
        lambda: CheckpointLeaf("/absolute", SCRIPT),
        lambda: checkpoint_merkle_root(
            (CheckpointLeaf("model", SCRIPT), CheckpointLeaf("model", OTHER_DIGEST))
        ),
        lambda: CorrectnessPolicy(
            ((ProofKind.LOSS, NumericTolerance(Decimal("0"), Decimal("0"))),)
        ),
    ),
)
def test_invalid_numeric_shapes_paths_and_policies_are_rejected(factory: object) -> None:
    with pytest.raises(CorrectnessFailure) as error:
        factory()  # type: ignore[operator]
    assert error.value.code is CorrectnessErrorCode.INVALID


def test_duplicate_indices_candidates_and_probe_results_are_rejected() -> None:
    with pytest.raises(CorrectnessFailure):
        ProbeRequest(ProofKind.GRADIENT, 1, "weight.grad", (0, 0))

    with pytest.raises(CorrectnessFailure):
        generate_correctness_challenge(
            seed=SEED,
            script_digest=SCRIPT,
            hyperparameters_digest=HYPERPARAMETERS,
            checkpoint_step=10,
            candidate_steps=(5, 5),
            targets=targets(),
            selected_step_count=1,
            values_per_target=1,
        )

    honest = bundle()
    with pytest.raises(CorrectnessFailure):
        replace(honest, probes=(honest.probes[0], honest.probes[0]))
