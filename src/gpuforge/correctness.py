"""Deterministic training-correctness challenges and checkpoint commitments."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HEX_256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TENSOR_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_MAX_PROBES = 4_096
_MAX_VALUES_PER_PROBE = 64
_MAX_CANDIDATE_STEPS = 100_000
_MAX_TENSOR_ELEMENTS = 2**63 - 1
_MAX_CHECKPOINT_LEAVES = 65_536
_STEP_DOMAIN = b"gpuforge/correctness/steps/v1\x00"
_SLICE_DOMAIN = b"gpuforge/correctness/slice/v1\x00"
_LEAF_DOMAIN = b"gpuforge/checkpoint/leaf/v1\x00"
_NODE_DOMAIN = b"gpuforge/checkpoint/node/v1\x00"


class ProofKind(str, Enum):
    """Training signals sampled by a correctness challenge."""

    LOSS = "loss"
    GRADIENT = "gradient"
    PARAMETER_DELTA = "parameter_delta"
    OPTIMIZER_STATE = "optimizer_state"
    CANARY_OUTPUT = "canary_output"


class CorrectnessErrorCode(str, Enum):
    """Stable correctness failures that contain no tensor values."""

    INVALID = "invalid_correctness_proof"
    IDENTITY_MISMATCH = "training_identity_mismatch"
    STALE_CHECKPOINT = "stale_checkpoint"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    MISSING_PROBE = "missing_probe"
    UNEXPECTED_PROBE = "unexpected_probe"
    NUMERIC_MISMATCH = "numeric_mismatch"


class CorrectnessFailure(RuntimeError):
    """Sanitized verification failure containing only a stable code."""

    def __init__(self, code: CorrectnessErrorCode) -> None:
        if not isinstance(code, CorrectnessErrorCode):
            raise ValueError("Correctness failure code is invalid")
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class NumericTolerance:
    """Absolute and relative bounds applied to one proof kind."""

    absolute: Decimal
    relative: Decimal

    def __post_init__(self) -> None:
        _decimal(self.absolute)
        _decimal(self.relative)
        if self.absolute < 0 or self.relative < 0:
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)

    def accepts(self, expected: Decimal, observed: Decimal) -> bool:
        """Return whether one finite value is inside the deterministic bound."""
        _decimal(expected)
        _decimal(observed)
        difference = abs(observed - expected)
        bound = max(self.absolute, self.relative * abs(expected))
        return difference <= bound


@dataclass(frozen=True, slots=True)
class ProbeTarget:
    """A logical tensor or scalar family eligible for hidden sampling."""

    kind: ProofKind
    tensor_name: str
    element_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProofKind):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        _tensor_name(self.tensor_name)
        _integer(self.element_count, 1, _MAX_TENSOR_ELEMENTS)


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """Exact step and tensor indices selected by the validator."""

    kind: ProofKind
    step: int
    tensor_name: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProofKind):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        _integer(self.step, 1, 2**63 - 1)
        _tensor_name(self.tensor_name)
        if (
            not isinstance(self.indices, tuple)
            or not 1 <= len(self.indices) <= _MAX_VALUES_PER_PROBE
        ):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        for index in self.indices:
            _integer(index, 0, _MAX_TENSOR_ELEMENTS - 1)
        if len(set(self.indices)) != len(self.indices):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)

    @property
    def key(self) -> tuple[ProofKind, int, str, tuple[int, ...]]:
        """Return the exact challenge identity without any numeric values."""
        return (self.kind, self.step, self.tensor_name, self.indices)


@dataclass(frozen=True, slots=True)
class CorrectnessChallenge:
    """Lease-time hidden selection bound to immutable training identities."""

    seed: str = field(repr=False)
    script_digest: str
    hyperparameters_digest: str
    checkpoint_step: int
    requests: tuple[ProbeRequest, ...]

    def __post_init__(self) -> None:
        _hex_256(self.seed)
        _digest(self.script_digest)
        _digest(self.hyperparameters_digest)
        _integer(self.checkpoint_step, 1, 2**63 - 1)
        if not isinstance(self.requests, tuple) or not 1 <= len(self.requests) <= _MAX_PROBES:
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        if not all(isinstance(request, ProbeRequest) for request in self.requests):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        keys = [request.key for request in self.requests]
        if len(keys) != len(set(keys)):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        if any(request.step > self.checkpoint_step for request in self.requests):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


def generate_correctness_challenge(
    *,
    seed: str,
    script_digest: str,
    hyperparameters_digest: str,
    checkpoint_step: int,
    candidate_steps: tuple[int, ...],
    targets: tuple[ProbeTarget, ...],
    selected_step_count: int,
    values_per_target: int,
) -> CorrectnessChallenge:
    """Select deterministic but seed-unpredictable steps and tensor slices."""
    _hex_256(seed)
    _digest(script_digest)
    _digest(hyperparameters_digest)
    _integer(checkpoint_step, 1, 2**63 - 1)
    if (
        not isinstance(candidate_steps, tuple)
        or not candidate_steps
        or len(candidate_steps) > _MAX_CANDIDATE_STEPS
    ):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    for step in candidate_steps:
        _integer(step, 1, checkpoint_step)
    if len(set(candidate_steps)) != len(candidate_steps):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    if not isinstance(targets, tuple) or not targets:
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    if not all(isinstance(target, ProbeTarget) for target in targets):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    target_keys = [(target.kind, target.tensor_name) for target in targets]
    if len(target_keys) != len(set(target_keys)):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    if {target.kind for target in targets} != set(ProofKind):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    _integer(selected_step_count, 1, len(candidate_steps))
    _integer(values_per_target, 1, _MAX_VALUES_PER_PROBE)
    if selected_step_count * len(targets) > _MAX_PROBES:
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)

    seed_bytes = bytes.fromhex(seed)
    selected_steps = _ranked(seed_bytes, _STEP_DOMAIN, candidate_steps)[:selected_step_count]
    requests: list[ProbeRequest] = []
    for step in sorted(selected_steps):
        for target in sorted(targets, key=lambda item: (item.kind.value, item.tensor_name)):
            sample_count = min(values_per_target, target.element_count)
            context = f"{target.kind.value}\x00{target.tensor_name}\x00{step}".encode("ascii")
            indices = _sample_indices(
                seed_bytes,
                _SLICE_DOMAIN + context + b"\x00",
                target.element_count,
                sample_count,
            )
            requests.append(
                ProbeRequest(
                    kind=target.kind,
                    step=step,
                    tensor_name=target.tensor_name,
                    indices=tuple(sorted(indices)),
                )
            )
    return CorrectnessChallenge(
        seed=seed,
        script_digest=script_digest,
        hyperparameters_digest=hyperparameters_digest,
        checkpoint_step=checkpoint_step,
        requests=tuple(requests),
    )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Finite numeric values for exactly one requested tensor slice."""

    request: ProbeRequest
    values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, ProbeRequest):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        if not isinstance(self.values, tuple) or len(self.values) != len(self.request.indices):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        for value in self.values:
            _decimal(value)


@dataclass(frozen=True, slots=True)
class TrainingProofBundle:
    """Identity, checkpoint, and selected numeric evidence for one run."""

    script_digest: str
    hyperparameters_digest: str
    checkpoint_step: int
    checkpoint_root: str
    probes: tuple[ProbeResult, ...]

    def __post_init__(self) -> None:
        _digest(self.script_digest)
        _digest(self.hyperparameters_digest)
        _integer(self.checkpoint_step, 1, 2**63 - 1)
        _digest(self.checkpoint_root)
        if not isinstance(self.probes, tuple) or not 1 <= len(self.probes) <= _MAX_PROBES:
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        if not all(isinstance(probe, ProbeResult) for probe in self.probes):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        keys = [probe.request.key for probe in self.probes]
        if len(keys) != len(set(keys)):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


@dataclass(frozen=True, slots=True)
class CorrectnessPolicy:
    """Per-signal numeric bounds required by the validator."""

    tolerances: tuple[tuple[ProofKind, NumericTolerance], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tolerances, tuple):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
        values: dict[ProofKind, NumericTolerance] = {}
        for item in self.tolerances:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], ProofKind)
                or not isinstance(item[1], NumericTolerance)
                or item[0] in values
            ):
                raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
            values[item[0]] = item[1]
        if set(values) != set(ProofKind):
            raise CorrectnessFailure(CorrectnessErrorCode.INVALID)

    def tolerance_for(self, kind: ProofKind) -> NumericTolerance:
        """Return the configured bound for one required proof kind."""
        for selected_kind, tolerance in self.tolerances:
            if selected_kind is kind:
                return tolerance
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """Sanitized accepted-proof summary."""

    checked_probes: int
    checked_values: int
    checkpoint_root: str


@dataclass(frozen=True, slots=True)
class TrainingCorrectnessVerifier:
    """Compare selected miner observations with a trusted reference run."""

    policy: CorrectnessPolicy

    def verify(
        self,
        challenge: CorrectnessChallenge,
        expected: TrainingProofBundle,
        observed: TrainingProofBundle,
    ) -> CorrectnessResult:
        """Fail closed on identity, freshness, commitment, shape, or value mismatch."""
        for bundle in (expected, observed):
            if (
                bundle.script_digest != challenge.script_digest
                or bundle.hyperparameters_digest != challenge.hyperparameters_digest
            ):
                raise CorrectnessFailure(CorrectnessErrorCode.IDENTITY_MISMATCH)
            if bundle.checkpoint_step != challenge.checkpoint_step:
                raise CorrectnessFailure(CorrectnessErrorCode.STALE_CHECKPOINT)
        if not hmac.compare_digest(expected.checkpoint_root, observed.checkpoint_root):
            raise CorrectnessFailure(CorrectnessErrorCode.CHECKPOINT_MISMATCH)

        required = {request.key: request for request in challenge.requests}
        expected_values = {probe.request.key: probe for probe in expected.probes}
        observed_values = {probe.request.key: probe for probe in observed.probes}
        if not set(required).issubset(expected_values) or not set(required).issubset(
            observed_values
        ):
            raise CorrectnessFailure(CorrectnessErrorCode.MISSING_PROBE)
        if set(expected_values) != set(required) or set(observed_values) != set(required):
            raise CorrectnessFailure(CorrectnessErrorCode.UNEXPECTED_PROBE)

        checked_values = 0
        for request in challenge.requests:
            expected_probe = expected_values[request.key]
            observed_probe = observed_values[request.key]
            tolerance = self.policy.tolerance_for(request.kind)
            for expected_value, observed_value in zip(
                expected_probe.values, observed_probe.values, strict=True
            ):
                checked_values += 1
                if not tolerance.accepts(expected_value, observed_value):
                    raise CorrectnessFailure(CorrectnessErrorCode.NUMERIC_MISMATCH)
        return CorrectnessResult(
            checked_probes=len(challenge.requests),
            checked_values=checked_values,
            checkpoint_root=observed.checkpoint_root,
        )


@dataclass(frozen=True, slots=True)
class CheckpointLeaf:
    """One logical checkpoint component committed by content digest."""

    name: str
    digest: str

    def __post_init__(self) -> None:
        _tensor_name(self.name)
        _digest(self.digest)


def checkpoint_merkle_root(leaves: tuple[CheckpointLeaf, ...]) -> str:
    """Return a canonical binary Merkle root for logical checkpoint components."""
    if not isinstance(leaves, tuple) or not 1 <= len(leaves) <= _MAX_CHECKPOINT_LEAVES:
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    if not all(isinstance(leaf, CheckpointLeaf) for leaf in leaves):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    ordered = sorted(leaves, key=lambda leaf: leaf.name)
    if len({leaf.name for leaf in ordered}) != len(ordered):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)
    level = [
        hashlib.sha256(
            _LEAF_DOMAIN
            + len(leaf.name.encode("ascii")).to_bytes(2, "big")
            + leaf.name.encode("ascii")
            + bytes.fromhex(leaf.digest.removeprefix("sha256:"))
        ).digest()
        for leaf in ordered
    ]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(_NODE_DOMAIN + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return "sha256:" + level[0].hex()


def _ranked(seed: bytes, domain: bytes, values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hmac.digest(seed, domain + value.to_bytes(8, "big"), "sha256"),
                value,
            ),
        )
    )


def _sample_indices(seed: bytes, domain: bytes, size: int, count: int) -> tuple[int, ...]:
    selected: set[int] = set()
    counter = 0
    modulus = 1 << 256
    unbiased_limit = modulus - (modulus % size)
    while len(selected) < count:
        candidate = int.from_bytes(
            hmac.digest(seed, domain + counter.to_bytes(8, "big"), "sha256"),
            "big",
        )
        counter += 1
        if candidate < unbiased_limit:
            selected.add(candidate % size)
    return tuple(sorted(selected))


def _digest(value: object) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


def _hex_256(value: object) -> None:
    if not isinstance(value, str) or _HEX_256_PATTERN.fullmatch(value) is None:
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


def _tensor_name(value: object) -> None:
    if (
        not isinstance(value, str)
        or _TENSOR_PATTERN.fullmatch(value) is None
        or value.startswith(("/", "."))
        or ".." in value.split("/")
        or "\\" in value
    ):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


def _integer(value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


def _decimal(value: object) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or len(value.as_tuple().digits) > 64
        or not -1_000 <= value.adjusted() <= 1_000
    ):
        raise CorrectnessFailure(CorrectnessErrorCode.INVALID)


__all__ = [
    "CheckpointLeaf",
    "CorrectnessChallenge",
    "CorrectnessErrorCode",
    "CorrectnessFailure",
    "CorrectnessPolicy",
    "CorrectnessResult",
    "NumericTolerance",
    "ProbeRequest",
    "ProbeResult",
    "ProbeTarget",
    "ProofKind",
    "TrainingCorrectnessVerifier",
    "TrainingProofBundle",
    "checkpoint_merkle_root",
    "generate_correctness_challenge",
]
