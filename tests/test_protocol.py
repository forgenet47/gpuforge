"""Tests for versioned protocol messages and canonical wire encoding."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from gpuforge.config import EvidenceTier, NetworkPolicy
from gpuforge.protocol import (
    MAX_WIRE_BYTES,
    PROTOCOL_VERSION,
    CapabilityClaim,
    CheckpointCommitment,
    ExecutionEvidence,
    JobManifest,
    Message,
    ProtocolDecodeError,
    ProtocolMessage,
    ProtocolValidationError,
    ResourcePolicy,
    SoftwareVersion,
    ValidationReceipt,
    VerificationPolicy,
    WorkLease,
    decode_message,
)

SIGNATURE = "ab" * 64
NONCE = "cd" * 32
MINER_HOTKEY = "5MinerHotkeyExample111111111111111111"
VALIDATOR_HOTKEY = "5ValidatorHotkeyExample11111111111111"
PUBLISHER_HOTKEY = "5PublisherHotkeyExample11111111111111"

GOLDEN_VECTORS = {
    "job_manifest": (
        893,
        "sha256:f23b5d67c86ebf79f6053950c4c72e30e4e289b38d08f6f2f07443105be12894",
    ),
    "capability_claim": (
        588,
        "sha256:6c55ec574083116cd205bea3d2615ddff287fa6352ee27044bc03213097473c7",
    ),
    "work_lease": (
        646,
        "sha256:c2d08d2cc208dfe760194a3c804867e685b91fd9b13e3d3b5d5a11cefe34bfce",
    ),
    "execution_evidence": (
        1024,
        "sha256:c83806b4779b037b14e0d8d7a7e3f3a5f79935fa728a86682d0ba839c3b87460",
    ),
    "validation_receipt": (
        472,
        "sha256:011f358e75e4f2b6ec9ccdfd60ef4d443ffe5c6be86c9092253d81c227fb4714",
    ),
}


def digest(character: str) -> str:
    """Return a syntactically valid SHA-256 content digest."""
    return f"sha256:{character * 64}"


def resource_policy() -> ResourcePolicy:
    """Return a representative bounded resource policy."""
    return ResourcePolicy(
        gpu_count=1,
        gpu_memory_mb=81_920,
        cpu_cores=16,
        memory_mb=131_072,
        max_runtime_seconds=3_600,
        network_policy=NetworkPolicy.DENY,
    )


def verification_policy() -> VerificationPolicy:
    """Return a representative local verification policy."""
    return VerificationPolicy(
        minimum_evidence_tier=EvidenceTier.C,
        challenge_kind="gradient_slice",
        checkpoint_interval_steps=100,
    )


def job_manifest() -> JobManifest:
    """Return a representative job manifest."""
    return JobManifest(
        job_id="job.example-001",
        container_digest=digest("1"),
        entrypoint_digest=digest("2"),
        input_root=digest("3"),
        framework="pytorch",
        resource_policy=resource_policy(),
        verification_policy=verification_policy(),
        lease_seconds=3_600,
        publisher_hotkey=PUBLISHER_HOTKEY,
        expires_at_block=1_000,
        signature=SIGNATURE,
    )


def capability_claim(*, reverse: bool = False) -> CapabilityClaim:
    """Return a representative miner capability claim."""
    versions: tuple[SoftwareVersion, ...] = (
        SoftwareVersion("cuda", "12.8"),
        SoftwareVersion("pytorch", "2.8.0"),
    )
    tiers: tuple[EvidenceTier, ...] = (EvidenceTier.A, EvidenceTier.C)
    if reverse:
        versions = tuple(reversed(versions))
        tiers = tuple(reversed(tiers))
    return CapabilityClaim(
        miner_hotkey=MINER_HOTKEY,
        gpu_count=1,
        gpu_model="NVIDIA H100 SXM",
        gpu_memory_mb=81_920,
        runtime_versions=versions,
        supported_evidence_tiers=tiers,
        available_gpu_seconds=7_200,
        nonce=NONCE,
        observed_at_block=900,
        signature=SIGNATURE,
    )


def work_lease() -> WorkLease:
    """Return a representative validator work lease."""
    return WorkLease(
        manifest_digest=job_manifest().digest(),
        miner_hotkey=MINER_HOTKEY,
        validator_hotkey=VALIDATOR_HOTKEY,
        shard_id="shard-0001",
        challenge_commitment=digest("4"),
        start_block=910,
        deadline_block=970,
        nonce=NONCE,
        signature=SIGNATURE,
    )


def execution_evidence(*, reverse: bool = False) -> ExecutionEvidence:
    """Return representative execution evidence."""
    checkpoints: tuple[CheckpointCommitment, ...] = (
        CheckpointCommitment(step=100, digest=digest("5")),
        CheckpointCommitment(step=200, digest=digest("6")),
    )
    if reverse:
        checkpoints = tuple(reversed(checkpoints))
    return ExecutionEvidence(
        lease_digest=work_lease().digest(),
        miner_hotkey=MINER_HOTKEY,
        attestation_digest=digest("7"),
        image_digest=digest("1"),
        challenge_response_digest=digest("8"),
        checkpoints=checkpoints,
        result_digest=digest("9"),
        work_units=4_096,
        active_seconds_ms=120_000,
        submitted_at_block=960,
        sequence=1,
        signature=SIGNATURE,
    )


def validation_receipt() -> ValidationReceipt:
    """Return a representative accepted validation receipt."""
    return ValidationReceipt(
        evidence_digest=execution_evidence().digest(),
        validator_hotkey=VALIDATOR_HOTKEY,
        accepted=True,
        reason_codes=(),
        verified_work_units=4_096,
        confidence_tier=EvidenceTier.A,
        validated_at_block=980,
        signature=SIGNATURE,
    )


def rejected_receipt(*, reverse: bool = False) -> ValidationReceipt:
    """Return a representative rejected validation receipt."""
    reasons: tuple[str, ...] = ("bad_challenge", "stale_evidence")
    if reverse:
        reasons = tuple(reversed(reasons))
    return ValidationReceipt(
        evidence_digest=execution_evidence().digest(),
        validator_hotkey=VALIDATOR_HOTKEY,
        accepted=False,
        reason_codes=reasons,
        verified_work_units=0,
        confidence_tier=None,
        validated_at_block=980,
        signature=SIGNATURE,
    )


def messages() -> tuple[Message, ...]:
    """Return one valid instance of every version-1 message type."""
    return (
        job_manifest(),
        capability_claim(),
        work_lease(),
        execution_evidence(),
        validation_receipt(),
    )


def encode_primitive(value: object) -> bytes:
    """Encode test mutations using the protocol's documented JSON profile."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@pytest.mark.parametrize("message", messages())
def test_all_messages_round_trip(message: Message) -> None:
    """Canonical bytes decode to the original immutable typed value."""
    encoded = message.canonical_bytes()
    decoded = decode_message(encoded)

    assert decoded == message
    assert decoded.canonical_bytes() == encoded
    assert decoded.digest() == message.digest()
    with pytest.raises(FrozenInstanceError):
        decoded.protocol_version = 2  # type: ignore[misc]


@pytest.mark.parametrize("message", messages())
def test_canonical_encoding_golden_vectors(message: Message) -> None:
    """Canonical sizes and domain-separated digests remain wire-compatible."""
    expected_size, expected_digest = GOLDEN_VECTORS[message.message_type]

    assert len(message.canonical_bytes()) == expected_size
    assert message.digest() == expected_digest


def test_set_like_and_keyed_sequences_have_semantic_ordering() -> None:
    """Equivalent input order produces identical canonical bytes and digests."""
    assert capability_claim().canonical_bytes() == capability_claim(reverse=True).canonical_bytes()
    assert execution_evidence().digest() == execution_evidence(reverse=True).digest()
    assert rejected_receipt().canonical_bytes() == rejected_receipt(reverse=True).canonical_bytes()


def test_digest_is_identical_in_a_separate_process() -> None:
    """Canonical digest identity is independent of the current Python process."""
    message = execution_evidence()
    script = (
        "import sys; "
        "from gpuforge.protocol import decode_message; "
        "print(decode_message(sys.stdin.buffer.read()).digest())"
    )
    result = subprocess.run(  # noqa: S603 - interpreter and script are test constants.
        [sys.executable, "-c", script],
        input=message.canonical_bytes(),
        capture_output=True,
        check=True,
    )

    assert result.stderr == b""
    assert result.stdout.decode("ascii").strip() == message.digest()


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value.update({"unexpected": True}),
        lambda value: cast(dict[str, object], value["payload"]).update({"unexpected": True}),
        lambda value: value.pop("protocol_version"),
    ),
)
def test_unknown_and_missing_fields_are_rejected(
    mutator: Callable[[dict[str, object]], object],
) -> None:
    """Top-level and payload schemas use an exact-field policy."""
    primitive = job_manifest().to_primitive()
    mutator(primitive)

    with pytest.raises((ProtocolDecodeError, ProtocolValidationError)):
        decode_message(encode_primitive(primitive))


def test_nested_unknown_field_is_rejected() -> None:
    """Strict field policy also applies to nested structures."""
    primitive = job_manifest().to_primitive()
    payload = cast(dict[str, object], primitive["payload"])
    policy = cast(dict[str, object], payload["resource_policy"])
    policy["debug"] = True

    with pytest.raises(ProtocolDecodeError, match="unknown fields"):
        decode_message(encode_primitive(primitive))


def test_duplicate_keys_are_rejected_before_schema_parsing() -> None:
    """Duplicate JSON fields cannot create signature or parser ambiguity."""
    encoded = work_lease().canonical_bytes()
    duplicate = encoded.replace(
        b'{"message_type"',
        b'{"message_type":"work_lease","message_type"',
        1,
    )

    with pytest.raises(ProtocolDecodeError, match="duplicate"):
        decode_message(duplicate)


@pytest.mark.parametrize(
    "invalid",
    (
        b"",
        b"not-json",
        b"\xff",
        b"[]",
        b"{}",
        b"{" + b"x" * MAX_WIRE_BYTES + b"}",
    ),
    ids=("empty", "not-json", "invalid-utf8", "array-root", "empty-object", "oversized"),
)
def test_malformed_and_oversized_inputs_are_rejected(invalid: bytes) -> None:
    """The decoder rejects invalid UTF-8, JSON, roots, schemas, and sizes."""
    with pytest.raises((ProtocolDecodeError, ProtocolValidationError)):
        decode_message(invalid)


def test_noncanonical_json_is_rejected() -> None:
    """Whitespace and equivalent alternate JSON representations are not accepted."""
    encoded = job_manifest().canonical_bytes()

    with pytest.raises(ProtocolDecodeError, match="not canonical"):
        decode_message(b" " + encoded)


def test_floats_constants_unknown_types_and_versions_are_rejected() -> None:
    """Wire messages cannot silently change numeric or protocol semantics."""
    primitive = job_manifest().to_primitive()
    payload = cast(dict[str, object], primitive["payload"])
    payload["lease_seconds"] = 60.0
    with pytest.raises(ProtocolDecodeError, match="floating-point"):
        decode_message(encode_primitive(primitive))

    primitive = job_manifest().to_primitive()
    primitive["message_type"] = "unknown"
    with pytest.raises(ProtocolDecodeError, match="Unsupported"):
        decode_message(encode_primitive(primitive))

    primitive = job_manifest().to_primitive()
    primitive["protocol_version"] = PROTOCOL_VERSION + 1
    with pytest.raises(ProtocolDecodeError, match="version"):
        decode_message(encode_primitive(primitive))

    encoded = job_manifest().canonical_bytes().replace(b"3600", b"NaN", 1)
    with pytest.raises(ProtocolDecodeError, match="non-finite"):
        decode_message(encoded)


def test_bool_is_not_accepted_as_an_integer() -> None:
    """JSON booleans cannot pass Python's integer subtype relationship."""
    primitive = job_manifest().to_primitive()
    payload = cast(dict[str, object], primitive["payload"])
    payload["lease_seconds"] = True

    with pytest.raises(ProtocolDecodeError, match="integer"):
        decode_message(encode_primitive(primitive))


def test_invalid_unicode_scalar_is_rejected() -> None:
    """Escaped surrogate code points cannot enter canonical UTF-8 fields."""
    primitive = job_manifest().to_primitive()
    payload = cast(dict[str, object], primitive["payload"])
    payload["framework"] = "\ud800"
    encoded = json.dumps(primitive, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    with pytest.raises(ProtocolDecodeError, match="UTF-8"):
        decode_message(encoded.encode("ascii"))


@pytest.mark.parametrize(
    "factory",
    (
        lambda: JobManifest(
            job_id="x" * 65,
            container_digest=digest("1"),
            entrypoint_digest=digest("2"),
            input_root=digest("3"),
            framework="pytorch",
            resource_policy=resource_policy(),
            verification_policy=verification_policy(),
            lease_seconds=3_600,
            publisher_hotkey=PUBLISHER_HOTKEY,
            expires_at_block=1_000,
            signature=SIGNATURE,
        ),
        lambda: CapabilityClaim(
            miner_hotkey=MINER_HOTKEY,
            gpu_count=1,
            gpu_model="H100-é",
            gpu_memory_mb=81_920,
            runtime_versions=(SoftwareVersion("cuda", "12.8"),),
            supported_evidence_tiers=(EvidenceTier.A,),
            available_gpu_seconds=1,
            nonce=NONCE,
            observed_at_block=1,
            signature=SIGNATURE,
        ),
        lambda: WorkLease(
            manifest_digest=digest("1"),
            miner_hotkey=MINER_HOTKEY,
            validator_hotkey=VALIDATOR_HOTKEY,
            shard_id="shard",
            challenge_commitment=digest("2"),
            start_block=10,
            deadline_block=10,
            nonce=NONCE,
            signature=SIGNATURE,
        ),
        lambda: CheckpointCommitment(step=0, digest=digest("1")),
    ),
)
def test_field_bounds_and_cross_field_rules(factory: Callable[[], object]) -> None:
    """Direct constructors enforce byte, numeric, text, and ordering bounds."""
    with pytest.raises(ProtocolValidationError):
        factory()


def test_capability_and_checkpoint_duplicates_are_rejected() -> None:
    """Keyed collections cannot carry ambiguous duplicate entries."""
    version = SoftwareVersion("cuda", "12.8")
    with pytest.raises(ProtocolValidationError, match="unique"):
        CapabilityClaim(
            miner_hotkey=MINER_HOTKEY,
            gpu_count=1,
            gpu_model="NVIDIA H100",
            gpu_memory_mb=81_920,
            runtime_versions=(version, version),
            supported_evidence_tiers=(EvidenceTier.A,),
            available_gpu_seconds=1,
            nonce=NONCE,
            observed_at_block=1,
            signature=SIGNATURE,
        )

    checkpoint = CheckpointCommitment(1, digest("1"))
    with pytest.raises(ProtocolValidationError, match="unique"):
        ExecutionEvidence(
            lease_digest=digest("1"),
            miner_hotkey=MINER_HOTKEY,
            attestation_digest=None,
            image_digest=digest("2"),
            challenge_response_digest=digest("3"),
            checkpoints=(checkpoint, checkpoint),
            result_digest=digest("4"),
            work_units=1,
            active_seconds_ms=1,
            submitted_at_block=1,
            sequence=0,
            signature=SIGNATURE,
        )


@pytest.mark.parametrize(
    "receipt",
    (
        lambda: ValidationReceipt(
            evidence_digest=digest("1"),
            validator_hotkey=VALIDATOR_HOTKEY,
            accepted=True,
            reason_codes=("unexpected",),
            verified_work_units=1,
            confidence_tier=EvidenceTier.A,
            validated_at_block=1,
            signature=SIGNATURE,
        ),
        lambda: ValidationReceipt(
            evidence_digest=digest("1"),
            validator_hotkey=VALIDATOR_HOTKEY,
            accepted=False,
            reason_codes=(),
            verified_work_units=0,
            confidence_tier=None,
            validated_at_block=1,
            signature=SIGNATURE,
        ),
    ),
)
def test_receipt_result_fields_are_consistent(receipt: Callable[[], ValidationReceipt]) -> None:
    """Accepted and rejected receipts cannot contain contradictory score fields."""
    with pytest.raises(ProtocolValidationError, match="inconsistent"):
        receipt()


def test_signature_hex_must_encode_exactly_64_bytes() -> None:
    """Protocol signatures use the fixed 64-byte hotkey signature width."""
    primitive = job_manifest().to_primitive()
    values = cast(dict[str, object], primitive["payload"])
    values["signature"] = "abc"

    with pytest.raises(ProtocolValidationError, match="byte-length"):
        JobManifest.from_payload(values, PROTOCOL_VERSION)


def test_wire_encoder_enforces_total_size_limit() -> None:
    """Even custom message implementations cannot bypass the wire ceiling."""

    class OversizedMessage(ProtocolMessage):
        message_type = "oversized"
        protocol_version = PROTOCOL_VERSION

        def _payload(self) -> dict[str, object]:
            return {"value": "x" * MAX_WIRE_BYTES}

    with pytest.raises(ProtocolValidationError, match="size limit"):
        OversizedMessage().canonical_bytes()


def test_wire_encoder_rejects_values_outside_canonical_profile() -> None:
    """Custom messages cannot introduce floats or unsupported JSON values."""

    class FloatMessage(ProtocolMessage):
        message_type = "float"
        protocol_version = PROTOCOL_VERSION

        def _payload(self) -> dict[str, object]:
            return {"value": 1.5}

    with pytest.raises(ProtocolValidationError, match="canonical JSON profile"):
        FloatMessage().canonical_bytes()


def test_decoder_requires_bytes() -> None:
    """Text input is rejected so callers cannot introduce implicit encodings."""
    with pytest.raises(ProtocolDecodeError, match="must be bytes"):
        decode_message(cast(bytes, "not-bytes"))
