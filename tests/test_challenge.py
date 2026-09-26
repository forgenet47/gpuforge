from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from gpuforge.challenge import (
    ChallengeCommitment,
    ChallengeErrorCode,
    ChallengeFailure,
    ChallengeReveal,
    ChallengeVerifier,
    ValidatorChallengeGenerator,
)

JOB = "sha256:" + "a" * 64
OTHER_JOB = "sha256:" + "b" * 64
NONCE = "c" * 64
OTHER_NONCE = "d" * 64
SECRET = b"validator entropy used only in deterministic tests"
CHALLENGE_ID = "e" * 64


def generator() -> ValidatorChallengeGenerator:
    return ValidatorChallengeGenerator(SECRET)


def commitment(**overrides: object) -> ChallengeCommitment:
    values: dict[str, object] = {
        "job_digest": JOB,
        "lease_nonce": NONCE,
        "deadline_block": 500,
        "shard_count": 7,
        "canary_pool_size": 11,
        "canary_count": 3,
        "challenge_id": CHALLENGE_ID,
    }
    values.update(overrides)
    return generator().commit(**values)  # type: ignore[arg-type]


def reveal(value: ChallengeCommitment | None = None) -> ChallengeReveal:
    selected = value or commitment()
    return generator().reveal(
        selected,
        job_digest=JOB,
        lease_nonce=NONCE,
        accepted=True,
        current_block=400,
    )


def test_commit_reveal_is_deterministic_and_well_formed() -> None:
    public = commitment()
    first = reveal(public)
    second = reveal(public)

    assert first == second
    assert sorted(first.shard_order) == list(range(public.shard_count))
    assert len(first.canary_indices) == public.canary_count
    assert len(set(first.canary_indices)) == public.canary_count
    assert all(index < public.canary_pool_size for index in first.canary_indices)


def test_distinct_ids_make_commitments_and_selections_unpredictably_distinct() -> None:
    ids = iter(("1" * 64, "2" * 64))
    source = ValidatorChallengeGenerator(SECRET, id_factory=ids.__next__)
    first = source.commit(
        job_digest=JOB,
        lease_nonce=NONCE,
        deadline_block=500,
        shard_count=7,
        canary_pool_size=11,
        canary_count=3,
    )
    second = source.commit(
        job_digest=JOB,
        lease_nonce=NONCE,
        deadline_block=500,
        shard_count=7,
        canary_pool_size=11,
        canary_count=3,
    )

    assert first.challenge_id != second.challenge_id
    assert first.commitment_digest != second.commitment_digest
    assert (
        source.reveal(
            first,
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=True,
            current_block=400,
        ).seed
        != source.reveal(
            second,
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=True,
            current_block=400,
        ).seed
    )


def test_reveal_requires_acceptance_and_unexpired_commitment() -> None:
    public = commitment()
    with pytest.raises(ChallengeFailure) as not_accepted:
        generator().reveal(
            public,
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=False,
            current_block=400,
        )
    assert not_accepted.value.code is ChallengeErrorCode.NOT_ACCEPTED

    with pytest.raises(ChallengeFailure) as expired:
        generator().reveal(
            public,
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=True,
            current_block=501,
        )
    assert expired.value.code is ChallengeErrorCode.EXPIRED


def test_commitment_is_bound_to_job_lease_and_deadline() -> None:
    public = commitment()
    disclosed = reveal(public)
    verifier = ChallengeVerifier()

    for job_digest, lease_nonce, selected in (
        (OTHER_JOB, NONCE, public),
        (JOB, OTHER_NONCE, public),
        (JOB, NONCE, replace(public, deadline_block=501)),
    ):
        with pytest.raises(ChallengeFailure) as error:
            verifier.verify_and_consume(
                selected,
                disclosed,
                job_digest=job_digest,
                lease_nonce=lease_nonce,
                accepted=True,
                current_block=400,
            )
        assert error.value.code is ChallengeErrorCode.MISMATCH


def test_verifier_accepts_once_then_rejects_replay() -> None:
    public = commitment()
    disclosed = reveal(public)
    verifier = ChallengeVerifier()

    verifier.verify_and_consume(
        public,
        disclosed,
        job_digest=JOB,
        lease_nonce=NONCE,
        accepted=True,
        current_block=500,
    )
    with pytest.raises(ChallengeFailure) as replayed:
        verifier.verify_and_consume(
            public,
            disclosed,
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=True,
            current_block=500,
        )
    assert replayed.value.code is ChallengeErrorCode.REPLAY


def test_tampered_selection_fails_without_consuming_commitment() -> None:
    public = commitment()
    disclosed = reveal(public)
    tampered = replace(
        disclosed,
        shard_order=tuple(reversed(disclosed.shard_order)),
    )
    verifier = ChallengeVerifier()

    with pytest.raises(ChallengeFailure) as mismatch:
        verifier.verify_and_consume(
            public,
            tampered,
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=True,
            current_block=400,
        )
    assert mismatch.value.code is ChallengeErrorCode.MISMATCH
    verifier.verify_and_consume(
        public,
        disclosed,
        job_digest=JOB,
        lease_nonce=NONCE,
        accepted=True,
        current_block=400,
    )


def test_replay_guard_fails_closed_at_capacity() -> None:
    first = commitment(challenge_id="1" * 64)
    second = commitment(challenge_id="2" * 64)
    verifier = ChallengeVerifier(max_consumed=1)
    verifier.verify_and_consume(
        first,
        reveal(first),
        job_digest=JOB,
        lease_nonce=NONCE,
        accepted=True,
        current_block=400,
    )

    with pytest.raises(ChallengeFailure) as full:
        verifier.verify_and_consume(
            second,
            reveal(second),
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=True,
            current_block=400,
        )
    assert full.value.code is ChallengeErrorCode.CAPACITY


def test_secret_material_is_redacted_from_representations_and_errors() -> None:
    source = generator()
    disclosed = reveal()
    assert SECRET.decode() not in repr(source)
    assert disclosed.seed not in repr(disclosed)

    with pytest.raises(ChallengeFailure) as error:
        source.reveal(
            replace(commitment(), commitment_digest="sha256:" + "0" * 64),
            job_digest=JOB,
            lease_nonce=NONCE,
            accepted=True,
            current_block=400,
        )
    assert disclosed.seed not in str(error.value)
    assert SECRET.decode() not in str(error.value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ValidatorChallengeGenerator(b"short"),
        lambda: ChallengeVerifier(max_consumed=0),
        lambda: ChallengeCommitment("bad", "sha256:" + "0" * 64, 1, 1, 1, 1),
        lambda: ChallengeReveal(CHALLENGE_ID, "0" * 64, (0, 0), (0,)),
        lambda: ChallengeReveal(
            CHALLENGE_ID,
            "0" * 64,
            cast(tuple[int, ...], ([0],)),
            (0,),
        ),
    ],
)
def test_invalid_inputs_use_one_sanitized_error(factory: object) -> None:
    with pytest.raises(ChallengeFailure) as error:
        factory()  # type: ignore[operator]
    assert error.value.code is ChallengeErrorCode.INVALID
