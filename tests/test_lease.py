"""Persistent lease lifecycle and exactly-once marker tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from gpuforge.lease import (
    AtomicLeaseStateWriter,
    LeaseState,
    LeaseStateError,
    LeaseStore,
)

LEASE = "sha256:" + "11" * 32
MANIFEST = "sha256:" + "22" * 32
DEADLINE = 1_000


def offered(store: LeaseStore) -> None:
    store.offer(lease_digest=LEASE, manifest_digest=MANIFEST, deadline_block=DEADLINE)


def ready(store: LeaseStore) -> None:
    offered(store)
    store.transition(LEASE, LeaseState.ACCEPTED, current_block=100)
    store.transition(LEASE, LeaseState.FETCHING, current_block=101)
    store.transition(LEASE, LeaseState.READY, current_block=102)


def test_complete_lifecycle_is_ordered_and_duplicate_transitions_are_idempotent() -> None:
    store = LeaseStore()
    ready(store)
    assert store.get(LEASE).state is LeaseState.READY
    assert store.transition(LEASE, LeaseState.READY, current_block=103).revision == 3

    assert store.claim_execution(LEASE, current_block=104)
    assert not store.claim_execution(LEASE, current_block=104)
    store.transition(LEASE, LeaseState.CHECKPOINTING, current_block=105)
    store.transition(LEASE, LeaseState.RUNNING, current_block=106)
    completed = store.transition(LEASE, LeaseState.COMPLETED, current_block=107)

    assert completed.state is LeaseState.COMPLETED
    assert completed.execution_started
    assert store.transition(LEASE, LeaseState.COMPLETED, current_block=108) == completed
    assert store.claim_evidence_submission(LEASE)
    assert not store.claim_evidence_submission(LEASE)


def test_duplicate_offer_is_idempotent_but_conflicts_fail() -> None:
    store = LeaseStore()
    first = store.offer(lease_digest=LEASE, manifest_digest=MANIFEST, deadline_block=DEADLINE)
    assert (
        store.offer(lease_digest=LEASE, manifest_digest=MANIFEST, deadline_block=DEADLINE) == first
    )
    with pytest.raises(LeaseStateError, match="conflicts"):
        store.offer(
            lease_digest=LEASE,
            manifest_digest="sha256:" + "33" * 32,
            deadline_block=DEADLINE,
        )


def test_restart_preserves_each_state_and_exactly_once_markers(tmp_path: Path) -> None:
    state_path = tmp_path / "lease-state.json"
    store = LeaseStore(path=state_path)
    ready(store)

    restarted = LeaseStore(path=state_path)
    assert restarted.get(LEASE).state is LeaseState.READY
    assert restarted.claim_execution(LEASE, current_block=200)

    restarted = LeaseStore(path=state_path)
    assert restarted.get(LEASE).state is LeaseState.RUNNING
    assert not restarted.claim_execution(LEASE, current_block=201)
    restarted.transition(LEASE, LeaseState.CHECKPOINTING, current_block=202)

    restarted = LeaseStore(path=state_path)
    assert restarted.get(LEASE).state is LeaseState.CHECKPOINTING
    restarted.transition(LEASE, LeaseState.COMPLETED, current_block=203)
    assert restarted.claim_evidence_submission(LEASE)

    restarted = LeaseStore(path=state_path)
    assert restarted.get(LEASE).state is LeaseState.COMPLETED
    assert not restarted.claim_evidence_submission(LEASE)


def test_cancellation_race_does_not_override_terminal_state() -> None:
    store = LeaseStore()
    ready(store)
    cancelled = store.transition(LEASE, LeaseState.CANCELLED, current_block=110)
    assert cancelled.state is LeaseState.CANCELLED
    with pytest.raises(LeaseStateError, match="not ready"):
        store.claim_execution(LEASE, current_block=111)

    other = LeaseStore()
    ready(other)
    other.claim_execution(LEASE, current_block=110)
    other.transition(LEASE, LeaseState.COMPLETED, current_block=111)
    with pytest.raises(LeaseStateError, match="out of order"):
        other.transition(LEASE, LeaseState.CANCELLED, current_block=112)


def test_deadline_expiry_prevents_execution() -> None:
    store = LeaseStore()
    ready(store)

    assert not store.claim_execution(LEASE, current_block=DEADLINE + 1)
    assert store.get(LEASE).state is LeaseState.EXPIRED


@pytest.mark.parametrize(
    ("current", "target"),
    (
        (LeaseState.OFFERED, LeaseState.READY),
        (LeaseState.ACCEPTED, LeaseState.RUNNING),
        (LeaseState.FETCHING, LeaseState.COMPLETED),
    ),
)
def test_out_of_order_transitions_are_rejected(current: LeaseState, target: LeaseState) -> None:
    store = LeaseStore()
    offered(store)
    if current is LeaseState.ACCEPTED:
        store.transition(LEASE, LeaseState.ACCEPTED, current_block=100)
    elif current is LeaseState.FETCHING:
        store.transition(LEASE, LeaseState.ACCEPTED, current_block=100)
        store.transition(LEASE, LeaseState.FETCHING, current_block=101)
    with pytest.raises(LeaseStateError, match="out of order|claimed"):
        store.transition(LEASE, target, current_block=102)


def test_failed_transition_requires_stable_machine_code() -> None:
    store = LeaseStore()
    offered(store)
    with pytest.raises(LeaseStateError, match="requires"):
        store.transition(LEASE, LeaseState.FAILED, current_block=100)
    failed = store.transition(
        LEASE,
        LeaseState.FAILED,
        current_block=100,
        failure_code="artifact_unavailable",
    )
    assert failed.failure_code == "artifact_unavailable"
    assert (
        store.transition(
            LEASE,
            LeaseState.FAILED,
            current_block=101,
            failure_code="artifact_unavailable",
        )
        == failed
    )
    with pytest.raises(LeaseStateError, match="different reason"):
        store.transition(
            LEASE,
            LeaseState.FAILED,
            current_block=101,
            failure_code="different_failure",
        )


@dataclass
class ToggleWriter:
    """Persist normally until a deterministic disk failure is enabled."""

    fail: bool = False

    def write(self, path: Path, payload: bytes) -> None:
        if self.fail:
            raise LeaseStateError("Lease state could not be persisted")
        AtomicLeaseStateWriter().write(path, payload)


def test_disk_failure_rolls_back_candidate_state(tmp_path: Path) -> None:
    writer = ToggleWriter()
    store = LeaseStore(path=tmp_path / "state.json", writer=writer)
    offered(store)
    writer.fail = True

    with pytest.raises(LeaseStateError, match="persisted"):
        store.transition(LEASE, LeaseState.ACCEPTED, current_block=100)
    assert store.get(LEASE).state is LeaseState.OFFERED
    assert LeaseStore(path=tmp_path / "state.json").get(LEASE).state is LeaseState.OFFERED


def test_persisted_state_contains_no_role_identity_nonce_or_failure_detail(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    store = LeaseStore(path=state_path)
    offered(store)
    store.transition(
        LEASE,
        LeaseState.FAILED,
        current_block=100,
        failure_code="runtime_failed",
    )
    content = state_path.read_text(encoding="utf-8")

    assert "hotkey" not in content
    assert "nonce" not in content
    assert "challenge" not in content
    assert "sensitive runtime detail" not in content
    assert "runtime_failed" in content


@pytest.mark.parametrize(
    "content",
    (
        "not-json",
        '{"version":2,"records":[]}',
        '{"version":1,"records":[{}]}',
    ),
)
def test_malformed_recovery_state_fails_closed(tmp_path: Path, content: str) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(content, encoding="utf-8")
    with pytest.raises(LeaseStateError):
        LeaseStore(path=state_path)
