"""Persistent, idempotent miner lease lifecycle management."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol, cast

_STATE_VERSION = 1
_MAX_STATE_BYTES = 4 * 1024 * 1024
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_CODE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]{0,63}")


class LeaseStateError(ValueError):
    """Raised when lease state is invalid, unavailable, or out of order."""


class LeaseState(str, Enum):
    """Persisted miner states for one immutable work lease."""

    OFFERED = "offered"
    ACCEPTED = "accepted"
    FETCHING = "fetching"
    READY = "ready"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


_TERMINAL_STATES = frozenset(
    {LeaseState.COMPLETED, LeaseState.FAILED, LeaseState.EXPIRED, LeaseState.CANCELLED}
)
_TRANSITIONS: dict[LeaseState, frozenset[LeaseState]] = {
    LeaseState.OFFERED: frozenset(
        {LeaseState.ACCEPTED, LeaseState.FAILED, LeaseState.EXPIRED, LeaseState.CANCELLED}
    ),
    LeaseState.ACCEPTED: frozenset(
        {LeaseState.FETCHING, LeaseState.FAILED, LeaseState.EXPIRED, LeaseState.CANCELLED}
    ),
    LeaseState.FETCHING: frozenset(
        {LeaseState.READY, LeaseState.FAILED, LeaseState.EXPIRED, LeaseState.CANCELLED}
    ),
    LeaseState.READY: frozenset(
        {LeaseState.RUNNING, LeaseState.FAILED, LeaseState.EXPIRED, LeaseState.CANCELLED}
    ),
    LeaseState.RUNNING: frozenset(
        {
            LeaseState.CHECKPOINTING,
            LeaseState.COMPLETED,
            LeaseState.FAILED,
            LeaseState.EXPIRED,
            LeaseState.CANCELLED,
        }
    ),
    LeaseState.CHECKPOINTING: frozenset(
        {
            LeaseState.RUNNING,
            LeaseState.COMPLETED,
            LeaseState.FAILED,
            LeaseState.EXPIRED,
            LeaseState.CANCELLED,
        }
    ),
    LeaseState.COMPLETED: frozenset(),
    LeaseState.FAILED: frozenset(),
    LeaseState.EXPIRED: frozenset(),
    LeaseState.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    """Non-secret recovery state for one signed lease digest."""

    lease_digest: str
    manifest_digest: str
    deadline_block: int
    state: LeaseState = LeaseState.OFFERED
    revision: int = 0
    execution_started: bool = False
    evidence_submitted: bool = False
    failure_code: str | None = None

    def __post_init__(self) -> None:
        _digest("lease digest", self.lease_digest)
        _digest("manifest digest", self.manifest_digest)
        _block("deadline block", self.deadline_block)
        if not isinstance(self.state, LeaseState):
            raise LeaseStateError("Lease state is invalid")
        _bounded_int("lease revision", self.revision, 0, 2**63 - 1)
        if not isinstance(self.execution_started, bool) or not isinstance(
            self.evidence_submitted, bool
        ):
            raise LeaseStateError("Lease recovery flags are invalid")
        if self.failure_code is not None:
            _code("failure code", self.failure_code)
        if self.state is LeaseState.FAILED and self.failure_code is None:
            raise LeaseStateError("Failed lease requires a stable failure code")
        if self.state is not LeaseState.FAILED and self.failure_code is not None:
            raise LeaseStateError("Failure code is only valid for a failed lease")
        if (
            self.state
            in {
                LeaseState.RUNNING,
                LeaseState.CHECKPOINTING,
                LeaseState.COMPLETED,
            }
            and not self.execution_started
        ):
            raise LeaseStateError("Executing lease must retain its start marker")
        if self.evidence_submitted and self.state is not LeaseState.COMPLETED:
            raise LeaseStateError("Evidence can be submitted only for a completed lease")

    def to_primitive(self) -> dict[str, object]:
        """Return the complete non-secret persisted representation."""
        return {
            "deadline_block": self.deadline_block,
            "evidence_submitted": self.evidence_submitted,
            "execution_started": self.execution_started,
            "failure_code": self.failure_code,
            "lease_digest": self.lease_digest,
            "manifest_digest": self.manifest_digest,
            "revision": self.revision,
            "state": self.state.value,
        }

    @classmethod
    def from_primitive(cls, value: object) -> LeaseRecord:
        """Parse one strict persisted lease record."""
        if not isinstance(value, dict) or set(value) != {
            "deadline_block",
            "evidence_submitted",
            "execution_started",
            "failure_code",
            "lease_digest",
            "manifest_digest",
            "revision",
            "state",
        }:
            raise LeaseStateError("Lease record has an invalid schema")
        data = cast(dict[str, object], value)
        try:
            state = LeaseState(data["state"])
        except (TypeError, ValueError):
            raise LeaseStateError("Lease record has an invalid state") from None
        failure_code = data["failure_code"]
        if failure_code is not None and not isinstance(failure_code, str):
            raise LeaseStateError("Lease record has an invalid failure code")
        return cls(
            lease_digest=_text(data["lease_digest"]),
            manifest_digest=_text(data["manifest_digest"]),
            deadline_block=_integer(data["deadline_block"]),
            state=state,
            revision=_integer(data["revision"]),
            execution_started=_boolean(data["execution_started"]),
            evidence_submitted=_boolean(data["evidence_submitted"]),
            failure_code=failure_code,
        )


class LeaseStateWriter(Protocol):
    """Atomically persist a complete encoded lease-state snapshot."""

    def write(self, path: Path, payload: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class AtomicLeaseStateWriter:
    """Write one bounded state snapshot through an atomic replacement."""

    def write(self, path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        except OSError:
            raise LeaseStateError("Lease state could not be persisted") from None


class LeaseStore:
    """Bounded lease registry with restart-safe exactly-once claim markers."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        max_records: int = 10_000,
        writer: LeaseStateWriter | None = None,
    ) -> None:
        _bounded_int("lease record limit", max_records, 1, 1_000_000)
        if path is not None and not isinstance(path, Path):
            raise LeaseStateError("Lease state path is invalid")
        self._path = path
        self._max_records = max_records
        self._writer = writer or AtomicLeaseStateWriter()
        self._records: dict[str, LeaseRecord] = {}
        if path is not None and path.exists():
            self._load()

    def offer(
        self,
        *,
        lease_digest: str,
        manifest_digest: str,
        deadline_block: int,
    ) -> LeaseRecord:
        """Persist a new offer or return an identical duplicate idempotently."""
        candidate = LeaseRecord(lease_digest, manifest_digest, deadline_block)
        existing = self._records.get(lease_digest)
        if existing is not None:
            if (
                existing.manifest_digest != manifest_digest
                or existing.deadline_block != deadline_block
            ):
                raise LeaseStateError("Duplicate lease offer conflicts with persisted state")
            return existing
        if len(self._records) >= self._max_records:
            raise LeaseStateError("Lease state store is full")
        records = self._records.copy()
        records[lease_digest] = candidate
        self._commit(records)
        return candidate

    def get(self, lease_digest: str) -> LeaseRecord:
        """Return one persisted lease without exposing other records."""
        _digest("lease digest", lease_digest)
        try:
            return self._records[lease_digest]
        except KeyError:
            raise LeaseStateError("Lease state is unavailable") from None

    def transition(
        self,
        lease_digest: str,
        target: LeaseState,
        *,
        current_block: int,
        failure_code: str | None = None,
    ) -> LeaseRecord:
        """Apply one ordered state transition or an idempotent duplicate."""
        _block("current block", current_block)
        if not isinstance(target, LeaseState):
            raise LeaseStateError("Target lease state is invalid")
        record = self.get(lease_digest)
        if current_block > record.deadline_block and record.state not in _TERMINAL_STATES:
            target = LeaseState.EXPIRED
            failure_code = None
        if target is record.state:
            if target is LeaseState.FAILED and failure_code != record.failure_code:
                raise LeaseStateError("Duplicate failed transition has a different reason")
            return record
        if record.state in _TERMINAL_STATES or target not in _TRANSITIONS[record.state]:
            raise LeaseStateError("Lease transition is out of order")
        if record.state is LeaseState.READY and target is LeaseState.RUNNING:
            raise LeaseStateError("Execution must be claimed atomically")
        if target is LeaseState.FAILED:
            if failure_code is None:
                raise LeaseStateError("Failed transition requires a stable failure code")
            _code("failure code", failure_code)
        elif failure_code is not None:
            raise LeaseStateError("Failure code is invalid for this transition")
        updated = replace(
            record,
            state=target,
            revision=record.revision + 1,
            failure_code=failure_code,
        )
        return self._replace(updated)

    def claim_execution(self, lease_digest: str, *, current_block: int) -> bool:
        """Atomically claim execution once, including after process restart."""
        _block("current block", current_block)
        record = self.get(lease_digest)
        if current_block > record.deadline_block:
            if record.state not in _TERMINAL_STATES:
                self.transition(
                    lease_digest,
                    LeaseState.EXPIRED,
                    current_block=current_block,
                )
            return False
        if record.execution_started:
            return False
        if record.state is not LeaseState.READY:
            raise LeaseStateError("Lease is not ready for execution")
        updated = replace(
            record,
            state=LeaseState.RUNNING,
            revision=record.revision + 1,
            execution_started=True,
        )
        self._replace(updated)
        return True

    def claim_evidence_submission(self, lease_digest: str) -> bool:
        """Atomically reserve the single evidence submission for a completed lease."""
        record = self.get(lease_digest)
        if record.state is not LeaseState.COMPLETED:
            raise LeaseStateError("Lease is not complete")
        if record.evidence_submitted:
            return False
        self._replace(replace(record, revision=record.revision + 1, evidence_submitted=True))
        return True

    def _replace(self, record: LeaseRecord) -> LeaseRecord:
        records = self._records.copy()
        records[record.lease_digest] = record
        self._commit(records)
        return record

    def _commit(self, records: dict[str, LeaseRecord]) -> None:
        self._persist(records)
        self._records = records

    def _persist(self, records: dict[str, LeaseRecord]) -> None:
        if self._path is None:
            return
        payload = json.dumps(
            {
                "records": [records[key].to_primitive() for key in sorted(records)],
                "version": _STATE_VERSION,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(payload) > _MAX_STATE_BYTES:
            raise LeaseStateError("Lease state exceeds the size limit")
        self._writer.write(self._path, payload)

    def _load(self) -> None:
        if self._path is None:  # pragma: no cover - guarded by caller
            return
        try:
            if self._path.stat().st_size > _MAX_STATE_BYTES:
                raise LeaseStateError("Lease state exceeds the size limit")
            decoded = json.loads(self._path.read_bytes())
        except LeaseStateError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise LeaseStateError("Lease state is unavailable or malformed") from None
        if not isinstance(decoded, dict) or set(decoded) != {"records", "version"}:
            raise LeaseStateError("Lease state has an invalid schema")
        if decoded["version"] != _STATE_VERSION:
            raise LeaseStateError("Lease state has an unsupported version")
        values = decoded["records"]
        if not isinstance(values, list) or len(values) > self._max_records:
            raise LeaseStateError("Lease state record collection is invalid")
        records: dict[str, LeaseRecord] = {}
        for value in values:
            record = LeaseRecord.from_primitive(value)
            if record.lease_digest in records:
                raise LeaseStateError("Lease state contains a duplicate record")
            records[record.lease_digest] = record
        self._records = records


def _digest(label: str, value: object) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise LeaseStateError(f"{label} is invalid")


def _code(label: str, value: object) -> None:
    if not isinstance(value, str) or _CODE_PATTERN.fullmatch(value) is None:
        raise LeaseStateError(f"{label} is invalid")


def _block(label: str, value: object) -> None:
    _bounded_int(label, value, 0, 2**63 - 1)


def _bounded_int(label: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise LeaseStateError(f"{label} is invalid")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise LeaseStateError("Lease record text field is invalid")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LeaseStateError("Lease record integer field is invalid")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise LeaseStateError("Lease record boolean field is invalid")
    return value


__all__ = [
    "AtomicLeaseStateWriter",
    "LeaseRecord",
    "LeaseState",
    "LeaseStateError",
    "LeaseStateWriter",
    "LeaseStore",
]
