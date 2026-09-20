"""Hotkey-compatible message signing, freshness, and replay authentication."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from typing import Protocol

from gpuforge.protocol import (
    CapabilityClaim,
    ExecutionEvidence,
    JobManifest,
    Message,
    ValidationReceipt,
    WorkLease,
)
from gpuforge.replay import ReplayCache

SIGNATURE_BYTES = 64


class AuthenticationError(ValueError):
    """Raised when identity, signature, freshness, or replay validation fails."""


class MessageSigner(Protocol):
    """Structural interface implemented by hotkey signing adapters."""

    @property
    def hotkey(self) -> str: ...

    def sign(self, payload: bytes) -> bytes: ...


class SignatureVerifier(Protocol):
    """Structural interface for hotkey-addressed signature verification."""

    def verify(self, hotkey: str, payload: bytes, signature: bytes) -> bool: ...


class BittensorKeypair(Protocol):
    """Minimal shape exposed by Bittensor hotkey keypair objects."""

    ss58_address: str

    def sign(self, payload: bytes) -> bytes: ...

    def verify(self, payload: bytes, signature: bytes) -> bool: ...


@dataclass(frozen=True, slots=True)
class BittensorHotkeyAdapter:
    """Adapt a Bittensor keypair-like object to signing and verification interfaces."""

    keypair: BittensorKeypair

    @property
    def hotkey(self) -> str:
        """Return the public SS58 hotkey address."""
        return self.keypair.ss58_address

    def sign(self, payload: bytes) -> bytes:
        """Sign exact domain-separated bytes with the wrapped hotkey."""
        signature = self.keypair.sign(payload)
        if not isinstance(signature, bytes) or len(signature) != SIGNATURE_BYTES:
            raise AuthenticationError("Hotkey signer returned an invalid signature")
        return signature

    def verify(self, hotkey: str, payload: bytes, signature: bytes) -> bool:
        """Verify exact bytes only when the requested hotkey matches this keypair."""
        if hotkey != self.hotkey or len(signature) != SIGNATURE_BYTES:
            return False
        return bool(self.keypair.verify(payload, signature))


def generate_nonce() -> str:
    """Return a cryptographically random 256-bit lowercase hexadecimal nonce."""
    return secrets.token_hex(32)


def signer_hotkey(message: Message) -> str:
    """Return the role identity that must authenticate a protocol message."""
    if isinstance(message, JobManifest):
        return message.publisher_hotkey
    if isinstance(message, CapabilityClaim | ExecutionEvidence):
        return message.miner_hotkey
    if isinstance(message, WorkLease | ValidationReceipt):
        return message.validator_hotkey
    raise AuthenticationError("Unsupported message type")


def sign_message(message: Message, signer: MessageSigner) -> Message:
    """Sign a message and return an immutable copy containing signature hex."""
    if signer.hotkey != signer_hotkey(message):
        raise AuthenticationError("Signer hotkey does not match the message role identity")
    signature = signer.sign(message.signing_bytes())
    if not isinstance(signature, bytes) or len(signature) != SIGNATURE_BYTES:
        raise AuthenticationError("Signer returned an invalid signature")
    return replace(message, signature=signature.hex())


def verify_message_signature(message: Message, verifier: SignatureVerifier) -> None:
    """Verify the role hotkey signature over exact domain-separated canonical bytes."""
    try:
        signature = bytes.fromhex(message.signature)
    except ValueError:
        raise AuthenticationError("Message signature is not valid hexadecimal") from None
    if len(signature) != SIGNATURE_BYTES or not verifier.verify(
        signer_hotkey(message), message.signing_bytes(), signature
    ):
        raise AuthenticationError("Message signature verification failed")


def validate_freshness(
    message: Message,
    *,
    current_block: int,
    max_age_blocks: int = 64,
    max_future_blocks: int = 2,
) -> int:
    """Validate block freshness and return the block at which replay state expires."""
    _bounded_block("current_block", current_block)
    if not 1 <= max_age_blocks <= 100_000 or not 0 <= max_future_blocks <= 1_000:
        raise AuthenticationError("Freshness policy is outside the permitted range")
    if isinstance(message, JobManifest):
        if current_block > message.expires_at_block:
            raise AuthenticationError("Job manifest has expired")
        return message.expires_at_block
    if isinstance(message, WorkLease):
        if current_block + max_future_blocks < message.start_block:
            raise AuthenticationError("Work lease is not active yet")
        if current_block > message.deadline_block:
            raise AuthenticationError("Work lease has expired")
        return message.deadline_block
    observed_block = (
        message.observed_at_block
        if isinstance(message, CapabilityClaim)
        else message.submitted_at_block
        if isinstance(message, ExecutionEvidence)
        else message.validated_at_block
    )
    if observed_block > current_block + max_future_blocks:
        raise AuthenticationError("Message block is too far in the future")
    if current_block - observed_block > max_age_blocks:
        raise AuthenticationError("Message is stale")
    return observed_block + max_age_blocks


@dataclass(slots=True)
class MessageAuthenticator:
    """Authenticate signatures, freshness, replay tokens, and evidence sequences."""

    verifier: SignatureVerifier
    replay_cache: ReplayCache
    max_age_blocks: int = 64
    max_future_blocks: int = 2

    def authenticate(self, message: Message, *, current_block: int) -> None:
        """Accept one authenticated fresh transition or fail without partial success."""
        verify_message_signature(message, self.verifier)
        expiry = validate_freshness(
            message,
            current_block=current_block,
            max_age_blocks=self.max_age_blocks,
            max_future_blocks=self.max_future_blocks,
        )
        hotkey = signer_hotkey(message)
        token = (
            message.nonce
            if isinstance(message, CapabilityClaim | WorkLease)
            else message.content_digest()
        )
        if isinstance(message, ExecutionEvidence):
            self.replay_cache.accept_once_with_sequence(
                scope=f"{message.message_type}:{hotkey}",
                token=token,
                current_block=current_block,
                expires_at_block=expiry,
                sequence_scope=f"{message.miner_hotkey}:{message.lease_digest}",
                sequence=message.sequence,
            )
        else:
            self.replay_cache.accept_once(
                scope=f"{message.message_type}:{hotkey}",
                token=token,
                current_block=current_block,
                expires_at_block=expiry,
            )


def _bounded_block(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**63 - 1:
        raise AuthenticationError(f"{name} is outside the permitted range")
