# GPUForge Subnet

> **Development status:** This codebase is under active development and is not ready for miners, validators, production workloads, testnet deployment, or mainnet deployment. Interfaces, scoring rules, and security assumptions may change without notice. Do not run untrusted training jobs or use production wallet credentials with this repository.

GPUForge is a proposed Bittensor subnet for verifiable GPU training work. A job publisher supplies a signed, immutable training workload. Miners execute that workload on eligible NVIDIA H100 hardware. Validators measure correct, useful training throughput and translate verified results into miner scores.

## Intended protocol

1. The publisher creates a job manifest containing immutable container and input digests, resource limits, checkpoint rules, and an expiry.
2. The publisher signs the manifest. Miners reject unsigned, expired, or altered jobs.
3. A miner reports its capabilities and, where supported, supplies hardware-backed GPU attestation evidence.
4. A validator assigns a short-lived challenge and workload lease.
5. The miner runs the pinned workload in an isolated environment with restricted privileges, network access, storage, and runtime.
6. The miner returns signed results, checkpoint commitments, timing evidence, and challenge outputs.
7. Validators verify attestation, manifest identity, correctness, freshness, and performance before calculating a score.
8. Validators publish normalized weights through the Bittensor network.

## Verification model

The protocol is designed to score verified work rather than self-reported hardware specifications. Verification is expected to combine:

- hardware-backed attestation when the deployment platform supports it;
- container, model, dataset-shard, and training-script digests;
- validator-generated nonces and unpredictable challenge inputs;
- deterministic or tolerance-bounded correctness checks;
- signed checkpoint and output commitments;
- throughput plausibility bounds for the required hardware class;
- redundant challenges and statistical anomaly detection; and
- penalties or score exclusion for stale, inconsistent, or unverifiable evidence.

No single software measurement proves physical GPU identity or exact code execution on a hostile machine. The implementation must treat attestation and challenge verification as layered evidence, document their trust assumptions, and fail closed when required evidence is missing.

## Safety boundaries

Publisher-provided training code is untrusted input. Miner execution must use pinned images, least privilege, resource quotas, read-only mounts where possible, explicit outbound-network policy, short-lived credentials, and auditable termination controls. Validators must not accept raw secrets, wallet material, personal data, or unrestricted host telemetry as evidence.

## Repository policy

Public documentation is limited to technical behavior, interfaces, security assumptions, operational safety, and reproducible testing. Private plans, development notes, credentials, local datasets, logs, checkpoints, and wallet files are excluded from version control.

## Current availability

There is no supported miner or validator release yet. Installation and network-operation instructions will be added only after the protocol, sandbox, adversarial tests, and testnet acceptance criteria are complete.

## Offline configuration validation

The checked-in local configurations contain no credentials and do not identify a live network. They can be validated without importing Bittensor or opening a network connection:

```text
python -m gpuforge miner --config config/miner.local.toml --check-config
python -m gpuforge validator --config config/validator.local.toml --check-config
```

Configuration files must never contain passwords, tokens, private keys, seed phrases, or other credentials. The only supported secret inputs are `GPUFORGE_ARTIFACT_ACCESS_TOKEN` and `GPUFORGE_ATTESTATION_ACCESS_TOKEN`, supplied to the process at runtime. Configuration summaries report only whether these values are present, and the logging formatter redacts their contents.

The local evidence tier is for development tests only. Non-local configurations reject that tier and require fail-closed verification. The sandbox settings are policy declarations at this stage; workload isolation is not implemented and no publisher-supplied code should be run.

## Protocol encoding

Protocol version 1 defines typed `JobManifest`, `CapabilityClaim`, `WorkLease`, `ExecutionEvidence`, and `ValidationReceipt` messages. Messages use a deliberately restricted canonical JSON profile:

- UTF-8 with NFC-normalized bounded text;
- lexicographically sorted object keys and no insignificant whitespace;
- integers only, with no floating-point or non-finite numbers;
- strict required fields with unknown and duplicate fields rejected;
- deterministic ordering for set-like fields; and
- a 64 KiB maximum encoded message size.

Wire decoders accept canonical bytes only. Each message digest is SHA-256 over a versioned GPUForge domain separator and the exact canonical bytes. This provides deterministic identities and prevents alternate JSON representations from producing ambiguous signed data.

## Message authentication

Protocol signatures cover a separate versioned signing domain, the message type, protocol version, and canonical payload with the signature field omitted. Signatures are fixed-width 64-byte values encoded as lowercase hexadecimal. The signing adapter uses the public SS58 hotkey as the role identity and is intentionally compatible with Bittensor keypair-shaped objects without requiring Bittensor during offline tests.

Freshness validation uses bounded block windows. Capability claims, execution evidence, and validation receipts reject stale or implausibly future block observations; manifests and leases reject expired transitions. Capability and lease nonces are 256-bit random values. Other message types use their unsigned content digest as a replay token.

The replay cache is bounded and fails closed instead of evicting active entries. Optional on-disk state is replaced atomically, stores hashed cache keys rather than raw hotkeys or nonces, and preserves nonce and evidence-sequence decisions across process restarts. Operators must place replay state on durable private storage. This authenticates protocol messages; it does not yet provide workload isolation, GPU attestation, or proof that a miner executed a training job correctly.

Wallet seed phrases, private keys, and passwords must never be passed in protocol messages, configuration files, command arguments, logs, or replay state. Only public hotkey addresses and signatures belong in these messages.

## Development checks

GPUForge supports Python 3.10 through 3.14. Create an isolated environment and install the development tools:

```text
python -m venv .venv
python -m pip install -e ".[dev]"
```

Run the local checks before proposing a public change:

```text
ruff check .
ruff format --check .
mypy
pytest --cov=gpuforge --cov-report=term-missing
python -m build
```

The `gpuforge publication-check` command is currently a fail-closed placeholder. It does not certify a change as safe to publish. Until the complete gate is implemented, manually inspect every staged file and staged line for credentials, personal data, private infrastructure details, internal development material, and unsafe operational defaults.
