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
