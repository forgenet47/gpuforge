"""Command-line entry point for development utilities."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gpuforge import __version__
from gpuforge.config import (
    ConfigurationError,
    EvidenceTier,
    NetworkPolicy,
    Role,
    RuntimeSettings,
)
from gpuforge.protocol import ResourcePolicy, VerificationPolicy
from gpuforge.publisher import (
    PackageRequest,
    load_bittensor_hotkey,
    package_job,
    write_manifest,
)

PUBLICATION_CHECK_UNAVAILABLE = 2


def _publication_check(_args: argparse.Namespace) -> int:
    """Fail closed until the complete publication gate is implemented."""
    print("Publication check is not implemented yet; do not publish based on this command.")
    return PUBLICATION_CHECK_UNAVAILABLE


def _check_role_configuration(args: argparse.Namespace) -> int:
    """Validate role configuration without opening a network connection."""
    if not args.check_config:
        print(
            "Network operation is not implemented; use --check-config for offline validation.",
            file=sys.stderr,
        )
        return PUBLICATION_CHECK_UNAVAILABLE

    config_path = args.config
    expected_role = args.expected_role
    if not isinstance(config_path, Path) or not isinstance(expected_role, Role):
        print("Configuration error: invalid command state", file=sys.stderr)
        return PUBLICATION_CHECK_UNAVAILABLE

    try:
        settings = RuntimeSettings.from_toml(config_path, expected_role=expected_role)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return PUBLICATION_CHECK_UNAVAILABLE

    payload = {
        "network_connection_attempted": False,
        "settings": settings.to_safe_dict(),
        "status": "configuration valid",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _add_role_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    role: Role,
) -> None:
    """Register an offline role configuration command."""
    role_parser = subparsers.add_parser(
        role.value,
        help=f"Validate {role.value} configuration; network operation is unavailable.",
    )
    role_parser.add_argument("--config", required=True, type=Path)
    role_parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration without connecting to a network.",
    )
    role_parser.set_defaults(handler=_check_role_configuration, expected_role=role)


def _package_publisher_job(args: argparse.Namespace) -> int:
    """Build and write one signed manifest without opening a network connection."""
    try:
        signer = load_bittensor_hotkey(args.wallet_name, args.hotkey_name)
        packaged = package_job(
            PackageRequest(
                root=args.root,
                entrypoint=args.entrypoint,
                inputs=tuple(args.inputs),
                container_image=args.container,
                job_id=args.job_id,
                framework=args.framework,
                resource_policy=ResourcePolicy(
                    gpu_count=args.gpu_count,
                    gpu_memory_mb=args.gpu_memory_mb,
                    cpu_cores=args.cpu_cores,
                    memory_mb=args.memory_mb,
                    max_runtime_seconds=args.max_runtime_seconds,
                    network_policy=NetworkPolicy(args.network_policy),
                ),
                verification_policy=VerificationPolicy(
                    minimum_evidence_tier=EvidenceTier(args.minimum_evidence_tier),
                    challenge_kind=args.challenge_kind,
                    checkpoint_interval_steps=args.checkpoint_interval_steps,
                ),
                lease_seconds=args.lease_seconds,
                current_block=args.current_block,
                expires_at_block=args.expires_at_block,
            ),
            signer,
        )
        write_manifest(packaged, args.output)
    except ValueError as error:
        print(f"Packaging error: {error}", file=sys.stderr)
        return PUBLICATION_CHECK_UNAVAILABLE

    print(
        json.dumps(
            {
                "input_count": len(packaged.inputs),
                "job_id": packaged.manifest.job_id,
                "manifest_digest": packaged.manifest.digest(),
                "network_connection_attempted": False,
                "package_digest": packaged.package_digest,
                "status": "signed manifest written",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_publisher_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register deterministic offline publisher packaging."""
    publisher_parser = subparsers.add_parser(
        "publisher", help="Build signed publisher artifacts without network access."
    )
    publisher_subparsers = publisher_parser.add_subparsers(dest="publisher_command")
    package_parser = publisher_subparsers.add_parser(
        "package", help="Hash declared files and write a signed job manifest."
    )
    package_parser.add_argument("--root", required=True, type=Path)
    package_parser.add_argument("--entrypoint", required=True, type=Path)
    package_parser.add_argument("--input", dest="inputs", required=True, action="append", type=Path)
    package_parser.add_argument("--container", required=True)
    package_parser.add_argument("--job-id", required=True)
    package_parser.add_argument("--framework", default="pytorch")
    package_parser.add_argument("--current-block", required=True, type=int)
    package_parser.add_argument("--expires-at-block", required=True, type=int)
    package_parser.add_argument("--lease-seconds", default=3_600, type=int)
    package_parser.add_argument("--gpu-count", default=1, type=int)
    package_parser.add_argument("--gpu-memory-mb", default=81_920, type=int)
    package_parser.add_argument("--cpu-cores", default=16, type=int)
    package_parser.add_argument("--memory-mb", default=131_072, type=int)
    package_parser.add_argument("--max-runtime-seconds", default=3_600, type=int)
    package_parser.add_argument(
        "--network-policy",
        choices=tuple(value.value for value in NetworkPolicy),
        default=NetworkPolicy.DENY.value,
    )
    package_parser.add_argument(
        "--minimum-evidence-tier",
        choices=tuple(value.value for value in EvidenceTier),
        default=EvidenceTier.C.value,
    )
    package_parser.add_argument("--challenge-kind", default="gradient_slice")
    package_parser.add_argument("--checkpoint-interval-steps", default=100, type=int)
    package_parser.add_argument("--wallet-name", required=True)
    package_parser.add_argument("--hotkey-name", required=True)
    package_parser.add_argument("--output", required=True, type=Path)
    package_parser.set_defaults(handler=_package_publisher_job)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="gpuforge",
        description="GPUForge development utilities (not ready for network operation).",
    )
    parser.add_argument("--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(dest="command")
    publication_parser = subparsers.add_parser(
        "publication-check",
        help="Run the pre-publication safety gate (currently unavailable).",
    )
    publication_parser.set_defaults(handler=_publication_check)
    _add_role_command(subparsers, Role.MINER)
    _add_role_command(subparsers, Role.VALIDATOR)
    _add_publisher_command(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return its process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
