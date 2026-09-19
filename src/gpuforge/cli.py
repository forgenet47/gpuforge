"""Command-line entry point for development utilities."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gpuforge import __version__
from gpuforge.config import ConfigurationError, Role, RuntimeSettings

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
