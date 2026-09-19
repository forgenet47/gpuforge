"""Command-line entry point for development utilities."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from gpuforge import __version__

PUBLICATION_CHECK_UNAVAILABLE = 2


def _publication_check(_args: argparse.Namespace) -> int:
    """Fail closed until the complete publication gate is implemented."""
    print("Publication check is not implemented yet; do not publish based on this command.")
    return PUBLICATION_CHECK_UNAVAILABLE


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
