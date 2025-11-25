"""Command line interface for GABRIEL.

The CLI intentionally exposes a light-touch interface focused on discovery
and verification.  It is designed to help users confirm the installation,
identify available helpers, and navigate toward the Python API for day-to-day
use.  Full task execution is provided by the Python functions rather than the
CLI so that users can compose prompts, callbacks, and checkpoints in code.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List, Optional

from gabriel import __version__
from gabriel import tasks as _tasks


def _iter_task_names() -> Iterable[str]:
    """Yield the public task helpers exposed by ``gabriel.tasks``."""

    return sorted(_tasks.__all__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gabriel",
        description=(
            "GABRIEL provides GPT-powered helpers for social-science analysis. "
            "Use the Python API for full control; run `gabriel --list-tasks` "
            "to see the available helpers."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the installed gabriel version and exit.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List the Python helpers bundled with gabriel.",
    )

    args = parser.parse_args(argv)

    if args.list_tasks:
        print("Available helpers (importable from `gabriel`):")
        for name in _iter_task_names():
            print(f"- {name}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
