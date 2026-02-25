#!/usr/bin/env python3
#
# Bayesian BM25
#
# Copyright (c) 2023-2026 Cognica, Inc.
#

"""Sync CITATION.cff version and date-released from pyproject.toml.

Usage:
    python scripts/sync_citation.py          # update CITATION.cff in place
    python scripts/sync_citation.py --check  # exit 1 if out of sync (CI mode)
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        sys.exit(
            "ERROR: Python < 3.11 requires the 'tomli' package. "
            "Install it with: pip install tomli"
        )

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"


def read_pyproject_version() -> str:
    """Read the version string from pyproject.toml."""
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def sync_citation(*, check_only: bool = False) -> bool:
    """Update CITATION.cff version and date-released.

    Returns True if CITATION.cff was (or would be) changed.
    """
    version = read_pyproject_version()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    text = CITATION.read_text(encoding="utf-8")

    new_text = re.sub(
        r"^version:\s+.*$",
        f"version: {version}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    new_text = re.sub(
        r'^date-released:\s+.*$',
        f'date-released: "{today}"',
        new_text,
        count=1,
        flags=re.MULTILINE,
    )

    changed = new_text != text

    if check_only:
        if changed:
            print(
                f"CITATION.cff is out of sync with pyproject.toml "
                f"(expected version: {version}, date-released: {today})."
            )
            print("Run 'python scripts/sync_citation.py' to fix.")
        else:
            print("CITATION.cff is in sync.")
        return changed

    if changed:
        CITATION.write_text(new_text, encoding="utf-8")
        print(f"Updated CITATION.cff: version={version}, date-released={today}")
    else:
        print("CITATION.cff already up to date.")

    return changed


def main() -> None:
    check_only = "--check" in sys.argv
    changed = sync_citation(check_only=check_only)
    if check_only and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
