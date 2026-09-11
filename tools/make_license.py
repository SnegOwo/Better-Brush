#!/usr/bin/env python
"""Generate a Better Brush Pro license key (offline, no network).

Usage (from the repo root):
    python tools/make_license.py "Sneg"
    python tools/make_license.py "Sneg" expires:2027-12-31

The printed key can be pasted into the Activate Pro dialog in-app.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on the path.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.core.licensing import issue_key  # noqa: E402


def _parse_expiry(exp: str) -> int:
    """Return a unix timestamp from a YYYY-MM-DD string, or 0 for 'never'."""
    exp = (exp or "").strip().lower().removeprefix("expires:").strip()
    if not exp or exp in ("never", "0"):
        return 0
    return int(datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def main() -> None:
    if len(sys.argv) < 2:
        print(textwrap.dedent("""\
            Usage:  python tools/make_license.py "Owner Name" [expires:YYYY-MM-DD]

            Args:
              name          — licensee / buyer name shown in the activation.
              expires       — optional expiry date. Omit or 0/never for permanent keys.
        """))
        sys.exit(2)

    name = sys.argv[1].strip()
    expires_str = sys.argv[2] if len(sys.argv) > 2 else ""
    expires = _parse_expiry(expires_str)

    key = issue_key(name, expires=expires)
    print(f"\n  Licensee  : {name}")
    print(f"  Expires   : {expires_str or 'never (permanent)'}")
    print(f"  Key       : {key}\n")


if __name__ == "__main__":
    main()
