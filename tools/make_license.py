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
    count = 1
    expires_str = ""
    positional = []
    argv = list(sys.argv[1:])
    while argv:
        a = argv.pop(0)
        if a.startswith("--count="):
            count = max(1, int(a.split("=", 1)[1]))
        elif a in ("-n", "--count"):
            if argv:
                count = max(1, int(argv.pop(0)))
            else:
                count = 1
        elif a.lower().startswith("expires:"):
            expires_str = a
        else:
            positional.append(a)

    if not positional:
        print(textwrap.dedent("""\
            Usage:  python tools/make_license.py "Owner Name" [expires:YYYY-MM-DD] [--count N]

            Args:
              name          — licensee / buyer name shown in the activation.
              expires       — optional expiry date. Omit or 0/never for permanent keys.
              --count N     — print N keys at once (e.g. a batch of 10).

            Example:
              python tools/make_license.py "Buyer" --count 10     # names "Buyer 1".."Buyer 10"
        """))
        sys.exit(2)

    name = positional[0].strip()
    expires = _parse_expiry(expires_str)

    for i in range(count):
        licensee = name if count == 1 else f"{name} {i + 1}"
        key = issue_key(licensee, expires=expires)
        print(f"[{i + 1:>2}] {licensee:<24} {key}")

    print()
    print(f"Generated {count} key(s) · expiry: {expires_str or 'never (permanent)'}")
    print("Send each key to its buyer; they paste it into Help → Activate Pro…")


if __name__ == "__main__":
    main()
