"""Offline, HMAC-signed entitlement system for Better Brush.

The app ships in two editions:

  * Lite — free, forever. Full masking engine, single-object projects only,
    H.264 MP4 + ProRes/QuickTime MOV + PNG alpha exports capped at
    1920×1080, chroma-key backgrounds and a small watermark on every video
    export.
  * Pro  — once per purchase (€5 at https://sneg.dev/BetterBrush). Removes
    the watermark and unlocks multi-object editing, exports above 1080p
    (up to 4K/8K), original-audio muxing and Library recoding.

Activation is OFFLINE: a Pro license is a self-signed key such as

    BBP-<payload>-<signature>

whose signature is an HMAC-SHA256 of the payload under a shared secret baked
into the build. The distributor issues keys with ``tools/make_license.py``.
Only the signature is persisted locally — never the raw key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from src.core.project import ProjectManager

# Buying / pricing constants for the in-app upsell texts.
PRO_URL = "https://sneg.dev/BetterBrush"
PRO_PRICE_EUR = "5"

# The signing secret. MUST stay identical between tools/make_license.py and
# every shipped build so previously issued keys keep validating.
_SECRET = b"REDACTED-REMOVED-SIGNING-SECRET"

_PREFIX = "BBP-"
_GROUP = 4
_SIG_B32_LEN = 20  # 12 raw digest bytes -> 20 base32 chars (no padding)

# Lite export caps — anything beyond these is a Pro feature.
LITE_MAX_WIDTH = 1920
LITE_MAX_HEIGHT = 1080

ACTIVATION_FILE = "activation.json"


class Entitlement(str, Enum):
    """Editions the app can run in."""

    LITE = "lite"
    PRO = "pro"


# --------------------------------------------------------------------- #
# Key encoding / decoding
# --------------------------------------------------------------------- #
def _b32(data: bytes) -> str:
    return base64.b32encode(data).decode("ascii").rstrip("=")


def _b32decode(s: str) -> bytes:
    s += "=" * (-len(s) % 8)
    return base64.b32decode(s.encode("ascii"))


def _digest(payload: bytes) -> bytes:
    return hmac.new(_SECRET, payload, hashlib.sha256).digest()[:12]


def issue_key(licensee: str, edition: str = "pro", expires: int = 0) -> str:
    """Build a license key. ``expires`` = unix ts (0 = never). Distributor use."""
    payload_obj = {
        "n": licensee.strip(),
        "e": edition,
        "x": int(expires),
        "i": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    payload = json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")
    sig = _b32(_digest(payload))
    body = _b32(payload) + "-" + sig
    key = _PREFIX
    for i in range(0, len(body), _GROUP):
        key += body[i : i + _GROUP] + "-"
    return key.rstrip("-")


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #
class _LicenseError(ValueError):
    pass


def validate_key(key: str) -> dict:
    """Parse + verify a key. Raises a descriptive :class:`ValueError`."""
    text = (key or "").strip().upper()
    if not text.startswith(_PREFIX):
        raise _LicenseError("This does not look like a Better Brush Pro key.")
    body = text[len(_PREFIX) :].replace("-", "")
    if len(body) <= _SIG_B32_LEN:
        raise _LicenseError(
            "The key has the wrong length — check it was copied in full."
        )
    sig_raw, payload_b32 = body[-_SIG_B32_LEN:], body[:-_SIG_B32_LEN]
    try:
        payload = _b32decode(payload_b32)
        sig = _b32decode(sig_raw)
    except (ValueError, TypeError) as exc:
        raise _LicenseError(
            "The key looks corrupted — re-check the characters."
        ) from exc
    if not hmac.compare_digest(sig, _digest(payload)):
        raise _LicenseError("This key is not valid for this copy of Better Brush.")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise _LicenseError("The key could not be read.") from exc
    if not isinstance(data, dict) or data.get("e") != "pro":
        raise _LicenseError("This key does not unlock the Pro edition.")
    if not data.get("n"):
        raise _LicenseError("The key is missing its owner field.")
    exp = int(data.get("x", 0) or 0)
    if exp and exp < int(datetime.now(UTC).timestamp()):
        raise _LicenseError("This key has expired.")
    return data


# --------------------------------------------------------------------- #
# Persisted activation (local, signature-only — never the raw key)
# --------------------------------------------------------------------- #
def _activation_path() -> Path:
    return ProjectManager.settings_dir() / ACTIVATION_FILE


def activate(key: str) -> tuple[bool, str]:
    """Validate *key* and persist the activation. Returns (ok, message)."""
    try:
        data = validate_key(key)
    except ValueError as exc:
        return False, str(exc)
    payload = json.dumps(
        {"n": data["n"], "e": "pro", "x": int(data.get("x", 0) or 0)},
        separators=(",", ":"),
    ).encode("utf-8")
    record = {
        "licensee": data["n"],
        "payload": _b32(payload),
        "activated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "expires": int(data.get("x", 0) or 0),
    }
    _activation_path().write_text(json.dumps(record, indent=2), encoding="utf-8")
    return True, f"Pro activated for {data['n']}"


def deactivate() -> None:
    _activation_path().unlink(missing_ok=True)


def is_pro() -> bool:
    record = _read_activation()
    if not record:
        return False
    try:
        payload = _b32decode(record.get("payload", ""))
        data = json.loads(payload.decode("utf-8"))
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict) or data.get("e") != "pro":
        return False
    exp = int(record.get("expires", 0) or 0)
    if exp and exp < int(datetime.now(UTC).timestamp()):
        return False
    return str(record.get("licensee", "")).strip() == str(data.get("n", "")).strip()


def entitlement() -> Entitlement:
    return Entitlement.PRO if is_pro() else Entitlement.LITE


def current_licensee() -> str | None:
    return _read_activation().get("licensee") if is_pro() else None


def activated_at() -> str | None:
    return _read_activation().get("activated_at") if is_pro() else None


def _read_activation() -> dict | None:
    path = _activation_path()
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
