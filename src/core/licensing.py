"""Edition helper for Better Brush.

Better Brush is **free**. There is no Lite/Pro split, no activation keys and no
pricing — every capability (multi-object projects, exports above 1080p up to
4K/8K, original-audio muxing, watermark-free output and Library recoding) is
unlocked for everyone, offline, with nothing to buy and nothing to activate.

The module survives as a tiny, dependency-free surface so the rest of the app
(and the Adobe panels, through the engine's ``/health`` payload) has a single
place to ask "is this a fully-unlocked copy?". The answer is always *yes*.

The previous offline HMAC key system — and the signing secret it depended on —
has been removed, so the shipped build contains no secret material and there is
nothing to bypass.
"""

from __future__ import annotations

from enum import Enum


class Entitlement(str, Enum):
    """How unlocked the running copy is. Better Brush only has one: full."""

    FREE = "free"


def is_pro() -> bool:
    """Return ``True`` — every copy ships fully unlocked and free."""
    return True


def entitlement() -> Entitlement:
    """The running edition (always :attr:`Entitlement.FREE`)."""
    return Entitlement.FREE


def current_licensee() -> str | None:
    """No activation exists, so there is no licensee name."""
    return None


def activated_at() -> str | None:
    """No activation exists, so there is no activation timestamp."""
    return None
