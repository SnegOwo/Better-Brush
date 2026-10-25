"""Better Brush <-> EffectCraft bridge (MVP).

Standard-library-first: only :mod:`.pipeline` pulls in numpy/cv2, and only
when a job actually runs, so the protocol and the command list stay testable
on a machine with no GPU.
"""

from __future__ import annotations

from .protocol import PROTOCOL_VERSION, Job, Status

__all__ = ["PROTOCOL_VERSION", "Job", "Status"]
