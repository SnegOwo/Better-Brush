"""Makes the MVP importable when pytest is run from the repository root.

The repository's own ``pytest`` configuration uses ``testpaths = ["tests"]``,
so this folder is invisible to the existing suite by design. Run these with::

    python -m pytest extensions/effectcraft/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

MVP = Path(__file__).resolve().parents[1]
if str(MVP) not in sys.path:
    sys.path.insert(0, str(MVP))
