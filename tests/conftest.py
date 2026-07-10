"""Test harness: prefer vendored PyPokerEngine fork over site-packages."""

import sys
from pathlib import Path

_VENDORED_ROOT = (
    Path(__file__).resolve().parents[1] / "src" / "engine_integration" / "pypokerengine"
)
_VENDORED_PACKAGE = _VENDORED_ROOT / "pypokerengine"

if _VENDORED_PACKAGE.is_dir():
    vendored_root = str(_VENDORED_ROOT)
    if vendored_root not in sys.path:
        sys.path.insert(0, vendored_root)

    # Drop a previously imported site-packages copy so tests use the fork.
    for name in list(sys.modules):
        if name == "pypokerengine" or name.startswith("pypokerengine."):
            del sys.modules[name]
