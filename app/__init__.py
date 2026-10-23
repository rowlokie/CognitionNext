"""
App package init - ensures src is importable from app context.
"""
import sys
from pathlib import Path

# Ensure src is on the path
_root = Path(__file__).parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
