from __future__ import annotations

import sys
from pathlib import Path

ARTICLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ARTICLE_ROOT.parent
for name in list(sys.modules):
    if name == "gdex_bufr" or name.startswith("gdex_bufr."):
        del sys.modules[name]
while str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(ARTICLE_ROOT))
