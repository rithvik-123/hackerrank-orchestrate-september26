"""
Multi-Modal Vision Extractor for the "Buy or Wait?" System.

Root-level re-export for ImageAmountExtractor from code.vision_extractor.
"""

from __future__ import annotations

import os
import sys

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_ROOT_DIR, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from code.vision_extractor import ImageAmountExtractor

__all__ = ["ImageAmountExtractor"]
