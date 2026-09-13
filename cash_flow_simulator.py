"""
Re-export shim for CashFlowSimulator to ensure seamless imports.
"""

from __future__ import annotations

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_CURRENT_DIR, "code")

if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from code.cash_flow_simulator import CashFlowSimulator

__all__ = ["CashFlowSimulator"]
