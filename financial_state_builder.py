"""
Financial State Builder Pipeline for the "Buy or Wait?" System.

Root-level module re-exporting FinancialStateBuilder from code.financial_state_builder.
"""

from __future__ import annotations

import os
import sys

# Ensure code directory is in sys.path
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_ROOT_DIR, "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from code.financial_state_builder import FinancialStateBuilder

__all__ = ["FinancialStateBuilder"]
