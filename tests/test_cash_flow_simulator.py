"""
Unit and Integration Tests for CashFlowSimulator (90-Day Cash Flow Forecast).
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta

# Add project root and code/ to sys.path
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
_CODE_DIR = os.path.abspath(os.path.join(_TESTS_DIR, "..", "code"))

if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from financial_state_builder import FinancialStateBuilder
except ImportError:
    from code.financial_state_builder import FinancialStateBuilder

try:
    from vision_extractor import ImageAmountExtractor
except ImportError:
    from code.vision_extractor import ImageAmountExtractor

try:
    from cash_flow_simulator import CashFlowSimulator
except ImportError:
    from code.cash_flow_simulator import CashFlowSimulator


class TestCashFlowSimulator(unittest.TestCase):
    """Test suite verifying 90-day cash flow forecast and safety logic."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = FinancialStateBuilder()
        cls.extractor = ImageAmountExtractor()
        cls.simulator = CashFlowSimulator(forecast_days=90)

    def test_user_01_evaluation(self) -> None:
        """
        Verify user_01 evaluation matches ground truth:
        Requested: 25,256.0 ZAR
        Safe today: 25,256.0 ZAR
        Earliest date: 2024-03-03
        """
        state = self.builder.get_user_state("user_01", "2024-03-03")
        state["events"] = self.extractor.extract_amounts_for_events(state["events"])

        res = self.simulator.evaluate_request(state, requested_amount=25256.0)

        self.assertEqual(res["amount_safe_to_pay"], 25256.0)
        self.assertEqual(res["earliest_date_for_full_payment"], "2024-03-03")
        self.assertTrue(res["is_safe_today"])
        self.assertGreaterEqual(res["minimum_buffer"], 25256.0)

    def test_baseline_forecast_structure(self) -> None:
        """Verify baseline forecast trajectory length, dates, and balance buffers."""
        state = self.builder.get_user_state("user_01", "2024-03-03")
        state["events"] = self.extractor.extract_amounts_for_events(state["events"])

        dates, balances, buffers, deltas = self.simulator.simulate_baseline_forecast(state)

        self.assertEqual(len(dates), 90)
        self.assertEqual(len(balances), 90)
        self.assertEqual(len(buffers), 90)
        self.assertEqual(len(deltas), 90)

        # First day matches request_date
        self.assertEqual(dates[0], "2024-03-03")
        # 90th day matches start + 89 days
        expected_end = (datetime.strptime("2024-03-03", "%Y-%m-%d") + timedelta(days=89)).strftime("%Y-%m-%d")
        self.assertEqual(dates[-1], expected_end)

        # Buffer invariant: buffer = balance - minimum_balance_to_keep
        min_bal = state["minimum_balance_to_keep"]
        for bal, buf in zip(balances, buffers):
            self.assertAlmostEqual(buf, bal - min_bal, places=2)

    def test_amount_safe_to_pay_bounds(self) -> None:
        """Verify 0 <= amount_safe_to_pay <= requested_amount."""
        state = self.builder.get_user_state("user_01", "2024-03-03")
        state["events"] = self.extractor.extract_amounts_for_events(state["events"])

        # Small request -> capped by requested_amount
        res_small = self.simulator.evaluate_request(state, requested_amount=100.0)
        self.assertEqual(res_small["amount_safe_to_pay"], 100.0)
        self.assertEqual(res_small["earliest_date_for_full_payment"], "2024-03-03")

        # Huge request -> capped by minimum buffer
        res_huge = self.simulator.evaluate_request(state, requested_amount=500000.0)
        self.assertLess(res_huge["amount_safe_to_pay"], 500000.0)
        self.assertEqual(res_huge["amount_safe_to_pay"], res_huge["minimum_buffer"])
        self.assertFalse(res_huge["is_safe_today"])

    def test_forward_scan_earliest_date(self) -> None:
        """Verify forward scanning identifies the first day safe for full payment."""
        state = self.builder.get_user_state("user_01", "2024-03-03")
        state["events"] = self.extractor.extract_amounts_for_events(state["events"])

        min_buf = self.simulator.evaluate_request(state, requested_amount=1.0)["minimum_buffer"]
        # Amount slightly above minimum buffer cannot be paid today
        excess_amount = min_buf + 2000.0
        res = self.simulator.evaluate_request(state, requested_amount=excess_amount)

        self.assertLess(res["amount_safe_to_pay"], excess_amount)
        # Should find a future date (e.g. after salary settlement) or empty string
        if res["earliest_date_for_full_payment"]:
            self.assertGreater(res["earliest_date_for_full_payment"], "2024-03-03")

    def test_ground_truth_samples_exact(self) -> None:
        """Verify exact matches against ground truth sample requests."""
        test_cases = [
            ("user_09", "2026-07-04", 166.61, 166.61, "2026-07-04"),
            ("user_12", "2026-04-05", 65164.0, 65164.0, "2026-04-05"),
            ("user_16", "2023-08-12", 122500.0, 122500.0, "2023-08-12"),
        ]
        for u_id, req_date, req_amt, expected_safe, expected_date in test_cases:
            state = self.builder.get_user_state(u_id, req_date)
            state["events"] = self.extractor.extract_amounts_for_events(state["events"])
            res = self.simulator.evaluate_request(state, requested_amount=req_amt)

            self.assertEqual(res["amount_safe_to_pay"], expected_safe, f"Failed for {u_id}")
            self.assertEqual(res["earliest_date_for_full_payment"], expected_date, f"Failed for {u_id}")


if __name__ == "__main__":
    unittest.main()
