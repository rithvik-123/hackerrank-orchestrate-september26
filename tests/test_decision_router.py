"""
Unit and Integration Tests for DecisionRouter (Payment Method Ranking & Routing).
"""

from __future__ import annotations

import os
import sys
import unittest

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

try:
    from decision_router import DecisionRouter
except ImportError:
    from code.decision_router import DecisionRouter


class TestDecisionRouter(unittest.TestCase):
    """Test suite verifying DecisionRouter routing, tie-breaking, and recommendations."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = FinancialStateBuilder()
        cls.extractor = ImageAmountExtractor()
        cls.simulator = CashFlowSimulator()
        cls.router = DecisionRouter(simulator=cls.simulator)

    def test_user_01_full_payment_recommendation(self) -> None:
        """
        Verify user_01 receives full_payment / affordable_now:
        Requested: 25256.0
        Status: affordable_now
        Method: full_payment
        Plan: 2024-03-03:25256
        """
        state = self.builder.get_user_state("user_01", "2024-03-03")
        state["events"] = self.extractor.extract_amounts_for_events(state["events"])
        sim_res = self.simulator.evaluate_request(state, requested_amount=25256.0)

        req_info = {
            "request_id": "request_01",
            "request_date": "2024-03-03",
            "desired_completion_date": "2024-03-20",
            "allows_partial_payment": True,
        }
        rec = self.router.get_recommendation(
            user_state=state,
            requested_amount=25256.0,
            amount_safe_to_pay=sim_res["amount_safe_to_pay"],
            earliest_date_for_full_payment=sim_res["earliest_date_for_full_payment"],
            request_info=req_info,
        )

        self.assertEqual(rec["affordability_status"], "affordable_now")
        self.assertEqual(rec["recommended_payment_method"], "full_payment")
        self.assertEqual(rec["payment_plan"], "2024-03-03:25256")
        self.assertEqual(rec["spending_changes_needed"], "none")

    def test_user_02_installments_recommendation(self) -> None:
        """
        Verify user_02 receives installments / affordable_with_plan:
        Status: affordable_with_plan
        Method: installments
        Option: payment_option_05 (3 payments of 15952906.67)
        """
        state = self.builder.get_user_state("user_02", "2025-08-05")
        state["events"] = self.extractor.extract_amounts_for_events(state["events"])
        sim_res = self.simulator.evaluate_request(state, requested_amount=46018000.0)

        req_info = {
            "request_id": "request_02",
            "request_date": "2025-08-05",
            "desired_completion_date": "2025-10-10",
            "allows_partial_payment": False,
        }
        rec = self.router.get_recommendation(
            user_state=state,
            requested_amount=46018000.0,
            amount_safe_to_pay=sim_res["amount_safe_to_pay"],
            earliest_date_for_full_payment=sim_res["earliest_date_for_full_payment"],
            request_info=req_info,
        )

        self.assertEqual(rec["affordability_status"], "affordable_with_plan")
        self.assertEqual(rec["recommended_payment_method"], "installments")
        self.assertIn("2025-08-08:15952906.67", rec["payment_plan"])
        self.assertEqual(len(rec["payment_plan"].split("|")), 3)

    def test_user_05_not_recommended(self) -> None:
        """
        Verify user_05 receives not_recommended / not_affordable when payment cannot be completed.
        """
        state = self.builder.get_user_state("user_05", "2025-11-06")
        state["events"] = self.extractor.extract_amounts_for_events(state["events"])

        req_info = {
            "request_id": "request_05",
            "request_date": "2025-11-06",
            "desired_completion_date": "2026-01-12",
            "allows_partial_payment": False,
        }
        rec = self.router.get_recommendation(
            user_state=state,
            requested_amount=15488.0,
            amount_safe_to_pay=737.0,
            earliest_date_for_full_payment="",
            request_info=req_info,
        )

        self.assertEqual(rec["affordability_status"], "not_affordable")
        self.assertEqual(rec["recommended_payment_method"], "not_recommended")
        self.assertEqual(rec["payment_plan"], "none")

    def test_partial_payment_route(self) -> None:
        """
        Verify Route 2 (partial_payment):
        When allows_partial=True, partial_payment considered, safe amount > 0,
        and earliest_date <= deadline.
        """
        mock_state = {
            "request_date": "2024-09-04",
            "payment_methods_user_will_consider": ["partial_payment", "installments"],
            "max_installment_months": 6,
            "minimum_balance_to_keep": 1000.0,
            "summary": {"effective_available_cash": 5000.0},
            "events": [],
        }
        req_info = {
            "request_id": "req_mock_partial",
            "request_date": "2024-09-04",
            "desired_completion_date": "2024-10-04",
            "allows_partial_payment": True,
        }
        rec = self.router.get_recommendation(
            user_state=mock_state,
            requested_amount=1000.0,
            amount_safe_to_pay=400.0,
            earliest_date_for_full_payment="2024-09-20",
            request_info=req_info,
            payment_options=[],
        )

        self.assertEqual(rec["affordability_status"], "affordable_with_plan")
        self.assertEqual(rec["recommended_payment_method"], "partial_payment")
        self.assertEqual(rec["payment_plan"], "2024-09-04:400|2024-09-20:600")

    def test_wait_route(self) -> None:
        """
        Verify Route 4 (wait):
        When full payment is considered and safe on a future date.
        """
        mock_state = {
            "request_date": "2024-06-04",
            "payment_methods_user_will_consider": ["full_payment"],
            "minimum_balance_to_keep": 1000.0,
            "summary": {"effective_available_cash": 2000.0},
            "events": [],
        }
        req_info = {
            "request_id": "req_mock_wait",
            "request_date": "2024-06-04",
            "desired_completion_date": "2024-06-30",
            "allows_partial_payment": False,
        }
        rec = self.router.get_recommendation(
            user_state=mock_state,
            requested_amount=5000.0,
            amount_safe_to_pay=500.0,
            earliest_date_for_full_payment="2024-06-15",
            request_info=req_info,
            payment_options=[],
        )

        self.assertEqual(rec["affordability_status"], "affordable_later")
        self.assertEqual(rec["recommended_payment_method"], "wait")
        self.assertEqual(rec["payment_plan"], "2024-06-15:5000")

    def test_tie_breaker_ranking(self) -> None:
        """
        Verify that 6 tie-breakers rank candidates strictly according to rules:
        1. Complete by desired_completion_date
        2. Require no spending changes
        3. Minimize total amount paid
        4. Start payment earlier
        5. Use fewer payments
        6. Lowest payment_option_id
        """
        cand1 = {
            "method": "installments",
            "completes_by_deadline": True,
            "num_spending_changes": 0,
            "total_amount_paid": 1200.0,
            "start_payment_date": "2024-01-01",
            "number_of_payments": 3,
            "payment_option_id": "payment_option_02",
        }
        cand2 = {
            "method": "partial_payment",
            "completes_by_deadline": True,
            "num_spending_changes": 0,
            "total_amount_paid": 1000.0,  # lower total amount paid!
            "start_payment_date": "2024-01-01",
            "number_of_payments": 2,
            "payment_option_id": "payment_option_01",
        }
        ranked = self.router.rank_safe_plans([cand1, cand2])
        self.assertEqual(ranked[0]["method"], "partial_payment")


if __name__ == "__main__":
    unittest.main()
