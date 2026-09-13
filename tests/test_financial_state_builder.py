"""
Comprehensive Unit and Integration Tests for FinancialStateBuilder.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add both project root and code/ directories to sys.path
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


class TestFinancialStateBuilder(unittest.TestCase):
    """Test suite verifying all strict business constraints and edge cases."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = FinancialStateBuilder()

    def test_initialization(self) -> None:
        """Verify builder loads profiles, events, and exchange rates correctly."""
        self.assertFalse(self.builder.profiles_df.empty)
        self.assertFalse(self.builder.events_df.empty)
        self.assertFalse(self.builder.exchange_rates_df.empty)
        self.assertGreater(len(self.builder._fx_lookup), 0)

    def test_constraint_1_currency_conversion(self) -> None:
        """
        STRICT CONSTRAINT 1:
        All amounts must be evaluated in the user's home_currency.
        If foreign currency, look up exact multiplier in exchange_rates.csv.
        """
        # User 25 has home_currency IDR and USD events (e.g. event_2167, 1800 USD on 2023-10-15)
        state_25 = self.builder.get_user_state(user_id="user_25", request_date="2024-03-06")
        self.assertEqual(state_25["home_currency"], "IDR")

        usd_event = next(
            (e for e in state_25["events"] if e["event_id"] == "event_2167"),
            None,
        )
        self.assertIsNotNone(usd_event)
        self.assertEqual(usd_event["original_currency"], "USD")
        self.assertEqual(usd_event["currency"], "IDR")
        self.assertEqual(usd_event["original_amount"], 1800.0)
        # Expected rate on 2023-10-15 for USD -> IDR is 15833.33
        self.assertAlmostEqual(usd_event["exchange_rate"], 15833.33, places=2)
        # Converted amount = 1800 * 15833.33 = 28499994.0
        self.assertAlmostEqual(usd_event["amount"], 28499994.0, places=1)

        # User 48 has home_currency INR and foreign USD events (e.g. event_4407 on 2026-02-15)
        state_48 = self.builder.get_user_state(user_id="user_48", request_date="2026-07-25")
        self.assertEqual(state_48["home_currency"], "INR")
        ev_4407 = next((e for e in state_48["events"] if e["event_id"] == "event_4407"), None)
        self.assertIsNotNone(ev_4407)
        self.assertEqual(ev_4407["original_currency"], "USD")
        self.assertEqual(ev_4407["currency"], "INR")
        self.assertAlmostEqual(ev_4407["exchange_rate"], 83.33, places=2)

    def test_constraint_2_missing_amounts_preservation(self) -> None:
        """
        STRICT CONSTRAINT 2:
        If an event has a blank amount, preserve it as None or NaN.
        Do NOT convert it to 0.
        """
        # Test event_253 for user_03 (blank amount)
        state_03 = self.builder.get_user_state(user_id="user_03", request_date="2019-09-03")
        ev_253 = next((e for e in state_03["events"] if e["event_id"] == "event_253"), None)
        self.assertIsNotNone(ev_253)
        self.assertTrue(ev_253["amount"] is None or math.isnan(ev_253["amount"]))
        self.assertNotEqual(ev_253["amount"], 0)

        # Test event_1442 for user_16 (blank amount)
        state_16 = self.builder.get_user_state(user_id="user_16", request_date="2023-08-12")
        ev_1442 = next((e for e in state_16["events"] if e["event_id"] == "event_1442"), None)
        self.assertIsNotNone(ev_1442)
        self.assertTrue(ev_1442["amount"] is None or math.isnan(ev_1442["amount"]))
        self.assertNotEqual(ev_1442["amount"], 0)

        # Test event_7307 for user_78 (blank amount with foreign currency USD -> home currency INR)
        state_78 = self.builder.get_user_state(user_id="user_78", request_date="2025-10-02")
        ev_7307 = next((e for e in state_78["events"] if e["event_id"] == "event_7307"), None)
        self.assertIsNotNone(ev_7307)
        self.assertTrue(ev_7307["amount"] is None or math.isnan(ev_7307["amount"]))
        self.assertNotEqual(ev_7307["amount"], 0)

    def test_constraint_3_event_deduplication_and_conflicts(self) -> None:
        """
        STRICT CONSTRAINT 3:
        Conflict prioritization order:
        a) Explicit cancellations, settlements, or amendments.
        b) Newer record from same source.
        c) Settled event over estimate or forecast.
        d) Financially safer interpretation.
        """
        # Scenario A: Parent cancelled authorization vs Child settled purchase
        # For user_01: event_100 (cancelled) linked to event_101 (settled)
        state_01 = self.builder.get_user_state(user_id="user_01", request_date="2024-03-03")
        event_ids_01 = {e["event_id"] for e in state_01["events"]}
        self.assertIn("event_101", event_ids_01)
        self.assertNotIn("event_100", event_ids_01)  # Cancelled parent dropped

        # Scenario B: Parent failed debit vs Child scheduled retry
        # For user_55: event_5168 (failed) linked to event_5169 (scheduled retry)
        state_55 = self.builder.get_user_state(user_id="user_55", request_date="2026-06-08")
        event_ids_55 = {e["event_id"] for e in state_55["events"]}
        self.assertIn("event_5169", event_ids_55)
        self.assertNotIn("event_5168", event_ids_55)  # Failed parent dropped

        # Scenario C: Settled original card charge vs Pending duplicate card charge
        # For user_138: event_12708 (settled) linked to event_12709 (pending duplicate)
        state_138 = self.builder.get_user_state(user_id="user_138", request_date="2024-09-03")
        event_ids_138 = {e["event_id"] for e in state_138["events"]}
        self.assertIn("event_12708", event_ids_138)
        self.assertNotIn("event_12709", event_ids_138)  # Duplicate estimate dropped

        # Scenario D: Standalone cancelled transaction
        # For user_06: event_557 is standalone 'Cancelled card authorization'
        state_06 = self.builder.get_user_state(user_id="user_06", request_date="2026-01-03")
        event_ids_06 = {e["event_id"] for e in state_06["events"]}
        self.assertNotIn("event_557", event_ids_06)

    def test_constraint_4_cash_flow_rules(self) -> None:
        """
        STRICT CONSTRAINT 4:
        Cash flow rules:
        - "recurring": tag recurring expenses vs. one-time events.
        - "pending": reserve pending debits (treat as active cash drains),
          completely IGNORE pending credits, bonuses, refunds, or investment gains.
        - "salary": count confirmed salary ONLY on its exact settlement date.
        - "unrealized": do not treat unrealized investment value as available cash.
        """
        state_01 = self.builder.get_user_state(user_id="user_01", request_date="2024-03-03")

        # 1. Recurring tagging
        # event_01 (rent) and event_05 (subscription) must be tagged recurring
        ev_01 = next(e for e in state_01["events"] if e["event_id"] == "event_01")
        self.assertTrue(ev_01["is_recurring"])
        self.assertEqual(ev_01["recurrence_type"], "recurring")

        ev_05 = next(e for e in state_01["events"] if e["event_id"] == "event_05")
        self.assertTrue(ev_05["is_recurring"])

        # One-time event: event_101 (settled card purchase)
        ev_101 = next(e for e in state_01["events"] if e["event_id"] == "event_101")
        self.assertFalse(ev_101["is_recurring"])
        self.assertEqual(ev_101["recurrence_type"], "one_time")

        # 2. Pending debits reserved as active cash drain
        # event_102: 'Pending fuel authorization' (amount 567.6, pending debit)
        ev_102 = next(e for e in state_01["events"] if e["event_id"] == "event_102")
        self.assertTrue(ev_102["is_pending_debit"])
        self.assertTrue(ev_102["is_cash_drain"])
        self.assertEqual(state_01["summary"]["total_reserved_pending_debits"], 567.6)
        expected_effective_cash = round(state_01["current_available_balance"] - 567.6, 2)
        self.assertEqual(state_01["summary"]["effective_available_cash"], expected_effective_cash)

        # Pending credits completely IGNORED
        # For user_20: event_1785 is 'Pending merchant refund' (credit)
        state_20 = self.builder.get_user_state(user_id="user_20", request_date="2026-02-07")
        ev_1785 = next((e for e in state_20["events"] if e["event_id"] == "event_1785"), None)
        if ev_1785:
            self.assertTrue(ev_1785["is_pending_credit_ignored"])
            self.assertEqual(ev_1785["cash_flow_effective_amount"], 0.0)

        # 3. Confirmed salary counted ONLY on exact settlement date
        # event_103: 'Next confirmed salary', event_date 2024-03-15, settlement_date 2024-03-15
        ev_103 = next(e for e in state_01["events"] if e["event_id"] == "event_103")
        self.assertTrue(ev_103["is_confirmed_salary"])
        self.assertEqual(ev_103["effective_date"], "2024-03-15")
        self.assertEqual(state_01["summary"]["upcoming_confirmed_salary"]["event_id"], "event_103")
        self.assertEqual(state_01["summary"]["upcoming_confirmed_salary"]["settlement_date"], "2024-03-15")

        # 4. Unrealized investment value NOT treated as available cash
        # For user_21: event_1856 is 'Current portfolio valuation' (unrealized, non_cash)
        state_21 = self.builder.get_user_state(user_id="user_21", request_date="2026-04-03")
        ev_1856 = next((e for e in state_21["events"] if e["event_id"] == "event_1856"), None)
        self.assertIsNotNone(ev_1856)
        self.assertTrue(ev_1856["is_unrealized"])
        self.assertEqual(ev_1856["cash_flow_effective_amount"], 0.0)

    def test_dataframe_output(self) -> None:
        """Verify get_user_events_df returns a valid DataFrame."""
        df = self.builder.get_user_events_df(user_id="user_01", request_date="2024-03-03")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("amount", df.columns)
        self.assertIn("currency", df.columns)
        self.assertIn("is_recurring", df.columns)
        self.assertIn("is_pending_debit", df.columns)

    def test_synthetic_conflicts_and_safe_interpretation(self) -> None:
        """
        Verify synthetic conflict resolution for tie-breaking and Rule 3d (safer interpretation):
        Higher debit = safer; Lower credit = safer.
        """
        profiles_data = pd.DataFrame([
            {
                "user_id": "test_u1",
                "home_currency": "USD",
                "current_available_balance": 1000.0,
                "minimum_balance_to_keep": 200.0,
                "financial_priorities": "emergency_savings",
                "expense_categories_to_protect": "rent",
                "expense_categories_user_is_willing_to_reduce": "",
                "expense_categories_user_is_willing_to_stop": "",
                "payment_methods_user_will_consider": "full_payment",
                "max_installment_months": None,
            }
        ])

        # Conflicting debits on same date with same description:
        # Rule 3d: Financially safer interpretation prefers higher debit
        events_data = pd.DataFrame([
            {
                "event_id": "ev_test_1",
                "user_id": "test_u1",
                "event_type": "expense",
                "description": "Conflicting utility charge",
                "category": "utilities",
                "direction": "debit",
                "amount": 100.0,
                "currency": "USD",
                "event_date": "2024-05-01",
                "settlement_date": "2024-05-01",
                "status": "settled",
                "linked_event_id": np.nan,
                "flexibility": "fixed",
                "minimum_allowed_amount": np.nan,
            },
            {
                "event_id": "ev_test_2",
                "user_id": "test_u1",
                "event_type": "expense",
                "description": "Conflicting utility charge",
                "category": "utilities",
                "direction": "debit",
                "amount": 150.0,
                "currency": "USD",
                "event_date": "2024-05-01",
                "settlement_date": "2024-05-01",
                "status": "settled",
                "linked_event_id": np.nan,
                "flexibility": "fixed",
                "minimum_allowed_amount": np.nan,
            }
        ])

        rates_data = pd.DataFrame(columns=["rate_date", "from_currency", "to_currency", "rate"])

        custom_builder = FinancialStateBuilder(
            profiles_df=profiles_data,
            events_df=events_data,
            exchange_rates_df=rates_data,
        )

        state = custom_builder.get_user_state("test_u1", "2024-05-02")
        self.assertEqual(len(state["events"]), 1)
        # Higher amount (150.0) chosen under Rule 3d
        self.assertEqual(state["events"][0]["amount"], 150.0)


    def test_all_sample_requests_users(self) -> None:
        """Verify that get_user_state executes cleanly for all 25 sample requests."""
        samples_path = Path("dataset/sample_requests.csv")
        if not samples_path.exists():
            return

        samples = pd.read_csv(samples_path)
        for _, req in samples.iterrows():
            user_id = req["user_id"]
            req_date = req["request_date"]
            state = self.builder.get_user_state(user_id=user_id, request_date=req_date)
            
            self.assertEqual(state["user_id"], user_id)
            self.assertEqual(state["request_date"], req_date)
            self.assertIn("home_currency", state)
            self.assertIn("events", state)
            self.assertIn("summary", state)
            self.assertIsInstance(state["events"], list)
            
            # Check consistency of summary metrics
            avail = state["current_available_balance"]
            pending = state["summary"]["total_reserved_pending_debits"]
            eff = state["summary"]["effective_available_cash"]
            self.assertAlmostEqual(eff, avail - pending, places=2)


    def test_all_250_requests(self) -> None:
        """Verify that get_user_state executes cleanly for all 250 evaluation requests."""
        requests_path = Path("dataset/requests.csv")
        if not requests_path.exists():
            return

        requests_df = pd.read_csv(requests_path)
        for _, req in requests_df.iterrows():
            user_id = req["user_id"]
            req_date = req["request_date"]
            state = self.builder.get_user_state(user_id=user_id, request_date=req_date)
            
            self.assertEqual(state["user_id"], user_id)
            self.assertEqual(state["request_date"], req_date)
            self.assertIn("home_currency", state)
            self.assertIn("events", state)
            self.assertIn("summary", state)


if __name__ == "__main__":
    unittest.main()
