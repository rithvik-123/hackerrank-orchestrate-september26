"""
Decision Router for the "Buy or Wait?" Financial Agent.

Determines the optimal recommended payment method, affordability status,
and chronological payment plan by evaluating eligibility, 90-day safety,
and the 6 hierarchical tie-breaker ranking rules.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from cash_flow_simulator import CashFlowSimulator
except ImportError:
    from code.cash_flow_simulator import CashFlowSimulator


class DecisionRouter:
    """
    Ranks safe and eligible payment methods to generate the final recommendation.
    """

    def __init__(
        self,
        payment_options_path: Optional[Union[str, Path]] = None,
        simulator: Optional[CashFlowSimulator] = None,
    ) -> None:
        """
        Initialize the DecisionRouter.

        Args:
            payment_options_path: Path to request_payment_options.csv.
            simulator: Optional CashFlowSimulator instance.
        """
        self.simulator = simulator or CashFlowSimulator()

        if payment_options_path is None:
            # Default to dataset/request_payment_options.csv
            base_dir = Path(__file__).resolve().parent.parent
            payment_options_path = base_dir / "dataset" / "request_payment_options.csv"

        self.options_path = Path(payment_options_path)
        if self.options_path.exists():
            self.options_df = pd.read_csv(self.options_path)
        else:
            self.options_df = pd.DataFrame()

    def format_amount(self, amt: float) -> str:
        """
        Format monetary amounts matching benchmark precision style:
        Integers as plain numbers (e.g. 25256), floats with cents (e.g. 166.61).
        """
        amt_float = float(amt)
        if abs(amt_float - round(amt_float)) < 1e-5:
            return str(int(round(amt_float)))
        return f"{amt_float:.2f}"

    def get_payment_options_for_request(self, request_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve available payment options for a specific request.
        """
        if self.options_df.empty or "request_id" not in self.options_df.columns:
            return []

        matched = self.options_df[self.options_df["request_id"] == str(request_id).strip()]
        return matched.to_dict(orient="records")

    def is_installment_plan_safe(
        self,
        user_state: Dict[str, Any],
        option: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Check if an installment plan option maintains the user's minimum balance
        throughout the 90-day simulation window.

        Returns:
            Tuple of (is_safe: bool, payment_dates: List[str])
        """
        dates, balances, buffers, deltas = self.simulator.simulate_baseline_forecast(user_state)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        init_cash = float(user_state["summary"]["effective_available_cash"])
        min_balance = float(user_state["minimum_balance_to_keep"])

        num_payments = int(option["number_of_payments"])
        amt = float(option["payment_amount"])
        freq_days = int(float(option.get("payment_frequency_days") or 30))
        first_date_str = str(option["first_payment_date"]).strip()

        try:
            first_dt = datetime.strptime(first_date_str, "%Y-%m-%d")
        except ValueError:
            return False, []

        pay_dates = [
            (first_dt + timedelta(days=k * freq_days)).strftime("%Y-%m-%d")
            for k in range(num_payments)
        ]

        # Simulate deduction of installment debits falling within the 90-day window
        cur_deltas = deltas.copy()
        for p_date in pay_dates:
            if p_date in date_to_idx:
                cur_deltas[date_to_idx[p_date]] -= amt

        cur_balances = init_cash + np.cumsum(cur_deltas)
        cur_buffers = cur_balances - min_balance
        is_safe = bool(np.min(cur_buffers) >= 0)

        return is_safe, pay_dates

    def rank_safe_plans(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank candidate safe payment plans using the 6 hierarchical tie-breakers:
        1. Complete full request by desired_completion_date.
        2. Require no spending changes.
        3. Minimize total amount paid.
        4. Start payment earlier.
        5. Use fewer payments.
        6. Lowest payment_option_id as final tie-breaker.
        """
        if not candidates:
            return []

        def sort_key(c: Dict[str, Any]) -> Tuple[Any, ...]:
            return (
                0 if c.get("completes_by_deadline", False) else 1,
                int(c.get("num_spending_changes", 0)),
                round(float(c.get("total_amount_paid", 0.0)), 2),
                str(c.get("start_payment_date", "")),
                int(c.get("number_of_payments", 1)),
                str(c.get("payment_option_id", "") or "zzzz"),
            )

        return sorted(candidates, key=sort_key)

    def get_recommendation(
        self,
        user_state: Dict[str, Any],
        requested_amount: float,
        amount_safe_to_pay: float,
        earliest_date_for_full_payment: Optional[str] = None,
        request_info: Optional[Dict[str, Any]] = None,
        payment_options: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Determine the recommended payment method, affordability status, and payment plan.

        Args:
            user_state: Reconstructed user state dictionary from FinancialStateBuilder.
            requested_amount: Total purchase/transfer amount.
            amount_safe_to_pay: Safe amount calculated by CashFlowSimulator.
            earliest_date_for_full_payment: Earliest safe full payment date.
            request_info: Optional dict with request metadata (request_id, request_date, desired_completion_date, allows_partial_payment).
            payment_options: Optional explicit list of option dicts from request_payment_options.csv.

        Returns:
            Dictionary with:
                - affordability_status
                - recommended_payment_method
                - payment_plan
                - amount_safe_to_pay
                - earliest_date_for_full_payment
                - spending_changes_needed
        """
        req_info = request_info or {}
        req_id = str(req_info.get("request_id", "")).strip()
        req_date = str(req_info.get("request_date") or user_state.get("request_date", "")).strip()
        deadline = str(req_info.get("desired_completion_date") or req_date).strip()
        allows_partial = bool(req_info.get("allows_partial_payment", False))

        req_amount = float(requested_amount)
        safe_amount = float(amount_safe_to_pay)
        earliest_date = str(earliest_date_for_full_payment).strip() if earliest_date_for_full_payment else ""

        considered_methods = [
            m.strip().lower()
            for m in user_state.get("payment_methods_user_will_consider", [])
        ]
        max_inst_months = user_state.get("max_installment_months")

        # Retrieve payment options
        if payment_options is None and req_id:
            payment_options = self.get_payment_options_for_request(req_id)
        elif payment_options is None:
            payment_options = []

        candidates: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Route 1: Full Payment
        # -------------------------------------------------------------
        if "full_payment" in considered_methods and safe_amount >= req_amount:
            # Look up option_id if available
            full_opt_id = ""
            for opt in payment_options:
                if str(opt.get("payment_method", "")).lower() == "full_payment":
                    full_opt_id = str(opt.get("payment_option_id", ""))
                    break

            candidates.append({
                "method": "full_payment",
                "status": "affordable_now",
                "plan": f"{req_date}:{self.format_amount(req_amount)}",
                "completes_by_deadline": (req_date <= deadline),
                "num_spending_changes": 0,
                "total_amount_paid": req_amount,
                "start_payment_date": req_date,
                "number_of_payments": 1,
                "payment_option_id": full_opt_id or "option_full",
                "spending_changes_needed": "none",
            })

        # -------------------------------------------------------------
        # Route 2: Partial Payment
        # -------------------------------------------------------------
        if (
            "partial_payment" in considered_methods
            and allows_partial
            and 0 < safe_amount < req_amount
            and earliest_date
            and earliest_date <= deadline
        ):
            rem_amount = req_amount - safe_amount
            partial_plan = (
                f"{req_date}:{self.format_amount(safe_amount)}|"
                f"{earliest_date}:{self.format_amount(rem_amount)}"
            )
            candidates.append({
                "method": "partial_payment",
                "status": "affordable_with_plan",
                "plan": partial_plan,
                "completes_by_deadline": True,
                "num_spending_changes": 0,
                "total_amount_paid": req_amount,
                "start_payment_date": req_date,
                "number_of_payments": 2,
                "payment_option_id": "option_partial",
                "spending_changes_needed": "none",
            })

        # -------------------------------------------------------------
        # Route 3: Installments
        # -------------------------------------------------------------
        if "installments" in considered_methods and payment_options:
            for opt in payment_options:
                method_type = str(opt.get("payment_method", "")).strip().lower()
                if method_type != "installments":
                    continue

                num_payments = int(opt["number_of_payments"])
                if max_inst_months is not None and num_payments > max_inst_months:
                    continue

                is_safe, pay_dates = self.is_installment_plan_safe(user_state, opt)
                if is_safe and pay_dates:
                    p_amt_str = self.format_amount(float(opt["payment_amount"]))
                    plan_str = "|".join(f"{d}:{p_amt_str}" for d in pay_dates)
                    last_pay_date = pay_dates[-1]
                    completes = (last_pay_date <= deadline)
                    tot_paid = float(opt.get("total_payable_amount") or (num_payments * float(opt["payment_amount"])))

                    candidates.append({
                        "method": "installments",
                        "status": "affordable_with_plan",
                        "plan": plan_str,
                        "completes_by_deadline": completes,
                        "num_spending_changes": 0,
                        "total_amount_paid": tot_paid,
                        "start_payment_date": pay_dates[0],
                        "number_of_payments": num_payments,
                        "payment_option_id": str(opt.get("payment_option_id", "")),
                        "spending_changes_needed": "none",
                    })

        # -------------------------------------------------------------
        # Route 4: Wait
        # -------------------------------------------------------------
        if (
            "full_payment" in considered_methods
            and earliest_date
            and earliest_date > req_date
        ):
            completes = (earliest_date <= deadline)
            candidates.append({
                "method": "wait",
                "status": "affordable_later",
                "plan": f"{earliest_date}:{self.format_amount(req_amount)}",
                "completes_by_deadline": completes,
                "num_spending_changes": 0,
                "total_amount_paid": req_amount,
                "start_payment_date": earliest_date,
                "number_of_payments": 1,
                "payment_option_id": "option_wait",
                "spending_changes_needed": "none",
            })

        # -------------------------------------------------------------
        # Selection & Tie-Breaking
        # -------------------------------------------------------------
        if candidates:
            ranked = self.rank_safe_plans(candidates)
            best_plan = ranked[0]
            rec_status = best_plan["status"]
            rec_method = best_plan["method"]
            plan_str = best_plan["plan"]
            spending_changes = best_plan.get("spending_changes_needed", "none")
            chosen_details = best_plan
        else:
            # Route 5: Not Recommended
            rec_status = "not_affordable"
            rec_method = "not_recommended"
            plan_str = "none"
            spending_changes = "none"
            chosen_details = {}

        return {
            "affordability_status": rec_status,
            "recommended_payment_method": rec_method,
            "payment_plan": plan_str,
            "amount_safe_to_pay": safe_amount,
            "earliest_date_for_full_payment": earliest_date,
            "spending_changes_needed": spending_changes,
            "selected_plan_details": chosen_details,
            "total_candidates_evaluated": len(candidates),
        }
