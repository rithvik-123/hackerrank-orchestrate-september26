"""
90-Day Cash Flow Simulator for the "Buy or Wait?" System.

Simulates day-by-day account balance over a 90-day forecast horizon from request_date,
incorporating starting available cash, confirmed future salary settlements,
and scheduled/recurring commitments on their expected monthly cadence.
Calculates amount_safe_to_pay and earliest_date_for_full_payment.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class CashFlowSimulator:
    """
    Simulates 90-day point-in-time financial trajectories to determine
    safe expenditure limits and earliest safe payment dates.
    """

    def __init__(self, forecast_days: int = 90) -> None:
        """
        Initialize the simulator.

        Args:
            forecast_days: Number of forecast days to simulate (default 90).
        """
        self.forecast_days = forecast_days

    def _get_clamped_day(self, year: int, month: int, target_day: int) -> int:
        """Clamp day of month to maximum days in the given year/month."""
        max_days = calendar.monthrange(year, month)[1]
        return min(target_day, max_days)

    MONTHLY_COMMITMENT_CATEGORIES = {
        "rent",
        "utilities",
        "education",
        "debt_repayment",
        "insurance",
        "healthcare",
        "family_support",
        "housing",
        "music_subscription",
        "delivery_membership",
        "cloud_storage",
        "streaming",
        "gym",
    }

    FLEXIBLE_STATUSES = {"stoppable", "reducible", "reducible_or_stoppable"}

    def _extract_recurring_templates(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract unique recurring debit templates from normalized events.
        Captures recurring monthly commitments (rent, utilities, debt, subscriptions,
        insurance, healthcare, family support, housing) and flexible expenses,
        excluding ad-hoc/discretionary daily groceries, dining, and transport.
        Uses the most recent historical occurrence for recurring amount and day of month.
        """
        templates: Dict[str, Dict[str, Any]] = {}

        for e in events:
            if (
                e.get("direction") == "debit"
                and e.get("amount") is not None
                and not e.get("is_confirmed_salary")
                and not e.get("is_unrealized")
            ):
                cat = str(e.get("category", "")).strip().lower()
                ev_type = str(e.get("event_type", "")).strip().lower()
                flex = str(e.get("flexibility", "")).strip().lower()
                is_rec = bool(e.get("is_recurring"))

                # Identify if event represents a monthly recurring commitment
                is_monthly_commitment = (
                    ev_type in {"subscription", "debt_payment"}
                    or cat in self.MONTHLY_COMMITMENT_CATEGORIES
                    or flex in self.FLEXIBLE_STATUSES
                    or (is_rec and cat not in {"groceries", "transport", "dining", "shopping", "entertainment"})
                )

                # Discretionary groceries and transport are not monthly scheduled bills
                if cat in {"groceries", "transport"}:
                    is_monthly_commitment = False
                if cat == "dining" and flex not in self.FLEXIBLE_STATUSES:
                    is_monthly_commitment = False

                if is_monthly_commitment:
                    desc = str(e.get("description", "")).strip()
                    ev_date_str = str(e.get("event_date", "")).strip()
                    try:
                        ev_dt = datetime.strptime(ev_date_str, "%Y-%m-%d")
                        day = ev_dt.day
                    except ValueError:
                        day = 1

                    # Overwrite with latest occurrence
                    templates[desc] = {
                        "amount": float(e["amount"]),
                        "day": day,
                        "category": cat,
                        "description": desc,
                        "flexibility": flex,
                        "event_id": e.get("event_id"),
                    }

        return templates

    def _extract_salary_template(
        self,
        events: List[Dict[str, Any]],
        request_date_str: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract confirmed salary schedule and settlement day.
        Prioritizes scheduled/future confirmed salary on or after request_date.
        """
        salary_events = [
            e for e in events
            if e.get("is_confirmed_salary") and e.get("amount") is not None
        ]

        if not salary_events:
            return None

        # Prefer future/scheduled confirmed salaries on or after request_date
        if request_date_str:
            future_salaries = [
                e for e in salary_events
                if str(e.get("effective_date") or e.get("event_date", "")).strip() >= request_date_str
            ]
            if future_salaries:
                future_salaries.sort(key=lambda s: str(s.get("effective_date") or s.get("event_date", "")))
                target_sal = future_salaries[0]
            else:
                # Check if user had a final payroll (job end)
                salary_events.sort(key=lambda s: str(s.get("effective_date") or s.get("event_date", "")))
                latest = salary_events[-1]
                if "final" in str(latest.get("description", "")).lower():
                    return None
                target_sal = latest
        else:
            salary_events.sort(key=lambda s: str(s.get("effective_date") or s.get("event_date", "")))
            target_sal = salary_events[-1]

        eff_date_str = str(target_sal.get("effective_date") or target_sal.get("event_date", "")).strip()
        try:
            eff_dt = datetime.strptime(eff_date_str, "%Y-%m-%d")
            sal_day = eff_dt.day
        except ValueError:
            sal_day = 15

        return {
            "amount": float(target_sal["amount"]),
            "day": sal_day,
            "first_date": eff_date_str,
            "description": target_sal.get("description", "Confirmed salary"),
        }

    def simulate_baseline_forecast(
        self,
        user_state: Dict[str, Any],
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 90-day baseline day-by-day cash flow forecast.

        Returns:
            Tuple of (dates_list, daily_balances, daily_buffers, daily_deltas)
        """
        request_date_str = str(user_state["request_date"]).strip()
        start_dt = datetime.strptime(request_date_str, "%Y-%m-%d")
        min_balance = float(user_state["minimum_balance_to_keep"])
        init_cash = float(user_state["summary"]["effective_available_cash"])
        events = user_state.get("events", [])

        # Generate list of 90 dates
        dates = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(self.forecast_days)]
        date_to_idx = {d: i for i, d in enumerate(dates)}

        daily_delta = np.zeros(self.forecast_days, dtype=float)

        # 1. Schedule confirmed salary on exact settlement day for each month in the 90 days
        salary_info = self._extract_salary_template(events, request_date_str=request_date_str)
        if salary_info:
            sal_amt = salary_info["amount"]
            sal_day = salary_info["day"]

            for d_str in dates:
                cur_dt = datetime.strptime(d_str, "%Y-%m-%d")
                target_day = self._get_clamped_day(cur_dt.year, cur_dt.month, sal_day)
                # Count confirmed salary only on or after request_date
                if cur_dt.day == target_day and d_str >= request_date_str:
                    daily_delta[date_to_idx[d_str]] += sal_amt

        # 2. Add explicit future scheduled events (debts, retries, etc.) that have specific future dates
        for e in events:
            eff_d = str(e.get("effective_date") or e.get("event_date", "")).strip()
            if (
                eff_d in date_to_idx
                and eff_d > request_date_str
                and not e.get("is_confirmed_salary")
                and e.get("status") in {"scheduled", "confirmed"}
            ):
                amt = float(e["amount"]) if e.get("amount") is not None else 0.0
                if e.get("is_cash_drain") or e.get("direction") == "debit":
                    daily_delta[date_to_idx[eff_d]] -= amt
                elif e.get("direction") == "credit":
                    daily_delta[date_to_idx[eff_d]] += amt

        # 3. Project recurring expenses on their monthly cadence
        rec_templates = self._extract_recurring_templates(events)
        for d_str in dates:
            # Recurring expenses occur strictly after request_date cutoff
            if d_str > request_date_str:
                cur_dt = datetime.strptime(d_str, "%Y-%m-%d")
                for desc, t in rec_templates.items():
                    target_day = self._get_clamped_day(cur_dt.year, cur_dt.month, t["day"])
                    if cur_dt.day == target_day:
                        daily_delta[date_to_idx[d_str]] -= t["amount"]

        # Compute cumulative daily balances
        daily_balances = np.zeros(self.forecast_days, dtype=float)
        cur_bal = init_cash
        for i in range(self.forecast_days):
            cur_bal += daily_delta[i]
            daily_balances[i] = cur_bal

        daily_buffers = daily_balances - min_balance
        return dates, daily_balances, daily_buffers, daily_delta

    def evaluate_request(
        self,
        user_state: Dict[str, Any],
        requested_amount: float,
    ) -> Dict[str, Any]:
        """
        Simulate 90-day baseline cash flow and calculate amount_safe_to_pay
        and earliest_date_for_full_payment according to hackathon rules.

        Args:
            user_state: Reconstructed financial state dictionary from FinancialStateBuilder.
            requested_amount: Total expense amount the user wants to commit.

        Returns:
            Dictionary containing:
                - amount_safe_to_pay (float)
                - earliest_date_for_full_payment (Optional[str])
                - minimum_buffer (float)
                - is_safe_today (bool)
                - daily_forecast (List[Dict[str, Any]])
        """
        req_amount = float(requested_amount)
        request_date_str = str(user_state["request_date"]).strip()
        min_balance = float(user_state["minimum_balance_to_keep"])
        init_cash = float(user_state["summary"]["effective_available_cash"])

        dates, daily_balances, daily_buffers, daily_deltas = self.simulate_baseline_forecast(user_state)

        # 1. Calculate amount_safe_to_pay:
        # Minimum buffer across the entire 90-day forecast
        min_buffer = float(np.min(daily_buffers))
        amount_safe = min(req_amount, max(0.0, min_buffer))
        amount_safe = round(amount_safe, 2)

        # 2. Calculate earliest_date_for_full_payment:
        # If full amount is safe today, earliest date is exactly request_date
        if amount_safe == req_amount:
            earliest_date: Optional[str] = request_date_str
        else:
            earliest_date = None
            for t in range(self.forecast_days):
                # Check if balance remains >= min_balance for remainder of the 90-day window (days t..89)
                if np.min(daily_buffers[t:]) >= req_amount:
                    earliest_date = dates[t]
                    break

        return {
            "amount_safe_to_pay": amount_safe,
            "earliest_date_for_full_payment": earliest_date if earliest_date else "",
            "requested_amount": req_amount,
            "minimum_buffer": round(min_buffer, 2),
            "minimum_balance_to_keep": min_balance,
            "effective_available_cash": init_cash,
            "is_safe_today": (amount_safe == req_amount),
            "forecast_period_days": self.forecast_days,
            "daily_forecast": [
                {
                    "day_index": i,
                    "date": dates[i],
                    "balance": round(float(daily_balances[i]), 2),
                    "buffer": round(float(daily_buffers[i]), 2),
                    "delta": round(float(daily_deltas[i]), 2),
                }
                for i in range(self.forecast_days)
            ],
        }
