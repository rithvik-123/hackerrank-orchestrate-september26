"""
Financial State Builder Pipeline for the "Buy or Wait?" System.

Deterministic financial data ingestion, multi-currency normalization,
event deduplication, conflict resolution, and cash flow state reconstruction using Pandas.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


class FinancialStateBuilder:
    """
    Deterministic financial data ingestion and state reconstruction pipeline built with Pandas.

    Loads and normalizes financial profiles, historical/pending transactions,
    and dated exchange rates. Provides normalized ledgers with explicit
    currency conversion into the user's home currency, missing amount preservation,
    priority-based conflict resolution, and conservative cash flow tagging.
    """

    RECURRING_CATEGORIES: Set[str] = {
        "rent",
        "utilities",
        "debt_repayment",
        "education",
        "insurance",
        "gym",
        "cloud_storage",
        "streaming",
        "music_subscription",
        "delivery_membership",
        "housing",
        "family_support",
    }

    FLEXIBLE_STATUSES: Set[str] = {
        "reducible",
        "stoppable",
        "reducible_or_stoppable",
    }

    ONE_TIME_KEYWORDS: Set[str] = {
        "one-time",
        "reversal",
        "authorization",
        "reimbursement",
        "windfall",
        "prize",
        "repair",
        "trip",
        "order",
        "purchase",
        "deposit",
        "laptop",
        "tote bag",
        "duplicate",
    }

    def __init__(
        self,
        profiles_path: Union[str, Path] = "dataset/financial_profiles.csv",
        events_path: Union[str, Path] = "dataset/financial_events.csv",
        exchange_rates_path: Union[str, Path] = "dataset/exchange_rates.csv",
        messages_path: Optional[Union[str, Path]] = "dataset/messages.csv",
        profiles_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        exchange_rates_df: Optional[pd.DataFrame] = None,
        messages_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the FinancialStateBuilder.

        Args:
            profiles_path: Path to financial_profiles.csv.
            events_path: Path to financial_events.csv.
            exchange_rates_path: Path to exchange_rates.csv.
            messages_path: Optional path to messages.csv.
            profiles_df: Optional pre-loaded DataFrame for profiles.
            events_df: Optional pre-loaded DataFrame for events.
            exchange_rates_df: Optional pre-loaded DataFrame for exchange rates.
            messages_df: Optional pre-loaded DataFrame for messages.
        """
        self.profiles_df = (
            profiles_df.copy()
            if profiles_df is not None
            else self._load_csv(profiles_path)
        )
        self.events_df = (
            events_df.copy()
            if events_df is not None
            else self._load_csv(events_path)
        )
        self.exchange_rates_df = (
            exchange_rates_df.copy()
            if exchange_rates_df is not None
            else self._load_csv(exchange_rates_path)
        )
        self.messages_df = (
            messages_df.copy()
            if messages_df is not None
            else (self._load_csv(messages_path) if messages_path and Path(messages_path).exists() else None)
        )

        self._preprocess_data()

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a file path relative to cwd or project root."""
        p = Path(path)
        if p.exists():
            return p

        # Check relative to repo root (parent of 'code' directory)
        module_dir = Path(__file__).resolve().parent
        repo_root = module_dir.parent if module_dir.name == "code" else module_dir
        repo_candidate = repo_root / path
        if repo_candidate.exists():
            return repo_candidate

        return p

    def _load_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        """Safely load a CSV file into a DataFrame."""
        resolved = self._resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Required CSV file not found: {resolved}")
        return pd.read_csv(resolved)

    def _preprocess_data(self) -> None:
        """Clean and index datasets for deterministic Pandas joins and lookups."""
        if "user_id" in self.profiles_df.columns:
            self.profiles_df["user_id"] = self.profiles_df["user_id"].astype(str).str.strip()
            self._profiles_by_user = self.profiles_df.set_index("user_id")
        else:
            self._profiles_by_user = pd.DataFrame()

        if not self.events_df.empty:
            self.events_df["event_id"] = self.events_df["event_id"].astype(str).str.strip()
            self.events_df["user_id"] = self.events_df["user_id"].astype(str).str.strip()
            self.events_df["currency"] = self.events_df["currency"].astype(str).str.strip().str.upper()
            self.events_df["event_date"] = self.events_df["event_date"].astype(str).str.strip()
            if "settlement_date" in self.events_df.columns:
                self.events_df["settlement_date"] = (
                    self.events_df["settlement_date"]
                    .astype(str)
                    .str.strip()
                    .replace({"nan": None, "None": None, "": None})
                )
            if "linked_event_id" in self.events_df.columns:
                self.events_df["linked_event_id"] = (
                    self.events_df["linked_event_id"]
                    .astype(str)
                    .str.strip()
                    .replace({"nan": None, "None": None, "": None})
                )

        # Build fast exchange rate lookup dictionary: (rate_date, from_currency, to_currency) -> rate
        self._fx_lookup: Dict[Tuple[str, str, str], float] = {}
        if not self.exchange_rates_df.empty:
            self.exchange_rates_df["rate_date"] = self.exchange_rates_df["rate_date"].astype(str).str.strip()
            self.exchange_rates_df["from_currency"] = self.exchange_rates_df["from_currency"].astype(str).str.strip().str.upper()
            self.exchange_rates_df["to_currency"] = self.exchange_rates_df["to_currency"].astype(str).str.strip().str.upper()
            self.exchange_rates_df["rate"] = pd.to_numeric(self.exchange_rates_df["rate"], errors="coerce")

            for _, row in self.exchange_rates_df.iterrows():
                r_date = str(row["rate_date"]).strip()
                from_c = str(row["from_currency"]).strip().upper()
                to_c = str(row["to_currency"]).strip().upper()
                rate = float(row["rate"])
                self._fx_lookup[(r_date, from_c, to_c)] = rate
                if (r_date, to_c, from_c) not in self._fx_lookup and rate != 0:
                    self._fx_lookup[(r_date, to_c, from_c)] = 1.0 / rate

    def get_exchange_rate(
        self,
        date_str: str,
        from_currency: str,
        to_currency: str,
        fallback_date: Optional[str] = None,
    ) -> float:
        """
        Look up deterministic exchange rate multiplier matching currency pair and date.
        """
        from_c = from_currency.strip().upper()
        to_c = to_currency.strip().upper()
        if from_c == to_c:
            return 1.0

        d_str = date_str.strip()
        key = (d_str, from_c, to_c)
        if key in self._fx_lookup:
            return self._fx_lookup[key]

        if fallback_date and pd.notna(fallback_date):
            fb_key = (str(fallback_date).strip(), from_c, to_c)
            if fb_key in self._fx_lookup:
                return self._fx_lookup[fb_key]

        year_month = d_str[:7]
        candidates = [
            (k, v) for k, v in self._fx_lookup.items()
            if k[0].startswith(year_month) and k[1] == from_c and k[2] == to_c
        ]
        if candidates:
            candidates.sort(key=lambda item: abs(int(item[0][0].replace("-", "")) - int(d_str.replace("-", ""))))
            return candidates[0][1]

        raise ValueError(
            f"No exchange rate found for {from_c} -> {to_c} on date {date_str} (or fallback {fallback_date})"
        )

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve and format profile data for a specific user.

        Args:
            user_id: Unique user identifier.

        Returns:
            Dictionary containing parsed user profile fields.
        """
        user_id_clean = user_id.strip()
        if user_id_clean not in self._profiles_by_user.index:
            raise KeyError(f"User {user_id_clean} not found in financial profiles.")

        row = self._profiles_by_user.loc[user_id_clean]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        def _split_pipe(val: Any) -> List[str]:
            if pd.isna(val) or not str(val).strip():
                return []
            return [item.strip() for item in str(val).split("|") if item.strip()]

        max_installments = None
        if "max_installment_months" in row and pd.notna(row["max_installment_months"]):
            try:
                max_installments = int(float(row["max_installment_months"]))
            except (ValueError, TypeError):
                max_installments = None

        return {
            "user_id": user_id_clean,
            "home_currency": str(row["home_currency"]).strip().upper(),
            "current_available_balance": float(row["current_available_balance"]),
            "minimum_balance_to_keep": float(row["minimum_balance_to_keep"]),
            "financial_priorities": _split_pipe(row.get("financial_priorities")),
            "expense_categories_to_protect": _split_pipe(row.get("expense_categories_to_protect")),
            "expense_categories_user_is_willing_to_reduce": _split_pipe(row.get("expense_categories_user_is_willing_to_reduce")),
            "expense_categories_user_is_willing_to_stop": _split_pipe(row.get("expense_categories_user_is_willing_to_stop")),
            "payment_methods_user_will_consider": _split_pipe(row.get("payment_methods_user_will_consider")),
            "max_installment_months": max_installments,
        }

    def _join_exchange_rates_pandas(
        self,
        events_df: pd.DataFrame,
        home_currency: str,
    ) -> pd.DataFrame:
        """
        Implement Constraint 1 via exact Pandas exchange rate joining logic.
        Converts all events to user's home_currency and preserves missing amounts as NaN (Constraint 2).
        """
        df = events_df.copy()
        df["home_currency"] = home_currency

        # Step 1: Exact join on (event_date, from_currency, to_currency)
        merged = df.merge(
            self.exchange_rates_df,
            left_on=["event_date", "currency", "home_currency"],
            right_on=["rate_date", "from_currency", "to_currency"],
            how="left",
        )

        # Same currency events multiplier is exactly 1.0
        same_curr = merged["currency"] == merged["home_currency"]
        merged.loc[same_curr, "rate"] = 1.0

        # Fallback to settlement_date if event_date rate is missing
        missing_rate = merged["rate"].isna() & merged["settlement_date"].notna()
        if missing_rate.any():
            fb = merged[missing_rate].merge(
                self.exchange_rates_df,
                left_on=["settlement_date", "currency", "home_currency"],
                right_on=["rate_date", "from_currency", "to_currency"],
                how="left",
                suffixes=("", "_fb"),
            )
            merged.loc[missing_rate, "rate"] = fb["rate_fb"].values

        # Reciprocal rate fallback: (rate_date, to_currency, from_currency) -> 1.0 / rate
        missing_recip = merged["rate"].isna()
        if missing_recip.any():
            recip = merged[missing_recip].merge(
                self.exchange_rates_df,
                left_on=["event_date", "home_currency", "currency"],
                right_on=["rate_date", "from_currency", "to_currency"],
                how="left",
                suffixes=("", "_recip"),
            )
            recip_mask = recip["rate_recip"].notna() & (recip["rate_recip"] != 0)
            if recip_mask.any():
                merged.loc[missing_recip, "rate"] = np.where(
                    recip_mask,
                    1.0 / recip["rate_recip"],
                    merged.loc[missing_recip, "rate"],
                )

        # Retain original amount and original currency
        merged["original_amount"] = merged["amount"]
        merged["original_currency"] = merged["currency"]
        merged["exchange_rate"] = merged["rate"].fillna(1.0)

        # Constraint 2: Missing amounts preserved as NaN (never converted to 0)
        merged["amount"] = np.where(
            merged["original_amount"].isna(),
            np.nan,
            (merged["original_amount"] * merged["exchange_rate"]).round(2),
        )

        if "minimum_allowed_amount" in merged.columns:
            merged["minimum_allowed_amount"] = np.where(
                merged["minimum_allowed_amount"].isna(),
                np.nan,
                (pd.to_numeric(merged["minimum_allowed_amount"], errors="coerce") * merged["exchange_rate"]).round(2),
            )

        merged["currency"] = home_currency

        # Drop temporary exchange rate join columns
        drop_cols = [c for c in ["rate_date", "from_currency", "to_currency", "rate"] if c in merged.columns]
        return merged.drop(columns=drop_cols)

    def _resolve_conflicts_and_deduplicate_pandas(
        self,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Implement Constraint 3 via exact Pandas filtering and deduplication logic:
            a) Explicit cancellations, settlements, or amendments.
            b) A newer record from the same source.
            c) A settled event over an estimate or forecast.
            d) The financially safer interpretation.
        """
        if events_df.empty:
            return events_df.copy()

        df = events_df.copy()

        # Step 1: Identify linked event supersession (parent -> child lifecycles)
        # Self-merge to compare parent and child status/description
        superseded_ids: Set[str] = set()

        linked_mask = df["linked_event_id"].notna()
        if linked_mask.any():
            children = df[linked_mask]
            parents = df[df["event_id"].isin(children["linked_event_id"])]

            linked_pairs = children.merge(
                parents,
                left_on="linked_event_id",
                right_on="event_id",
                suffixes=("_child", "_parent"),
            )

            for _, row in linked_pairs.iterrows():
                p_id = row["event_id_parent"]
                c_id = row["event_id_child"]
                p_status = str(row["status_parent"]).strip().lower()
                c_status = str(row["status_child"]).strip().lower()
                c_desc = str(row["description_child"]).strip().lower()

                # Rule 3a: Explicit cancellations & settlements
                # Cancelled authorization superseded by settled purchase
                if p_status == "cancelled" and c_status == "settled":
                    superseded_ids.add(p_id)
                # Failed bill payment superseded by scheduled retry
                elif p_status == "failed" and c_status == "scheduled":
                    superseded_ids.add(p_id)
                # Rule 3c: Settled original charge over pending duplicate estimate
                elif p_status == "settled" and ("duplicate" in c_desc or "possible duplicate" in c_desc):
                    superseded_ids.add(c_id)

        # Filter out superseded linked transactions
        if superseded_ids:
            df = df[~df["event_id"].isin(superseded_ids)].copy()

        # Step 2: Also exclude standalone cancelled and failed transactions that did not settle
        active_status_mask = ~df["status"].isin(["cancelled", "failed"])
        df = df[active_status_mask].copy()

        # Step 3: Exact duplicate records deduplication using Pandas sort and drop_duplicates
        # Establish priority ranks:
        # a) Settlement priority: settled (rank 1) > scheduled (rank 2) > pending (rank 3) > other
        status_rank_map = {"settled": 1, "scheduled": 2, "pending": 3}
        df["_status_rank"] = df["status"].map(lambda s: status_rank_map.get(str(s).lower(), 99))

        # b) Recency: settlement_date or event_date
        df["_effective_date"] = np.where(df["settlement_date"].notna(), df["settlement_date"], df["event_date"])

        # d) Financially safer interpretation:
        # For debit: higher amount is safer (rank higher = descending)
        # For credit: lower amount is safer (rank higher = ascending)
        df["_safer_rank"] = np.where(
            df["direction"] == "debit",
            -df["amount"].fillna(-1),
            df["amount"].fillna(float("inf")),
        )

        # Sort values deterministically by priority criteria
        df = df.sort_values(
            by=["event_date", "description", "category", "direction", "_status_rank", "_effective_date", "_safer_rank", "event_id"],
            ascending=[True, True, True, True, True, False, True, True],
        )

        # Deduplicate identical transaction signatures in Pandas
        dup_subset = ["event_date", "description", "category", "direction"]
        df = df.drop_duplicates(subset=dup_subset, keep="first").copy()

        # Drop temporary sorting helper columns
        df = df.drop(columns=["_status_rank", "_effective_date", "_safer_rank"])

        # Final chronological sort
        return df.sort_values(by=["event_date", "event_id"]).reset_index(drop=True)

    def _apply_cash_flow_rules_pandas(
        self,
        events_df: pd.DataFrame,
        user_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Implement Constraint 4 via vectorized Pandas cash flow rules:
            - "recurring": tag recurring expenses vs. one-time events.
            - "pending": reserve pending debits (treat as active cash drains),
              completely IGNORE pending credits, bonuses, refunds, or investment gains.
            - "salary": count confirmed salary ONLY on its exact settlement date.
            - "unrealized": do not treat unrealized investment value as available cash.
        """
        if events_df.empty:
            return events_df.copy()

        df = events_df.copy()

        # 1. Recurring tagging
        # Detect repetition count in user history
        desc_counts = user_history["description"].value_counts()
        rep_counts = df["description"].map(lambda d: desc_counts.get(d, 0))

        is_sub = df["event_type"].str.lower() == "subscription"
        is_flex = df["flexibility"].str.lower().isin(self.FLEXIBLE_STATUSES)
        is_rec_cat = df["category"].str.lower().isin(self.RECURRING_CATEGORIES)
        is_salary_rec = (df["category"].str.lower() == "salary") | df["description"].str.contains("salary|payroll", case=False, na=False)
        is_repeated = rep_counts >= 2

        # One-time event indicators
        is_one_time_type = df["event_type"].str.lower().isin({"refund", "investment_purchase", "investment_sale", "investment_valuation"})
        has_one_time_keyword = df["description"].apply(
            lambda d: any(kw in str(d).lower() for kw in self.ONE_TIME_KEYWORDS)
        )

        is_recurring_series = (
            (is_sub | is_flex | is_rec_cat | is_salary_rec | is_repeated)
            & (~is_one_time_type)
            & (~(has_one_time_keyword & (~is_sub)))
        )
        df["is_recurring"] = is_recurring_series
        df["recurrence_type"] = np.where(is_recurring_series, "recurring", "one_time")

        # 2. Pending Rules
        is_pending = df["status"].str.lower() == "pending"
        is_debit = df["direction"].str.lower() == "debit"
        is_credit_or_non_cash = df["direction"].str.lower().isin(["credit", "non_cash"])

        df["is_pending"] = is_pending
        df["is_pending_debit"] = is_pending & is_debit
        df["is_pending_credit_ignored"] = is_pending & is_credit_or_non_cash

        # 3. Salary Rules: Count confirmed salary ONLY on exact settlement date
        is_salary = is_salary_rec
        df["is_confirmed_salary"] = is_salary
        df["effective_date"] = np.where(df["settlement_date"].notna(), df["settlement_date"], df["event_date"])

        # 4. Unrealized Investment Rules: Do not treat as available cash
        is_unrealized = (
            (df["status"].str.lower() == "unrealized")
            | (df["direction"].str.lower() == "non_cash")
            | (df["event_type"].str.lower() == "investment_valuation")
        )
        df["is_unrealized"] = is_unrealized

        # Deterministic cash flow impact calculation
        # Cancelled/failed, ignored pending credits, and unrealized have 0.0 cash flow
        conditions = [
            df["is_pending_credit_ignored"] | df["is_unrealized"] | df["status"].isin(["cancelled", "failed"]),
            is_debit,
            (df["status"] == "settled") & (~is_debit),
            is_salary & df["status"].isin(["scheduled", "confirmed"]),
        ]
        choices = [
            0.0,
            -df["amount"],
            df["amount"],
            df["amount"],
        ]
        df["cash_flow_effective_amount"] = np.select(conditions, choices, default=0.0)
        # Preserve NaN if amount was NaN
        df["cash_flow_effective_amount"] = np.where(
            df["amount"].isna() & (df["cash_flow_effective_amount"] != 0.0),
            np.nan,
            df["cash_flow_effective_amount"],
        )
        df["is_cash_drain"] = is_debit & (~df["is_unrealized"])

        return df

    def get_user_events_df(
        self,
        user_id: str,
        request_date: str,
    ) -> pd.DataFrame:
        """
        Return the fully normalized and deduplicated financial events as a pandas DataFrame.

        Args:
            user_id: Target user ID.
            request_date: Evaluation date string (YYYY-MM-DD).

        Returns:
            pd.DataFrame of normalized events in chronological order.
        """
        profile = self.get_user_profile(user_id)
        home_currency = profile["home_currency"]

        user_events = self.events_df[self.events_df["user_id"] == user_id].copy()
        if user_events.empty:
            return pd.DataFrame()

        # Step 1: Currency Conversion via Pandas join (Constraint 1 & 2)
        norm_df = self._join_exchange_rates_pandas(user_events, home_currency)

        # Step 2: Conflict resolution & deduplication in Pandas (Constraint 3)
        dedup_df = self._resolve_conflicts_and_deduplicate_pandas(norm_df)

        # Step 3: Cash flow rules application in Pandas (Constraint 4)
        tagged_df = self._apply_cash_flow_rules_pandas(dedup_df, user_events)

        return tagged_df

    def get_user_state(
        self,
        user_id: str,
        request_date: str,
    ) -> Dict[str, Any]:
        """
        Build and return a clean dictionary containing the user's profile data
        and their normalized, deduplicated list of financial events converted to home_currency.

        Args:
            user_id: Unique user identifier (e.g. 'user_01').
            request_date: Request evaluation date (YYYY-MM-DD).

        Returns:
            Clean dictionary with profile data, normalized events ledger,
            and cash flow summary metrics.
        """
        profile = self.get_user_profile(user_id)
        events_df = self.get_user_events_df(user_id, request_date)

        # Serialize DataFrame rows to clean Python dicts preserving missing amounts as None/NaN
        if events_df.empty:
            events_list: List[Dict[str, Any]] = []
        else:
            records = events_df.to_dict(orient="records")
            events_list = []
            for r in records:
                clean_r: Dict[str, Any] = {}
                for k, v in r.items():
                    if pd.isna(v):
                        clean_r[k] = None
                    else:
                        clean_r[k] = v
                # Specifically guarantee amount preservation (Constraint 2)
                if pd.isna(r["amount"]):
                    clean_r["amount"] = None
                events_list.append(clean_r)

        # Compute summary cash flow metrics
        pending_debits = [
            e for e in events_list
            if e.get("is_pending_debit") and e.get("amount") is not None
        ]
        total_pending_debits = sum(e["amount"] for e in pending_debits)

        # Confirmed salary on exact settlement date
        salary_events = [
            e for e in events_list
            if e.get("is_confirmed_salary") and e.get("amount") is not None
        ]
        next_salary = None
        for s in salary_events:
            eff_d = s.get("effective_date")
            if eff_d and eff_d >= request_date:
                next_salary = {
                    "event_id": s["event_id"],
                    "settlement_date": eff_d,
                    "amount": s["amount"],
                    "currency": s["currency"],
                    "description": s["description"],
                }
                break

        # Recurring monthly expenses estimate
        recurring_expenses = [
            e for e in events_list
            if e.get("is_recurring")
            and e.get("direction") == "debit"
            and e.get("amount") is not None
            and not e.get("is_unrealized")
        ]
        recurring_by_desc: Dict[str, float] = {}
        for re in recurring_expenses:
            recurring_by_desc[re["description"]] = re["amount"]
        recurring_monthly_total = round(sum(recurring_by_desc.values()), 2)

        # Total unrealized investments
        unrealized_events = [
            e for e in events_list
            if e.get("is_unrealized") and e.get("amount") is not None
        ]
        total_unrealized = round(sum(e["amount"] for e in unrealized_events), 2)

        available_bal = profile["current_available_balance"]
        effective_cash = round(available_bal - total_pending_debits, 2)

        summary = {
            "current_available_balance": available_bal,
            "minimum_balance_to_keep": profile["minimum_balance_to_keep"],
            "total_reserved_pending_debits": round(total_pending_debits, 2),
            "effective_available_cash": effective_cash,
            "upcoming_confirmed_salary": next_salary,
            "recurring_monthly_expenses_estimate": recurring_monthly_total,
            "total_unrealized_investments": total_unrealized,
            "total_active_events_count": len(events_list),
        }

        return {
            "user_id": user_id,
            "request_date": request_date,
            "home_currency": profile["home_currency"],
            "current_available_balance": profile["current_available_balance"],
            "minimum_balance_to_keep": profile["minimum_balance_to_keep"],
            "financial_priorities": profile["financial_priorities"],
            "expense_categories_to_protect": profile["expense_categories_to_protect"],
            "expense_categories_user_is_willing_to_reduce": profile["expense_categories_user_is_willing_to_reduce"],
            "expense_categories_user_is_willing_to_stop": profile["expense_categories_user_is_willing_to_stop"],
            "payment_methods_user_will_consider": profile["payment_methods_user_will_consider"],
            "max_installment_months": profile["max_installment_months"],
            "profile": profile,
            "events": events_list,
            "summary": summary,
        }
