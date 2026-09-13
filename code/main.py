"""
Main pipeline execution and batch processor for the "Buy or Wait?" Financial Agent.

Iterates over all evaluation requests in dataset/requests.csv, running:
1. FinancialStateBuilder (State Reconstruction & FX normalization)
2. ImageAmountExtractor (Multi-Modal missing amount extraction)
3. CashFlowSimulator (90-Day daily cash flow forecast & safety limits)
4. DecisionRouter (Hierarchical ranking, routing & payment plans)
5. Explanation Generator (Fact-based, deterministic decision rationale)

Outputs the completed 8-column predictions to output.csv in the project root.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Add project code directory and project root to sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))

if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
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

__all__ = [
    "FinancialStateBuilder",
    "ImageAmountExtractor",
    "CashFlowSimulator",
    "DecisionRouter",
    "generate_explanation",
    "process_batch",
    "main",
]


def generate_explanation(
    status: str,
    method: str,
    currency: str = "",
    requested_amount: float = 0.0,
    amount_safe_to_pay: float = 0.0,
    earliest_date_for_full_payment: str = "",
    minimum_balance: float = 0.0,
    desired_completion_date: str = "",
    plan_details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a deterministic, fact-based financial explanation for the recommendation.

    Args:
        status: Affordability status (affordable_now, affordable_with_plan, affordable_later, not_affordable).
        method: Recommended payment method (full_payment, partial_payment, installments, wait, not_recommended).
        currency: User's home currency.
        requested_amount: Total requested expenditure.
        amount_safe_to_pay: Maximum safe amount payable today.
        earliest_date_for_full_payment: Earliest safe date for single full payment.
        minimum_balance: Preferred minimum balance to keep.
        desired_completion_date: User's requested completion deadline.
        plan_details: Details of the chosen payment plan candidate.

    Returns:
        A concise, fact-based explanation string.
    """
    def _fmt(val: float) -> str:
        if abs(val - round(val)) < 1e-5:
            return f"{int(round(val)):,}"
        return f"{val:,.2f}"

    c = f"{currency} " if currency else ""
    amt_str = _fmt(requested_amount)
    min_str = _fmt(minimum_balance)
    safe_str = _fmt(amount_safe_to_pay)

    if status == "affordable_now" or method == "full_payment":
        return f"Pay {c}{amt_str} today. This leaves at least {c}{min_str} available over the next 90 days."

    elif method == "installments":
        details = plan_details or {}
        num_p = details.get("number_of_payments", 3)
        start_d = details.get("start_payment_date", "")
        plan_str = str(details.get("plan", ""))
        inst_amt_raw = plan_str.split("|")[0].split(":")[1] if ":" in plan_str else amt_str
        try:
            inst_amt_fmt = _fmt(float(inst_amt_raw))
        except (ValueError, TypeError):
            inst_amt_fmt = inst_amt_raw
        return f"Use {num_p} installments of {c}{inst_amt_fmt}, starting {start_d}. This leaves at least {c}{min_str} available."

    elif method == "partial_payment":
        rem = requested_amount - amount_safe_to_pay
        rem_str = _fmt(rem)
        return (
            f"Pay {c}{safe_str} today and the remaining {c}{rem_str} on {earliest_date_for_full_payment}. "
            f"This completes the full request and keeps the {c}{min_str} minimum protected."
        )

    elif method == "wait" or status == "affordable_later":
        if earliest_date_for_full_payment:
            return (
                f"Pay {c}{amt_str} in full on {earliest_date_for_full_payment}. "
                f"Paying earlier would take the balance below the {c}{min_str} minimum."
            )
        return f"Wait until cash reserves replenish before completing payment. Paying earlier would dip below the {c}{min_str} minimum."

    else:  # not_affordable / not_recommended
        if desired_completion_date:
            return f"Do not make this payment by {desired_completion_date}. None of the available options keeps the {c}{min_str} minimum protected."
        return f"Do not proceed with the {c}{amt_str} request. None of the available options keeps the {c}{min_str} minimum protected."


def process_batch(
    requests_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    verbose_step: int = 50,
) -> pd.DataFrame:
    """
    Process all evaluation requests and output results to CSV.

    Args:
        requests_path: Path to dataset/requests.csv.
        output_path: Path to write root output.csv.
        verbose_step: Print progress every N requests.

    Returns:
        DataFrame containing compiled 8-column predictions.
    """
    base_dir = Path(_PROJECT_ROOT)
    req_file = Path(requests_path) if requests_path else base_dir / "dataset" / "requests.csv"
    out_file = Path(output_path) if output_path else base_dir / "output.csv"
    dataset_out_file = base_dir / "dataset" / "output.csv"

    if not req_file.exists():
        raise FileNotFoundError(f"Requests file not found at: {req_file}")

    print(f"Loading evaluation requests from: {req_file.resolve()}")
    requests_df = pd.read_csv(req_file)
    total_requests = len(requests_df)
    print(f"Found {total_requests} evaluation requests to process.")

    # Initialize pipeline modules
    print("Initializing pipeline modules...")
    builder = FinancialStateBuilder()
    extractor = ImageAmountExtractor()
    simulator = CashFlowSimulator()
    router = DecisionRouter(simulator=simulator)

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    print("\nStarting batch processing loop...")
    for i, (_, row) in enumerate(requests_df.iterrows()):
        req_id = str(row["request_id"]).strip()
        user_id = str(row["user_id"]).strip()
        req_date = str(row["request_date"]).strip()
        req_amt = float(row["requested_amount"])
        desired_date = str(row.get("desired_completion_date") or req_date).strip()
        allows_partial = str(row.get("allows_partial_payment", "")).strip().lower() == "true"

        # 1. State reconstruction
        user_state = builder.get_user_state(user_id=user_id, request_date=req_date)

        # 2. Multi-modal image amount extraction
        user_state["events"] = extractor.extract_amounts_for_events(user_state["events"])

        # 3. 90-Day Cash Flow Simulation
        sim_res = simulator.evaluate_request(user_state=user_state, requested_amount=req_amt)
        amount_safe = sim_res["amount_safe_to_pay"]
        earliest_date = sim_res["earliest_date_for_full_payment"]

        # 4. Decision Router
        req_info = {
            "request_id": req_id,
            "request_date": req_date,
            "desired_completion_date": desired_date,
            "allows_partial_payment": allows_partial,
        }
        rec = router.get_recommendation(
            user_state=user_state,
            requested_amount=req_amt,
            amount_safe_to_pay=amount_safe,
            earliest_date_for_full_payment=earliest_date,
            request_info=req_info,
        )

        status = rec["affordability_status"]
        method = rec["recommended_payment_method"]
        plan = rec["payment_plan"]
        rec_earliest = rec["earliest_date_for_full_payment"]
        spending_changes = rec["spending_changes_needed"]
        plan_details = rec.get("selected_plan_details", {})

        # 5. Generate Explanation
        explanation = generate_explanation(
            status=status,
            method=method,
            currency=user_state.get("home_currency", ""),
            requested_amount=req_amt,
            amount_safe_to_pay=amount_safe,
            earliest_date_for_full_payment=rec_earliest,
            minimum_balance=float(user_state["minimum_balance_to_keep"]),
            desired_completion_date=desired_date,
            plan_details=plan_details,
        )

        # Compile row with exact 8 columns
        results.append({
            "request_id": req_id,
            "amount_safe_to_pay": amount_safe,
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": plan,
            "earliest_date_for_full_payment": rec_earliest,
            "spending_changes_needed": spending_changes,
            "decision_explanation": explanation,
        })

        if (i + 1) % verbose_step == 0 or (i + 1) == total_requests:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"Processed {i + 1}/{total_requests} requests ({(i + 1) / total_requests * 100:.1f}%) in {elapsed:.1f}s ({rate:.1f} req/s)...")

    # Construct final DataFrame
    out_df = pd.DataFrame(results)

    # Required column order
    required_columns = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]
    out_df = out_df[required_columns]

    # Save to root output.csv
    out_df.to_csv(out_file, index=False)
    print(f"\nSuccessfully wrote {len(out_df)} predictions to: {out_file.resolve()}")

    # Also mirror to dataset/output.csv if dataset dir exists
    if dataset_out_file.parent.exists():
        out_df.to_csv(dataset_out_file, index=False)
        print(f"Mirrored predictions to: {dataset_out_file.resolve()}")

    return out_df


def main() -> None:
    """Entry point to run the full batch processing pipeline."""
    print("=" * 70)
    print("Initializing Buy or Wait Financial Engine...")
    print("=" * 70)

    # 1. Single demonstration check for user_01
    builder = FinancialStateBuilder()
    extractor = ImageAmountExtractor()
    simulator = CashFlowSimulator()
    router = DecisionRouter(simulator=simulator)

    user_id = "user_01"
    request_id = "request_01"
    request_date = "2024-03-03"
    requested_amount = 25256.0

    state = builder.get_user_state(user_id=user_id, request_date=request_date)
    state["events"] = extractor.extract_amounts_for_events(state["events"])
    sim_res = simulator.evaluate_request(state, requested_amount=requested_amount)
    rec = router.get_recommendation(
        user_state=state,
        requested_amount=requested_amount,
        amount_safe_to_pay=sim_res["amount_safe_to_pay"],
        earliest_date_for_full_payment=sim_res["earliest_date_for_full_payment"],
        request_info={
            "request_id": request_id,
            "request_date": request_date,
            "desired_completion_date": "2024-03-20",
            "allows_partial_payment": True,
        },
    )

    print(f"\n[Validation] {user_id} ({request_id}):")
    print(f"  amount_safe_to_pay: {rec['amount_safe_to_pay']} {state['home_currency']}")
    print(f"  affordability_status: {rec['affordability_status']}")
    print(f"  recommended_payment_method: {rec['recommended_payment_method']}")
    print(f"  payment_plan: {rec['payment_plan']}")
    print(f"  earliest_date_for_full_payment: {rec['earliest_date_for_full_payment']}")
    print(f"  explanation: {generate_explanation(rec['affordability_status'], rec['recommended_payment_method'], state['home_currency'], requested_amount, rec['amount_safe_to_pay'], rec['earliest_date_for_full_payment'], state['minimum_balance_to_keep'], '2024-03-20', rec['selected_plan_details'])}")
    print("-" * 70)

    # 2. Run full batch processing over dataset/requests.csv
    out_df = process_batch(verbose_step=50)

    # Verify line count
    root_output = Path(_PROJECT_ROOT) / "output.csv"
    with open(root_output, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    print("-" * 70)
    print(f"VERIFICATION: {root_output.name} contains {line_count} lines (1 header + {line_count - 1} data rows).")
    print("=" * 70)


if __name__ == "__main__":
    main()
