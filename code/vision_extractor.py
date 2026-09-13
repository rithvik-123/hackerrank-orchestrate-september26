"""
Multi-Modal Vision Extractor for the "Buy or Wait?" System.

Extracts transaction amounts from financial documents (receipts, bills, invoices, statements)
associated with financial events having missing amounts.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd


class ImageAmountExtractor:
    """
    Multi-modal image amount extractor.

    Maps financial events with missing amounts to corresponding document images
    in dataset/images.csv, retrieves the PNG media file, and extracts the numeric
    total via vision model integration.
    """

    # Production prompt to be sent to a multi-modal LLM (e.g. Gemini 1.5 Pro or GPT-4o)
    VISION_EXTRACTION_PROMPT: str = (
        "You are an expert financial document parser. "
        "Analyze the provided image (which may be an invoice, bill, receipt, payroll slip, or account statement). "
        "Identify and extract the final total transaction amount or net payable/receivable figure. "
        "Respond ONLY with the single numeric float amount (e.g. 1250.00 or 45.50). "
        "Do not include any currency symbols, commas, words, units, or markdown formatting."
    )

    def __init__(
        self,
        images_path: Union[str, Path] = "dataset/images.csv",
        media_dir: Union[str, Path] = "dataset/media/images",
        images_df: Optional[pd.DataFrame] = None,
        default_mock_amount: float = 99.99,
        mock_amount_map: Optional[Dict[str, float]] = None,
        vision_model_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        """
        Initialize the ImageAmountExtractor.

        Args:
            images_path: Path to dataset/images.csv mapping related_event_id to image_id.
            media_dir: Directory containing <image_id>.png files.
            images_df: Optional pre-loaded DataFrame for images.
            default_mock_amount: Fallback float returned by mock vision extractor.
            mock_amount_map: Optional dict mapping image_id or event_id to specific float values.
            vision_model_fn: Optional callable (image_path, prompt) -> float for live vision inference.
        """
        self.images_path = self._resolve_path(images_path)
        self.media_dir = self._resolve_path(media_dir)
        self.default_mock_amount = default_mock_amount
        self.mock_amount_map = mock_amount_map or {}
        self.vision_model_fn = vision_model_fn

        if images_df is not None:
            self.images_df = images_df.copy()
        elif self.images_path.exists():
            self.images_df = pd.read_csv(self.images_path)
        else:
            self.images_df = pd.DataFrame(columns=["image_id", "user_id", "request_id", "related_event_id"])

        self._preprocess_images()

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a file path relative to cwd or project root."""
        p = Path(path)
        if p.exists():
            return p

        module_dir = Path(__file__).resolve().parent
        repo_root = module_dir.parent if module_dir.name == "code" else module_dir
        repo_candidate = repo_root / path
        if repo_candidate.exists():
            return repo_candidate

        return p

    def _preprocess_images(self) -> None:
        """Index images by related_event_id for fast lookup."""
        self._event_to_image: Dict[str, Dict[str, Any]] = {}
        if not self.images_df.empty and "related_event_id" in self.images_df.columns:
            for _, row in self.images_df.iterrows():
                event_id = str(row["related_event_id"]).strip()
                image_id = str(row["image_id"]).strip()
                self._event_to_image[event_id] = {
                    "image_id": image_id,
                    "user_id": str(row.get("user_id", "")).strip(),
                    "request_id": str(row.get("request_id", "")).strip(),
                    "image_path": str(self.media_dir / f"{image_id}.png"),
                }

    def _extract_amount_from_image(self, image_path: str) -> float:
        """
        Extract numeric total amount from an image.

        If a live vision model callable is provided, invokes it with the
        VISION_EXTRACTION_PROMPT. Otherwise, uses deterministic mock resolution.

        Args:
            image_path: Absolute or relative path to the image PNG.

        Returns:
            Extracted float amount.
        """
        # Validate that image exists
        resolved_img = Path(image_path)
        if not resolved_img.exists():
            resolved_img = self._resolve_path(image_path)

        # In production with an active API key:
        if self.vision_model_fn is not None:
            return float(self.vision_model_fn(str(resolved_img), self.VISION_EXTRACTION_PROMPT))

        # Mock / Placeholder mode
        # 1. Check if mock_amount_map has an entry by filename stem (image_id)
        image_id = resolved_img.stem
        if image_id in self.mock_amount_map:
            return float(self.mock_amount_map[image_id])

        # 2. Return default mock amount
        return float(self.default_mock_amount)

    def extract_amounts_for_events(
        self,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process normalized events list. For any event with a missing amount (None or NaN),
        look up the matching image in images.csv, extract the amount, and update the event.

        Args:
            events: List of event dictionaries from FinancialStateBuilder.

        Returns:
            Updated list of event dictionaries where missing amounts are populated.
        """
        updated_events: List[Dict[str, Any]] = []

        for e in events:
            item = dict(e)
            amt = item.get("amount")

            # Check if amount is missing (None, NaN, or blank)
            is_missing = amt is None or (isinstance(amt, float) and math.isnan(amt))

            if is_missing:
                event_id = item.get("event_id", "")
                img_info = self._event_to_image.get(event_id)

                if img_info:
                    image_path = img_info["image_path"]
                    extracted_raw_amount = self._extract_amount_from_image(image_path)

                    # Multi-currency adjustment: if event has exchange rate to home currency
                    rate = float(item.get("exchange_rate", 1.0))
                    amount_in_home_currency = round(extracted_raw_amount * rate, 2)

                    item["amount"] = amount_in_home_currency
                    item["original_amount"] = extracted_raw_amount
                    item["extracted_from_image"] = img_info["image_id"]
                    item["image_path"] = image_path

                    # Update cash flow impact
                    direction = item.get("direction", "debit")
                    if item.get("is_cash_drain", False) or direction == "debit":
                        item["cash_flow_effective_amount"] = -amount_in_home_currency
                    elif direction == "credit" and not item.get("is_pending_credit_ignored", False):
                        item["cash_flow_effective_amount"] = amount_in_home_currency

            updated_events.append(item)

        return updated_events

    def process(self, state_or_events: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convenience method accepting either a full user state dict or an events list.

        Args:
            state_or_events: User state dict from FinancialStateBuilder or events list.

        Returns:
            Updated user state dict or events list with extracted amounts populated.
        """
        if isinstance(state_or_events, dict) and "events" in state_or_events:
            state_copy = dict(state_or_events)
            state_copy["events"] = self.extract_amounts_for_events(state_copy["events"])
            return state_copy
        elif isinstance(state_or_events, list):
            return self.extract_amounts_for_events(state_or_events)
        else:
            raise TypeError("Expected user state dict or list of event dictionaries.")
