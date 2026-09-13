"""
Unit and Integration Tests for ImageAmountExtractor (Multi-Modal Vision Pipeline).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

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


class TestImageAmountExtractor(unittest.TestCase):
    """Test suite verifying image-to-event join logic and extraction behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = FinancialStateBuilder()
        cls.extractor = ImageAmountExtractor(default_mock_amount=150.0)

    def test_initialization_and_image_mapping(self) -> None:
        """Verify that images.csv is properly ingested and mapped."""
        self.assertFalse(self.extractor.images_df.empty)
        self.assertEqual(len(self.extractor._event_to_image), 16)
        self.assertIn("event_253", self.extractor._event_to_image)
        self.assertEqual(self.extractor._event_to_image["event_253"]["image_id"], "image_01")

    def test_all_16_image_files_exist_on_disk(self) -> None:
        """Verify all referenced PNG images exist in dataset/media/images/."""
        for event_id, info in self.extractor._event_to_image.items():
            img_path = info["image_path"]
            self.assertTrue(
                os.path.exists(img_path),
                f"Image file for {event_id} does not exist at {img_path}",
            )

    def test_pipeline_join_and_amount_replacement(self) -> None:
        """
        Test end-to-end integration:
        FinancialStateBuilder -> ImageAmountExtractor
        Checks that a missing amount event is correctly matched and updated.
        """
        # user_03 on 2019-09-03 has event_253 with blank amount
        state = self.builder.get_user_state(user_id="user_03", request_date="2019-09-03")
        events_before = state["events"]

        ev_before = next((e for e in events_before if e["event_id"] == "event_253"), None)
        self.assertIsNotNone(ev_before)
        self.assertIsNone(ev_before["amount"])

        # Pipe into ImageAmountExtractor
        events_after = self.extractor.extract_amounts_for_events(events_before)
        ev_after = next((e for e in events_after if e["event_id"] == "event_253"), None)

        self.assertIsNotNone(ev_after)
        self.assertEqual(ev_after["amount"], 150.0)
        self.assertEqual(ev_after["extracted_from_image"], "image_01")
        self.assertTrue(os.path.exists(ev_after["image_path"]))

    def test_custom_mock_amount_mapping(self) -> None:
        """Verify extractor respects specific custom amounts per image."""
        custom_map = {
            "image_01": 2500000.0,  # for event_253 (IDR)
            "image_02": 45000.0,     # for event_1442 (INR)
        }
        custom_extractor = ImageAmountExtractor(mock_amount_map=custom_map)

        state = self.builder.get_user_state(user_id="user_03", request_date="2019-09-03")
        events = custom_extractor.extract_amounts_for_events(state["events"])

        ev = next(e for e in events if e["event_id"] == "event_253")
        self.assertEqual(ev["amount"], 2500000.0)
        self.assertEqual(ev["extracted_from_image"], "image_01")

    def test_foreign_currency_image_conversion(self) -> None:
        """
        Verify multi-currency handling:
        event_7307 (user_78): currency USD, home_currency INR, rate 83.33.
        Extracted amount in USD should be converted to home_currency INR.
        """
        custom_map = {"image_12": 100.0}  # $100 USD
        custom_extractor = ImageAmountExtractor(mock_amount_map=custom_map)

        state = self.builder.get_user_state(user_id="user_78", request_date="2025-10-02")
        events = custom_extractor.extract_amounts_for_events(state["events"])

        ev = next(e for e in events if e["event_id"] == "event_7307")
        self.assertEqual(ev["original_amount"], 100.0)
        self.assertEqual(ev["original_currency"], "USD")
        self.assertEqual(ev["currency"], "INR")
        # 100.0 * 83.33 = 8333.0 INR
        self.assertAlmostEqual(ev["amount"], 8333.0, places=1)


if __name__ == "__main__":
    unittest.main()
