import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from pipeline import ExtremeHandwritingPipeline, FLAG


class FakeTesseract:
    Output = SimpleNamespace(DICT="dict")

    @staticmethod
    def image_to_data(image, config, output_type):
        return {
            "text": ["Patient", "BP"],
            "conf": ["92", "88"],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 2],
            "left": [10, 80],
            "top": [10, 70],
            "width": [40, 10],
            "height": [10, 10],
        }


class PipelineTests(unittest.TestCase):
    def test_unconfigured_backend_does_not_return_fixture_transcriptions(self):
        pipeline = ExtremeHandwritingPipeline()
        pipeline.vlm_api_key = None
        pipeline.tesseract_available = False

        result = pipeline.process(Image.new("RGB", (1200, 720), "white"))

        self.assertIn(FLAG, result["full_text"])
        self.assertNotIn("Paracetamol", result["full_text"])
        self.assertEqual(result["engine"], "OCR not configured")
        self.assertEqual(result["flagged_blocks"], 1)

    def test_tesseract_extracts_page_lines_and_separates_margin_notes(self):
        pipeline = ExtremeHandwritingPipeline()
        pipeline.vlm_api_key = None
        pipeline.tesseract = FakeTesseract()
        pipeline.tesseract_available = True

        result = pipeline.process(Image.new("RGB", (100, 100), "white"))

        self.assertEqual(result["total_blocks"], 2)
        self.assertEqual(result["blocks"][0]["raw_text"], "Patient")
        self.assertEqual(result["blocks"][0]["region_type"], "main_body")
        self.assertEqual(result["blocks"][1]["raw_text"], "BP")
        self.assertEqual(result["blocks"][1]["region_type"], "margin_note")
        self.assertIn("two-pass text agreement", result["blocks"][0]["reasons"])

    def test_vision_uncertainty_is_flagged_and_crossed_out_text_is_omitted(self):
        pipeline = ExtremeHandwritingPipeline()
        pipeline.vlm_api_key = "configured"
        pipeline.tesseract_available = False
        regions = [
            {
                "text": "Take one tablet",
                "confidence": 0.91,
                "bbox": [0, 0, 100, 20],
                "region_type": "main_body",
                "uncertain": False,
                "crossed_out": False,
                "reasons": ["clear writing"],
            },
            {
                "text": "Wrong dose 1000mg",
                "confidence": 0.2,
                "bbox": [0, 20, 100, 20],
                "region_type": "main_body",
                "uncertain": True,
                "crossed_out": True,
                "reasons": ["crossed out"],
            },
        ]
        with patch.object(pipeline, "_recognize_vision_model", return_value=regions):
            result = pipeline.process(Image.new("RGB", (100, 100), "white"))

        self.assertEqual(result["crossed_out_blocks"], 1)
        self.assertNotIn("Wrong dose", result["full_text"])
        self.assertIn("Take one tablet", result["full_text"])
        self.assertTrue(result["blocks"][1]["crossed_out"])


if __name__ == "__main__":
    unittest.main()
