"""OCR pipeline with optional local Tesseract and an opt-in vision API."""
from __future__ import annotations

import base64
import difflib
import io
import json
import os
import shutil
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

FLAG = "[REVIEW_REQUIRED]"
MAX_IMAGE_PIXELS = 40_000_000


class RecognitionError(RuntimeError):
    """A configured recognition service failed or returned an invalid response."""


class ExtremeHandwritingPipeline:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.vlm_api_key = os.environ.get("INKLENS_VLM_API_KEY")
        self.vlm_base_url = os.environ.get(
            "INKLENS_VLM_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.vlm_model = os.environ.get("INKLENS_VLM_MODEL", "gpt-4o-mini")

        self.tesseract = None
        self.tesseract_available = False
        self.tesseract_error = None
        try:
            import pytesseract

            command = os.environ.get("TESSERACT_CMD")
            if command:
                pytesseract.pytesseract.tesseract_cmd = command
            if shutil.which(pytesseract.pytesseract.tesseract_cmd):
                self.tesseract = pytesseract
                self.tesseract_available = True
        except ImportError:
            self.tesseract_error = "Install the optional pytesseract package."

    @property
    def engine_status(self) -> dict[str, Any]:
        if self.vlm_api_key:
            return {
                "available": True,
                "engine": f"Vision model · {self.vlm_model}",
                "remote": True,
                "message": "Uploads are sent to the configured vision API.",
            }
        if self.tesseract_available:
            return {
                "available": True,
                "engine": "Local Tesseract · two-pass OCR",
                "remote": False,
                "message": "Images are processed locally with Tesseract.",
            }
        return {
            "available": False,
            "engine": "OCR not configured",
            "remote": False,
            "message": (
                "Install Tesseract for local OCR or configure an opt-in vision API. "
                "No image text will be guessed."
            ),
        }

    def _quality(self, image: Image.Image) -> dict[str, float]:
        gray = ImageOps.grayscale(image)
        gray.thumbnail((900, 900))
        pixels = np.asarray(gray)
        contrast = float(np.std(pixels))
        dark = float(np.mean(pixels < 170))
        blur = (
            float(np.var(np.diff(pixels.astype(np.float32), axis=0)))
            if pixels.shape[0] > 1
            else 0.0
        )
        return {
            "contrast": round(min(1.0, contrast / 85), 3),
            "ink_density": round(min(1.0, dark * 2.2), 3),
            "edge_energy": round(min(1.0, blur / 6000), 3),
        }

    def _tesseract_lines(self, image: Image.Image, page_mode: int) -> list[dict[str, Any]]:
        data = self.tesseract.image_to_data(
            image,
            config=f"--psm {page_mode}",
            output_type=self.tesseract.Output.DICT,
        )
        lines: dict[tuple[str, str, str], dict[str, Any]] = {}
        for i, raw_word in enumerate(data["text"]):
            word = str(raw_word).strip()
            if not word:
                continue
            key = (
                str(data["block_num"][i]),
                str(data["par_num"][i]),
                str(data["line_num"][i]),
            )
            line = lines.setdefault(
                key,
                {
                    "words": [],
                    "confidences": [],
                    "left": int(data["left"][i]),
                    "top": int(data["top"][i]),
                    "right": 0,
                    "bottom": 0,
                },
            )
            left, top = int(data["left"][i]), int(data["top"][i])
            right, bottom = left + int(data["width"][i]), top + int(data["height"][i])
            line["words"].append(word)
            line["left"] = min(line["left"], left)
            line["top"] = min(line["top"], top)
            line["right"] = max(line["right"], right)
            line["bottom"] = max(line["bottom"], bottom)
            try:
                confidence = float(data["conf"][i])
            except (TypeError, ValueError):
                confidence = -1
            if confidence >= 0:
                line["confidences"].append(confidence)

        result = []
        for line in lines.values():
            average = (
                sum(line["confidences"]) / len(line["confidences"]) / 100
                if line["confidences"]
                else 0.0
            )
            result.append(
                {
                    "text": " ".join(line["words"]),
                    "confidence": average,
                    "bbox": [
                        line["left"],
                        line["top"],
                        line["right"] - line["left"],
                        line["bottom"] - line["top"],
                    ],
                }
            )
        return sorted(result, key=lambda line: (line["bbox"][1], line["bbox"][0]))

    def _recognize_tesseract(self, image: Image.Image) -> list[dict[str, Any]]:
        gray = ImageOps.grayscale(image)
        enhanced = ImageOps.autocontrast(gray)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)

        primary = self._tesseract_lines(enhanced, page_mode=11)
        secondary = self._tesseract_lines(gray, page_mode=6)
        primary_text = " ".join(line["text"] for line in primary).casefold()
        secondary_text = " ".join(line["text"] for line in secondary).casefold()
        consensus = difflib.SequenceMatcher(
            None, " ".join(primary_text.split()), " ".join(secondary_text.split())
        ).ratio()
        agrees = consensus >= 0.85

        regions = []
        for line in primary:
            x, y, width, height = line["bbox"]
            reasons = ["Tesseract word confidence"]
            if agrees:
                reasons.append("two-pass text agreement")
            else:
                reasons.append("OCR passes disagree")
            regions.append(
                {
                    "text": line["text"],
                    "confidence": line["confidence"] * (0.6 + 0.4 * consensus),
                    "bbox": [x, y, width, height],
                    "region_type": (
                        "margin_note" if x >= image.width * 0.72 else "main_body"
                    ),
                    "uncertain": not agrees,
                    "crossed_out": False,
                    "reasons": reasons,
                }
            )
        return regions

    def _recognize_vision_model(self, image: Image.Image) -> list[dict[str, Any]]:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
        prompt = (
            "Transcribe this entire handwritten page exactly as visible. It may contain "
            "very messy or mixed-script handwriting, crossed-out text, margin notes, "
            "tables, or forms. Do not correct spelling, expand abbreviations, infer "
            "medical terms, or use outside context. Preserve reading order and keep "
            "margin notes and distinct layout regions separate. If any text is not "
            "readable, use [REVIEW_REQUIRED] for that text; never guess. Return only "
            "a JSON object of the form "
            '{"regions":[{"text":"verbatim text or [REVIEW_REQUIRED]",'
            '"confidence":0.0,"uncertain":true,"crossed_out":false,'
            '"region_type":"main_body","bbox":[0,0,1000,100],'
            '"reasons":["brief visual evidence"]}]}. '
            "bbox is [x,y,width,height] normalized to 0..1000. confidence is a "
            "conservative visual estimate, not a calibrated probability. Set "
            "crossed_out only when the writing is visibly crossed out; do not include "
            "crossed-out words in the transcription."
        )
        payload = {
            "model": self.vlm_model,
            "temperature": 0,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
        request = Request(
            f"{self.vlm_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.vlm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RecognitionError(
                f"Vision API returned HTTP {error.code}; check the model, key, and quota."
            ) from error
        except (URLError, TimeoutError) as error:
            raise RecognitionError(
                "Could not reach the configured vision API; check the endpoint and network."
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RecognitionError("Vision API returned an invalid JSON response.") from error

        try:
            content = response_data["choices"][0]["message"]["content"]
            decoded = json.loads(content)
            raw_regions = decoded["regions"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RecognitionError(
                "Vision API response did not contain the expected JSON regions."
            ) from error
        if not isinstance(raw_regions, list):
            raise RecognitionError("Vision API response 'regions' must be a list.")

        regions = []
        for item in raw_regions:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise RecognitionError("Vision API returned an invalid region.")
            try:
                confidence = float(item.get("confidence", 0.0))
                bbox = item.get("bbox", [0, 0, 1000, 100])
                if len(bbox) != 4:
                    raise ValueError
                x, y, width, height = (float(value) for value in bbox)
            except (TypeError, ValueError) as error:
                raise RecognitionError(
                    "Vision API returned invalid confidence or bounding-box values."
                ) from error
            region_type = str(item.get("region_type", "unknown"))
            if region_type not in {"main_body", "margin_note", "table", "unknown"}:
                region_type = "unknown"
            reasons = item.get("reasons", [])
            if not isinstance(reasons, list) or not all(
                isinstance(reason, str) for reason in reasons
            ):
                reasons = ["Vision model supplied no readable evidence notes"]
            regions.append(
                {
                    "text": item["text"],
                    "confidence": max(0.0, min(1.0, confidence)),
                    "bbox": [
                        round(x * image.width / 1000),
                        round(y * image.height / 1000),
                        round(width * image.width / 1000),
                        round(height * image.height / 1000),
                    ],
                    "region_type": region_type,
                    "uncertain": bool(item.get("uncertain", False)),
                    "crossed_out": bool(item.get("crossed_out", False)),
                    "reasons": reasons,
                }
            )
        return regions

    def process(self, source_image: Image.Image) -> dict[str, Any]:
        image = ImageOps.exif_transpose(source_image).convert("RGB")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel processing limit."
            )
        quality = self._quality(image)

        if self.vlm_api_key:
            regions = self._recognize_vision_model(image)
            engine = f"Vision model · {self.vlm_model}"
        elif self.tesseract_available:
            regions = self._recognize_tesseract(image)
            engine = "Local Tesseract · two-pass OCR"
        else:
            regions = [
                {
                    "text": f"{FLAG} OCR is not configured",
                    "confidence": 0.0,
                    "bbox": [0, 0, image.width, image.height],
                    "region_type": "unknown",
                    "uncertain": True,
                    "crossed_out": False,
                    "reasons": [self.engine_status["message"]],
                }
            ]
            engine = "OCR not configured"

        blocks = []
        for region in regions:
            text = region["text"].strip()
            confidence = max(0.0, min(1.0, float(region["confidence"])))
            crossed_out = bool(region.get("crossed_out", False))
            uncertain = bool(region.get("uncertain", False))
            flagged = uncertain or confidence < self.threshold
            if crossed_out:
                final_text = f"{FLAG} crossed-out text omitted"
            elif flagged and not text.startswith(FLAG):
                final_text = f"{FLAG} {text}"
            else:
                final_text = text or f"{FLAG} no readable text detected"
                flagged = True
            blocks.append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "region_type": region.get("region_type", "unknown"),
                    "bbox": region.get("bbox", [0, 0, 0, 0]),
                    "raw_text": text,
                    "final_text": final_text,
                    "confidence": round(confidence, 3),
                    "is_flagged": flagged,
                    "crossed_out": crossed_out,
                    "reasons": region.get("reasons", []),
                }
            )

        if not blocks:
            blocks.append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "region_type": "unknown",
                    "bbox": [0, 0, image.width, image.height],
                    "raw_text": "",
                    "final_text": f"{FLAG} no readable text detected",
                    "confidence": 0.0,
                    "is_flagged": True,
                    "crossed_out": False,
                    "reasons": ["The recognition engine returned no text regions"],
                }
            )

        visible = [
            block["final_text"] for block in blocks if not block["crossed_out"]
        ]
        return {
            "full_text": "\n".join(visible),
            "blocks": blocks,
            "total_blocks": len(blocks),
            "flagged_blocks": sum(block["is_flagged"] for block in blocks),
            "crossed_out_blocks": sum(block["crossed_out"] for block in blocks),
            "overall_confidence": round(
                sum(block["confidence"] for block in blocks) / len(blocks), 3
            ),
            "quality": quality,
            "engine": engine,
            "threshold": self.threshold,
        }
