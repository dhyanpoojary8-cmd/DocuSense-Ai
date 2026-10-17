"""Tesseract OCR engine implementation with auto-discovery and error handling."""
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image, ImageOps
import pytesseract

from app.extraction.base import BaseOCREngine
from app.models.extraction import ExtractionSource


DEFAULT_WINDOWS_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def resolve_tesseract_binary() -> Optional[str]:
    """Find the Tesseract executable path from environment, standard Windows path, or PATH."""
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and Path(env_path).is_file():
        return env_path

    if Path(DEFAULT_WINDOWS_TESSERACT_PATH).is_file():
        return DEFAULT_WINDOWS_TESSERACT_PATH

    system_which = shutil.which("tesseract")
    if system_which:
        return system_which

    return None


class TesseractOCREngine(BaseOCREngine):
    """Local Tesseract OCR engine with guaranteed provenance tracking."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.cmd_path = tesseract_cmd or resolve_tesseract_binary()
        if self.cmd_path:
            pytesseract.pytesseract.tesseract_cmd = self.cmd_path

    def is_available(self) -> bool:
        """Check if Tesseract binary is located and executable."""
        if not self.cmd_path:
            return False
        return Path(self.cmd_path).is_file()

    def get_source_identifier(self) -> ExtractionSource:
        return ExtractionSource.OCR

    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """General image preprocessing with Pillow: grayscale, upscale small images, autocontrast, simple threshold."""
        # 1. Grayscale
        gray = image.convert("L")

        # 2. Upscale small images
        w, h = gray.size
        if min(w, h) < 800:
            scale = 2
            gray = gray.resize((w * scale, h * scale), Image.Resampling.BILINEAR)

        # 3. Autocontrast
        contrasted = ImageOps.autocontrast(gray)

        # 4. Simple threshold
        threshold = 140
        thresholded = contrasted.point(lambda p: 255 if p > threshold else 0)

        return thresholded

    @staticmethod
    def reconstruct_lines_from_data(image: Image.Image, psm: int = 6) -> str:
        """Run OCR with image_to_data and reconstruct lines by vertical position so labels and values stay on one line."""
        config = f"--psm {psm}"
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        words: List[Dict[str, Any]] = []
        n = len(data["text"])
        for i in range(n):
            t = data["text"][i].strip()
            if not t or data["conf"][i] < 0:
                continue
            words.append({
                "text": t,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "cy": data["top"][i] + data["height"][i] / 2.0,
            })

        if not words:
            return ""

        # Sort words primarily by top/center_y
        words.sort(key=lambda w: (w["cy"], w["left"]))
        lines: List[List[Dict[str, Any]]] = []
        for w in words:
            placed = False
            for line in lines:
                line_cy = sum(item["cy"] for item in line) / len(line)
                line_h = sum(item["height"] for item in line) / len(line)
                if abs(w["cy"] - line_cy) <= max(10, line_h * 0.55):
                    line.append(w)
                    placed = True
                    break
            if not placed:
                lines.append([w])

        # Sort lines vertically from top to bottom
        lines.sort(key=lambda l: sum(w["cy"] for w in l) / len(l))

        # Sort words horizontally in each line left to right
        text_lines = []
        for l in lines:
            l.sort(key=lambda w: w["left"])
            text_lines.append(" ".join(w["text"] for w in l))

        return "\n".join(text_lines).strip()

    def extract_text_from_image(
        self,
        image: Union[str, Path, Image.Image],
        psm: int = 6,
        reconstruct_lines: bool = False,
    ) -> str:
        """Run OCR on image and return extracted text string."""
        if not self.is_available():
            raise RuntimeError(
                f"Tesseract binary not found at '{self.cmd_path}'. "
                "Ensure Tesseract is installed or set TESSERACT_CMD in .env"
            )

        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image))
        else:
            pil_image = image

        processed_image = self.preprocess_image(pil_image)

        if reconstruct_lines:
            return self.reconstruct_lines_from_data(processed_image, psm=psm)

        config = f"--psm {psm}"
        raw_text = pytesseract.image_to_string(processed_image, config=config)
        return raw_text.strip()
