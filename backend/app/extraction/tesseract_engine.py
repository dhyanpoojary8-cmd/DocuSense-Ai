"""Tesseract OCR engine implementation with auto-discovery and error handling."""
import os
import shutil
from pathlib import Path
from typing import Optional, Union
from PIL import Image
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

    def extract_text_from_image(self, image: Union[str, Path, Image.Image], psm: int = 3) -> str:
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

        # Convert to RGB if palette/alpha
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        config = f"--psm {psm}"
        raw_text = pytesseract.image_to_string(pil_image, config=config)
        return raw_text.strip()
