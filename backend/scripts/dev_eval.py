"""Evaluate extraction pipeline (Tesseract OCR + DeterministicRuleExtractor) on dev split.

Usage:
    .venv\\Scripts\\python.exe scripts\\dev_eval.py
    python backend/scripts/dev_eval.py

Compares extracted 'subtotal', 'tax', and 'total' fields against labels.csv in the CORD validation split.
"""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure stdout handles unicode/OCR symbols on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from app.extraction.rule_extractor import DeterministicRuleExtractor
from app.extraction.tesseract_engine import TesseractOCREngine
from app.models.extraction import ExtractionSource
from app.validation.money_validator import parse_monetary_value


TARGET_FIELDS = ("subtotal", "tax", "total")


def evaluate_dev_set(data_dir: Path) -> int:
    """Run evaluation on the dev set directory and print per-field accuracy and misses."""
    labels_file = data_dir / "labels.csv"
    if not labels_file.is_file():
        print(f"Error: labels.csv not found in {data_dir}", file=sys.stderr)
        return 1

    engine = TesseractOCREngine()
    if not engine.is_available():
        print(f"Error: Tesseract OCR engine is not available at '{engine.cmd_path}'", file=sys.stderr)
        return 1

    extractor = DeterministicRuleExtractor()

    with open(labels_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Tracking metrics per field
    labelled_counts: Dict[str, int] = {f: 0 for f in TARGET_FIELDS}
    correct_counts: Dict[str, int] = {f: 0 for f in TARGET_FIELDS}
    wrong_counts: Dict[str, int] = {f: 0 for f in TARGET_FIELDS}
    abstained_counts: Dict[str, int] = {f: 0 for f in TARGET_FIELDS}
    no_label_counts: Dict[str, int] = {f: 0 for f in TARGET_FIELDS}

    misses: List[Dict[str, Any]] = []

    for row in rows:
        filename = row["file"]
        image_path = data_dir / filename
        if not image_path.is_file():
            print(f"Warning: Image file not found: {image_path}", file=sys.stderr)
            continue

        raw_ocr_text = engine.extract_text_from_image(image_path)
        extracted_fields = extractor.extract_fields(
            raw_ocr_text,
            source=ExtractionSource.OCR,
            currency_hint="IDR",
        )

        for field_name in TARGET_FIELDS:
            raw_label = (row.get(field_name) or "").strip()
            if not raw_label:
                # Skip fields with an empty label and count them as "no label"
                no_label_counts[field_name] += 1
                continue

            labelled_counts[field_name] += 1

            # Parse ground truth label amount using existing Decimal money parser with IDR hint
            label_val, l_status, _, _ = parse_monetary_value(
                raw_label,
                allow_negative=(field_name == "tax"),
                currency_hint="IDR",
                field_name=field_name,
            )

            # Extract predicted amount
            pred_field = extracted_fields.get(field_name)
            pred_val: Optional[Decimal] = None
            pred_status: Optional[str] = None
            if pred_field:
                pred_status = pred_field.status.value if hasattr(pred_field.status, "value") else str(pred_field.status)
                if pred_field.normalized_value is not None and pred_status != "AMBIGUOUS":
                    try:
                        pred_val = Decimal(str(pred_field.normalized_value))
                    except Exception:
                        pred_val = None
                elif pred_field.raw_value and pred_status not in ("AMBIGUOUS", "UNREADABLE", "MISSING"):
                    pv, pstat, _, _ = parse_monetary_value(
                        pred_field.raw_value,
                        allow_negative=(field_name == "tax"),
                        currency_hint="IDR",
                        field_name=field_name,
                    )
                    if pstat != FieldStatus.AMBIGUOUS:
                        pred_val = pv

            # Compare predicted value against ground truth: correct / wrong / abstained
            if pred_val is not None:
                if pred_val == label_val:
                    correct_counts[field_name] += 1
                else:
                    wrong_counts[field_name] += 1
                    misses.append(
                        {
                            "file": filename,
                            "field": field_name,
                            "miss_type": "WRONG",
                            "predicted": str(pred_val),
                            "label": str(label_val) if label_val is not None else raw_label,
                            "raw_label": raw_label,
                            "status": pred_status,
                            "raw_ocr_text": raw_ocr_text,
                        }
                    )
            else:
                abstained_counts[field_name] += 1
                misses.append(
                    {
                        "file": filename,
                        "field": field_name,
                        "miss_type": "ABSTAINED",
                        "predicted": "None",
                        "label": str(label_val) if label_val is not None else raw_label,
                        "raw_label": raw_label,
                        "status": pred_status or "MISSING",
                        "raw_ocr_text": raw_ocr_text,
                    }
                )

    # Output results labeled "DEV SET results"
    print("=" * 82)
    print("DEV SET results: Accuracy & Breakdown Summary")
    print("=" * 82)
    print(f"{'Field':<12} {'Correct':<9} {'Wrong':<8} {'Abstained':<11} {'Labelled':<10} {'Accuracy':<11} {'No Label':<9}")
    print("-" * 82)

    total_correct = 0
    total_wrong = 0
    total_abstained = 0
    total_labelled = 0
    total_no_label = 0

    for field_name in TARGET_FIELDS:
        correct = correct_counts[field_name]
        wrong = wrong_counts[field_name]
        abstained = abstained_counts[field_name]
        labelled = labelled_counts[field_name]
        no_label = no_label_counts[field_name]
        acc_str = f"{(correct / labelled * 100):.2f}%" if labelled > 0 else "N/A"
        print(f"{field_name:<12} {correct:<9} {wrong:<8} {abstained:<11} {labelled:<10} {acc_str:<11} {no_label:<9}")

        total_correct += correct
        total_wrong += wrong
        total_abstained += abstained
        total_labelled += labelled
        total_no_label += no_label

    print("-" * 82)
    overall_acc_str = f"{(total_correct / total_labelled * 100):.2f}%" if total_labelled > 0 else "N/A"
    print(f"{'Overall':<12} {total_correct:<9} {total_wrong:<8} {total_abstained:<11} {total_labelled:<10} {overall_acc_str:<11} {total_no_label:<9}")
    print("=" * 82)
    print()

    # Print misses with file name, predicted value, label, and raw OCR text
    print("=" * 70)
    print(f"DEV SET results: Detailed Misses ({len(misses)} total)")
    print("=" * 70)

    for i, miss in enumerate(misses, 1):
        print(f"\n--- [DEV SET results] Miss #{i} ---")
        print(f"File:            {miss['file']}")
        print(f"Field:           {miss['field']}")
        print(f"Predicted value: {miss['predicted']} (status: {miss['status']}, {miss['miss_type']})")
        print(f"Label:           {miss['label']} (raw label: '{miss['raw_label']}')")
        print("Raw OCR text:")
        print("-" * 50)
        print(miss["raw_ocr_text"] if miss["raw_ocr_text"] else "<EMPTY OCR TEXT>")
        print("-" * 50)

    return 0


def main() -> int:
    default_dev_dir = BACKEND_DIR / "dev_data" / "cord_validation"
    parser = argparse.ArgumentParser(description="Evaluate extraction pipeline on CORD validation dev set.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_dev_dir,
        help="Path to directory containing receipt images and labels.csv",
    )
    args = parser.parse_args()
    return evaluate_dev_set(args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
