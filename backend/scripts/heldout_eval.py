"""HELD-OUT evaluation of Tesseract + rules on the CORD *test* split (money fields only).

Run this ONLY after the rules are frozen and committed, and only once.

    .venv\\Scripts\\python.exe scripts\\heldout_eval.py --confirm-frozen

Safety checks (the script refuses to run otherwise):
  * --confirm-frozen must be given.
  * Code under app/ must be committed, so the report matches a git commit
    (override with --allow-dirty; the report will then say so).
  * A held-out run may happen once. A second run needs --rerun and is labelled as run #2, #3 ...

What is evaluated: subtotal, tax, total (the only fields CORD labels).
Not evaluated: vendor, document_number, date (no ground truth exists; none was invented).
Vision-model comparison: not run.
Receipts are the first N in dataset file order, never hand-picked.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
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
NOT_EVALUATED = ("vendor", "document_number", "date")
CURRENCY = "IDR"
REPORT_DIR = BACKEND_DIR / "evaluation_reports"
RUN_LOG = REPORT_DIR / "heldout_run_log.jsonl"
OCR_SETTINGS = ("Tesseract engine defaults at this commit: --psm 6, no line reconstruction; Pillow "
                "preprocessing (grayscale, 2x upscale of small images, autocontrast, threshold 140)")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """95% Wilson score interval for k successes out of n (honest about small samples)."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def predicted_amount(fields: Dict[str, Any], name: str) -> Tuple[Optional[Decimal], str]:
    """Only a VALID field with a value counts as an answer. Everything else is an abstention."""
    f = fields.get(name)
    if f is None:
        return None, "MISSING"
    status = f.status.value if hasattr(f.status, "value") else str(f.status)
    if status == "VALID" and f.normalized_value is not None:
        try:
            return Decimal(str(f.normalized_value)), status
        except Exception:
            return None, "UNPARSEABLE"
    return None, status


def run_evaluation(rows: List[Dict[str, str]], data_dir: Path,
                   ocr_fn: Callable[[Path], str], extractor: Any, n: int) -> Dict[str, Any]:
    stats = {f: {"labelled": 0, "correct": 0, "wrong": 0, "abstained": 0, "no_label": 0,
                 "bad_label": 0, "reasons": Counter()} for f in TARGET_FIELDS}
    wrong: List[Dict[str, Any]] = []
    abstained: List[Dict[str, Any]] = []
    files: List[str] = []
    missing_images: List[str] = []
    docs_all_correct = 0
    docs_with_labels = 0

    for row in rows[:n]:
        name = row["file"]
        path = data_dir / name
        if not path.is_file():
            missing_images.append(name)
            continue
        files.append(name)
        ocr_error = None
        try:
            text = ocr_fn(path)
        except Exception as exc:  # one bad image must not abort the one-shot run
            text, ocr_error = "", f"{type(exc).__name__}"
        fields = extractor.extract_fields(text, source=ExtractionSource.OCR, currency_hint=CURRENCY)

        doc_labelled = 0
        doc_correct = 0
        for fld in TARGET_FIELDS:
            raw_label = (row.get(fld) or "").strip()
            s = stats[fld]
            if not raw_label:
                s["no_label"] += 1
                continue
            label_val, _, _, _ = parse_monetary_value(
                raw_label, allow_negative=(fld == "tax"), currency_hint=CURRENCY, field_name=fld)
            if label_val is None:
                s["bad_label"] += 1
                continue
            s["labelled"] += 1
            doc_labelled += 1
            pred, status = predicted_amount(fields, fld)
            if ocr_error:
                pred, status = None, "OCR_ERROR"
            if pred is None:
                s["abstained"] += 1
                s["reasons"][status] += 1
                abstained.append({"file": name, "field": fld, "label": str(label_val), "reason": status})
            elif pred == label_val:
                s["correct"] += 1
                doc_correct += 1
            else:
                s["wrong"] += 1
                wrong.append({"file": name, "field": fld, "predicted": str(pred),
                              "label": str(label_val), "ocr_text": text[:600]})
        if doc_labelled:
            docs_with_labels += 1
            if doc_correct == doc_labelled:
                docs_all_correct += 1

    return {"stats": stats, "wrong": wrong, "abstained": abstained, "files": files,
            "missing_images": missing_images, "docs_all_correct": docs_all_correct,
            "docs_with_labels": docs_with_labels}


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def build_report(result: Dict[str, Any], meta: Dict[str, Any]) -> str:
    st = result["stats"]
    L: List[str] = []
    L.append("# HELD-OUT evaluation: CORD test split (money fields)\n")
    L.append("**These are held-out results. They are not comparable with the DEV SET results, "
             "which were used to build the rules.**\n")
    L.append("## Run information\n")
    for k, v in meta.items():
        L.append(f"- **{k}:** {v}")
    L.append("")
    L.append("## What was and was not evaluated\n")
    L.append("- **Evaluated:** subtotal, tax, total. These are the only fields CORD provides labels for.")
    L.append("- **NOT EVALUATED:** vendor, document_number, date. No ground-truth labels exist for them "
             "and none were created, so no accuracy is claimed for these fields.")
    L.append("- **Vision-model comparison:** not run.")
    L.append(f"- Receipts evaluated: {len(result['files'])} (the first in dataset file order, not hand-picked).")
    if result["missing_images"]:
        L.append(f"- WARNING: {len(result['missing_images'])} listed image(s) were missing and skipped.")
    L.append("- A value counts as correct only if it exactly equals the label. "
             "Abstaining (flagging instead of guessing) counts as not correct, but not as wrong.")
    L.append("")
    L.append("## Field-level results\n")
    L.append("| Field | Labelled | Correct | Wrong | Abstained | Accuracy (correct / labelled) | 95% interval | "
             "Precision when it answers | Coverage |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    tot = {"labelled": 0, "correct": 0, "wrong": 0, "abstained": 0}
    for fld in TARGET_FIELDS:
        s = st[fld]
        lo, hi = wilson(s["correct"], s["labelled"])
        answered = s["correct"] + s["wrong"]
        prec = s["correct"] / answered if answered else None
        cov = answered / s["labelled"] if s["labelled"] else None
        acc = s["correct"] / s["labelled"] if s["labelled"] else None
        L.append(f"| {fld} | {s['labelled']} | {s['correct']} | {s['wrong']} | {s['abstained']} | "
                 f"{_pct(acc)} | {_pct(lo)} to {_pct(hi)} | {_pct(prec)} | {_pct(cov)} |")
        for k in tot:
            tot[k] += s[k]
    answered = tot["correct"] + tot["wrong"]
    lo, hi = wilson(tot["correct"], tot["labelled"])
    L.append(f"| **all three** | {tot['labelled']} | {tot['correct']} | {tot['wrong']} | {tot['abstained']} | "
             f"{_pct(tot['correct'] / tot['labelled'] if tot['labelled'] else None)} | {_pct(lo)} to {_pct(hi)} | "
             f"{_pct(tot['correct'] / answered if answered else None)} | "
             f"{_pct(answered / tot['labelled'] if tot['labelled'] else None)} |")
    for fld in NOT_EVALUATED:
        L.append(f"| {fld} | NOT EVALUATED | | | | | | | |")
    L.append("")
    L.append("The three fields come from the same receipts, so the 'all three' row is not an independent sample. "
             "The intervals are wide because the sample is small.\n")
    L.append(f"Receipts where every labelled field was correct: {result['docs_all_correct']} of "
             f"{result['docs_with_labels']}.\n")
    L.append("Labels with no value in CORD (skipped): " +
             ", ".join(f"{f} {st[f]['no_label']}" for f in TARGET_FIELDS) + ".\n")
    L.append("## Why the app abstained\n")
    L.append("| Field | Reasons (status of the extracted field) |")
    L.append("|---|---|")
    for fld in TARGET_FIELDS:
        reasons = ", ".join(f"{k}: {v}" for k, v in st[fld]["reasons"].most_common()) or "none"
        L.append(f"| {fld} | {reasons} |")
    L.append("")
    L.append(f"## Wrong answers ({len(result['wrong'])})\n")
    if not result["wrong"]:
        L.append("None.\n")
    for w in result["wrong"]:
        L.append(f"### {w['file']} / {w['field']}: predicted {w['predicted']}, label {w['label']}\n")
        L.append("```text\n" + (w["ocr_text"] or "<empty OCR text>") + "\n```\n")
    L.append(f"## Abstentions ({len(result['abstained'])})\n")
    L.append("| File | Field | Label | Reason |")
    L.append("|---|---|---|---|")
    for a in result["abstained"]:
        L.append(f"| {a['file']} | {a['field']} | {a['label']} | {a['reason']} |")
    L.append("")
    return "\n".join(L)


def git_info() -> Tuple[str, Optional[bool]]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "app"], cwd=BACKEND_DIR,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "unknown", None


def previous_runs() -> int:
    if not RUN_LOG.is_file():
        return 0
    return sum(1 for line in RUN_LOG.read_text(encoding="utf-8").splitlines() if line.strip())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="One-shot held-out evaluation on the CORD test split.")
    ap.add_argument("--confirm-frozen", action="store_true",
                    help="I confirm the rules are frozen and committed.")
    ap.add_argument("--data-dir", type=Path, default=BACKEND_DIR / "dev_data" / "cord_test")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--allow-dirty", action="store_true", help="Run even if app/ has uncommitted changes.")
    ap.add_argument("--rerun", action="store_true", help="Allow a second or later held-out run.")
    args = ap.parse_args(argv)

    if not args.confirm_frozen:
        print("REFUSING TO RUN: pass --confirm-frozen to confirm the rules are frozen and committed.\n"
              "The test split must be used once, after the last rule change.", file=sys.stderr)
        return 2
    commit, dirty = git_info()
    if dirty and not args.allow_dirty:
        print("REFUSING TO RUN: files under app/ have uncommitted changes. Commit them first so this "
              "report matches a git commit (or pass --allow-dirty).", file=sys.stderr)
        return 3
    runs_before = previous_runs()
    if runs_before > 0 and not args.rerun:
        print(f"REFUSING TO RUN: {runs_before} held-out run(s) already exist in {RUN_LOG.name}. "
              "Use --rerun only if you must; the report will be labelled as a repeat run.", file=sys.stderr)
        return 4
    labels = args.data_dir / "labels.csv"
    if not labels.is_file():
        print(f"labels.csv not found in {args.data_dir}", file=sys.stderr)
        return 1
    engine = TesseractOCREngine()
    if not engine.is_available():
        print(f"Tesseract not available at '{engine.cmd_path}'", file=sys.stderr)
        return 1

    with open(labels, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = run_evaluation(rows, args.data_dir, engine.extract_text_from_image,
                            DeterministicRuleExtractor(), args.n)

    try:
        import pytesseract
        tess_version = str(pytesseract.get_tesseract_version())
    except Exception:
        tess_version = "unknown"
    now = datetime.now(timezone.utc)
    meta = {
        "Date (UTC)": now.isoformat(timespec="seconds"),
        "Run number": runs_before + 1,
        "Git commit": commit + (" (app/ HAD UNCOMMITTED CHANGES)" if dirty else ""),
        "Dataset": f"CORD v2 test split, folder {args.data_dir.name}, first {args.n} receipts (licence CC BY 4.0)",
        "Currency hint": CURRENCY,
        "OCR / extractor": OCR_SETTINGS,
        "Tesseract": tess_version,
        "Python": platform.python_version(),
    }
    if runs_before > 0:
        meta["WARNING"] = "This is a repeat held-out run; earlier results may have influenced later changes."
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    md_path = REPORT_DIR / f"heldout_{stamp}.md"
    md_path.write_text(build_report(result, meta), encoding="utf-8")
    raw = dict(result)
    raw["stats"] = {k: {**v, "reasons": dict(v["reasons"])} for k, v in result["stats"].items()}
    raw["meta"] = meta
    (REPORT_DIR / f"heldout_{stamp}.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now.isoformat(timespec="seconds"), "commit": commit,
                            "report": md_path.name}) + "\n")

    print("HELD-OUT results (CORD test split)")
    print(f"{'Field':<10} {'Labelled':>8} {'Correct':>8} {'Wrong':>6} {'Abstained':>10}")
    for fld in TARGET_FIELDS:
        s = result["stats"][fld]
        print(f"{fld:<10} {s['labelled']:>8} {s['correct']:>8} {s['wrong']:>6} {s['abstained']:>10}")
    print(f"\nReport written to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
