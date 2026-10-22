"""DocuSense AI backend: upload, extract, validate, correct and export documents.

Run from the backend folder:
    .venv\\Scripts\\uvicorn.exe app.main:app --port 8000
Then open http://127.0.0.1:8000  (interactive API docs: /docs)

Extraction is local OCR (Tesseract) or embedded PDF text, plus deterministic rules.
No paid API is needed. The optional vision-model adapter is NOT wired in here.
"""
import csv
import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.extraction.rule_extractor import DeterministicRuleExtractor
from app.extraction.tesseract_engine import TesseractOCREngine
from app.models.extraction import ExtractionSource
from app.models.field import ExtractedField
from app.models.validation import ArithmeticStatus
from app.validation.rules_engine import validate_document_fields
import os
import tempfile

BASE_DIR = Path(__file__).resolve().parent.parent


def _ensure_upload_dir() -> Path:
    """Ensure upload directory exists and is writable, falling back to tempdir if read-only."""
    target = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
    try:
        target.mkdir(parents=True, exist_ok=True)
        test_file = target / f".write_test_{uuid.uuid4().hex[:6]}"
        test_file.touch()
        test_file.unlink(missing_ok=True)
        return target
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "docusense_uploads"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except OSError:
            return target


UPLOAD_DIR = _ensure_upload_dir()          # git-ignored; tolerates wiped or read-only disk
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024        # 10 MB
FIELD_ORDER = ["vendor", "document_number", "date", "subtotal", "tax", "total"]
MONEY_FIELDS = ["subtotal", "tax", "total"]

app = FastAPI(title="DocuSense AI", version="0.1.0")
ocr_engine = TesseractOCREngine()
rule_extractor = DeterministicRuleExtractor()
_STORE: Dict[str, Dict[str, Any]] = {}     # in-memory cache; JSON copies live in UPLOAD_DIR


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


# ---------------------------------------------------------------- helpers
def detect_kind(head: bytes) -> Optional[Tuple[str, str, str]]:
    """Identify the file by its first bytes (never by file name). Returns (kind, ext, media_type)."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", ".png", "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image", ".jpg", "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", ".webp", "image/webp"
    if head.startswith(b"%PDF-"):
        return "pdf", ".pdf", "application/pdf"
    return None


def field_to_dict(f: ExtractedField) -> Dict[str, Any]:
    return {
        "name": f.name,
        "value": None if f.normalized_value is None else str(f.normalized_value),
        "raw_value": f.raw_value,
        "status": f.status.value if hasattr(f.status, "value") else str(f.status),
        "confidence": round(float(f.confidence.score), 2),
        "confidence_method": f.confidence.method,
        "confidence_calibrated": bool(f.confidence.is_calibrated),
        "flags": list(f.flags),
        "message": f.message,
        "source": f.source,
    }


def summarize(fields: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    to_review = [n for n in FIELD_ORDER if fields[n]["status"] != "VALID"]
    return {
        "needs_review": bool(to_review),
        "fields_to_review": to_review,
        "valid_count": len(FIELD_ORDER) - len(to_review),
    }


def normalize_currency(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip() or value.strip().lower() == "auto":
        return None
    code = value.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise HTTPException(422, "Currency must be a 3-letter code such as IDR or USD, or 'auto'.")
    return code


def _uuid_or_404(doc_id: str) -> str:
    try:
        return str(uuid.UUID(doc_id))
    except ValueError:
        raise HTTPException(404, "Document not found.")


def _save(rec: Dict[str, Any]) -> None:
    _STORE[rec["id"]] = rec
    try:
        upload_dir = UPLOAD_DIR
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            upload_dir = Path(tempfile.gettempdir()) / "docusense_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
    except OSError:
        pass  # in-memory copy still works


def get_record(doc_id: str) -> Dict[str, Any]:
    doc_id = _uuid_or_404(doc_id)
    rec = _STORE.get(doc_id)
    if rec is None:
        candidate_dirs = [UPLOAD_DIR, Path(tempfile.gettempdir()) / "docusense_uploads"]
        for cdir in candidate_dirs:
            path = cdir / f"{doc_id}.json"
            if path.is_file():
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                    _STORE[doc_id] = rec
                    break
                except (OSError, ValueError):
                    rec = None
    if rec is None:
        raise HTTPException(404, "Document not found.")
    return rec


def public_view(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k != "stored_name"}


def _csv_safe(value: Any) -> str:
    """Stop spreadsheet formula injection from OCR text (cells starting with = + - @)."""
    text = "" if value is None else str(value)
    if text and text[0] in "=+-@\t\r" and not re.fullmatch(r"-?\d+(\.\d+)?", text):
        return "'" + text
    return text


# ------------------------------------------------------------- processing
def _ocr_image(img: Image.Image) -> str:
    if not ocr_engine.is_available():
        raise HTTPException(503, "Tesseract OCR is not installed or was not found on this server.")
    return ocr_engine.extract_text_from_image(img)


def _read_image(data: bytes) -> Image.Image:
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        raise HTTPException(422, "This image could not be read. It may be corrupted.")


def _text_from_pdf(path: Path) -> Tuple[str, ExtractionSource]:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        if reader.is_encrypted:
            raise HTTPException(422, "Password-protected PDFs are not supported.")
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:3])
        if len(text.strip()) >= 20:
            return text, ExtractionSource.PDF_TEXT
        import pypdfium2 as pdfium  # scanned PDF: draw page 1 and OCR it
        page = pdfium.PdfDocument(str(path))[0]
        return _ocr_image(page.render(scale=3).to_pil()), ExtractionSource.OCR
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "This PDF could not be read. It may be corrupted.")


def process_document(data: bytes, kind: str, ext: str, media_type: str,
                     display_name: str, currency: Optional[str]) -> Dict[str, Any]:
    doc_id = str(uuid.uuid4())
    upload_dir = UPLOAD_DIR
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        upload_dir = Path(tempfile.gettempdir()) / "docusense_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{doc_id}{ext}"
    path = upload_dir / stored_name
    try:
        path.write_bytes(data)  # stored under a random name; the user's file name is never used on disk
    except OSError:
        upload_dir = Path(tempfile.gettempdir()) / "docusense_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / stored_name
        path.write_bytes(data)

    try:
        if kind == "image":
            raw_text, source = _ocr_image(_read_image(data)), ExtractionSource.OCR
        else:
            raw_text, source = _text_from_pdf(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    extracted = rule_extractor.extract_fields(raw_text, source=source, currency_hint=currency)
    fields = {n: field_to_dict(extracted[n]) for n in FIELD_ORDER}
    rec = {
        "id": doc_id,
        "filename": Path(display_name or "upload").name[:100],
        "kind": kind,
        "media_type": media_type,
        "stored_name": stored_name,
        "source": source.value,
        "currency": currency,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": fields,
        "corrected": [],
        "summary": summarize(fields),
        "raw_text": raw_text[:20000],
        "notes": "Confidence values are heuristic indicators, not calibrated probabilities.",
    }
    _save(rec)
    return rec


# -------------------------------------------------------------- endpoints
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "tesseract_available": ocr_engine.is_available(), "vision_model": "not_run"}


@app.post("/api/documents/upload")
async def upload(file: UploadFile = File(...), currency: Optional[str] = Form(None)):
    cur = normalize_currency(currency)
    data = b""
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is too large. The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    detected = detect_kind(data[:16])
    if detected is None:
        raise HTTPException(415, "Unsupported file type. Please upload a PNG, JPG, WEBP or PDF.")
    kind, ext, media_type = detected
    rec = await run_in_threadpool(process_document, data, kind, ext, media_type, file.filename or "upload", cur)
    return public_view(rec)


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    return public_view(get_record(doc_id))


@app.get("/api/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    rec = get_record(doc_id)
    candidate_paths = [
        UPLOAD_DIR / rec["stored_name"],
        Path(tempfile.gettempdir()) / "docusense_uploads" / rec["stored_name"],
    ]
    path = next((p for p in candidate_paths if p.is_file()), None)
    if path is None:
        raise HTTPException(404, "The stored file is no longer available.")
    return FileResponse(path, media_type=rec["media_type"])


class CorrectionRequest(BaseModel):
    field: str
    value: Optional[str] = Field(default=None, max_length=200)


@app.post("/api/documents/{doc_id}/correct")
def correct_field(doc_id: str, req: CorrectionRequest):
    rec = get_record(doc_id)
    if req.field not in FIELD_ORDER:
        raise HTTPException(422, f"Unknown field. Use one of: {', '.join(FIELD_ORDER)}.")
    raw = {n: rec["fields"][n]["raw_value"] for n in FIELD_ORDER}
    raw[req.field] = (req.value or "").strip() or None
    corrected = sorted(set(rec.get("corrected", [])) | {req.field})

    result = validate_document_fields(raw, currency_hint=rec.get("currency"))
    new_fields: Dict[str, Dict[str, Any]] = {}
    for n in FIELD_ORDER:
        d = field_to_dict(result.fields[n])
        if n in corrected:
            d["source"] = "manual"
            d["confidence_method"] = "manual_correction"
            if d["status"] == "VALID":
                d["confidence"] = 1.0
        else:
            d["source"] = rec["fields"][n]["source"]
        new_fields[n] = d
    if result.arithmetic.status == ArithmeticStatus.MISMATCH:
        for n in MONEY_FIELDS:
            if new_fields[n]["status"] == "VALID":
                new_fields[n]["status"] = "INCONSISTENT"
                new_fields[n]["flags"] = sorted(set(new_fields[n]["flags"]) | {"ARITHMETIC_MISMATCH"})
                new_fields[n]["message"] = f"Subtotal + tax does not equal total ({result.arithmetic.message})."
    rec["fields"] = new_fields
    rec["corrected"] = corrected
    rec["summary"] = summarize(new_fields)
    _save(rec)
    return public_view(rec)


@app.get("/api/documents/{doc_id}/export")
def export_document(doc_id: str, format: str = "json"):
    rec = get_record(doc_id)
    if format == "json":
        return JSONResponse(
            public_view(rec),
            headers={"Content-Disposition": f'attachment; filename="docusense_{rec["id"]}.json"'},
        )
    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["document_id", "filename", "document_source", "field", "value",
                         "raw_value", "status", "confidence", "flags", "source"])
        for n in FIELD_ORDER:
            f = rec["fields"][n]
            writer.writerow([_csv_safe(x) for x in (
                rec["id"], rec["filename"], rec["source"], n, f["value"], f["raw_value"],
                f["status"], f["confidence"], ";".join(f["flags"]), f["source"])])
        return Response(
            buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="docusense_{rec["id"]}.csv"'},
        )
    raise HTTPException(422, "format must be 'csv' or 'json'.")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
