---
title: DocuSense AI
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# DocuSense AI — Intelligent Document Verification

DocuSense AI is a fast, deployable web application and API for extracting, validating, correcting, and exporting structured data from invoices, receipts, and forms.

## Key Features
- **Local OCR & Digital PDF Ingestion**: Extracts text via Tesseract OCR for images/scans and directly extracts text from digital PDFs without paid external APIs.
- **6 Core Fields**: Vendor, document number, date, subtotal, tax, and total.
- **Deterministic Validation**: Pydantic schemas and deterministic rules using `Decimal` arithmetic for exact financial calculations.
- **Data Integrity & Confidence**: Identifies missing, unreadable, ambiguous, and inconsistent values with field-level confidence indicators and provenance tracking (`ocr`, `pdf_text`, or `manual`).
- **Interactive Review & Manual Correction**: Edit and correct extracted fields directly before exporting.
- **Sanitized Export**: Export clean structured data as CSV (with spreadsheet formula injection protection) and JSON.
- **Cloud-Ready**: Prepared for one-click Docker container deployment to Hugging Face Spaces (Docker SDK) or Render.

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Tesseract OCR (v5+ recommended) installed on system PATH or configured via `TESSERACT_CMD`.

### Setup
```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the application
uvicorn app.main:app --port 8000
```

Open `http://localhost:8000` in your browser. API documentation is available at `http://localhost:8000/docs`.

## Docker Deployment

Detailed instructions for deploying the container to free hosts (Hugging Face Spaces and Render) are available in [DEPLOY.md](DEPLOY.md).

```bash
# Build Docker image
docker build -t docusense-ai .

# Run container locally
docker run -p 7860:7860 docusense-ai
```

## Running Tests

```bash
cd backend
pytest
```

## Project Structure
```text
DocuSense-Ai/
├── Dockerfile                   # Production container definition (python:3.11-slim)
├── .dockerignore                # Image build ignore rules
├── .env.example                 # Environment configuration template
├── DEPLOY.md                    # Verified deployment steps for HF Spaces & Render
├── README.md                    # Project documentation & Space metadata
└── backend/
    ├── requirements.txt         # Pydantic, FastAPI, Tesseract, Pillow, PyPDF
    ├── app/
    │   ├── main.py              # FastAPI endpoints & resilient storage logic
    │   ├── static/              # Dashboard web interface
    │   ├── models/              # Pydantic schemas & enums
    │   ├── extraction/          # Tesseract engine & deterministic rule extractor
    │   └── validation/          # Validation engine & arithmetic checker
    └── tests/                   # Pytest test suite (90 passing tests)
```
