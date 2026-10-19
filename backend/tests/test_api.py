"""API tests. OCR is replaced by a fake so these tests check the API itself, not Tesseract."""
import csv
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main

GOOD_TEXT = "STORE ABC\nInvoice No: INV-001\nDate: 25/12/2018\nSubtotal: 30,000\nPB1: 3,000\nTotal: 33,000\n"
BAD_TOTAL_TEXT = "STORE ABC\nSubtotal: 30,000\nPB1: 3,000\nTotal: 40,000\n"


class FakeOCR:
    def __init__(self, text):
        self.text = text

    def is_available(self):
        return True

    def extract_text_from_image(self, image, *args, **kwargs):
        return self.text


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main, "ocr_engine", FakeOCR(GOOD_TEXT))
    main._STORE.clear()
    return TestClient(main.app)


def upload(client, data=None, name="receipt.png", currency="IDR"):
    return client.post(
        "/api/documents/upload",
        files={"file": (name, data if data is not None else png_bytes(), "application/octet-stream")},
        data={"currency": currency},
    )


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["vision_model"] == "not_run"


def test_upload_returns_all_six_fields_with_provenance(client):
    r = upload(client)
    assert r.status_code == 200
    body = r.json()
    assert list(body["fields"]) == main.FIELD_ORDER
    assert body["source"] == "ocr"
    assert body["fields"]["subtotal"]["value"] == "30000"
    assert body["fields"]["tax"]["value"] == "3000"
    assert body["fields"]["total"]["value"] == "33000"
    assert all(f["confidence_calibrated"] is False for f in body["fields"].values())
    assert "stored_name" not in body


def test_get_document_and_preview_file(client):
    doc = upload(client).json()
    again = client.get(f"/api/documents/{doc['id']}")
    assert again.status_code == 200 and again.json()["id"] == doc["id"]
    f = client.get(f"/api/documents/{doc['id']}/file")
    assert f.status_code == 200 and f.headers["content-type"] == "image/png"


def test_user_filename_is_never_used_on_disk(client, tmp_path):
    upload(client, name="../../evil.png")
    assert not any("evil" in p.name for p in tmp_path.iterdir())
    assert not (tmp_path.parent.parent / "evil.png").exists()


def test_export_json_and_csv(client):
    doc = upload(client).json()
    j = client.get(f"/api/documents/{doc['id']}/export?format=json")
    assert j.status_code == 200 and "attachment" in j.headers["content-disposition"]
    assert j.json()["fields"]["total"]["value"] == "33000"
    c = client.get(f"/api/documents/{doc['id']}/export?format=csv")
    assert c.status_code == 200 and c.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(c.text)))
    assert rows[0][:5] == ["document_id", "filename", "document_source", "field", "value"]
    assert len(rows) == 1 + 6


def test_export_rejects_unknown_format(client):
    doc = upload(client).json()
    assert client.get(f"/api/documents/{doc['id']}/export?format=xml").status_code == 422


def test_rejects_text_file_even_with_image_name(client):
    assert upload(client, data=b"hello, not an image", name="fake.png").status_code == 415


def test_rejects_empty_file(client):
    assert upload(client, data=b"").status_code == 400


def test_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    assert upload(client, data=b"\x89PNG\r\n\x1a\n" + b"0" * 500).status_code == 413


def test_rejects_corrupted_image(client):
    assert upload(client, data=b"\x89PNG\r\n\x1a\n" + b"garbage" * 20).status_code == 422


def test_rejects_corrupted_pdf(client):
    assert upload(client, data=b"%PDF-1.4 this is not really a pdf", name="x.pdf").status_code == 422


def test_rejects_bad_currency(client):
    assert upload(client, currency="12345").status_code == 422


def test_unknown_and_malformed_ids_return_404(client):
    assert client.get("/api/documents/not-a-uuid").status_code == 404
    assert client.get("/api/documents/00000000-0000-4000-8000-000000000000").status_code == 404


def test_inconsistent_totals_are_flagged_not_guessed(client):
    main.ocr_engine.text = BAD_TOTAL_TEXT
    body = upload(client).json()
    for n in main.MONEY_FIELDS:
        assert body["fields"][n]["status"] == "INCONSISTENT"
        assert body["fields"][n]["value"] is None
    assert body["summary"]["needs_review"] is True


def test_manual_correction_revalidates_and_marks_source(client):
    main.ocr_engine.text = BAD_TOTAL_TEXT
    doc = upload(client).json()
    r = client.post(f"/api/documents/{doc['id']}/correct", json={"field": "total", "value": "33,000"})
    assert r.status_code == 200
    fields = r.json()["fields"]
    assert fields["total"]["source"] == "manual"
    assert fields["total"]["value"] == "33000"
    assert all(fields[n]["status"] == "VALID" for n in main.MONEY_FIELDS)


def test_wrong_manual_correction_keeps_flag(client):
    doc = upload(client).json()
    r = client.post(f"/api/documents/{doc['id']}/correct", json={"field": "total", "value": "99,000"})
    fields = r.json()["fields"]
    assert all(fields[n]["status"] == "INCONSISTENT" for n in main.MONEY_FIELDS)


def test_correction_rejects_unknown_field(client):
    doc = upload(client).json()
    r = client.post(f"/api/documents/{doc['id']}/correct", json={"field": "colour", "value": "red"})
    assert r.status_code == 422


def test_record_survives_a_restart(client):
    doc = upload(client).json()
    main._STORE.clear()  # simulate the server restarting
    assert client.get(f"/api/documents/{doc['id']}").status_code == 200


def test_csv_cells_cannot_run_formulas():
    assert main._csv_safe("=1+1") == "'=1+1"
    assert main._csv_safe("@SUM(A1)") == "'@SUM(A1)"
    assert main._csv_safe("-100") == "-100"
    assert main._csv_safe(None) == ""


def test_detect_kind_uses_content_not_name():
    assert main.detect_kind(b"%PDF-1.7")[0] == "pdf"
    assert main.detect_kind(b"\xff\xd8\xff\xe0abc")[1] == ".jpg"
    assert main.detect_kind(b"RIFF\x00\x00\x00\x00WEBP")[1] == ".webp"
    assert main.detect_kind(b"MZ\x90\x00") is None
