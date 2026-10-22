"""Tests for the held-out evaluation script. OCR is faked, so these check the logic and the safety guards."""
import importlib.util
import io
from pathlib import Path

from PIL import Image

from app.extraction.rule_extractor import DeterministicRuleExtractor

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "heldout_eval.py"
spec = importlib.util.spec_from_file_location("heldout_eval", SCRIPT)
heldout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heldout)

TEXTS = {
    "a.png": "STORE A\nSubtotal: 30,000\nPB1: 3,000\nTotal: 33,000\n",   # perfect
    "b.png": "nothing useful here",                                      # abstains
    "c.png": "STORE C\nTotal 45,500\n",                                  # wrong total (label 50000)
}
ROWS = [
    {"file": "a.png", "subtotal": "30,000", "tax": "3,000", "total": "33,000"},
    {"file": "b.png", "subtotal": "10,000", "tax": "1,000", "total": "11,000"},
    {"file": "c.png", "subtotal": "", "tax": "", "total": "50,000"},
]


def make_dir(tmp_path):
    for r in ROWS:
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), "white").save(buf, "PNG")
        (tmp_path / r["file"]).write_bytes(buf.getvalue())
    return tmp_path


def fake_ocr(path):
    return TEXTS[path.name]


def test_refuses_without_confirm_flag(tmp_path):
    assert heldout.main(["--data-dir", str(tmp_path)]) == 2


def test_counts_correct_wrong_and_abstained(tmp_path):
    res = heldout.run_evaluation(ROWS, make_dir(tmp_path), fake_ocr, DeterministicRuleExtractor(), 30)
    s = res["stats"]
    assert (s["total"]["labelled"], s["total"]["correct"], s["total"]["wrong"], s["total"]["abstained"]) == (3, 1, 1, 1)
    assert (s["subtotal"]["labelled"], s["subtotal"]["correct"], s["subtotal"]["abstained"]) == (2, 1, 1)
    assert s["subtotal"]["no_label"] == 1 and s["tax"]["no_label"] == 1
    assert res["docs_all_correct"] == 1 and res["docs_with_labels"] == 3
    assert [w["file"] for w in res["wrong"]] == ["c.png"]


def test_only_first_n_receipts_are_used(tmp_path):
    res = heldout.run_evaluation(ROWS, make_dir(tmp_path), fake_ocr, DeterministicRuleExtractor(), 1)
    assert res["files"] == ["a.png"]


def test_ocr_crash_becomes_abstention_not_a_crash(tmp_path):
    def boom(path):
        raise RuntimeError("bad image")
    res = heldout.run_evaluation(ROWS[:1], make_dir(tmp_path), boom, DeterministicRuleExtractor(), 30)
    assert res["stats"]["total"]["abstained"] == 1
    assert res["stats"]["total"]["reasons"]["OCR_ERROR"] == 1


def test_missing_image_is_reported(tmp_path):
    res = heldout.run_evaluation(ROWS, tmp_path, fake_ocr, DeterministicRuleExtractor(), 30)
    assert len(res["missing_images"]) == 3 and res["files"] == []


def test_report_is_honest_about_scope(tmp_path):
    res = heldout.run_evaluation(ROWS, make_dir(tmp_path), fake_ocr, DeterministicRuleExtractor(), 30)
    text = heldout.build_report(res, {"Git commit": "abc123", "Run number": 1})
    assert "NOT EVALUATED" in text and "not run" in text and "abc123" in text
    assert "not comparable with the DEV SET" in text
    for field in ("vendor", "document_number", "date"):
        assert f"| {field} | NOT EVALUATED" in text


def test_wilson_interval():
    assert heldout.wilson(0, 0) == (None, None)
    lo, hi = heldout.wilson(5, 10)
    assert lo < 0.5 < hi
    assert heldout.wilson(10, 10)[1] == 1.0
    assert heldout.wilson(0, 10)[0] == 0.0
