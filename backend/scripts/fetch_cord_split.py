"""Download a CORD-v2 split from Hugging Face and export the first N receipts.

CORD-v2 is licensed CC BY 4.0 (Naver Clova AI). Credit it in your README.
Only subtotal, tax and total are used as labels (CORD has no vendor/date labels).

Usage, from the backend/ folder:
    .venv\\Scripts\\python.exe scripts\\fetch_cord_split.py --split validation --n 12
    .venv\\Scripts\\python.exe scripts\\fetch_cord_split.py --split test --n 30

Use the validation split as the DEV set (tune rules on it).
Use the test split only for the final held-out run, after rules are frozen.
Receipts are taken in file order (first N), never hand-picked.

Output (git-ignored):  backend/dev_data/cord_<split>/
    cord_<split>_000.<ext>, ...       receipt images
    cord_<split>_000.gt.json, ...     raw CORD ground truth
    labels.csv                        file, subtotal, tax, total, menu_items
An empty cell in labels.csv means CORD has no label for that field on that
receipt. Treat it as "no label", not as "field absent from the receipt".
"""
import argparse
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data/"
FILES = {
    "validation": "validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "test": "test-00000-of-00001-9c204eb3f4e11791.parquet",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached file: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
    print()
    tmp.rename(dest)


def first_value(v) -> str:
    """CORD values are usually strings, sometimes lists. Return one string."""
    if isinstance(v, list):
        v = v[0] if v else ""
    return "" if v is None else str(v).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(FILES), required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--root", default="dev_data", help="output root (relative to cwd)")
    args = ap.parse_args()

    root = Path(args.root)
    parquet_path = root / "_cache" / FILES[args.split]
    download(BASE + FILES[args.split], parquet_path)

    out_dir = root / f"cord_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(parquet_path)
    rows = []
    count = 0
    for batch in pf.iter_batches(batch_size=8):
        for rec in batch.to_pylist():
            if count >= args.n:
                break
            img_bytes = rec["image"]["bytes"]
            try:
                fmt = (Image.open(io.BytesIO(img_bytes)).format or "PNG").lower()
            except Exception:
                fmt = "png"
            ext = "jpg" if fmt == "jpeg" else fmt
            stem = f"cord_{args.split}_{count:03d}"
            (out_dir / f"{stem}.{ext}").write_bytes(img_bytes)

            gt = json.loads(rec["ground_truth"])
            (out_dir / f"{stem}.gt.json").write_text(
                json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            parse = gt.get("gt_parse", {}) or {}
            sub = parse.get("sub_total", {}) or {}
            tot = parse.get("total", {}) or {}
            menu = parse.get("menu", [])
            rows.append(
                {
                    "file": f"{stem}.{ext}",
                    "subtotal": first_value(sub.get("subtotal_price")),
                    "tax": first_value(sub.get("tax_price")),
                    "total": first_value(tot.get("total_price")),
                    "menu_items": len(menu) if isinstance(menu, list) else 1,
                }
            )
            count += 1
        if count >= args.n:
            break

    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "subtotal", "tax", "total", "menu_items"]
        )
        w.writeheader()
        w.writerows(rows)

    have = {k: sum(1 for r in rows if r[k]) for k in ("subtotal", "tax", "total")}
    print(f"Exported {len(rows)} receipts to {out_dir}")
    print(f"Receipts with a label: {have}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
