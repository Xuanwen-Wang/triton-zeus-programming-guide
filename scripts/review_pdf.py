#!/usr/bin/env python3
"""Render every page and record structural checks against the exact PDF hash."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default="Triton-Zeus-Programming-Guide.pdf")
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()
    path = (ROOT / args.pdf).resolve()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    out = ROOT / "review" / f"render-{sha[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    document = fitz.open(path)
    reader = PdfReader(path, strict=True)
    pages = []
    failures = []
    total_links = 0
    for i, page in enumerate(document):
        image_path = out / f"page-{i+1:03}.png"
        page.get_pixmap(dpi=args.dpi, alpha=False).save(image_path)
        lines = [line for block in page.get_text("dict")["blocks"]
                 if "lines" in block for line in block["lines"]]
        spans = [span for line in lines for span in line["spans"]
                 if span["text"].strip()]
        outside = []
        outside_body = []
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            if x0 < 10 or x1 > page.rect.width-10 or y0 < 8 or y1 > page.rect.height-8:
                outside.append({"text": s["text"], "bbox": list(s["bbox"])})
            # CJK punctuation can legitimately hang beyond the 72 pt text edge.
            # Keep a small optical margin, but surface real column overflows.
            if i > 0 and 70 <= y0 < 775 and (x0 < 64 or x1 > page.rect.width-64):
                outside_body.append({"text": s["text"], "bbox": list(s["bbox"])})
        text = page.get_text()
        body = [s for s in spans if 68 <= s["bbox"][1] < 775]
        # A short chapter-ending page is a review flag, not an automatic failure.
        body_bottom = max((s["bbox"][3] for s in body), default=0)
        links = page.get_links()
        total_links += len(links)
        bad_links = [link for link in links if link["kind"] == fitz.LINK_GOTO
                     and not 0 <= link.get("page", -1) < len(document)]
        if outside or bad_links or not text.strip():
            failures.append(i+1)
        pages.append({
            "physical_page": i+1, "label": reader.page_labels[i],
            "image": str(image_path.relative_to(ROOT)),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "text_chars": len(text), "body_bottom_pt": round(body_bottom, 1),
            "short_page_flag": 0 < body_bottom < 440 and i > 0,
            "outside_page": outside, "broken_internal_links": bad_links,
            "outside_body_review_flags": outside_body,
            "first_lines": text.splitlines()[:5],
        })
    fonts = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    for start in range(0, len(pages), 4):
        sheet = Image.new("RGB", (1060, 1560), "#e5ebef")
        draw = ImageDraw.Draw(sheet)
        for pos, entry in enumerate(pages[start:start+4]):
            preview = Image.open(ROOT / entry["image"])
            preview.thumbnail((510, 724))
            x, y = 15+(pos % 2)*525, 38+(pos//2)*780
            sheet.paste(preview, (x, y))
            draw.text((x, y-28), f"PDF {entry['physical_page']} / {entry['label']}",
                      font=fonts, fill="#183d52")
        sheet.save(out / f"contact-{start+1:03}-{min(start+4,len(pages)):03}.png")
    text = "\n".join(page.get_text() for page in document)
    sensitive = re.findall(r"/data/[\w/.-]+|/home/[\w/.-]+|172\.16\.\d+\.\d+|TODO|TBD|待补", text)
    report = {
        "pdf": str(path.relative_to(ROOT)), "sha256": sha,
        "page_count": len(document), "toc_entries": len(document.get_toc()),
        "links": total_links, "structural_failure_pages": failures,
        "internal_path_or_placeholder_matches": sensitive,
        "pages": pages,
    }
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    (ROOT / "review/latest-render.json").write_text(json.dumps({
        "sha256": sha, "manifest": str((out/"manifest.json").relative_to(ROOT)),
        "page_count": len(document)}, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pages"},
                     ensure_ascii=False, indent=2))
    print(f"Page renders: {out}")


if __name__ == "__main__":
    main()
