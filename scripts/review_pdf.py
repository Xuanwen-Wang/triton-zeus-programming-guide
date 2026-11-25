#!/usr/bin/env python3
"""Render every page and record structural checks against the exact PDF hash."""

import argparse
import hashlib
from io import BytesIO
import json
import re
from pathlib import Path
from zipfile import ZipFile

import fitz
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from pypdf.generic import ArrayObject, ByteStringObject, DictionaryObject, IndirectObject, TextStringObject

ROOT = Path(__file__).resolve().parents[1]


def string_bytes(value):
    return bytes(value) if isinstance(value, ByteStringObject) else value.original_bytes


def named_destinations(node):
    if node is None:
        return
    node = node.get_object()
    for child in node.get("/Kids", ()):
        yield from named_destinations(child)
    pairs = node.get("/Names", ())
    for name, destination in zip(pairs[::2], pairs[1::2], strict=True):
        destination = destination.get_object()
        if isinstance(destination, DictionaryObject):
            destination = destination["/D"]
        yield string_bytes(name), destination


def invalid_destinations(reader):
    root = reader.trailer["/Root"]
    tree = root["/Names"].get("/Dests") if "/Names" in root else None
    names = dict(named_destinations(tree))
    pages = {(page.indirect_reference.idnum, page.indirect_reference.generation)
             for page in reader.pages}
    failures = {}
    for number, page in enumerate(reader.pages, 1):
        for annotation in page.get("/Annots", ()):
            link = annotation.get_object()
            if link.get("/Subtype") != "/Link":
                continue
            destination = link.get("/Dest")
            action = link["/A"] if "/A" in link else {}
            if destination is None and action.get("/S") == "/GoTo":
                destination = action["/D"]
            if destination is None:
                continue
            destination = destination.get_object()
            if isinstance(destination, (ByteStringObject, TextStringObject)):
                destination = names.get(string_bytes(destination))
            target = destination[0] if isinstance(destination, ArrayObject) and destination else None
            if not isinstance(target, IndirectObject) or (target.idnum, target.generation) not in pages:
                failures.setdefault(number, []).append(annotation.idnum)
    return failures


def check_example_bundle(reader, pdf_sha):
    build = json.loads((ROOT / "pdf-build.json").read_text())
    bundle = build.get("example_bundle")
    if pdf_sha != build["sha256"] or bundle is None:
        return None
    name = bundle["name"]
    attachments = reader.attachments
    if list(attachments) != [name] or len(attachments[name]) != 1:
        raise SystemExit("Unexpected PDF attachment set")
    data = attachments[name][0]
    if hashlib.sha256(data).hexdigest() != bundle["sha256"]:
        raise SystemExit("The PDF attachment does not match its manifest")
    with ZipFile(BytesIO(data)) as archive:
        if archive.namelist() != bundle["files"]:
            raise SystemExit("Unexpected example bundle contents")
        for path in bundle["files"]:
            if hashlib.sha256(archive.read(path)).hexdigest() != build["inputs"][path]:
                raise SystemExit(f"Example attachment source differs from manifest: {path}")
    return bundle


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
    broken = invalid_destinations(reader)
    bundle = check_example_bundle(reader, sha)
    outlines = document.get_toc()
    navigation = []
    for title in ("目录", "插图目录", "表格目录"):
        matches = [item for item in outlines if item[1] == title]
        if len(matches) != 1:
            raise SystemExit(f"Expected one PDF bookmark for {title}: {matches}")
        level, _, physical_page = matches[0]
        if level != 1 or title not in document[physical_page - 1].get_text():
            raise SystemExit(f"Invalid navigation bookmark target: {matches[0]}")
        navigation.append({"title": title, "physical_page": physical_page})
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
        bad_links.extend({"invalid_annotation": item} for item in broken.get(i + 1, ()))
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
        "navigation_bookmarks": navigation,
        "links": total_links, "structural_failure_pages": failures,
        "example_bundle": bundle,
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
