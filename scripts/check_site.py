#!/usr/bin/env python3
"""Check every local website link, anchor, asset and published PDF identity."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.ids, self.references, self.duplicates = set(), [], []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.references.append(attrs[key])


def main():
    pages = {path: Page(path) for path in SITE.glob("*.html")}
    errors, total = [], 0
    for path, page in pages.items():
        errors.extend(f"{path.name}: duplicate id {identifier}" for identifier in page.duplicates)
        for reference in page.references:
            parts = urlsplit(reference)
            if parts.scheme or parts.netloc:
                continue
            total += 1
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if not target.is_relative_to(SITE) or not target.is_file():
                errors.append(f"{path.name}: missing or invalid target {reference}")
            elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                errors.append(f"{path.name}: missing anchor {reference}")
    build = json.loads((SITE / "build-info.json").read_text())
    for name, expected in build["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            errors.append(f"Website source changed since build: {name}")
    if hashlib.sha256((SITE / "Triton-Zeus-Programming-Guide.pdf").read_bytes()).hexdigest() != build["pdf_sha256"]:
        errors.append("Website PDF differs from its manifest")
    if (ROOT / "pdf-build.json").is_file():
        pdf = json.loads((ROOT / "pdf-build.json").read_text())
        if pdf["sha256"] != build["pdf_sha256"]:
            errors.append("PDF build identity differs from website PDF")
        for name, expected in pdf["inputs"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                errors.append(f"PDF input changed since build: {name}")
    for entry in json.loads((SITE / "search-index.json").read_text()):
        parts = urlsplit(entry["url"])
        target = SITE / unquote(parts.path)
        if target not in pages or unquote(parts.fragment) not in pages[target].ids:
            errors.append(f"Search result is not reachable: {entry['url']}")
    print(json.dumps({"html_pages": len(pages), "local_links": total, "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
