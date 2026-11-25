#!/usr/bin/env python3
"""Verify the PDF publication, homepage and absence of the retired HTML reader."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"


class Redirect(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.refresh, self.canonical, self.links = [], [], []
        self.reader_content = False
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "refresh":
            self.refresh.append(attrs.get("content"))
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical.append(attrs.get("href"))
        if tag == "a":
            self.links.append(attrs.get("href"))
        if tag in {"article", "script", "iframe", "embed", "object"}:
            self.reader_content = True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = json.loads((SITE / "build-info.json").read_text())
    pdf = json.loads((ROOT / "pdf-build.json").read_text())
    errors = []
    if build["mode"] != "pdf-only" or build["version"] != pdf["version"]:
        errors.append("The publication mode or version is incorrect")
    if build["pdf_sha256"] != pdf["sha256"] or digest(SITE / build["pdf"]) != pdf["sha256"]:
        errors.append("Published PDF differs from the reviewed PDF")
    if digest(ROOT / build["pdf"]) != pdf["sha256"]:
        errors.append("Source PDF changed since build")
    for name, expected in build["source_sha256"].items():
        if digest(ROOT / name) != expected:
            errors.append(f"Publication source changed since build: {name}")
    for name, expected in pdf["inputs"].items():
        if digest(ROOT / name) != expected:
            errors.append(f"PDF input changed since build: {name}")

    page = Redirect(SITE / "index.html")
    target = build["pdf"]
    if page.refresh != ["0;url=" + target] or page.canonical != [target] or page.links != [target]:
        errors.append("Incorrect homepage destination or fallback link")
    if page.reader_content:
        errors.append("Unexpected HTML reader or embedded viewer")

    allowed = {"index.html", build["pdf"], "build-info.json", ".nojekyll"}
    actual = {p.relative_to(SITE).as_posix() for p in SITE.rglob("*") if p.is_file()}
    if actual != allowed:
        errors.append(f"Unexpected or missing publication files: {sorted(actual ^ allowed)}")
    print(json.dumps({"mode": build["mode"], "pdf_pages": pdf["page_count"],
                      "published_files": len(actual), "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
