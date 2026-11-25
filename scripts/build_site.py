#!/usr/bin/env python3
"""Publish the reviewed PDF with a single homepage entry."""

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
PDF = "Triton-Zeus-Programming-Guide.pdf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pdf = json.loads((ROOT / "pdf-build.json").read_text())
    if pdf["pdf"] != PDF or digest(ROOT / PDF) != pdf["sha256"]:
        raise SystemExit("The PDF does not match its build manifest")
    for name, expected in pdf["inputs"].items():
        if digest(ROOT / name) != expected:
            raise SystemExit(f"PDF source changed since build: {name}")

    template = (ROOT / "site/redirect.html").read_text()
    values = {"TARGET": PDF, "TITLE": "编程指南", "VERSION": html.escape(pdf["version"])}
    rendered = re.sub(r"\{\{([A-Z]+)\}\}", lambda m: values[m[1]], template)
    # Replace the generated directory so retired chapter pages and reader
    # assets cannot remain in the deployed publication.
    if OUT.exists():
        if OUT.is_symlink():
            raise SystemExit("Refusing to replace a symlinked publication directory")
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copyfile(ROOT / PDF, OUT / PDF)
    (OUT / ".nojekyll").write_text("")
    (OUT / "index.html").write_text(rendered)

    sources = dict(pdf["inputs"])
    for name in ("pdf-build.json", "scripts/build_site.py", "site/redirect.html"):
        sources[name] = digest(ROOT / name)
    manifest = {
        "version": pdf["version"], "mode": "pdf-only", "pdf": PDF,
        "pdf_pages": pdf["page_count"], "pdf_sha256": pdf["sha256"],
        "source_sha256": sources,
    }
    (OUT / "build-info.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
