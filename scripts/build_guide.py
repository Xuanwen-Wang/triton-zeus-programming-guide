#!/usr/bin/env python3
"""Build the guide locally and bind its inputs to a reproducible manifest."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    bundled = ROOT / ".tools/typst-0.14.2/typst"
    default_typst = str(bundled) if bundled.is_file() else "typst"
    parser.add_argument("--typst", default=os.environ.get("TYPST", default_typst))
    parser.add_argument("--output", default="Triton-Zeus-Programming-Guide.pdf")
    args = parser.parse_args()
    typst = shutil.which(args.typst)
    if not typst:
        raise SystemExit("Typst not found; set TYPST or pass --typst /path/to/typst")
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    subprocess.run(["python3", "scripts/draw_figures.py"], cwd=ROOT, check=True)
    subprocess.run(["python3", "scripts/draw_architecture.py"], cwd=ROOT, check=True)
    subprocess.run(["python3", "scripts/draw_interfaces.py"], cwd=ROOT, check=True)
    subprocess.run(["python3", "scripts/draw_practice_layouts.py"], cwd=ROOT, check=True)
    subprocess.run(["python3", "scripts/draw_ldg.py"], cwd=ROOT, check=True)
    front = [ROOT / "chapters/00-copyright.md", ROOT / "chapters/00-preface.md"]
    # Pandoc already owns YAML parsing; consume its typed metadata instead of
    # introducing a second YAML parser with different quoting behavior.
    metadata = json.loads(subprocess.check_output([
        "pandoc", "--from=markdown", "--to=json", "--metadata-file=book.yaml"
    ], input="", text=True, cwd=ROOT))["meta"]
    entries = metadata["chapters"]["c"]
    names = []
    for entry in entries:
        if entry["t"] == "MetaString":
            names.append(entry["c"])
        elif entry["t"] == "MetaInlines" and all(v["t"] == "Str" for v in entry["c"]):
            names.append("".join(v["c"] for v in entry["c"]))
        else:
            raise SystemExit("Each chapters entry must be a plain relative file path")
    chapters = [ROOT / name for name in names]
    if not chapters:
        raise SystemExit("book.yaml must list at least one chapter")
    if len(set(chapters)) != len(chapters) or not all(p.is_file() for p in chapters):
        raise SystemExit("Chapter paths must be unique existing files")
    chapter_paths = front + chapters
    common = ["pandoc", "--from=markdown+east_asian_line_breaks", "--to=typst",
              "--lua-filter=layout/guide.lua", "--fail-if-warnings"]
    frontmatter = [subprocess.check_output(common + [str(p)], cwd=ROOT, text=True)
                   for p in front]
    (build / "frontmatter.typ").write_text("\n#pagebreak()\n".join(frontmatter))
    command = common + [str(p) for p in chapters] + [
        "--standalone", "--number-sections", "--metadata-file=book.yaml",
        "--template=layout/book.typ", "--include-before-body=build/frontmatter.typ",
        "--output=build/guide.typ",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    epoch_match = re.search(r"^source-date-epoch:\s*(\d+)\s*$",
                            (ROOT/"book.yaml").read_text(), re.MULTILINE)
    if not epoch_match:
        raise SystemExit("book.yaml must declare source-date-epoch for reproducible metadata")
    result = subprocess.run([typst, "compile", "--root", str(ROOT),
                             "--creation-timestamp", epoch_match.group(1),
                             "build/guide.typ", str(output)],
                            cwd=ROOT, capture_output=True, text=True)
    (build / "typst.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise SystemExit(result.stderr)
    if result.stderr.strip():
        print(result.stderr)
    # Preserve Typst's destinations and outlines, and request the outline pane
    # when a PDF reader supports the catalog's PageMode preference.
    import pypdf
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter(clone_from=PdfReader(output))
    writer.page_mode = "/UseOutlines"
    writer.page_layout = "/OneColumn"
    staged = output.with_suffix(".navigation.pdf")
    with staged.open("wb") as stream:
        writer.write(stream)
    staged.replace(output)
    inputs = [ROOT / "book.yaml", ROOT / "layout/book.typ", ROOT / "layout/guide.lua",
              ROOT / "scripts/draw_figures.py", ROOT / "scripts/draw_architecture.py",
              ROOT / "scripts/draw_interfaces.py",
              ROOT / "scripts/draw_practice_layouts.py",
              ROOT / "scripts/draw_ldg.py",
              ROOT / "scripts/build_guide.py"]
    inputs += chapter_paths + sorted(p for p in (ROOT / "assets").iterdir()
                                    if p.suffix in {".svg", ".png", ".jpg"})
    inputs += sorted((ROOT / "examples").glob("*.py"))
    if (ROOT / "validation-summary.json").is_file():
        inputs.append(ROOT / "validation-summary.json")
    final_reader = PdfReader(output)
    manifest = {
        "pdf": str(output.relative_to(ROOT)), "sha256": digest(output),
        "version": re.search(r"^version:\s*(v[0-9.]+)\s*$",
                             (ROOT / "book.yaml").read_text(), re.MULTILINE).group(1),
        "page_count": len(final_reader.pages),
        "pandoc": subprocess.check_output(["pandoc", "--version"], text=True).splitlines()[0],
        "typst": subprocess.check_output([typst, "--version"], text=True).strip(),
        "pypdf": pypdf.__version__,
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
    }
    (build / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    (ROOT / "pdf-build.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
