#!/usr/bin/env python3
"""Reject private work files and machine-specific data from the public tree."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = {"README.md", "OUTLINE.md", "CONTRIBUTING.md", "CHANGELOG.md", ".gitignore", ".gitattributes",
       "book.yaml", "requirements-review.txt", "Triton-Zeus-Programming-Guide.pdf",
       "pdf-build.json", "validation-summary.json", "RELEASE-REVIEW.md"}
DIRECTORIES = {"chapters", "assets", "examples", "layout", "scripts", "site", ".github"}
COVER_FONTS = {
    "assets/fonts/CormorantGaramond-MediumItalic.ttf": "432864a497463aab9cb5edd77291149969880763c7501b68a9f61af415adb17e",
    "assets/fonts/CormorantGaramond-Semibold.ttf": "184fbdd536c3a199eb276230491da9633fe19a6429e59067924f90ecf14782d2",
}
PATTERNS = {
    "machine path": r"/(?:data|home|mnt)/[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+",
    "private host": r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b",
    "access token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{32,})\b",
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "internal RTL tag": r"tag_PN\d+_[A-Za-z0-9_]+",
}


def public_files():
    files = [p for p in ROOT.iterdir() if p.name in TOP and p.is_file()]
    for directory in sorted(DIRECTORIES):
        files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    return sorted(str(p.relative_to(ROOT)) for p in files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked", action="store_true")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    if args.tracked or args.staged:
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().strip("\0").split("\0")
    else:
        names = public_files()
    errors = []
    for name in filter(None, names):
        path = Path(name)
        if name not in TOP and path.parts[0] not in DIRECTORIES:
            errors.append(f"File is outside the publication list: {name}")
            continue
        if path.suffix in {".pyc", ".pyo", ".so", ".a", ".bin", ".zip"} or "__pycache__" in path.parts:
            errors.append(f"Unexpected generated or binary file: {name}")
            continue
        data = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT) if args.staged else (ROOT / name).read_bytes()
        if path.suffix == ".pdf":
            if name != "Triton-Zeus-Programming-Guide.pdf":
                errors.append(f"Unexpected third-party or historical PDF: {name}")
            continue
        if name == "assets/cover-art-v005.png":
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append(f"Invalid PNG cover: {name}")
            continue
        if name in COVER_FONTS:
            if hashlib.sha256(data).hexdigest() != COVER_FONTS[name]:
                errors.append(f"Unexpected cover font contents: {name}")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"Unexpected non-text source: {name}")
            continue
        for label, pattern in PATTERNS.items():
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{name}:{line}: {label}")
    print(json.dumps({"files_checked": len(names), "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
