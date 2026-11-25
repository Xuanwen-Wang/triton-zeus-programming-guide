#!/usr/bin/env python3
"""Build a self-contained, searchable website from the PDF's Markdown sources."""

import copy
import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
REPO = "https://github.com/Xuanwen-Wang/triton-zeus-programming-guide"
PDF = "Triton-Zeus-Programming-Guide.pdf"


def plain(value):
    if isinstance(value, list):
        return "".join(plain(v) for v in value)
    if not isinstance(value, dict):
        return ""
    kind, content = value.get("t"), value.get("c", [])
    if kind == "Str":
        return content
    if kind == "MetaString":
        return content
    if kind in {"Space", "SoftBreak", "LineBreak"}:
        return " "
    if kind in {"Code", "CodeBlock", "Math"}:
        return content[1]
    if kind == "Header":
        return plain(content[2])
    if kind in {"Image", "Link", "Span"}:
        return plain(content[1])
    return plain(content)


def walk(value, action):
    if isinstance(value, dict):
        action(value)
        for child in value.values():
            walk(child, action)
    elif isinstance(value, list):
        for child in value:
            walk(child, action)


def main():
    metadata = json.loads(subprocess.check_output([
        "pandoc", "-f", "markdown", "-t", "json", "--metadata-file=book.yaml"
    ], input="", text=True, cwd=ROOT))["meta"]
    version = plain(metadata["version"])
    names = [plain(x) for x in metadata["chapters"]["c"]]
    sources = ["chapters/00-copyright.md", "chapters/00-preface.md"] + names
    doc = json.loads(subprocess.check_output([
        "pandoc", "-f", "markdown+east_asian_line_breaks", "-t", "json", *sources
    ], text=True, cwd=ROOT))
    # Print-only keep groups must not hide headings from web numbering/search.
    def unwrap_keep(blocks):
        for block in blocks:
            if block["t"] == "Div" and "keep" in block["c"][0][1]:
                yield from unwrap_keep(block["c"][1])
            else:
                yield block
    doc["blocks"] = list(unwrap_keep(doc["blocks"]))
    source_headers = []
    walk(doc["blocks"], lambda node: source_headers.append(node) if node.get("t") == "Header" else None)
    pages, current = [], None
    counters = [0] * 6
    for block in doc["blocks"]:
        if block["t"] == "Header":
            level, attr, inlines = block["c"]
            if "unnumbered" not in attr[1]:
                counters[level - 1] += 1
                counters[level:] = [0] * (6 - level)
                number = ".".join(str(n) for n in counters[:level])
                inlines[:0] = [{"t": "Str", "c": number}, {"t": "Space"}]
            if level == 1:
                current = {"id": attr[0], "title": plain(inlines), "blocks": [],
                           "headings": [], "source": sources[len(pages)],
                           "prefix": str(counters[0]), "figures": 0, "tables": 0}
                pages.append(current)
            if level <= 3:
                current["headings"].append((level, attr[0], plain(inlines)))
            if current["id"] == "appendices" and level == 2:
                current["prefix"] = plain(inlines)[0]
                current["figures"] = current["tables"] = 0
        if current is None:
            raise SystemExit("Every source must start with a level-one heading")

        def caption(node):
            if node.get("t") in {"Figure", "Table"}:
                caption_blocks = node["c"][1][1]
                if caption_blocks:
                    key = "figures" if node["t"] == "Figure" else "tables"
                    current[key] += 1
                    label = "图" if key == "figures" else "表"
                    label += f" {current['prefix']}.{current[key]}　"
                    caption_blocks[0]["c"].insert(0, {"t": "Str", "c": label})
        walk(block, caption)
        current["blocks"].append(block)

    ids, images = {}, set()
    for page in pages:
        def collect(node):
            if node.get("t") == "Header":
                identifier = node["c"][1][0]
                if identifier in ids:
                    raise SystemExit(f"Duplicate heading ID: {identifier}")
                ids[identifier] = page["id"] + ".html"
            if node.get("t") == "Image":
                images.add(node["c"][2][0])
        walk(page["blocks"], collect)

    OUT.mkdir(exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    for asset in images:
        path = (ROOT / asset).resolve()
        if not path.is_relative_to(ROOT / "assets") or not path.is_file():
            raise SystemExit(f"Invalid figure path: {asset}")
        shutil.copyfile(path, OUT / asset)
    for asset in (ROOT / "site").glob("*"):
        if asset.suffix in {".css", ".js", ".svg"}:
            shutil.copyfile(asset, OUT / asset.name)
    shutil.copyfile(ROOT / PDF, OUT / PDF)
    (OUT / ".nojekyll").write_text("")
    template = (ROOT / "site/page.html").read_text()

    def link(page, current_id=None):
        active = ' aria-current="page"' if page["id"] == current_id else ""
        return f'<a href="{page["id"]}.html"{active}>{html.escape(page["title"])}</a>'

    def render(title, body, current_id="", toc="", pager="", source="README.md"):
        values = {
            "TITLE": html.escape(title), "VERSION": version, "BODY": body,
            "NAV": "\n".join(link(p, current_id) for p in pages),
            "TOC": toc, "PAGER": pager, "REPO": REPO, "PDF": PDF,
            "SOURCE": REPO + "/blob/main/" + source,
        }
        return re.sub(r"\{\{([A-Z]+)\}\}", lambda m: values[m[1]], template)

    search = []
    for index, page in enumerate(pages):
        blocks = copy.deepcopy(page["blocks"])

        def rewrite(node):
            if node.get("t") == "Link":
                target = node["c"][2][0]
                if target.startswith("#"):
                    identifier = target[1:]
                    if identifier not in ids:
                        raise SystemExit(f"Unknown cross-reference: {target}")
                    node["c"][2][0] = ids[identifier] + target
                elif target.startswith("examples/"):
                    node["c"][2][0] = REPO + "/blob/main/" + target
        walk(blocks, rewrite)
        converted = {"pandoc-api-version": doc["pandoc-api-version"], "meta": {}, "blocks": blocks}
        body = subprocess.check_output([
            "pandoc", "-f", "json", "-t", "html5", "--mathml", "--wrap=none"
        ], input=json.dumps(converted, ensure_ascii=False), text=True)
        body = re.sub(r"(<table\b.*?</table>)", r'<div class="table-scroll" tabindex="0" role="region" aria-label="表格，可横向滚动">\1</div>', body, flags=re.S)
        toc = '<p class="nav-label">本页内容</p>' + "".join(
            f'<a class="toc-level-{level}" href="#{identifier}">{html.escape(title)}</a>'
            for level, identifier, title in page["headings"] if level > 1)
        neighbors = []
        for offset, label in [(-1, "上一章"), (1, "下一章")]:
            ni = index + offset
            if 0 <= ni < len(pages):
                neighbor = pages[ni]
                neighbors.append(f'<a href="{neighbor["id"]}.html"><span>{label}</span>'
                                 f'{html.escape(neighbor["title"])}</a>')
        pager = '<nav class="pager" aria-label="相邻章节">' + "".join(neighbors) + "</nav>"
        (OUT / (page["id"] + ".html")).write_text(render(
            page["title"], body, page["id"], toc, pager, page["source"]))
        section = None
        for block in page["blocks"]:
            if block["t"] == "Header":
                level, attr, _ = block["c"]
                section = {"title": plain(block), "chapter": page["title"],
                           "url": page["id"] + ".html#" + attr[0], "text": ""}
                search.append(section)
            elif section is not None:
                section["text"] += plain(block) + "\n"

    if len(search) != len(source_headers):
        raise SystemExit("Every source heading must be numbered and included in search")
    contents = '<div class="chapter-list">' + "".join(link(p) for p in pages[2:]) + "</div>"
    home = f'''<div class="book-intro"><p class="eyebrow">POLAR BEAR TECH · 开发者文档</p>
<h1>Triton Zeus<br>编程指南</h1><p class="lead">从第一个算子到 LDG 近存编程，理解 Zeus V3 上的数据布局、计算组织与性能优化。</p>
<div class="intro-actions"><a class="primary" href="quick-start.html">开始编写第一个算子 <span aria-hidden="true">→</span></a>
<a class="secondary" href="{PDF}">下载 PDF</a></div>
<p class="edition">{version} · Zeus V3 · 中文</p></div>
<h2 id="reading">阅读路线</h2><div class="reading-routes">
<a href="quick-start.html"><strong>初次使用</strong><span>环境、向量加法与运行检查</span></a>
<a href="hardware-model.html"><strong>理解硬件</strong><span>执行单元、L2 Buffer 与 LDG</span></a>
<a href="api-reference.html"><strong>查阅接口</strong><span>参数、语义、约束与示例</span></a>
</div><h2 id="contents">全书目录</h2>{contents}
<div class="note"><p>网页版与 PDF 使用同一份 Markdown 正文。示例程序和版本记录随仓库维护；
算力与带宽的条件、接口的适用范围见对应章节。</p></div>'''
    # Resolve landing-page entry points from the source identities.
    for source, placeholder in [("chapters/03-hardware-model.md", "hardware-model.html"),
                                ("chapters/05a-api-reference.md", "api-reference.html")]:
        actual = next(p["id"] + ".html" for p in pages if p["source"] == source)
        home = home.replace(placeholder, actual)
    (OUT / "index.html").write_text(render("Triton Zeus 编程指南", home))
    (OUT / "search-index.json").write_text(json.dumps(search, ensure_ascii=False, separators=(",", ":")))
    manifest = {"version": version, "pages": len(pages) + 1, "search_sections": len(search),
                "figures": len(images), "pdf_sha256": hashlib.sha256((ROOT / PDF).read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    (OUT / "build-info.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
