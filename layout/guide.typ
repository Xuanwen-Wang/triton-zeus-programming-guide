#let guide-name = "Triton Zeus"
#let blue = rgb("#155c99")
#let link-blue = rgb("#285979")
#let page-rule = line(length: 100%, stroke: 0.5pt)

#set document(title: [$title$])
#set text(font: ("Noto Sans CJK SC", "DejaVu Sans"), size: 10.5pt, lang: "zh", region: "CN")
#set par(justify: true, leading: 0.65em, spacing: 0.85em)
#set page(paper: "a4", margin: (x: 25.5mm, top: 25mm, bottom: 24mm), header-ascent: 28%, footer-descent: 27%)

#let folio(pattern) = context {
  page-rule
  v(2pt)
  set text(size: 8.8pt)
  grid(columns: (1fr, auto), [$title$], counter(page).display(pattern))
}

#page(margin: (x: 25.5mm, top: 29mm, bottom: 55mm))[
  #page-rule
  #v(7mm)
  #align(center, text(size: 49pt, weight: "bold", fill: blue, guide-name))
  #v(28mm)
  #align(right, text(size: 23pt, weight: "bold")[$title$])
  #v(4mm)
  #align(right, text(size: 14pt, weight: "bold")[$edition$])
  #v(1fr)
  #align(right, text(size: 10pt, weight: "bold")[$date$])
]

#set page(
  numbering: "i",
  header: [
    #text(size: 10.5pt, fill: blue, guide-name)
    #v(1pt)
    #page-rule
  ],
  footer: folio("i"),
)
#counter(page).update(1)
#show outline.entry: entry => {
  let chapter = entry.level == 1
  set text(size: 10pt, weight: if chapter { "bold" } else { "regular" })
  let prefix = entry.prefix()
  let number = if prefix != none { text(fill: link-blue, prefix) }
  let title = text(fill: link-blue, entry.body())
  let leaders = box(width: 1fr, repeat(gap: 4pt, justify: false)[.])
  let body = title + h(0.55em) + leaders + h(0.55em) + entry.page()
  block(
    above: if chapter { 10pt } else { 5pt },
    below: 0pt,
    breakable: false,
    link(entry.element.location(), entry.indented(number, body)),
  )
}
#align(right, text(size: 19pt, weight: "bold")[目录])
#outline(title: none, depth: 2, indent: 1.55em)

#let heading-label(item) = {
  if item.numbering != none {
    numbering(item.numbering, ..counter(heading).at(item.location()))
    h(0.7em)
  }
  item.body
}

#let running-header = context {
  let page-index = here().page()
  let chapters = query(heading.where(level: 1, outlined: true))
  let chapter = chapters.filter(item => item.location().page() <= page-index).last(default: none)
  let chapter-start = if chapter != none { chapter.location().page() } else { page-index }
  let sections = query(heading.where(level: 2, outlined: true)).filter(item => {
    let section-page = item.location().page()
    section-page >= chapter-start and section-page <= page-index
  })
  let first-on-page = sections.find(item => item.location().page() == page-index)
  let section = if first-on-page != none { first-on-page } else { sections.last(default: none) }
  set text(size: 8.8pt)
  grid(
    columns: (auto, 1fr),
    gutter: 12pt,
    if chapter != none { heading-label(chapter) } else { guide-name },
    align(right, if section != none { heading-label(section) }),
  )
  v(1pt)
  page-rule
}

#set page(numbering: "1", header: running-header, footer: folio("1"))
#counter(page).update(1)
#set heading(numbering: "1.1")
#show heading.where(level: 1, outlined: true): item => {
  pagebreak(weak: true)
  block(sticky: true, width: 100%, above: 10pt, below: 19pt, align(right)[
    #if item.numbering != none [
      #context text(size: 12pt, weight: "bold")[第 #counter(heading).at(item.location()).first() 章]
      #v(4pt)
    ]
    #text(size: 23pt, weight: "bold", item.body)
  ])
}
#show heading.where(level: 2): set text(size: 13pt, weight: "bold")
#show heading.where(level: 2): set block(above: 13pt, below: 6pt)
#show link: set text(fill: link-blue)

#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans Mono CJK SC"), size: 9pt)
#show raw.where(block: true): item => block(
  width: 100%, inset: 8pt, fill: rgb("#e1e1e1"),
  above: 9pt, below: 9pt, breakable: true, item,
)

#set table(inset: (x: 6pt, y: 5pt), stroke: 0.4pt + rgb("#777777"))
#show table.header: set text(weight: "bold")
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: image): set figure.caption(position: bottom)
#show figure.caption: set text(size: 9pt)

$body$
