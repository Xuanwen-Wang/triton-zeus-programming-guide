#let navy = rgb("#204d6c")
#let teal = navy
#let muted = rgb("#555555")
#let pale = rgb("#f1f1f1")
#let rule-color = rgb("#aaaaaa")
#let guide-name = "Triton Zeus"
#let figure-prefix = state("guide-figure-prefix", "0")
#let heading-label(item) = {
  if item.numbering != none {
    numbering(item.numbering, ..counter(heading).at(item.location()))
    h(0.6em)
  }
  item.body
}

#set document(title: [$title$], author: "Polar Bear Tech (Xi'an) Co., Ltd", description: [$subtitle$])
#set text(font: ("Noto Sans CJK SC", "DejaVu Sans"), size: 10pt, lang: "zh", region: "CN", fill: rgb("#111111"))
#set par(justify: true, leading: 10.5pt, spacing: 13pt)
#set page(paper: "a4", margin: (left: 25.4mm, right: 25.4mm, top: 25.4mm, bottom: 24.4mm), header-ascent: 35%, footer-descent: 35%)
#set list(indent: 0pt, body-indent: 0.8em, spacing: 12pt)
#set enum(indent: 0pt, body-indent: 0.65em, spacing: 11pt)
#show list: set par(justify: false, leading: 8.5pt, spacing: 10pt)
#show enum: set par(justify: false, leading: 8.5pt, spacing: 10pt)

#let folio(pattern) = context {
  line(length: 100%, stroke: 0.5pt + rule-color)
  v(3pt)
  set text(size: 8pt, fill: muted)
  grid(columns: (1fr, auto), text("Copyright © $copyright-year$ Polar Bear Tech (Xi'an) Co., Ltd."), counter(page).display(pattern))
}

#let guide-note(body) = block(
  width: 100%, inset: (x: 10pt, y: 8pt),
  fill: rgb("#f4f7fa"), stroke: (left: 2pt + navy),
  above: 13pt, below: 13pt, breakable: true,
)[#set text(size: 9.5pt); #set par(leading: 8pt, spacing: 10pt); #body]

#show link: set text(fill: teal)
#show raw: set text(font: ("Liberation Mono", "Noto Sans Mono CJK SC"), size: 9.2pt, fill: black)
#show raw.where(block: true): item => block(
  width: 100%, inset: (x: 8pt, y: 8pt), fill: pale,
  above: 13pt, below: 13pt,
  breakable: item.text.split("\n").len() > 28,
  { set par(justify: false, leading: 7.5pt, spacing: 0pt); item },
)
#set table(
  inset: (x: 6pt, y: 6pt),
  stroke: 0.35pt + rgb("#bcbcbc"),
  fill: (x, y) => if y == 0 { rgb("#e9eef2") } else { white },
)
#show table: set text(size: 9pt)
#show table: set par(justify: false, leading: 6.7pt, spacing: 6pt)
#show table.header: set text(weight: "bold")
#show figure.where(kind: table): set block(breakable: false)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: image): set figure.caption(position: bottom)
#show figure.where(caption: none): set figure(outlined: false)
#show figure.caption: set text(size: 9.2pt, fill: black)
#show figure.caption: set par(justify: false, leading: 6pt)
#set figure(gap: 9pt)
#show figure: set block(above: 15pt, below: 15pt)
#set figure(numbering: n => context {
  figure-prefix.get() + "." + str(n)
})

#page(margin: 0pt, fill: rgb("#102d42"), background:
  image("/assets/cover-art-v005.png", width: 100%, height: 100%, fit: "cover"))[
  #set par(justify: false, leading: 7pt, spacing: 0pt)
  #set text(fill: white)
  #place(top + left, dx: 25.4mm, dy: 21mm)[
    #text(size: 16pt, weight: "bold")[北极雄芯]
    #v(3mm)
    #text(size: 10pt, tracking: 0.25pt, fill: rgb("#c1d8e4"))[Polar Bear Tech]
  ]
  #place(top + left, dx: 25.4mm, dy: 65mm)[
    #block(width: 159mm)[
      #text(size: 9pt, tracking: 1.6pt, fill: rgb("#91c9df"))[PROGRAMMING GUIDE]
      #v(7mm)
      #text(size: 43pt, weight: "bold")[Triton Zeus]
      #v(6mm)
      #text(size: 29pt, weight: "bold")[编程指南]
      #v(8mm)
      #text(size: 10pt, fill: rgb("#c1d8e4"))[$subtitle$]
    ]
  ]
  #place(top + left, dx: 25.4mm, dy: 261mm)[
    #block(width: 159mm)[
      #line(length: 100%, stroke: 0.5pt + rgb("#52778b"))
      #v(5mm)
      #grid(columns: (1fr, auto), gutter: 20pt,
        text(size: 13pt, weight: "bold")[$version$],
        text(size: 10pt, fill: rgb("#c1d8e4"))[$date$],
      )
      #v(6mm)
      #text(size: 9pt)[北极雄芯信息科技（西安）有限公司]
      #v(2mm)
      #text(size: 8.5pt, fill: rgb("#c1d8e4"), "Polar Bear Tech (Xi'an) Co., Ltd")
    ]
  ]
  #box(width: 1pt, height: 1pt)
]

#set page(
  numbering: "i",
  header: [
    #text(size: 8.5pt, fill: muted)[TRITON ZEUS]
    #h(1fr)
    #text(size: 8.5pt, fill: muted)[$edition$]
    #v(2pt)
    #line(length: 100%, stroke: 0.45pt + rule-color)
  ],
  footer: folio("i"),
)
#counter(page).update(1)
#show heading.where(level: 1): set text(size: 21pt, weight: "bold", fill: navy)
#show heading.where(level: 2): item => block(sticky: true, above: 23pt, below: 14pt)[
  #text(size: 14pt, weight: "bold", fill: navy)[#heading-label(item)]
]
#show heading.where(level: 3): item => block(sticky: true, above: 18pt, below: 12pt)[
  #text(size: 12pt, weight: "bold", fill: navy)[#heading-label(item)]
]
#show heading.where(level: 4): item => block(sticky: true, above: 15pt, below: 10pt)[
  #text(size: 10pt, weight: "bold", fill: navy)[#heading-label(item)]
]
$for(include-before)$
$include-before$
$endfor$
#pagebreak()
#heading(outlined: false, bookmarked: true, numbering: none)[目录]
#v(3mm)
#show outline.entry: entry => context {
  let chapter = entry.element.func() == heading and entry.level == 1
  set text(size: if chapter { 10.3pt } else { 9.8pt },
    tracking: 0.015em, weight: if chapter { "bold" } else { "regular" })
  set par(justify: false, leading: 8pt, spacing: 0pt)
  // Resolve a figure's chapter prefix at its source, not at the list of figures.
  let prefix = if entry.element.func() == figure {
    let item = entry.element
    let loc = item.location()
    let supplement = if item.kind == table { "表" } else { "图" }
    let n = counter(figure.where(kind: item.kind)).at(loc).first()
    [#supplement #figure-prefix.at(loc).#n]
  } else { entry.prefix() }
  let number = if prefix != none { text(fill: teal, prefix) }
  let title = text(fill: if chapter { navy } else { rgb("#263b48") }, entry.body())
  let leaders = box(width: 1fr, repeat(gap: 3.5pt, justify: false,
    text(size: 7pt, fill: rgb("#9baeba"))[·]))
  let body = title + h(0.55em) + leaders + h(0.55em) + entry.page()
  block(above: if chapter { 15pt } else { 8pt }, below: 0pt, breakable: false,
    sticky: entry.element.func() == heading and entry.level < 3,
    link(entry.element.location(), entry.indented(number, body)))
}
#outline(title: none, depth: 3, indent: 1.6em)
#pagebreak()
#heading(outlined: false, bookmarked: true, numbering: none)[插图目录]
#v(3mm)
#outline(title: none, target: figure.where(kind: image))
#pagebreak()
#heading(outlined: false, bookmarked: true, numbering: none)[表格目录]
#v(3mm)
#outline(title: none, target: figure.where(kind: table))
#pagebreak()

#let running-header = context {
  let p = here().page()
  let chapters = query(heading.where(level: 1, outlined: true))
  let chapter = chapters.filter(h => h.location().page() <= p).last(default: none)
  let chapter-page = if chapter != none { chapter.location().page() } else { p }
  let sections = query(heading.where(level: 2, outlined: true)).filter(h => h.location().page() >= chapter-page and h.location().page() <= p)
  let eligible = sections.filter(h => h.location().page() < p or h.location().position().y < 110pt)
  let section = eligible.last(default: none)
  set text(size: 8.5pt, fill: black)
  grid(columns: (1fr, 1fr), gutter: 10pt,
    if chapter != none { heading-label(chapter) } else { guide-name },
    align(right, if section != none { heading-label(section) }),
  )
  v(2pt)
  line(length: 100%, stroke: 0.45pt + rule-color)
}
#set page(numbering: "1", header: running-header, footer: folio("1"))
#counter(page).update(1)
#counter(heading).update(0)
#set heading(numbering: "1.1")
#show heading.where(level: 1, outlined: true): item => {
  pagebreak(weak: true)
  if item.numbering != none {
    context figure-prefix.update(str(counter(heading).at(item.location()).first()))
  } else {
    figure-prefix.update("附")
  }
  counter(figure.where(kind: image)).update(0)
  counter(figure.where(kind: table)).update(0)
  v(25mm)
  block(sticky: true, width: 100%, below: 30pt)[
    #align(right)[
      #text(size: 21pt, weight: "bold", fill: navy)[#heading-label(item)]
    ]
  ]
}

$body$
