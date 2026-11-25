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

#let guide-callout(kind, body) = {
  let style = (
    caution: (title: "注意", ink: rgb("#ad640f"), fill: rgb("#fff6e8")),
    tip: (title: "提示", ink: rgb("#167d83"), fill: rgb("#edf8f6")),
    note: (title: "注解", ink: rgb("#376f9f"), fill: rgb("#eef4fb")),
  ).at(kind)
  block(width: 100%, inset: 0pt, radius: 3pt, clip: true,
    fill: style.fill, stroke: 0.65pt + style.ink,
    above: 14pt, below: 14pt, breakable: false,
  )[
    #set par(justify: false, leading: 7.8pt, spacing: 7pt)
    #set text(size: 9.5pt)
    #set list(spacing: 6pt)
    #set enum(spacing: 6pt)
    #stack(dir: ttb, spacing: 0pt,
      block(width: 100%, inset: (x: 11pt, y: 5pt), fill: style.ink,
        text(size: 10pt, weight: "bold", fill: white, style.title + "：")),
      block(width: 100%, inset: (x: 11pt, y: 9pt), body),
    )
  ]
}

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

#page(margin: 0pt, fill: rgb("#f4f0e5"))[
  #let edge = 18mm
  #let ink = rgb("#292822")
  #let secondary = rgb("#57554c")
  #let song = "Noto Serif CJK SC"
  #let roman = "Cormorant Garamond"
  #let left-at(x, y, body) = place(top + left, dx: x, dy: y, body)
  #let right-at(y, body) = place(top + right, dx: -edge, dy: y, body)
  #let center-at(y, body) = place(top + center, dy: y, body)
  #let cover-rule = line(length: 210mm - 2 * edge, stroke: 0.5pt + rgb("#a79e8d"))
  #set text(font: "Noto Sans CJK SC", size: 11pt, fill: ink)
  #set par(justify: false, leading: 0.65em, spacing: 0pt)

  #left-at(edge - 0.2mm, 15.6mm, text(font: song, size: 18.8pt, weight: "bold", tracking: 0.5pt)[北极雄芯])
  #left-at(edge + 31.8mm, 17.7mm, text(font: roman, size: 13pt, weight: "semibold", tracking: 0.13pt, fill: secondary)[Polar Bear Tech])
  #right-at(18.7mm, text(font: "FreeSans", size: 8.6pt, tracking: 1.25pt, fill: rgb("#8a4b33"))[PROGRAMMING GUIDE])
  #left-at(edge, 28.8mm, cover-rule)

  #center-at(33.5mm, image("/assets/cover-art-v005.png", width: 193mm, height: 144.75mm, fit: "contain"))
  #center-at(184mm, text(font: roman, size: 66.5pt, weight: "medium", style: "italic", tracking: -0.65pt, fill: rgb("#583d2d"), guide-name))
  #center-at(209mm, text(font: song, size: 39pt, weight: "bold", tracking: 2.25pt)[编程指南])
  #center-at(230mm, text(size: 11.2pt, fill: secondary)[$subtitle$])

  #left-at(edge, 266.2mm, cover-rule)
  #left-at(edge, 272.6mm, text(size: 9.5pt, weight: "bold")[北极雄芯信息科技（西安）有限公司])
  #left-at(edge, 278.2mm, text(font: "FreeSans", size: 9pt, fill: secondary, "Polar Bear Tech (Xi'an) Co., Ltd"))
  #right-at(271.9mm, text(font: "Liberation Mono", size: 11.8pt, weight: "bold")[$version$])
  #right-at(278.2mm, text(size: 9.4pt, fill: secondary)[$date$])
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
