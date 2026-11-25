-- Keep asset paths stable when the generated Typst file lives in build/.
function Span(span)
  if FORMAT == 'typst' and span.classes:includes('company-literal') then
    local text = pandoc.utils.stringify(span):gsub('’', "'"):gsub('\\', '\\\\'):gsub('"', '\\"')
    return pandoc.RawInline('typst', '#text("' .. text .. '")')
  end
  return span
end

function Image(image)
  if FORMAT == 'typst' and image.src:match('^assets/') then
    image.src = '/' .. image.src
  end
  return image
end

function Table(tbl)
  if FORMAT == 'typst' and pandoc.utils.stringify(tbl.caption.long)
      == '编译策略的默认行为与使用前提' then
    tbl.colspecs = {
      {pandoc.AlignCenter, 0.37},
      {pandoc.AlignCenter, 0.30},
      {pandoc.AlignCenter, 0.33}
    }
  end
  return tbl
end

function Div(div)
  if FORMAT == 'typst' and div.classes:includes('keep') then
    local content = pandoc.write(pandoc.Pandoc(div.content), 'typst')
    return pandoc.RawBlock('typst', '#block(breakable: false)[\n' .. content .. '\n]')
  end
  if FORMAT == 'typst' and div.classes:includes('note') then
    local content = pandoc.write(pandoc.Pandoc(div.content), 'typst')
    return pandoc.RawBlock('typst', '#guide-note[\n' .. content .. '\n]')
  end
end

function Header(header)
  local appendix = {
    ['api-support'] = 'A', ['options'] = 'B',
    ['glossary'] = 'C', ['versions-and-examples'] = 'D'
  }
  local prefix = appendix[header.identifier]
  if FORMAT == 'typst' and header.level == 2 and prefix then
    return {
      pandoc.RawBlock('typst', '#figure-prefix.update("' .. prefix .. '")\n'
        .. '#counter(figure.where(kind: table)).update(0)\n'
        .. '#counter(figure.where(kind: image)).update(0)'),
      header
    }
  end
  return header
end
