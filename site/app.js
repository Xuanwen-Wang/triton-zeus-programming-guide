/* Progressive enhancements: the complete guide remains readable without JS. */
const menu = document.querySelector('#menu-toggle');
const sidebar = document.querySelector('#sidebar');
menu.addEventListener('click', () => {
  const open = sidebar.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
  menu.setAttribute('aria-label', open ? '关闭章节目录' : '打开章节目录');
});
document.addEventListener('click', event => {
  if (!sidebar.contains(event.target) && !menu.contains(event.target)) {
    sidebar.classList.remove('open');
    menu.setAttribute('aria-expanded', 'false');
  }
});
document.querySelectorAll('article table').forEach(table => {
  if (table.parentElement.classList.contains('table-scroll')) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'table-scroll';
  wrapper.tabIndex = 0;
  wrapper.setAttribute('role', 'region');
  wrapper.setAttribute('aria-label', table.caption?.textContent || '表格，可横向滚动');
  table.before(wrapper);
  wrapper.append(table);
});
document.querySelectorAll('div.sourceCode').forEach(block => {
  const code = block.querySelector('pre code');
  if (!code) return;
  const button = document.createElement('button');
  button.className = 'copy-code';
  button.type = 'button';
  button.textContent = '复制';
  button.setAttribute('aria-label', '复制代码');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = '已复制';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      button.textContent = '已选中，请复制';
    }
    setTimeout(() => { button.textContent = '复制'; }, 2000);
  });
  block.append(button);
});
const figureDialog = document.querySelector('#figure-dialog');
document.querySelectorAll('article figure img').forEach(img => {
  const open = () => {
    figureDialog.querySelector('img').src = img.src;
    figureDialog.querySelector('img').alt = img.alt;
    figureDialog.querySelector('p').textContent = img.closest('figure').querySelector('figcaption')?.textContent || img.alt;
    figureDialog.showModal();
  };
  img.tabIndex = 0;
  img.setAttribute('role', 'button');
  img.setAttribute('aria-label', `放大插图：${img.alt}`);
  img.addEventListener('click', open);
  img.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
  });
});
document.querySelectorAll('dialog').forEach(dialog => {
  dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
});
const searchDialog = document.querySelector('#search-dialog');
const input = document.querySelector('#search-input');
const results = document.querySelector('#search-results');
const status = document.querySelector('#search-status');
let indexPromise;
const loadIndex = () => indexPromise ||= fetch('search-index.json').then(response => {
  if (!response.ok) throw new Error('Search index unavailable');
  return response.json();
});
function openSearch() {
  searchDialog.showModal();
  input.focus();
  loadIndex().catch(() => { status.textContent = '搜索索引加载失败，请稍后重试或使用章节目录。'; indexPromise = undefined; });
}
document.querySelector('#search-open').addEventListener('click', openSearch);
document.addEventListener('keydown', event => {
  if (event.key === '/' && !event.ctrlKey && !event.metaKey && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
    event.preventDefault();
    if (!searchDialog.open) openSearch();
  }
});
input.addEventListener('input', async () => {
  const query = input.value.trim().toLocaleLowerCase();
  results.replaceChildren();
  if (!query) { status.textContent = '搜索章节标题与正文'; return; }
  try {
    const index = await loadIndex();
    if (input.value.trim().toLocaleLowerCase() !== query) return;
    const words = query.split(/\s+/).filter(Boolean);
    const matches = index.map(entry => {
      const title = entry.title.toLocaleLowerCase();
      const text = entry.text.toLocaleLowerCase();
      const haystack = title + ' ' + text;
      const matched = words.every(word => haystack.includes(word));
      const score = words.reduce((sum, word) => sum + (title.includes(word) ? 20 : 0), 0);
      return {entry, matched, score};
    }).filter(x => x.matched).sort((a,b) => b.score-a.score);
    status.textContent = `找到 ${matches.length} 个相关小节${matches.length > 20 ? '，显示前 20 项' : ''}`;
    matches.slice(0,20).forEach(({entry}) => {
      const a = document.createElement('a'); a.href = entry.url;
      const small = document.createElement('small'); small.textContent = entry.chapter;
      const title = document.createElement('strong'); title.textContent = entry.title;
      const excerpt = document.createElement('p');
      const position = Math.max(0, entry.text.toLocaleLowerCase().indexOf(words[0]) - 35);
      excerpt.textContent = (position ? '…' : '') + entry.text.slice(position, position + 150).replace(/\s+/g,' ') + (entry.text.length > position + 150 ? '…' : '');
      a.append(small,title,excerpt); results.append(a);
    });
  } catch { status.textContent = '搜索索引加载失败，请稍后重试或使用章节目录。'; indexPromise = undefined; }
});
const tocLinks = [...document.querySelectorAll('.page-toc a')];
const sections = tocLinks.map(link => document.getElementById(decodeURIComponent(link.hash.slice(1)))).filter(Boolean);
const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(entry => entry.isIntersecting);
  if (!visible.length) return;
  const id = visible[0].target.id;
  tocLinks.forEach(link => link.classList.toggle('active', decodeURIComponent(link.hash.slice(1)) === id));
}, {rootMargin: '-85px 0px -65% 0px'});
sections.forEach(section => observer.observe(section));
document.querySelector('.sidebar [aria-current]')?.scrollIntoView({block:'nearest'});
