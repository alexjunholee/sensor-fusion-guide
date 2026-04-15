#!/usr/bin/env python3
"""
Build script: assemble chapter_*.md files into a single guide.html.
Template based on robotics-practice/guide.html with KaTeX math support added.
Usage: python3 scripts/build_guide.py
"""

import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(PROJECT_ROOT, "guide.html")

def read_chapters():
    """Read all chapter markdown files in order."""
    pattern = os.path.join(PROJECT_ROOT, "chapter_*.md")
    files = sorted(glob.glob(pattern))
    parts = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            content = fh.read()
        parts.append(content)
    return "\n\n---\n\n".join(parts)

# Chapter groupings for TOC sidebar labels
# Ch.1-3: 기초, Ch.4-5: 방법론 기초, Ch.6-8: Odometry & Fusion,
# Ch.9-10: Loop Closure, Ch.11-13: 표현·실전·미래
TOC_GROUPS_JS = """\
    var groupFirstChapter = {
      '1.1': '기초 (Foundations)',
      '4.1': '방법론 기초 (Methods)',
      '6.1': 'Odometry & Fusion',
      '9.1': 'Place Recognition & Loop Closure',
      '11.1': '표현 · 실전 · 미래'
    };
"""

def build_html(md_content):
    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>센서 퓨전 심화 가이드</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg: #1a1a2e;
  --bg-sidebar: #151528;
  --bg-code: #0d1117;
  --bg-blockquote: #1e1e3a;
  --text: #e0e0e0;
  --text-muted: #9a9ab0;
  --text-heading: #ffffff;
  --accent: #4a6fa5;
  --link: #6fa8dc;
  --border: #2a2a4a;
  --sidebar-width: 280px;
  --topbar-height: 50px;
  --progress-height: 3px;
  --table-even: #1e1e38;
  --table-odd: #16162e;
  --table-header: #252545;
}}

html {{ scroll-behavior: smooth; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-size: 16px;
  line-height: 1.7;
  color: var(--text);
  background: var(--bg);
  overflow-x: hidden;
}}

/* Progress bar */
#progress-bar {{
  position: fixed;
  top: 0;
  left: 0;
  height: var(--progress-height);
  background: linear-gradient(90deg, #4a6fa5, #6fa8dc, #8fcbfa);
  z-index: 10000;
  transition: width 0.1s linear;
  width: 0%;
}}

/* Top bar */
#topbar {{
  position: fixed;
  top: var(--progress-height);
  left: 0;
  right: 0;
  height: var(--topbar-height);
  background: var(--bg-sidebar);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  z-index: 1000;
}}

#topbar .menu-toggle {{
  display: none;
  background: none;
  border: none;
  color: var(--text);
  font-size: 24px;
  cursor: pointer;
  margin-right: 12px;
  padding: 4px 8px;
  border-radius: 4px;
}}

#topbar .menu-toggle:hover {{ background: rgba(255,255,255,0.1); }}

#topbar .title {{
  font-size: 15px;
  font-weight: 700;
  color: var(--text-heading);
  letter-spacing: 0.5px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

#topbar .back-link {{
  margin-left: auto;
  color: var(--link);
  text-decoration: none;
  font-size: 13px;
  white-space: nowrap;
  padding: 6px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  transition: all 0.2s;
}}

#topbar .back-link:hover {{
  background: rgba(111, 168, 220, 0.1);
  border-color: var(--link);
}}

/* Sidebar */
#sidebar {{
  position: fixed;
  top: calc(var(--progress-height) + var(--topbar-height));
  left: 0;
  width: var(--sidebar-width);
  height: calc(100vh - var(--topbar-height) - var(--progress-height));
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  overflow-y: auto;
  z-index: 999;
  display: flex;
  flex-direction: column;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}}

#sidebar::-webkit-scrollbar {{ width: 5px; }}
#sidebar::-webkit-scrollbar-track {{ background: transparent; }}
#sidebar::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

#search-box {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}}

#search-input {{
  width: 100%;
  padding: 8px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  font-size: 13px;
  outline: none;
  transition: border-color 0.2s;
}}

#search-input:focus {{ border-color: var(--accent); }}
#search-input::placeholder {{ color: var(--text-muted); }}

#toc-container {{
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}}

#toc-container::-webkit-scrollbar {{ width: 4px; }}
#toc-container::-webkit-scrollbar-track {{ background: transparent; }}
#toc-container::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}

.toc-item {{
  display: block;
  padding: 6px 16px 6px 20px;
  color: var(--text-muted);
  text-decoration: none;
  font-size: 13px;
  line-height: 1.4;
  border-left: 3px solid transparent;
  transition: all 0.15s;
  cursor: pointer;
}}

.toc-item.toc-h3 {{ padding-left: 32px; font-size: 12px; }}

.toc-item:hover {{
  color: var(--text);
  background: rgba(255,255,255,0.03);
}}

.toc-item.active {{
  color: var(--link);
  border-left-color: var(--link);
  background: rgba(111, 168, 220, 0.08);
}}

.toc-item.toc-hidden {{ display: none; }}

.toc-group-label {{
  display: block;
  padding: 10px 16px 4px 16px;
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

.toc-group-label:first-child {{ padding-top: 4px; }}
.toc-group-label.toc-hidden {{ display: none; }}

.toc-no-results {{
  padding: 16px;
  color: var(--text-muted);
  font-size: 13px;
  text-align: center;
  display: none;
}}

/* Main content */
#main-content {{
  margin-left: var(--sidebar-width);
  margin-top: calc(var(--progress-height) + var(--topbar-height));
  min-height: calc(100vh - var(--topbar-height) - var(--progress-height));
  padding: 40px 20px 80px;
}}

#content {{
  max-width: 800px;
  margin: 0 auto;
}}

/* Loading state */
#loading {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  color: var(--text-muted);
}}

.spinner {{
  width: 36px;
  height: 36px;
  border: 3px solid var(--border);
  border-top-color: var(--link);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 16px;
}}

@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

/* Typography */
#content h1 {{
  font-size: 2em;
  color: var(--text-heading);
  margin: 0 0 24px 0;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--accent);
  line-height: 1.3;
}}

#content h1 + blockquote {{
  border-left: none;
  background: transparent;
  text-align: center;
  font-size: 17px;
  color: var(--text-muted);
  padding: 8px 0 16px;
}}

#content h2 {{
  font-size: 1.5em;
  color: var(--text-heading);
  margin: 48px 0 16px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
  line-height: 1.3;
}}

#content h3 {{
  font-size: 1.25em;
  color: var(--text-heading);
  margin: 32px 0 12px 0;
  line-height: 1.3;
}}

#content h4 {{
  font-size: 1.1em;
  color: var(--text-heading);
  margin: 24px 0 8px 0;
}}

#content h5, #content h6 {{
  font-size: 1em;
  color: var(--text-heading);
  margin: 20px 0 8px 0;
}}

#content p {{ margin: 0 0 16px 0; }}

#content a {{
  color: var(--link);
  text-decoration: none;
  transition: text-decoration 0.15s;
}}

#content a:hover {{ text-decoration: underline; }}

#content ul, #content ol {{
  margin: 0 0 16px 0;
  padding-left: 28px;
}}

#content li {{ margin-bottom: 4px; }}
#content li > ul, #content li > ol {{ margin-top: 4px; margin-bottom: 4px; }}

#content hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 32px 0;
}}

#content strong {{ color: var(--text-heading); }}

#content img {{
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  margin: 8px 0;
}}

/* Code */
#content code {{
  font-family: 'SF Mono', 'Fira Code', 'Fira Mono', 'Roboto Mono', 'Consolas', monospace;
  font-size: 0.9em;
}}

#content :not(pre) > code {{
  background: var(--bg-code);
  padding: 2px 6px;
  border-radius: 4px;
  color: #e6b450;
}}

#content pre {{
  background: var(--bg-code);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  overflow-x: auto;
  margin: 0 0 16px 0;
  line-height: 1.5;
}}

#content pre code {{
  background: none;
  padding: 0;
  color: inherit;
  font-size: 0.85em;
}}

/* Tables */
#content table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 16px 0;
  font-size: 0.95em;
  display: block;
  overflow-x: auto;
}}

#content thead {{ background: var(--table-header); }}

#content th {{
  padding: 10px 14px;
  text-align: left;
  color: var(--text-heading);
  font-weight: 600;
  border-bottom: 2px solid var(--border);
}}

#content td {{
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
}}

#content tbody tr:nth-child(even) {{ background: var(--table-even); }}
#content tbody tr:nth-child(odd) {{ background: var(--table-odd); }}
#content tbody tr:hover {{ background: rgba(74, 111, 165, 0.1); }}

/* Blockquotes */
#content blockquote {{
  border-left: 4px solid var(--accent);
  background: var(--bg-blockquote);
  padding: 14px 20px;
  margin: 0 0 16px 0;
  border-radius: 0 8px 8px 0;
}}

#content blockquote p {{ margin-bottom: 8px; }}
#content blockquote p:last-child {{ margin-bottom: 0; }}

#content blockquote.quote-person {{
  border-left: none;
  background: transparent;
  text-align: center;
  padding: 24px 32px;
  margin: 8px 0 20px 0;
}}

#content blockquote.quote-person p:first-child {{
  font-style: italic;
  font-size: 17px;
  color: var(--text);
  line-height: 1.8;
}}

#content blockquote.quote-person p:last-child {{
  font-size: 13px;
  color: var(--text-muted);
  margin-top: 8px;
}}

#content blockquote strong {{
  color: #8ab4f8;
}}

/* KaTeX overrides for dark theme */
.katex {{ font-size: 1.05em; }}
.katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 4px 0; }}
.katex .base {{ color: var(--text); }}

/* Search highlight */
mark.search-highlight {{
  background: rgba(255, 200, 0, 0.35);
  color: inherit;
  padding: 1px 2px;
  border-radius: 2px;
}}

/* Mobile */
@media (max-width: 900px) {{
  #topbar .menu-toggle {{ display: block; }}

  #sidebar {{
    transform: translateX(-100%);
    transition: transform 0.3s ease;
    box-shadow: none;
  }}

  #sidebar.open {{
    transform: translateX(0);
    box-shadow: 4px 0 20px rgba(0,0,0,0.5);
  }}

  #sidebar-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.5);
    z-index: 998;
  }}

  #sidebar-overlay.visible {{ display: block; }}

  #main-content {{ margin-left: 0; }}
}}

/* Print */
@media print {{
  #progress-bar, #topbar, #sidebar, #sidebar-overlay, #search-box, .menu-toggle {{ display: none !important; }}

  #main-content {{
    margin: 0 !important;
    padding: 0 !important;
  }}

  #content {{
    max-width: 100%;
    color: #000;
  }}

  #content h1, #content h2, #content h3, #content h4, #content strong {{ color: #000; }}
  #content a {{ color: #1a0dab; text-decoration: underline; }}
  #content pre, #content code {{ background: #f5f5f5; color: #000; border-color: #ccc; }}
  #content blockquote {{ background: #f9f9f9; border-left-color: #666; }}
  #content table {{ border: 1px solid #ccc; }}
  #content th, #content td {{ border: 1px solid #ccc; }}
  #content tbody tr {{ background: #fff !important; }}

  body {{ background: #fff; }}
}}
</style>
</head>
<body>

<div id="progress-bar"></div>

<div id="topbar">
  <button class="menu-toggle" id="menu-toggle" aria-label="Toggle navigation">&#9776;</button>
  <span class="title">센서 퓨전 심화 가이드</span>
</div>

<div id="sidebar-overlay"></div>

<nav id="sidebar">
  <div id="search-box">
    <input type="text" id="search-input" placeholder="Search headings... (Ctrl+K)">
  </div>
  <div id="toc-container"></div>
  <div class="toc-no-results" id="toc-no-results">No matching sections</div>
</nav>

<main id="main-content">
  <div id="loading">
    <div class="spinner"></div>
    <span>Rendering guide...</span>
  </div>
  <div id="content" style="display:none;"></div>
</main>

<textarea id="md-source" style="display:none;">
{md_content}
</textarea>

<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script>
(function() {{
  'use strict';

  // ---- Marked configuration ----
  marked.setOptions({{
    highlight: function(code, lang) {{
      if (lang && hljs.getLanguage(lang)) {{
        try {{ return hljs.highlight(code, {{ language: lang }}).value; }} catch(e) {{}}
      }}
      try {{ return hljs.highlightAuto(code).value; }} catch(e) {{}}
      return code;
    }},
    breaks: false,
    gfm: true
  }});
  // Disable strikethrough: ~ is used as range separator
  marked.use({{ renderer: {{ del: function(token) {{ return '~' + (token.text || token) + '~'; }} }} }});

  // ---- Protect math from marked.js ----
  // marked.js treats _ as emphasis, which breaks LaTeX subscripts.
  // Strategy: extract all math expressions before parsing, replace with placeholders,
  // parse Markdown, then restore math and render with KaTeX.
  function protectMath(src) {{
    var placeholders = [];
    var idx = 0;

    // Protect code blocks first (``` ... ```)
    var codeBlocks = [];
    src = src.replace(/```[\s\S]*?```/g, function(m) {{
      codeBlocks.push(m);
      return '%%CODEBLOCK' + (codeBlocks.length - 1) + '%%';
    }});

    // Protect inline code (` ... `)
    var inlineCodes = [];
    src = src.replace(/`[^`\\n]+`/g, function(m) {{
      inlineCodes.push(m);
      return '%%INLINECODE' + (inlineCodes.length - 1) + '%%';
    }});

    // Protect display math ($$...$$)
    src = src.replace(/\$\$([\s\S]*?)\$\$/g, function(m, math) {{
      placeholders.push({{ display: true, math: math }});
      return '%%MATH' + (idx++) + '%%';
    }});

    // Protect inline math ($...$) — but not lone $ signs
    src = src.replace(/\$([^\s$](?:[^$]*[^\s$])?)\$/g, function(m, math) {{
      placeholders.push({{ display: false, math: math }});
      return '%%MATH' + (idx++) + '%%';
    }});

    // Restore code blocks
    src = src.replace(/%%CODEBLOCK(\d+)%%/g, function(m, i) {{
      return codeBlocks[parseInt(i)];
    }});
    src = src.replace(/%%INLINECODE(\d+)%%/g, function(m, i) {{
      return inlineCodes[parseInt(i)];
    }});

    return {{ src: src, placeholders: placeholders }};
  }}

  function restoreAndRenderMath(html, placeholders) {{
    return html.replace(/%%MATH(\d+)%%/g, function(m, i) {{
      var p = placeholders[parseInt(i)];
      try {{
        return katex.renderToString(p.math, {{
          displayMode: p.display,
          throwOnError: false,
          trust: true
        }});
      }} catch(e) {{
        return p.display ? '$$' + p.math + '$$' : '$' + p.math + '$';
      }}
    }});
  }}

  // ---- Render markdown ----
  var mdSource = document.getElementById('md-source').value;
  // Fix: marked.js doesn't recognize **text** when immediately followed by Korean characters
  mdSource = mdSource.replace(/\\*\\*([^*\\n]+)\\*\\*(?=[가-힣])/g, '<strong>$1</strong>');
  var contentEl = document.getElementById('content');
  var loadingEl = document.getElementById('loading');

  requestAnimationFrame(function() {{
    var mathData = protectMath(mdSource);
    var html = marked.parse(mathData.src);
    html = restoreAndRenderMath(html, mathData.placeholders);
    contentEl.innerHTML = html;
    contentEl.style.display = 'block';
    loadingEl.style.display = 'none';

    // Tag person-quote blockquotes
    contentEl.querySelectorAll('blockquote').forEach(function(bq) {{
      var text = bq.textContent;
      if (text.indexOf('\\u2014') !== -1 && !bq.querySelector('strong')) {{
        bq.classList.add('quote-person');
      }}
    }});

    // Build TOC, setup interactions
    buildTOC();
    setupScrollTracking();
    setupSearch();
    setupProgressBar();
    setupMobileMenu();
    setupKeyboard();
  }});

  // ---- Build Table of Contents ----
  function buildTOC() {{
    var headings = contentEl.querySelectorAll('h2, h3');
    var tocContainer = document.getElementById('toc-container');
    var fragment = document.createDocumentFragment();

{TOC_GROUPS_JS}

    function getGroupLabel(text) {{
      var trimmed = text.trim();
      for (var prefix in groupFirstChapter) {{
        if (trimmed.indexOf(prefix) === 0) {{
          var label = groupFirstChapter[prefix];
          delete groupFirstChapter[prefix];
          return label;
        }}
      }}
      return null;
    }}

    headings.forEach(function(heading) {{
      var id = heading.id;
      if (!id) {{
        id = heading.textContent
          .toLowerCase()
          .replace(/[^\\w\\s가-힣-]/g, '')
          .replace(/\\s+/g, '-')
          .substring(0, 60);
        heading.id = id;
      }}

      if (heading.tagName === 'H2') {{
        var groupLabel = getGroupLabel(heading.textContent);
        if (groupLabel) {{
          var label = document.createElement('div');
          label.className = 'toc-group-label';
          label.textContent = groupLabel;
          fragment.appendChild(label);
        }}
      }}

      var link = document.createElement('a');
      link.className = 'toc-item toc-' + heading.tagName.toLowerCase();
      link.href = '#' + id;
      link.textContent = heading.textContent;
      link.dataset.headingId = id;

      link.addEventListener('click', function(e) {{
        e.preventDefault();
        var target = document.getElementById(id);
        if (target) {{
          var offset = 60;
          var top = target.getBoundingClientRect().top + window.scrollY - offset;
          window.scrollTo({{ top: top, behavior: 'smooth' }});
        }}
        closeMobileSidebar();
      }});

      fragment.appendChild(link);
    }});

    tocContainer.appendChild(fragment);
  }}

  // ---- Scroll tracking ----
  function setupScrollTracking() {{
    var headings = contentEl.querySelectorAll('h2, h3');
    var tocItems = document.querySelectorAll('.toc-item');
    if (headings.length === 0) return;

    var tocMap = {{}};
    tocItems.forEach(function(item) {{
      tocMap[item.dataset.headingId] = item;
    }});

    var currentActive = null;

    var observer = new IntersectionObserver(function(entries) {{
      var topHeading = null;
      var topOffset = Infinity;

      entries.forEach(function(entry) {{
        if (entry.isIntersecting) {{
          var rect = entry.target.getBoundingClientRect();
          if (rect.top < topOffset) {{
            topOffset = rect.top;
            topHeading = entry.target;
          }}
        }}
      }});

      if (topHeading) {{
        if (currentActive) currentActive.classList.remove('active');
        var item = tocMap[topHeading.id];
        if (item) {{
          item.classList.add('active');
          currentActive = item;
          var tocContainer = document.getElementById('toc-container');
          var itemRect = item.getBoundingClientRect();
          var containerRect = tocContainer.getBoundingClientRect();
          if (itemRect.top < containerRect.top || itemRect.bottom > containerRect.bottom) {{
            item.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
          }}
        }}
      }}
    }}, {{
      rootMargin: '-60px 0px -70% 0px',
      threshold: 0
    }});

    headings.forEach(function(h) {{ observer.observe(h); }});

    var ticking = false;
    window.addEventListener('scroll', function() {{
      if (!ticking) {{
        requestAnimationFrame(function() {{
          updateActiveFromScroll(headings, tocMap);
          ticking = false;
        }});
        ticking = true;
      }}
    }});

    function updateActiveFromScroll(headings, tocMap) {{
      var found = null;
      var scrollTop = window.scrollY + 80;

      for (var i = headings.length - 1; i >= 0; i--) {{
        if (headings[i].offsetTop <= scrollTop) {{
          found = headings[i];
          break;
        }}
      }}

      if (found) {{
        if (currentActive) currentActive.classList.remove('active');
        var item = tocMap[found.id];
        if (item) {{
          item.classList.add('active');
          currentActive = item;
        }}
      }}
    }}
  }}

  // ---- Search ----
  function setupSearch() {{
    var searchInput = document.getElementById('search-input');
    var tocItems = document.querySelectorAll('.toc-item');
    var groupLabels = document.querySelectorAll('.toc-group-label');
    var noResults = document.getElementById('toc-no-results');

    searchInput.addEventListener('input', function() {{
      var query = this.value.trim().toLowerCase();

      if (!query) {{
        tocItems.forEach(function(item) {{ item.classList.remove('toc-hidden'); }});
        groupLabels.forEach(function(label) {{ label.classList.remove('toc-hidden'); }});
        noResults.style.display = 'none';
        return;
      }}

      var matchCount = 0;
      tocItems.forEach(function(item) {{
        var text = item.textContent.toLowerCase();
        if (text.includes(query)) {{
          item.classList.remove('toc-hidden');
          matchCount++;
        }} else {{
          item.classList.add('toc-hidden');
        }}
      }});

      groupLabels.forEach(function(label) {{ label.classList.add('toc-hidden'); }});
      noResults.style.display = matchCount === 0 ? 'block' : 'none';
    }});
  }}

  // ---- Progress bar ----
  function setupProgressBar() {{
    var progressBar = document.getElementById('progress-bar');
    var ticking = false;

    function updateProgress() {{
      var scrollTop = window.scrollY;
      var docHeight = document.documentElement.scrollHeight - window.innerHeight;
      var progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      progressBar.style.width = Math.min(progress, 100) + '%';
    }}

    window.addEventListener('scroll', function() {{
      if (!ticking) {{
        requestAnimationFrame(function() {{
          updateProgress();
          ticking = false;
        }});
        ticking = true;
      }}
    }});

    updateProgress();
  }}

  // ---- Mobile menu ----
  function setupMobileMenu() {{
    var toggle = document.getElementById('menu-toggle');
    var sidebar = document.getElementById('sidebar');
    var overlay = document.querySelector('#sidebar-overlay');

    toggle.addEventListener('click', function() {{
      sidebar.classList.toggle('open');
      overlay.classList.toggle('visible');
    }});

    overlay.addEventListener('click', closeMobileSidebar);
  }}

  function closeMobileSidebar() {{
    document.getElementById('sidebar').classList.remove('open');
    document.querySelector('#sidebar-overlay').classList.remove('visible');
  }}

  // ---- Keyboard shortcut ----
  function setupKeyboard() {{
    document.addEventListener('keydown', function(e) {{
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{
        e.preventDefault();
        document.getElementById('search-input').focus();
      }}
    }});
  }}

}})();
</script>
<script data-goatcounter="https://alexjunholee.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
</body>
</html>'''


def main():
    md_content = read_chapters()
    # Escape textarea closing tag if it appears in content
    md_content = md_content.replace("</textarea>", "&lt;/textarea&gt;")
    html = build_html(md_content)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Built {OUTPUT}")
    print(f"  Markdown content: {len(md_content):,} chars")
    print(f"  HTML output: {len(html):,} chars")


if __name__ == "__main__":
    main()
