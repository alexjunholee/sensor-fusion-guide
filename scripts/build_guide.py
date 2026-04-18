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

# Guide overview — CSS for text-prose list at top of content
OVERVIEW_CSS = """\
/* Guide Overview (at top of content) */
.guide-overview {
  margin: 0 0 80px 0;
  padding: 0 0 48px 0;
  border-bottom: 1px solid var(--border);
}
.guide-overview .overview-header { margin-bottom: 44px; }
.guide-overview .overview-title {
  font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Pretendard', sans-serif;
  font-size: 48px;
  font-weight: 600;
  line-height: 1.08;
  letter-spacing: -0.018em;
  color: var(--text-heading);
  margin: 0 0 10px 0;
}
.guide-overview .overview-subtitle {
  font-size: 17px;
  color: var(--text-muted);
  letter-spacing: -0.014em;
  margin: 0;
}
.overview-group { margin-top: 36px; }
.overview-group:first-child { margin-top: 0; }
.overview-group-label {
  font-size: 11px;
  font-weight: 700;
  color: rgba(0, 0, 0, 0.42);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin: 0 0 18px 0;
}
.overview-chapter { margin: 0 0 22px 0; }
.overview-chapter:last-child { margin-bottom: 0; }
.overview-chapter-link {
  font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Pretendard', sans-serif;
  font-size: 19px;
  font-weight: 600;
  color: var(--text-heading);
  text-decoration: none;
  letter-spacing: -0.014em;
  display: inline-block;
}
.overview-chapter-link:hover { color: var(--accent); }
.overview-chapter-num {
  font-size: 12px;
  font-weight: 700;
  color: var(--accent);
  margin-right: 10px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
.overview-sections {
  margin-top: 4px;
  font-size: 14px;
  line-height: 1.55;
  color: var(--text-muted);
  letter-spacing: -0.01em;
}
.overview-sections a {
  color: var(--text-muted);
  text-decoration: none;
}
.overview-sections a:hover { color: var(--accent); }
.overview-sections .sep {
  margin: 0 8px;
  color: var(--border-strong);
}
"""

# Guide overview — JS for rendering (text prose, native hash nav)
OVERVIEW_JS = r"""
  function buildOverview() {
    var chapterGroups = [
      { label: '기초 (Foundations)', chapters: [1, 2, 3] },
      { label: '방법론 기초 (Methods)', chapters: [4, 5] },
      { label: 'Odometry & Fusion', chapters: [6, 7, 8] },
      { label: 'Place Recognition & Loop Closure', chapters: [9, 10] },
      { label: '표현 · 실전 · 미래', chapters: [11, 12, 13] }
    ];

    var h1s = contentEl.querySelectorAll('h1');
    var chapters = {};

    h1s.forEach(function(h1) {
      var m = h1.textContent.trim().match(/(?:Chapter|Ch\.?)\s*(\d+)\s*[\u2014\u2013-]\s*(.+)/);
      if (!m) return;
      var chNum = parseInt(m[1], 10);
      if (!h1.id) h1.id = 'chapter-' + chNum;

      var sections = [];
      var node = h1.nextElementSibling;
      while (node && node.tagName !== 'H1') {
        if (node.tagName === 'H2') {
          if (!node.id) {
            node.id = node.textContent.toLowerCase()
              .replace(/[^\w\s가-힣-]/g, '').replace(/\s+/g, '-').substring(0, 60);
          }
          sections.push({ text: node.textContent.trim(), id: node.id });
        }
        node = node.nextElementSibling;
      }

      chapters[chNum] = { title: m[2].trim(), id: h1.id, sections: sections };
    });

    var total = Object.keys(chapters).length;
    if (total === 0) return;

    var overview = document.createElement('section');
    overview.className = 'guide-overview';

    var header = document.createElement('div');
    header.className = 'overview-header';
    var oTitle = document.createElement('h1');
    oTitle.className = 'overview-title';
    oTitle.textContent = '센서 퓨전 심화 가이드';
    header.appendChild(oTitle);
    var oSub = document.createElement('p');
    oSub.className = 'overview-subtitle';
    oSub.textContent = '로컬라이제이션 · 매핑 · 멀티센서 융합 심화 레퍼런스 · 전 ' + total + '장';
    header.appendChild(oSub);
    overview.appendChild(header);

    chapterGroups.forEach(function(group) {
      var valid = group.chapters.filter(function(n) { return chapters[n]; });
      if (valid.length === 0) return;

      var groupEl = document.createElement('div');
      groupEl.className = 'overview-group';

      var label = document.createElement('div');
      label.className = 'overview-group-label';
      label.textContent = group.label;
      groupEl.appendChild(label);

      valid.forEach(function(chNum) {
        var ch = chapters[chNum];
        var chWrap = document.createElement('div');
        chWrap.className = 'overview-chapter';

        var chLink = document.createElement('a');
        chLink.className = 'overview-chapter-link';
        chLink.href = '#' + ch.id;
        var numSpan = document.createElement('span');
        numSpan.className = 'overview-chapter-num';
        numSpan.textContent = 'Ch.' + chNum;
        chLink.appendChild(numSpan);
        chLink.appendChild(document.createTextNode(ch.title));
        chWrap.appendChild(chLink);

        if (ch.sections.length) {
          var secsDiv = document.createElement('div');
          secsDiv.className = 'overview-sections';
          ch.sections.forEach(function(s, i) {
            if (i > 0) {
              var sep = document.createElement('span');
              sep.className = 'sep';
              sep.textContent = '·';
              secsDiv.appendChild(sep);
            }
            var sLink = document.createElement('a');
            sLink.href = '#' + s.id;
            sLink.textContent = s.text;
            secsDiv.appendChild(sLink);
          });
          chWrap.appendChild(secsDiv);
        }

        groupEl.appendChild(chWrap);
      });

      overview.appendChild(groupEl);
    });

    contentEl.insertBefore(overview, contentEl.firstChild);
  }
"""

def build_html(md_content):
    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>센서 퓨전 심화 가이드</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-light.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  /* Apple binary palette */
  --bg: #ffffff;
  --bg-alt: #f5f5f7;
  --bg-sidebar: #f5f5f7;
  --bg-code: #f5f5f7;
  --bg-surface: #f5f5f7;

  /* Text */
  --text: #1d1d1f;
  --text-body: rgba(0, 0, 0, 0.8);
  --text-muted: rgba(0, 0, 0, 0.56);
  --text-tertiary: rgba(0, 0, 0, 0.48);
  --text-heading: #1d1d1f;

  /* Single accent — Apple Blue */
  --accent: #0071e3;
  --link: #0066cc;

  /* Borders */
  --border: rgba(0, 0, 0, 0.08);
  --border-strong: rgba(0, 0, 0, 0.16);

  /* Layout */
  --sidebar-width: 300px;
  --topbar-height: 52px;
  --progress-height: 3px;

  /* Shadow */
  --shadow-elev: rgba(0, 0, 0, 0.22) 3px 5px 30px 0px;
}}

html {{ -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }}

body {{
  font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Pretendard', system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
  font-size: 17px;
  line-height: 1.47;
  letter-spacing: -0.022em;
  color: var(--text);
  background: var(--bg);
  overflow-x: hidden;
  font-feature-settings: "kern", "liga", "calt";
  word-break: keep-all;
  overflow-wrap: break-word;
}}

/* Progress bar */
#progress-bar {{
  position: fixed;
  top: 0;
  left: 0;
  height: var(--progress-height);
  background: linear-gradient(90deg, #0071e3, #2997ff);
  z-index: 10000;
  transition: width 0.1s linear;
  width: 0%;
}}

/* Topbar — Apple navigation glass */
#topbar {{
  position: fixed;
  top: var(--progress-height);
  left: 0;
  right: 0;
  height: var(--topbar-height);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 24px;
  z-index: 1000;
}}

#topbar .menu-toggle {{
  display: none;
  background: none;
  border: none;
  color: var(--text);
  font-size: 22px;
  cursor: pointer;
  margin-right: 12px;
  padding: 6px 10px;
  border-radius: 8px;
}}

#topbar .menu-toggle:hover {{ background: rgba(0,0,0,0.04); }}

#topbar .title {{
  font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', sans-serif;
  font-size: 17px;
  font-weight: 600;
  color: var(--text-heading);
  letter-spacing: -0.022em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

#topbar .back-link {{
  margin-left: auto;
  color: var(--link);
  text-decoration: none;
  font-size: 13.5px;
  font-weight: 500;
  white-space: nowrap;
  padding: 7px 14px;
  border-radius: 980px;
  border: 1px solid var(--border);
  letter-spacing: -0.014em;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}}

#topbar .back-link:hover {{
  background: rgba(0, 113, 227, 0.06);
  border-color: var(--accent);
  color: var(--accent);
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
  scrollbar-color: rgba(0,0,0,0.16) transparent;
}}

#sidebar::-webkit-scrollbar {{ width: 6px; }}
#sidebar::-webkit-scrollbar-track {{ background: transparent; }}
#sidebar::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.16); border-radius: 3px; }}

#search-box {{
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.6);
  flex-shrink: 0;
}}

#search-input {{
  width: 100%;
  padding: 9px 12px;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-size: 13.5px;
  outline: none;
  font-family: inherit;
  letter-spacing: -0.016em;
  transition: border-color 0.15s, box-shadow 0.15s;
}}

#search-input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15); }}
#search-input::placeholder {{ color: var(--text-muted); }}

#toc-container {{
  flex: 1;
  overflow-y: auto;
  padding: 8px 0 32px;
}}

#toc-container::-webkit-scrollbar {{ width: 4px; }}
#toc-container::-webkit-scrollbar-track {{ background: transparent; }}
#toc-container::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.12); border-radius: 2px; }}

/* Chapter level (h2) */
.toc-item {{
  display: block;
  padding: 8px 16px 8px 20px;
  color: var(--text);
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.4;
  letter-spacing: -0.014em;
  border-left: 3px solid transparent;
  transition: color 0.15s, background 0.15s, border-color 0.15s;
  cursor: pointer;
}}

/* Subsection (h3) */
.toc-item.toc-h3 {{
  padding: 5px 16px 5px 36px;
  font-size: 13px;
  font-weight: 400;
  color: var(--text-muted);
  letter-spacing: -0.012em;
}}

.toc-item.toc-h3 + .toc-item:not(.toc-h3) {{ margin-top: 6px; }}

.toc-item:hover {{
  color: var(--text);
  background: rgba(0, 0, 0, 0.04);
}}

.toc-item.toc-h3:hover {{ color: var(--text); }}

.toc-item.active {{
  color: var(--accent);
  border-left-color: var(--accent);
  background: rgba(0, 113, 227, 0.08);
  font-weight: 600;
}}

.toc-item.toc-h3.active {{
  font-weight: 500;
  color: var(--accent);
}}

.toc-item.toc-hidden {{ display: none; }}

.toc-group-label {{
  display: block;
  padding: 22px 16px 6px 20px;
  font-size: 11px;
  font-weight: 700;
  color: rgba(0, 0, 0, 0.42);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}}

.toc-group-label:first-child {{ padding-top: 8px; }}
.toc-group-label.toc-hidden {{ display: none; }}

.toc-no-results {{
  padding: 20px 16px;
  color: var(--text-muted);
  font-size: 13.5px;
  text-align: center;
  display: none;
  letter-spacing: -0.014em;
}}

/* Main content */
#main-content {{
  margin-left: var(--sidebar-width);
  margin-top: calc(var(--progress-height) + var(--topbar-height));
  min-height: calc(100vh - var(--topbar-height) - var(--progress-height));
  padding: 56px 56px 96px;
}}

#content {{
  max-width: 820px;
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
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 16px;
}}

@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

/* Typography — SF Pro Display headings */
#content h1, #content h2, #content h3, #content h4, #content h5, #content h6 {{
  font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Pretendard', sans-serif;
  color: var(--text-heading);
  font-feature-settings: "ss01", "kern";
  scroll-margin-top: 72px;
}}

#content h1 {{
  font-size: 48px;
  font-weight: 600;
  line-height: 1.08;
  letter-spacing: -0.018em;
  margin: 0 0 28px 0;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}}

#content h1 + blockquote {{
  border-left: none;
  background: transparent;
  text-align: center;
  font-size: 18px;
  color: var(--text-muted);
  padding: 8px 0 24px;
  letter-spacing: -0.018em;
}}

#content h2 {{
  font-size: 34px;
  font-weight: 600;
  line-height: 1.10;
  letter-spacing: -0.014em;
  margin: 64px 0 20px 0;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}}

#content h3 {{
  font-size: 24px;
  font-weight: 600;
  line-height: 1.16;
  letter-spacing: -0.012em;
  margin: 44px 0 14px 0;
}}

#content h4 {{
  font-size: 19px;
  font-weight: 600;
  line-height: 1.22;
  letter-spacing: -0.010em;
  margin: 32px 0 10px 0;
}}

#content h5, #content h6 {{
  font-size: 17px;
  font-weight: 600;
  line-height: 1.3;
  margin: 24px 0 8px 0;
}}

#content p {{ margin: 0 0 16px 0; color: var(--text-body); }}

#content a {{
  color: var(--link);
  text-decoration: none;
  transition: color 0.15s;
}}

#content a:hover {{ text-decoration: underline; text-underline-offset: 2px; }}

#content ul, #content ol {{
  margin: 0 0 16px 0;
  padding-left: 28px;
}}

#content li {{ margin-bottom: 6px; color: var(--text-body); }}
#content li > ul, #content li > ol {{ margin-top: 6px; margin-bottom: 6px; }}

#content hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 56px 0;
}}

#content strong {{ color: var(--text); font-weight: 600; }}

#content img {{
  max-width: 100%;
  height: auto;
  border-radius: 12px;
  margin: 12px 0;
}}

/* Code */
#content code {{
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
  font-size: 0.82em;
  letter-spacing: 0;
}}

#content :not(pre) > code {{
  background: var(--bg-code);
  padding: 2px 6px;
  border-radius: 5px;
  color: var(--text);
}}

#content pre {{
  background: var(--bg-code);
  border: none;
  border-radius: 12px;
  padding: 18px 20px;
  overflow-x: auto;
  margin: 0 0 20px 0;
  font-size: 13px;
  line-height: 1.55;
}}

#content pre code {{
  background: none;
  padding: 0;
  color: inherit;
  font-size: inherit;
}}

/* Tables — Apple compare-style */
#content table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 20px 0;
  font-size: 15px;
  display: block;
  overflow-x: auto;
}}

#content thead {{ background: var(--bg-surface); }}

#content th {{
  padding: 12px 16px;
  text-align: left;
  color: var(--text-heading);
  font-weight: 600;
  font-size: 14px;
  border-bottom: 1px solid var(--border-strong);
  letter-spacing: -0.012em;
}}

#content td {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  color: var(--text-body);
  letter-spacing: -0.016em;
}}

#content tbody tr:hover {{ background: rgba(0, 113, 227, 0.04); }}

/* Blockquotes */
#content blockquote {{
  border-left: 3px solid var(--accent);
  background: var(--bg-surface);
  padding: 14px 22px;
  margin: 0 0 20px 0;
  border-radius: 0 8px 8px 0;
  color: var(--text-body);
}}

#content blockquote p {{ margin-bottom: 8px; }}
#content blockquote p:last-child {{ margin-bottom: 0; }}

#content blockquote.quote-person {{
  border-left: none;
  background: transparent;
  text-align: center;
  padding: 28px 32px;
  margin: 12px 0 24px 0;
}}

#content blockquote.quote-person p:first-child {{
  font-style: italic;
  font-size: 19px;
  color: var(--text);
  line-height: 1.5;
  letter-spacing: -0.018em;
}}

#content blockquote.quote-person p:last-child {{
  font-size: 13px;
  color: var(--text-muted);
  margin-top: 10px;
  letter-spacing: -0.014em;
}}

#content blockquote strong {{ color: var(--accent); }}

/* KaTeX */
.katex {{ font-size: 1.05em; }}
.katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 4px 0; margin: 20px 0; }}

/* Selection */
::selection {{ background: rgba(0, 113, 227, 0.20); color: var(--text); }}

/* Search highlight */
mark.search-highlight {{
  background: rgba(0, 113, 227, 0.18);
  color: inherit;
  padding: 1px 3px;
  border-radius: 3px;
}}

{OVERVIEW_CSS}

/* Responsive */
@media (max-width: 1024px) {{
  #main-content {{ padding-left: 40px; padding-right: 40px; }}
}}

@media (max-width: 900px) {{
  #topbar .menu-toggle {{ display: block; }}

  #sidebar {{
    transform: translateX(-100%);
    transition: transform 0.3s ease;
    box-shadow: none;
  }}

  #sidebar.open {{
    transform: translateX(0);
    box-shadow: var(--shadow-elev);
  }}

  #sidebar-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.4);
    z-index: 998;
  }}

  #sidebar-overlay.visible {{ display: block; }}

  #main-content {{ margin-left: 0; padding: 24px 24px 56px; }}

  #content h1 {{ font-size: 36px; }}
  #content h2 {{ font-size: 28px; margin-top: 48px; }}
  #content h3 {{ font-size: 21px; }}
  #content h4 {{ font-size: 17px; }}
}}

@media (max-width: 600px) {{
  body {{ font-size: 16px; }}
  #content h1 {{ font-size: 30px; }}
  #content h2 {{ font-size: 24px; }}
  #content h3 {{ font-size: 19px; }}
  #main-content {{ padding: 20px 20px 48px; }}
}}

/* Print */
@media print {{
  #progress-bar, #topbar, #sidebar, #sidebar-overlay, #search-box, .menu-toggle {{ display: none !important; }}

  #main-content {{ margin: 0 !important; padding: 0 !important; }}
  #content {{ max-width: 100%; color: #000; }}

  #content h1, #content h2, #content h3, #content h4, #content strong {{ color: #000; }}
  #content a {{ color: #1a0dab; text-decoration: underline; }}
  #content pre, #content code {{ background: #f5f5f7; color: #000; border: none; }}
  #content blockquote {{ background: #f5f5f7; border-left-color: #999; }}
  #content table {{ border: 1px solid #ccc; }}
  #content th, #content td {{ border: 1px solid #ccc; }}

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
    buildOverview();
    buildTOC();
    setupScrollTracking();
    setupSearch();
    setupProgressBar();
    setupMobileMenu();
    setupKeyboard();

    // On initial load with a URL hash, scroll to target (content just rendered)
    if (window.location.hash) {{
      var initId = window.location.hash.slice(1);
      var initTarget = document.getElementById(initId);
      if (initTarget) {{
        var initTop = initTarget.getBoundingClientRect().top + window.scrollY - 60;
        window.scrollTo(0, initTop);
      }}
    }}
  }});

  // ---- Build Guide Overview (top of page) ----
{OVERVIEW_JS}

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
            item.scrollIntoView({{ block: 'center', behavior: 'auto' }});
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
