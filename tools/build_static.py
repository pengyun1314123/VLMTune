#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLMTune 静态站点构建器
======================
把需要 Flask 后端的工程改造成可直接托管在 GitHub Pages 的纯静态站点。

原理：
  1. 无参数 API  ->  调用后端 demos 预生成 JSON 文件
  2. 带参数 API  ->  复制 JS 版算法到前端（tools/static-api.js）
  3. Markdown    ->  转成带样式的 HTML 文档
  4. app.js      ->  替换数据获取层，保持页面其余逻辑不变

用法：
  python3 tools/build_static.py
输出：
  site/   可直接上传 GitHub Pages 的目录

本地预览：
  cd site && python3 -m http.server 8000
"""
import os
import re
import sys
import html
import json
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(BASE, "site")
BACKEND = os.path.join(BASE, "backend")
FRONTEND = os.path.join(BASE, "frontend")
DOCS = os.path.join(BASE, "docs")
TOOLS = os.path.join(BASE, "tools")

sys.path.insert(0, BACKEND)
from demos import pipeline as pipe_demo          # noqa: E402
from demos import finetuning as ft_demo          # noqa: E402
from demos import training as train_demo         # noqa: E402
from demos.pipeline import load_raw              # noqa: E402
from demos.tokenizer import SimpleBPE            # noqa: E402


def ensure_clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def write(path, text, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def dump_json(name, obj):
    path = os.path.join(SITE, "api", name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("  [api] %-14s %8d bytes" % (name + ".json", os.path.getsize(path)))


# ---------------------------------------------------------------- 数据预生成
def build_api_data():
    print("[1/5] 预生成 API 数据 ...")
    os.makedirs(os.path.join(SITE, "api"), exist_ok=True)

    dump_json("health", {
        "status": "ok",
        "service": "VLMTune 教学实验室 · 静态版（GitHub Pages）"
    })
    dump_json("pipeline", pipe_demo.build_pipeline())

    samples, rendered = pipe_demo.get_sft_samples()
    dump_json("sft", {
        "samples": samples,
        "rendered": rendered,
        "note": "训练时只对 assistant 段计算 loss，模板与 user 段 labels=-100。"
    })
    dump_json("multimodal", pipe_demo.get_multimodal_sample())
    dump_json("methods", {
        "methods": ft_demo.METHODS,
        "tips": ft_demo.training_tips()
    })
    dump_json("attention", ft_demo.attention_demo())

    # BPE：词表与合并规则是固定的（语料固定），预生成后浏览器只做 encode
    corpus = "\n".join(d["text"] for d in load_raw())
    tokenizer = SimpleBPE().train([corpus], vocab_size=300)
    merges = [[list(pair), rank]
              for pair, rank in sorted(tokenizer.merges.items(), key=lambda kv: kv[1])]
    dump_json("tokenizer", {
        "vocab": tokenizer.vocab,
        "merges": merges,
        "special_tokens": tokenizer.special_tokens,
        "note": "固定语料上训练的迷你 BPE；浏览器端复刻 encode 算法。"
    })
    print("  [api] train_demo / lora_calculator 已在浏览器端实时计算，无需预生成")


# ---------------------------------------------------------------- Markdown
def md_inline(text):
    s = html.escape(text, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    return s


def is_table_sep(line):
    s = line.strip()
    return s.startswith("|") and re.match(r"^\|[\s:\-|]+\|$", s) is not None


def md_to_html(md_text):
    lines = md_text.split("\n")
    out = []
    i = 0
    in_code = False
    code_lang = ""
    code_buf = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 代码块
        if stripped.startswith("```"):
            if in_code:
                cls = (' class="lang-%s"' % code_lang) if code_lang else ""
                out.append("<pre%s><code>%s</code></pre>" %
                           (cls, html.escape("\n".join(code_buf))))
                in_code = False
                code_buf = []
            else:
                close_lists()
                code_lang = stripped[3:].strip()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 公式块（保留 LaTeX，由 MathJax 渲染）
        if stripped.startswith("$$"):
            close_lists()
            buf = [stripped]
            if not (len(stripped) > 4 and stripped.endswith("$$")):
                while i + 1 < len(lines):
                    i += 1
                    buf.append(lines[i].strip())
                    if "$$" in lines[i]:
                        break
            out.append('<div class="math-block">%s</div>' %
                       html.escape("\n".join(buf), quote=False))
            i += 1
            continue

        # 表格
        if stripped.startswith("|") and i + 1 < len(lines) and is_table_sep(lines[i + 1]):
            close_lists()
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            thead = "".join("<th>%s</th>" % md_inline(c) for c in header)
            tbody = "".join(
                "<tr>%s</tr>" % "".join("<td>%s</td>" % md_inline(c) for c in row)
                for row in rows
            )
            out.append('<div class="table-wrap"><table><thead><tr>%s</tr></thead>'
                       '<tbody>%s</tbody></table></div>' % (thead, tbody))
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            close_lists()
            level = len(m.group(1))
            out.append("<h%d>%s</h%d>" % (level, md_inline(m.group(2)), level))
            i += 1
            continue

        # 水平线
        if stripped in ("---", "***", "___"):
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        # 引用
        if stripped.startswith(">"):
            close_lists()
            quoted = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quoted.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>%s</blockquote>" %
                       "<br>".join(md_inline(q) for q in quoted))
            continue

        # 无序列表
        if re.match(r"^[-*]\s+", stripped):
            if in_ol:
                close_lists()
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append("<li>%s</li>" % md_inline(re.sub(r"^[-*]\s+", "", stripped)))
            i += 1
            continue

        # 有序列表
        if re.match(r"^\d+\.\s+", stripped):
            if in_ul:
                close_lists()
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append("<li>%s</li>" % md_inline(re.sub(r"^\d+\.\s+", "", stripped)))
            i += 1
            continue

        # 空行
        if not stripped:
            close_lists()
            i += 1
            continue

        # 段落：合并连续普通行
        close_lists()
        para = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith("#") or nxt.startswith("```") or
                    nxt.startswith(">") or nxt.startswith("$$") or
                    nxt.startswith("|") or re.match(r"^[-*]\s+", nxt) or
                    re.match(r"^\d+\.\s+", nxt) or nxt in ("---", "***", "___")):
                break
            para.append(nxt)
            i += 1
        out.append("<p>%s</p>" % md_inline(" ".join(para)))

    close_lists()
    if in_code:
        out.append("<pre><code>%s</code></pre>" % html.escape("\n".join(code_buf)))
    return "\n".join(out)


DOC_CSS = """
:root{--bg:#0b1020;--panel:#121a2e;--line:#24304d;--text:#dbe4f5;--muted:#8fa0c0;
--cyan:#22d3ee;--violet:#a78bfa;--green:#34d399;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
line-height:1.85;font-size:15px;}
.doc-wrap{max-width:940px;margin:0 auto;padding:36px 28px 100px;}
.back{display:inline-block;color:var(--cyan);text-decoration:none;font-size:13px;
border:1px solid var(--line);padding:6px 14px;border-radius:8px;margin-bottom:28px;}
.back:hover{background:#16203a}
h1,h2,h3,h4,h5,h6{color:#fff;line-height:1.35;margin:34px 0 14px;}
h1{font-size:30px;border-bottom:1px solid var(--line);padding-bottom:16px;}
h2{font-size:23px;} h3{font-size:18px;} h4{font-size:16px;}
p{margin:12px 0;color:#c7d3e8;}
hr{border:0;border-top:1px solid var(--line);margin:30px 0;}
a{color:var(--cyan);}
strong{color:#fff;}
code{font-family:"SFMono-Regular",Consolas,Menlo,monospace;font-size:13px;
background:#1a2540;color:#7dd3fc;padding:2px 6px;border-radius:4px;}
pre{background:#0e1629;border:1px solid var(--line);border-radius:10px;
padding:16px 18px;overflow-x:auto;line-height:1.6;}
pre code{background:none;color:#b9c9e6;padding:0;font-size:12.5px;}
blockquote{border-left:3px solid var(--cyan);background:rgba(34,211,238,.06);
margin:16px 0;padding:10px 18px;color:#b6c8e4;border-radius:0 8px 8px 0;}
ul,ol{padding-left:24px;color:#c7d3e8;} li{margin:6px 0;}
.math-block{background:rgba(167,139,250,.07);border:1px solid rgba(167,139,250,.2);
border-radius:8px;padding:12px 18px;margin:16px 0;overflow-x:auto;
color:#ddd6fe;text-align:center;font-size:15px;}
.table-wrap{overflow-x:auto;margin:18px 0;}
table{border-collapse:collapse;width:100%;font-size:13px;}
th,td{border:1px solid var(--line);padding:8px 12px;text-align:left;}
th{background:#18223c;color:#fff;} tr:nth-child(even){background:rgba(255,255,255,.02)}
@media(max-width:640px){.doc-wrap{padding:20px 16px 60px}h1{font-size:24px}}
"""


def doc_html(title, body):
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>%s · VLMTune</title>
<style>%s</style>
<script>
window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],
displayMath:[['$$','$$'],['\\\\[','\\\\]']]}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
<div class="doc-wrap">
<a class="back" href="../index.html#docs">← 返回 VLMTune 实验室</a>
%s
</div>
</body>
</html>
""" % (html.escape(title), DOC_CSS, body)


def build_docs():
    print("[2/5] 转换 Markdown 文档 ...")
    outdir = os.path.join(SITE, "docs")
    os.makedirs(outdir, exist_ok=True)
    for name in ["01-architecture.md", "02-finetuning.md", "03-data-pipeline.md"]:
        src = os.path.join(DOCS, name)
        if not os.path.isfile(src):
            print("  [skip] %s 不存在" % name)
            continue
        with open(src, "r", encoding="utf-8") as f:
            md = f.read()
        title = md.split("\n", 1)[0].lstrip("#").strip() or name
        dst = os.path.join(outdir, name[:-3] + ".html")
        write(dst, doc_html(title, md_to_html(md)))
        print("  [doc] %s -> %s" % (name, os.path.basename(dst)))


# ---------------------------------------------------------------- 前端装配
STATIC_GET = """async function get(path) {
  const parts = path.split('?');
  const route = parts[0];
  const q = new URLSearchParams(parts[1] || '');
  const json = async name => {
    const r = await fetch('api/' + name + '.json');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  };
  switch (route) {
    case '/api/health':    return json('health');
    case '/api/pipeline':  return json('pipeline');
    case '/api/sft':       return json('sft');
    case '/api/multimodal':return json('multimodal');
    case '/api/methods':   return json('methods');
    case '/api/attention': return json('attention');
    case '/api/lora':
      return VLMStaticAPI.lora({
        r: q.get('r'), alpha: q.get('alpha'), d: q.get('d'), layers: q.get('layers')
      });
    case '/api/training/simulate':
      return VLMStaticAPI.simulate({
        steps: q.get('steps'), method: q.get('method'), quality: q.get('quality')
      });
    case '/api/tokenize':
      return VLMStaticAPI.tokenize(q.get('text') || '大模型训练');
    default:
      throw new Error('未知静态接口: ' + route);
  }
}"""

ORIGINAL_GET = """async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}"""


def build_frontend():
    print("[3/5] 装配静态前端 ...")
    # 静态资源
    for rel in ["index.html", "compression.html", "css/style.css", "js/visuals.js", "js/app.js",
                "vendor/three.min.js"]:
        src = os.path.join(FRONTEND, rel.replace("/", os.sep))
        dst = os.path.join(SITE, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # JS 计算模块
    shutil.copy2(os.path.join(TOOLS, "static-api.js"),
                 os.path.join(SITE, "js", "static-api.js"))

    # 替换数据获取层
    app_path = os.path.join(SITE, "js", "app.js")
    app = read(app_path)
    if ORIGINAL_GET not in app:
        raise RuntimeError("app.js 中未找到预期的 get() 实现，构建中止（请检查前端是否改动）")
    app = app.replace(ORIGINAL_GET, STATIC_GET, 1)
    app = app.replace("$('#backendStatus').textContent = '后端已连接 · ' + d.service;",
                      "$('#backendStatus').textContent = '静态版已就绪 · ' + d.service;")
    for name, title in [("01-architecture", "架构知识点与核心公式"),
                        ("02-finetuning", "微调方法原理与代码注解"),
                        ("03-data-pipeline", "训练数据全流程")]:
        app = app.replace("'docs/%s.md'" % name, "'docs/%s.html'" % name)
    write(app_path, app)

    # index.html：先加载静态 API，再加载 app.js
    idx_path = os.path.join(SITE, "index.html")
    idx = read(idx_path)
    idx = idx.replace('<script src="js/app.js"></script>',
                      '<script src="js/static-api.js"></script>\n'
                      '<script src="js/app.js"></script>')
    idx = idx.replace('<span class="tag">Python · Flask</span>',
                      '<span class="tag">GitHub Pages · 静态版</span>')
    write(idx_path, idx)
    print("  [ok] 数据层已切换为静态模式，交互算法已注入")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_readme():
    print("[4/5] 写入站点说明 ...")
    write(os.path.join(SITE, "README.txt"),
          "VLMTune 静态站点（自动生成，请勿手工修改）\n"
          "重新生成：python3 tools/build_static.py\n"
          "来源：https://github.com/pengyun1314123/VLMTune\n"
          "访问地址：https://pengyun1314123.github.io/VLMTune/\n"
          "专题页面：compression.html（剪枝 / 量化 / 全栈推理加速）\n"
          "入口位置：主站侧栏「资源」与「知识文档」页\n")


def main():
    print("=" * 58)
    print("VLMTune 静态站点构建器 -> site/")
    print("=" * 58)
    ensure_clean_dir(SITE)
    build_api_data()
    build_docs()
    build_frontend()
    write_readme()
    total = sum(os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(SITE) for f in files)
    print("[5/5] 完成：%s（共 %.1f KB）" % (SITE, total / 1024))
    print("本地预览：cd site && python3 -m http.server 8000")


if __name__ == "__main__":
    main()
