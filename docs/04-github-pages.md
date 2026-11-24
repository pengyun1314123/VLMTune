# 04 · 把 VLMTune 挂到 GitHub Pages 主页

> 目标：访问者不需要部署、不需要装 Python，点开你主页上的一个链接就能看到完整可交互的实验室。

## 一、为什么原来不行，现在可以

GitHub Pages **只托管静态文件**（HTML / CSS / JS / 图片），不能在服务器上运行 Python Flask。原工程的页面数据全部来自 `/api/*`，直接上传到 Pages 会一片空白。

现在做了三件事，把工程变成了纯静态站：

| 原来的接口 | 静态版做法 |
|---|---|
| `/api/pipeline`、`/api/sft`、`/api/multimodal`、`/api/methods`、`/api/attention`、`/api/health` | 运行后端函数，把结果预生成成 `site/api/*.json` |
| `/api/lora?r=...`、`/api/training/simulate?...` | 用 JS 复刻算法，浏览器里实时算（已校验与 Python 数值完全一致） |
| `/api/tokenize?text=...` | 预生成 BPE 词表，浏览器端复刻 encode 算法 |

三篇 Markdown 文档也转成了带样式的 HTML（公式用 MathJax 渲染）。Three.js 3D 场景走本地文件，不依赖外网。

## 二、上线步骤（只需做一次）

### 1. 开启 GitHub Pages（GitHub Actions 方式）

1. 打开 `https://github.com/pengyun1314123/VLMTune/settings/pages`
2. **Build and deployment → Source** 选择 **GitHub Actions**
3. 确认仓库里有 `.github/workflows/deploy-pages.yml`（本工程已自带）

### 2. 推送代码

```bash
cd VLMTune
git add .
git commit -m "Add static GitHub Pages build"
git push origin main
```

推送后进入仓库 **Actions** 标签页，可以看到 `Deploy VLMTune to GitHub Pages` 正在运行。跑完（约 1 分钟）后访问：

**https://pengyun1314123.github.io/VLMTune/**

> 注意 URL 大小写：仓库名是 `VLMTune`，Pages 路径保留原样。
> 以后每次 push 到主分支都会自动重新生成并发布，不用手动操作。

### 3. 在主页加一个链接

打开你的主页仓库 `pengyun1314123.github.io`，编辑 `index.html`，在合适的位置（比如项目列表里）加上：

**简洁文字版：**

```html
<a href="https://pengyun1314123.github.io/VLMTune/" target="_blank" rel="noopener">
  VLMTune · 大模型训练教学实验室 ↗
</a>
```

**推荐卡片版：**

```html
<a href="https://pengyun1314123.github.io/VLMTune/"
   target="_blank" rel="noopener"
   style="display:block;max-width:420px;padding:20px 22px;background:#121a2e;
          border:1px solid #24304d;border-radius:14px;text-decoration:none;
          color:#dbe4f5;font-family:system-ui,sans-serif">
  <div style="color:#22d3ee;font-size:12px;letter-spacing:.1em">INTERACTIVE LAB</div>
  <h3 style="margin:8px 0 6px;color:#fff">VLMTune</h3>
  <p style="margin:0;color:#8fa0c0;font-size:14px;line-height:1.6">
    从零学会大模型 / 多模态训练：3D 架构、真实注意力计算、12 阶段数据流水线、LoRA 调参模拟器。
  </p>
  <span style="display:inline-block;margin-top:14px;color:#22d3ee;font-size:14px">点进去开始 →</span>
</a>
```

改完提交主页仓库即可。链接指向的是完整实验室首页，带 `#architecture`、`#pipeline` 等 hash 还能直达对应板块，例如：

```html
<a href="https://pengyun1314123.github.io/VLMTune/#finetune">微调游乐场</a>
```

## 三、两种备选发布方式

如果不想用 Actions，也可以：

- **分支方式**：把 `site/` 目录内容推到 `gh-pages` 分支，然后在 Settings → Pages → Source 选 `Deploy from a branch` / `gh-pages` / `/ (root)`。
- **自定义域名**：在 Settings → Pages → Custom domain 填自己的域名，并在 DNS 加 CNAME 记录；本工程用的是相对路径，换域名不需要改代码。

## 四、本地改了内容怎么更新

后端算法或样例数据有改动后，重新生成静态文件：

```bash
python3 tools/build_static.py      # 重建 site/
python3 tools/verify_static.py     # 校验浏览器端算法与后端一致
```

然后 `git add . && git commit && git push`，Actions 会自动重新发布。

只想本地预览静态效果：

```bash
cd site && python3 -m http.server 8000
# 打开 http://127.0.0.1:8000
```

原来的 Flask 开发方式完全保留，不受影响：

```bash
python3 backend/app.py    # http://127.0.0.1:5000
```

## 五、常见问题

**页面打开是空白？** 先确认 Settings → Pages 的 Source 是 GitHub Actions，且 Actions 最后一次运行成功；再看浏览器控制台有没有 404（通常是路径大小写问题）。

**公式显示成 `$$...$$` 原文？** 文档页的公式依赖 jsDelivr 上的 MathJax；网络受限时会降级显示 LaTeX 源码，内容不丢。实验室页面里的公式是 HTML 写的，不受影响。

**数字和本地 Flask 版不一样？** 正常情况应完全一致（已校验 12 项）。如发现偏差，先跑 `python3 tools/verify_static.py`。训练模拟的随机数用 MT19937 逐位复刻，Loss 曲线数值与后端相同。
