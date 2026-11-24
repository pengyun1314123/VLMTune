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

## 二、实际发布方式（已完成）

**当前线上地址：https://pengyun1314123.github.io/VLMTune/**

`VLMTune` 仓库是**私有仓库**，免费账户无法为它单独开启 GitHub Pages（直接访问会返回 503 Unicorn 页）。因此采用更稳妥的方案：**把 `site/` 的全部内容放到主页仓库 `pengyun1314123.github.io` 的 `/VLMTune/` 子目录下发布**。主页仓库是 public 且 Pages 正常，Jekyll 会把这些静态文件原样复制，路径和链接都不用改。

已完成的事情：

1. 主页 README.md 顶部导航加入「教学实验室」链接，并新增「交互教学实验室 · VLMTune」章节（六个板块表格 + 在线 / 源码入口）。
2. 静态站文件已推送到主页仓库的 `VLMTune/` 目录。
3. 线上验证：首页与 `api/*.json`、`js/*.js`、`css`、`vendor/three.min.js`、`docs/*.html` 全部返回 200。

> 仓库里的 `.github/workflows/deploy-pages.yml` 保留作备用：如果将来把 VLMTune 仓库转为 public，可在 Settings → Pages → Source 选 GitHub Actions 切回独立发布。

### 如果要在别的主页加链接

打开主页仓库，编辑 `README.md`（Jekyll 站）或 `index.html`，在合适的位置加上：

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

## 三、本地改了内容怎么更新（重要）

后端算法或样例数据改动后，**一步重建并同步到线上**：

```bash
python3 tools/publish_to_homepage.py
```

这个脚本会依次：重建 `site/` → 跑 12 项一致性校验 → 把 `site/` 内容同步到主页仓库的 `VLMTune/` 目录 → 提交并推送主页仓库。约 1 分钟后线上生效。

也可以分步手动操作：

```bash
python3 tools/build_static.py      # 重建 site/
python3 tools/verify_static.py     # 校验浏览器端算法与后端一致
cp -R site/. ../pengyun1314123.github.io/VLMTune/
cd ../pengyun1314123.github.io && git add VLMTune && git commit -m "Update VLMTune" && git push
```

只想本地预览静态效果：

```bash
cd site && python3 -m http.server 8000
# 打开 http://127.0.0.1:8000
```

原来的 Flask 开发方式完全保留，不受影响：

```bash
python3 backend/app.py    # http://127.0.0.1:5000
```

## 四、常见问题

**页面打开是空白？** 先看浏览器控制台有没有 404（通常是路径大小写问题），再确认主页仓库 `VLMTune/` 目录下的文件是最新的。

**公式显示成 `$$...$$` 原文？** 文档页的公式依赖 jsDelivr 上的 MathJax；网络受限时会降级显示 LaTeX 源码，内容不丢。实验室页面里的公式是 HTML 写的，不受影响。

**数字和本地 Flask 版不一样？** 正常情况应完全一致（已校验 12 项）。如发现偏差，先跑 `python3 tools/verify_static.py`。训练模拟的随机数用 MT19937 逐位复刻，Loss 曲线数值与后端相同。
