# VLMTune · 大模型 / 多模态训练教学实验室

面向零基础学习者的交互式教学工程：从架构原理、核心公式、微调方法底层原理与代码注解，到完整的数据训练流水线，配合**真实运行的样例数据与计算**把每个阶段讲清楚。

## 在线体验（无需部署）

> 🌐 **直接看网页版**：[**https://pengyun1314123.github.io/VLMTune/**](https://pengyun1314123.github.io/VLMTune/)
>
> 无需安装、无需启动后端，点开链接即可完整体验所有交互演示。喜欢的话欢迎点进 [GitHub 仓库](https://github.com/pengyun1314123/VLMTune/) 点个 Star ⭐ 跟踪后续更新。

## 快速开始

```bash
# 1. 安装依赖（Python 3.9+）
pip install flask numpy

# 2. 启动后端（同时托管前端）
python backend/app.py

# 3. 浏览器打开
http://127.0.0.1:5000/
```

> 本工程使用可直接运行的教学级实现（迷你 BPE、numpy 注意力、真实数据清洗流水线、训练模拟器），不依赖 GPU 和 PyTorch，方便小白在任何机器上运行理解。想迁移到真实训练时，文档中的代码骨架给出了 HuggingFace / PEFT 的标准写法。

## 工程结构

```
VLMTune/
├── docs/                          # 系统化知识文档（先读这里）
│   ├── 01-architecture.md         # 架构知识点 + 核心公式速查
│   ├── 02-finetuning.md           # 微调方法原理 + 逐行代码注解
│   ├── 03-data-pipeline.md        # 数据全流程 + 样例数据各阶段产物
│   └── 04-github-pages.md         # 静态站发布 + 主页链接接入说明
├── data/                          # 内置样例数据集
│   ├── raw_corpus.jsonl           # 8 篇故意埋了脏数据的原始语料
│   ├── sft_instruct.jsonl         # SFT 指令样本
│   └── multimodal_instruct.json   # LLaVA 风格多模态指令样本
├── backend/                       # Python Flask 后端
│   ├── app.py                     # API 与静态文件托管
│   └── demos/
│       ├── pipeline.py            # 12 阶段数据流水线（真实处理）
│       ├── tokenizer.py           # 教学级 BPE 实现
│       ├── finetuning.py          # LoRA 参数/显存精算 + 注意力计算
│       └── training.py            # 训练过程模拟器
├── frontend/                      # HTML + JS + Three.js 前端
│   ├── index.html
│   ├── css/style.css
│   ├── js/app.js                  # 交互逻辑
│   ├── js/visuals.js              # Three.js 3D 可视化
│   └── vendor/three.min.js        # 本地 Three.js（离线可用）
├── tools/                         # 静态站构建与校验
│   ├── build_static.py            # 生成 site/
│   ├── static-api.js              # 浏览器端算法（LoRA/训练/BPE）
│   ├── verify_static.py           # 前后端数值一致性校验
│   └── publish_to_homepage.py     # 一键重建 + 同步 + 发布到主页
└── site/                          # 生成的静态站（同步到主页仓库 /VLMTune/）
```

## 静态站工作流

```bash
python3 tools/build_static.py      # 预生成 API 数据 + 文档转 HTML + 装配前端
python3 tools/verify_static.py     # 校验 JS 算法与 Python 后端数值一致
python3 tools/publish_to_homepage.py  # 一键：重建 + 校验 + 同步 + 推送上线
cd site && python3 -m http.server 8000    # 本地预览静态版
```

## 前端交互页面

| 页面 | 内容 |
|---|---|
| 总览首页 | 学习路线、核心概念速览、3D Transformer 信息流 |
| 架构实验室 | Transformer Block 3D 结构、真实注意力热力图、BPE 分词演示 |
| 数据流水线 | 12 阶段逐步导航，展示每阶段输入/处理/产物/真实样例片段；SFT 模板 |
| 微调游乐场 | LoRA 参数与显存实时计算器、方法全景对比、带注解代码、训练曲线模拟 |
| 多模态世界 | ViT+投影层+LLM 3D 架构、逐阶段产物、LLaVA 两阶段训练 |
| 知识文档 | 三篇 Markdown 文档的入口与学习路线 |

## 后端 API

| 端点 | 说明 |
|---|---|
| `GET /api/health` | 健康检查 |
| `GET /api/pipeline` | 12 阶段流水线的真实输入/产物/统计 |
| `GET /api/tokenize?text=...` | BPE 分词演示 |
| `GET /api/sft` | SFT 样本与 ChatML 模板渲染 |
| `GET /api/multimodal` | 多模态数据流转各阶段产物 |
| `GET /api/lora?r=16&alpha=32&d=4096&layers=32` | LoRA 参数量与显存精算 |
| `GET /api/methods` | 微调方法对比与训练技巧 |
| `GET /api/attention` | numpy 真实计算的注意力矩阵 |
| `GET /api/training/simulate?steps=200&method=lora&quality=0.85` | 训练 loss / LR 曲线模拟 |

## 建议学习顺序

1. 阅读 `docs/01-architecture.md` 建立 Transformer 与多模态架构直觉
2. 前端「架构实验室」看 3D 结构与注意力热力图
3. 阅读 `docs/03-data-pipeline.md` 并打开「数据流水线」跟着样例数据走一遍
4. 阅读 `docs/02-finetuning.md` 并在「微调游乐场」调参、读代码、看曲线
5. 打开「多模态世界」理解视觉-语言对齐与两阶段训练

## 知识点覆盖

- **预训练**：CLM/MLM 目标、数据清洗/去重(MinHash)/PII/去污染、BPE、packing、labels 构造、checkpoint、PPL
- **架构**：注意力/MHA/GQA、RoPE、RMSNorm、SwiGLU、KV Cache、ViT patch、MLP Projector
- **微调**：Full FT、LoRA(ΔW=BA, α/r, merge)、QLoRA(NF4/双重量化/分页优化器)、Adapter、Prompt/Prefix/P-Tuning v2、BitFit、RLHF/PPO、DPO
- **多模态**：CLIP/ViT、patch token、模态对齐、LLaVA 两阶段训练、多模态 loss 掩码
