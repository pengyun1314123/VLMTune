# 01 · 架构知识点与核心公式（LLM / 多模态大模型）

> 面向零基础读者：先建立"整体地图"，再逐层拆开。文中所有公式都配有"人话解释"和对应的可交互演示入口（前端页面）。

---

## 0. 一张图建立全局观

```
文本 / 图像 / 音频
      │
      ▼
[ Tokenization ]  → 把人类数据切成模型能读的整数 ID
      │
      ▼
[ Embedding + Position ]  → ID 变成向量，并注入"第几个"的信息
      │
      ▼
[ Transformer Block × N ]  → 注意力 + 前馈网络，反复提炼上下文
      │
      ▼
[ LM Head ]  → 输出词表上每个词的概率，采样出下一个 token
      │
      ▼
[ 训练目标 ]  → 预训练（预言下一个词） / 微调（听指令、对齐人类偏好）
```

**核心直觉**：大模型做的事情，本质上只有一个——"根据前文，预测下一个词"。预训练把它喂成博览群书的基础模型；微调教它按人的要求说话；多模态则是给它增加"眼睛"和"耳朵"，把图像/声音也变成一种"语言" token 送进同一个 Transformer。

---

## 1. 语言建模目标（一切的根）

### 1.1 自回归语言模型（Causal LM）

给定一段 token 序列 `x = (x₁, x₂, …, x_T)`，模型把联合概率链式分解：

$$P(x) = \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t})$$

训练目标 = 最大化训练语料的对数似然（等价于最小化交叉熵）：

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

**人话**：第 t 个位置只看它前面的词，预测自己。一句话里每个词都能产生一个训练信号，所以一本 1T token 的语料就有 1T 个"练习题"。

### 1.2 掩码语言模型（MLM，BERT 系，了解即可）

随机把输入中 15% 的 token 换成 `[MASK]`，让模型根据双向上下文猜回来：

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P_\theta(x_i \mid x_{\setminus \text{masked}})$$

> 现代主流生成式大模型（GPT / Llama / Qwen）几乎都用 CLM；MLM 多用于理解类任务。

---

## 2. Transformer Block：大模型的心脏

一个标准 Decoder-only Block：

```
输入 x
 ├─→ RMSNorm → Multi-Head Attention (带因果掩码) ─┐
 │                                                ├─→ + 残差 → y
 └───────────────────────────────────────────────┘
 y ├─→ RMSNorm → FFN (SwiGLU) ───────────────────┐
  └──────────────────────────────────────────────┘─→ + 残差 → 输出
```

### 2.1 缩放点积注意力（Scaled Dot-Product Attention）

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V$$

- `Q`（Query）：我在找什么；`K`（Key）：我有什么；`V`（Value）：我实际提供的内容。
- `d_k`：每个头的维度。除以 `√d_k` 是为了防止点积值过大导致 softmax 梯度消失。
- `M`：因果掩码（下三角为 0，上三角为 −∞），保证第 t 个位置看不到未来。

**人话**：每个词拿自己的 Q 去和所有词的 K 打分，分数越高说明越"相关"，然后按分数加权把 V 汇总起来。这就是"上下文理解"的数学本质。

### 2.2 多头注意力（MHA）

$$\text{head}_i = \text{Attention}(XW_i^Q,\; XW_i^K,\; XW_i^V), \quad
\text{MHA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h) W^O$$

每个头在不同子空间里关注不同的关系（语法、指代、位置……）。

### 2.3 GQA / MQA（省 KV Cache 的关键工程优化）

| 机制 | K/V 头数 | 效果 |
|---|---|---|
| MHA | h（与 Q 相同） | 质量最好，KV Cache 最大 |
| GQA | g（1 < g < h，分组共享） | Llama 2/3、Qwen 常用，几乎不掉点，显存省很多 |
| MQA | 1 | 最省显存，质量略有损失 |

**为什么重要**：推理时逐 token 生成，需要缓存之前所有 token 的 K/V。KV Cache 大小 ∝ 层数 × KV头数 × 序列长度 × d_k。GQA 把 KV 头数从 h 降到 g，是长上下文推理能跑起来的核心。

### 2.4 位置编码：RoPE（旋转位置编码）

$$f_{\text{RoPE}}(x_m, m) = x_m \, e^{i m \theta}, \quad \theta_j = 10000^{-2j/d}$$

实际实现中用旋转矩阵：对向量两两分组，按位置 m 旋转一个角度。

**人话**：把"绝对位置"编码成向量旋转的角度，使得两个 token 的注意力分数只依赖于它们的**相对距离**，同时保留绝对位置信息。优点是可外推到更长序列，被 Llama / Qwen / ChatGLM 广泛采用。

### 2.5 前馈网络 FFN（SwiGLU）

$$\text{FFN}_{\text{SwiGLU}}(x) = \big(\text{Swish}(xW_1) \odot xW_3\big) W_2$$

其中 `Swish(z) = z · σ(z)`。相比经典 ReLU FFN，SwiGLU 用门控机制提升表达能力，是当代主流配置。

### 2.6 RMSNorm 与残差

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot g$$

残差连接 `x + SubLayer(Norm(x))` 让梯度能直达底层，是训练深达几十上百层的关键。

---

## 3. 多模态大模型：给 LLM 装上眼睛

### 3.1 总体架构（以 LLaVA 为代表）

```
                    ┌──────────────────────────┐
   图像 → ViT 视觉编码器 → 图像特征 (patch tokens) │
                    └────────────┬─────────────┘
                                 ▼
                    [ 投影层 Projector (MLP) ]
                                 ▼
   文本 → Tokenizer → 文本 Embedding ─┐
                                      ├→ 拼接 → LLM (Vicuna/Llama) → 回答
   图像 token ────────────────────────┘
```

### 3.2 视觉编码器（Vision Transformer）

图像 `I ∈ ℝ^{H×W×3}` 被切成 `P×P` 的 patch：

$$v_i = \text{Flatten}(\text{Patch}_i)\, W_{\text{patch}} + b_{\text{patch}}, \quad i=1,\dots,N,\; N=\frac{HW}{P^2}$$

再加位置编码、 `[CLS]` token，送进标准 Transformer 得到视觉特征 `V ∈ ℝ^{N×d_v}`。CLIP 的 ViT-L/14 是最常用的视觉塔之一（patch=14×14）。

### 3.3 对齐投影层（Projector）

视觉特征维度 `d_v` 与 LLM 词嵌入维度 `d_l` 通常不同，需要投影：

$$H_v = \text{MLP}(V) \in \mathbb{R}^{N \times d_l}$$

简单的线性层或两层 MLP 即可。这一层是**模态对齐的桥梁**——训练初期只冻住两边、只训它，就能让 LLM "看懂"图像特征。

### 3.4 LLaVA 两阶段训练

| 阶段 | 训练参数 | 数据 | 目标 |
|---|---|---|---|
| Stage 1 特征对齐 | 只训 Projector | CC3M 等图文对（约 59.5 万） | 让图像 token 落在 LLM 的语义空间 |
| Stage 2 端到端指令微调 | Projector + LLM（或加 LoRA） | 158K 多模态指令数据（对话 58K / 详细描述 23K / 复杂推理 77K） | 学会按指令回答关于图像的问题 |

> Qwen-VL / InternVL 等多模态模型架构思想类似：视觉塔 + 对齐层 + LLM，差别在视觉编码器选择、对齐层设计（如 Q-Former、Perceiver Resampler）和训练策略。

### 3.5 多模态训练损失

与文本相同，仍是自回归交叉熵，只是序列中含有图像 token（这些位置通常不计算 loss，只对文本回答部分计算）：

$$\mathcal{L} = -\sum_{t \in \text{文本回答位置}} \log P_\theta(y_t \mid y_{<t}, H_v, x_{\text{instr}})$$

---

## 4. 核心公式速查表

| 主题 | 公式 | 一句话 |
|---|---|---|
| 语言建模 | $P(x)=\prod_t P(x_t\mid x_{<t})$ | 预测下一个词 |
| 预训练损失 | $\mathcal{L}=-\sum_t \log P_\theta(x_t\mid x_{<t})$ | 交叉熵 |
| 注意力 | $\text{softmax}(QK^\top/\sqrt{d_k}+M)V$ | 相关性加权汇总 |
| 多头 | $\text{Concat}(\text{head}_i)W^O$ | 多子空间并行 |
| RoPE | $x_m e^{im\theta}$ | 位置=旋转角 |
| SwiGLU | $(\text{Swish}(xW_1)\odot xW_3)W_2$ | 门控前馈 |
| LoRA | $W'=W_0+\frac{\alpha}{r}BA$ | 低秩增量 |
| QLoRA | NF4 量化基座 + LoRA | 4bit 存储 16bit 计算 |
| DPO | $-\log\sigma\big(\beta\log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\big)$ | 直接学偏好 |

> LoRA / QLoRA / DPO 等微调公式详见 `02-finetuning.md`。

---

## 5. 建议阅读顺序（小白路线）

1. 本文 §0–§2（建立 Transformer 直觉）
2. 前端交互页 "架构实验室"（3D 看 Block、注意力热力图）
3. `03-data-pipeline.md`（跟着样例数据走一遍全流程）
4. `02-finetuning.md`（理解各种微调方法 + 代码注解）
5. 前端 "微调游乐场"（亲手算 LoRA 参数量、看损失曲线）
