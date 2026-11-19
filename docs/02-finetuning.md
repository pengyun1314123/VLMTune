# 02 · 微调方法底层原理与核心代码注解

> 预训练模型像一个"博览群书但不懂规矩"的学生。微调就是给它上"专业课"：要么全面补课（全量微调），要么只给它塞一张小抄（PEFT）。

---

## 0. 为什么需要微调（PEFT）？

全量微调 7B 模型的显存账：

| 项目 | 显存（FP32 优化器状态） | 说明 |
|---|---|---|
| 模型权重（BF16） | ~14 GB | 7B × 2 bytes |
| 梯度（BF16） | ~14 GB | |
| AdamW 一阶动量 m | ~28 GB | 7B × 4 bytes |
| AdamW 二阶动量 v | ~28 GB | 7B × 4 bytes |
| **合计** | **~84 GB+** | 还不含激活值 |

**结论**：一张 80GB A100 都吃紧。PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）的思想是——冻结基座，只训练极少量新增参数（通常 0.01%–1%），却能逼近全量微调效果。

---

## 1. 全量微调（Full Fine-Tuning）

**原理**：加载预训练权重 θ₀，在任务数据上继续梯度下降。

$$\theta^* = \arg\min_{\theta} \sum_{(x,y)\in\mathcal{D}} \mathcal{L}\big(f(x;\theta), y\big)$$

**代码骨架**（注解版）：

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
model.gradient_checkpointing_enable()          # 用时间换显存：不保存中间激活，反向时重算
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)   # 所有参数都进优化器

for batch in dataloader:
    out = model(**batch)                        # forward：算 logits
    loss = out.loss                             # 交叉熵
    loss.backward()                             # 所有参数都算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防炸
    optimizer.step(); optimizer.zero_grad()     # 全员更新
```

**缺点**：显存巨大；容易灾难性遗忘；每个任务都要存一份完整模型（几十 GB）。

---

## 2. LoRA（Low-Rank Adaptation）★ 工业界默认选择

### 2.1 底层原理

论文 *LoRA: Low-Rank Adaptation of Large Language Models*（Hu et al., 2021）的核心假设：

> 模型适配新任务时，权重的变化量 ΔW 具有很低的**内在秩（intrinsic rank）**——真正有用的调整只发生在少数几个方向上。

于是不直接更新 `W₀ ∈ ℝ^{d×k}`，而是用两个小矩阵相乘来近似增量：

$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

- `A ∈ ℝ^{r×k}`：高斯随机初始化（通常用 kaiming）
- `B ∈ ℝ^{d×r}`：**初始化为全 0** → 训练开始时 ΔW=0，模型输出与原模型完全一致，不破坏预训练能力
- `r`：秩（rank），典型值 8 / 16 / 32 / 64
- `α`：缩放系数，实际生效的是 `α/r`，所以它和 r 会**互相牵制**（调 r 不改 α = 偷偷改了学习率）

**参数量对比**：

| 方法 | 可训练参数 |
|---|---|
| 全量 | `d × k` |
| LoRA | `r × (d + k)` |

以 d=k=4096、r=16 为例：`4096×4096 = 1678万` → `16×(4096+4096) = 13万`，**压缩约 128 倍**。7B 模型全体 attention 层加 LoRA 后，可训练参数约 700 万（0.1%）。

**为什么推理不慢**：训练完把 `BA` 加回 `W₀`，得到普通权重矩阵，前向传播没有任何额外算子 → **零推理延迟**（这是 LoRA 相比 Adapter / Prefix Tuning 的决定性优势）。

### 2.2 核心代码注解（手写版，帮助理解原理）

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """把一个 Linear 层改造成 LoRA 版本。"""
    def __init__(self, linear: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        # ① 冻结原始权重：requires_grad=False，优化器不会碰它
        self.base = linear
        for p in self.base.parameters():
            p.requires_grad = False

        d, k = linear.weight.shape          # out_features=d, in_features=k
        self.r, self.scaling = r, alpha / r # ② 真正的缩放是 alpha/r

        # ③ A: (r, k) 随机初始化；B: (d, r) 全 0 初始化
        self.A = nn.Parameter(torch.empty(r, k))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base_out = self.base(x)             # 原模型输出（W0 x）
        lora_out = self.dropout(x) @ self.A.T @ self.B.T * self.scaling
        return base_out + lora_out          # ④ 叠加低秩增量

    def merge(self):
        """训练后合并回原权重 → 推理零开销"""
        with torch.no_grad():
            self.base.weight.data += self.scaling * (self.B @ self.A)
        # 之后可丢弃 A/B，只保留 base
```

### 2.3 使用 HuggingFace PEFT 的真实写法

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                              # 秩：容量旋钮
    lora_alpha=32,                     # 缩放：通常设 2r
    lora_dropout=0.05,                 # 小数据集上防过拟合
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # 注意全覆盖比只调 q/v 效果更好
    bias="none",
)
model = get_peft_model(model, config)  # 自动冻结基座、注入 A/B
model.print_trainable_parameters()     # 例如：7.2M / 7.2B = 0.10%
# ...正常训练循环...
model.save_pretrained("./lora-adapter")  # 只存几十 MB 的适配器
```

---

## 3. QLoRA：消费级显卡微调大模型 ★ 资源受限首选

### 3.1 三大核心技术（论文 *QLoRA*, Dettmers et al., 2023）

1. **NF4（4-bit NormalFloat）**：针对正态分布权重设计的最优 4bit 数据类型。权重通常服从 N(0,σ)，NF4 按分位数均匀划分量化区间，比普通 INT4 信息损失更小。
2. **双重量化（Double Quantization）**：量化用的缩放常数（scale）本身也占显存（每个 64 权重块一个 fp32 scale ≈ 0.5 bit/权重）。对这些 scale 再做一次 8bit 量化，再省约 0.4 bit/权重。
3. **分页优化器（Paged Optimizers）**：用 NVIDIA 统一内存，显存溢出时自动把优化器状态换出到 CPU 内存，避免 OOM 崩溃。

**工作流**：基座权重以 4bit NF4 存储（冻结）→ 前向/反向时即时**反量化到 BF16** 计算 → 梯度只更新 BF16 的 LoRA 参数。即"**4bit 存储，16bit 计算**"。

$$\text{存储}: W_{\text{NF4}} \quad\rightarrow\quad \text{计算}: \hat{W}_{\text{BF16}} = \text{dequant}(W_{\text{NF4}}),\quad \Delta W = \tfrac{\alpha}{r}BA$$

**效果**：65B 模型微调显存从 >780GB 压到 <48GB，单张 48GB 显卡即可完成。

### 3.2 代码骨架

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # ① NF4
    bnb_4bit_use_double_quant=True,         # ② 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # ③ 计算时用 BF16
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B", quantization_config=bnb_config, device_map="auto"
)
model = get_peft_model(model, lora_config)  # 在 4bit 基座上挂 BF16 LoRA
```

---

## 4. Adapter：在层间插小模块

**原理**：在每个 Transformer Block 的 Attention / FFN 之后插入瓶颈结构：

$$\text{Adapter}(x) = x + W_{\uparrow}\,\sigma(W_{\downarrow}\,\text{LN}(x))$$

- `W_↓`：降维 `d → r`（r 如 64）；`W_↑`：升维 `r → d`
- 只训 Adapter 参数，基座冻结

**缺点**：多了一串串行小矩阵运算，**推理变慢且无法合并进原权重**。适合需要频繁切换多任务的场景（每个任务存一个小 Adapter）。

---

## 5. Prompt / Prefix / P-Tuning：只学"软提示"

### 5.1 Prompt Tuning

在输入 embedding 前拼接 `n` 个可训练向量（soft prompt）：

$$X' = [P_1, \dots, P_n;\, e(x_1), \dots, e(x_m)]$$

只训 `P ∈ ℝ^{n×d}`（可能只有几千个参数）。**缺点**：小模型上效果差；占用上下文窗口。

### 5.2 Prefix Tuning

不只改输入，而是在**每一层**的注意力 K/V 前都拼上可训练前缀：

$$\text{Attn}(Q, [P_K; K], [P_V; V])$$

表达力更强，但训练不稳定、参数更多（每层每头都有）。

### 5.3 P-Tuning v2

用轻量编码器（MLP/LSTM）生成 soft prompt，而非独立学习每个向量；同时在每层加前缀。训练更稳定，是软提示路线的代表。

> 一句话区分：**Adapter 改结构（推理慢）、Prefix/Prompt 改输入（占窗口）、LoRA 改权重（可合并、零延迟）**——这就是 LoRA 胜出的根本原因。

---

## 6. 对齐：RLHF 与 DPO

### 6.1 RLHF 三阶段（ChatGPT 背后的技术）

1. **SFT**：用人工示范数据做监督微调，得到初始策略 π^SFT。
2. **奖励模型（RM）**：收集人类对同一问题多个回答的偏好排序，训练奖励模型 `R(x,y)`。
3. **PPO 强化学习**：最大化奖励，同时用 KL 散度约束不要偏离 SFT 策略太远：

$$\max_{\pi}\; \mathbb{E}_{x\sim D,\, y\sim\pi(\cdot|x)}\big[R(x,y)\big] - \beta\, \mathbb{D}_{\mathrm{KL}}\big(\pi \,\|\, \pi_{\text{ref}}\big)$$

**缺点**：同时维护 4 个模型（策略/参考/RM/价值），训练复杂易崩，存在 reward hacking。

### 6.2 DPO（Direct Preference Optimization）★ 更简单的对齐

DPO 推导出上述 RLHF 目标的最优解有闭式形式，直接用偏好数据优化策略，跳过 RM 和 RL：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

- `y_w`（winner）：偏好的回答；`y_l`（loser）：不偏好的回答
- `β`：控制偏离参考模型的程度
- **直觉**：拉高好回答相对参考模型的概率，压低坏回答的相对概率

```python
# DPO 核心损失（概念版）
pi_logps_w = model.log_prob(prompt, chosen)      # 好回答的 log π
pi_logps_l = model.log_prob(prompt, rejected)    # 坏回答的 log π
ref_logps_w = ref_model.log_prob(prompt, chosen) # 冻结参考模型
ref_logps_l = ref_model.log_prob(prompt, rejected)

logits = beta * ((pi_logps_w - ref_logps_w) - (pi_logps_l - ref_logps_l))
loss = -F.logsigmoid(logits).mean()              # 让 logits 越大越好
```

---

## 7. 方法对比与选型指南

| 方法 | 可训练参数 | 推理延迟 | 显存需求 | 典型场景 |
|---|---|---|---|---|
| 全量微调 | 100% | 无 | 极高 | 有充足算力、追求极致效果 |
| **LoRA** | 0.1–1% | **零（可合并）** | 低 | **默认首选** |
| **QLoRA** | 0.1–1% | 零（可合并） | 极低 | 单卡消费级显卡 |
| Adapter | 0.5–5% | 有 | 低 | 多任务热切换 |
| Prompt Tuning | <0.01% | 无 | 极低 | 大模型(11B+)、简单任务 |
| Prefix / P-Tuning v2 | 0.1–1% | 略占上下文 | 低 | NLP 理解类任务 |
| RLHF / DPO | 取决于底座 | 无 | 中–高 | 人类偏好对齐、聊天助手 |

**实践口诀**：
- 首选 LoRA；显存不够 → QLoRA。
- `r` 从 16 起，`alpha = 2r`，target 覆盖所有线性层通常更稳。
- 先跑通小数据过拟合测试（确认流程无误），再扩大数据。
- 对齐阶段用 DPO 比 RLHF 简单得多，效果 often 相当。

---

## 8. 训练稳定性工程要点

| 技巧 | 作用 |
|---|---|
| 混合精度（BF16/FP16） | 前向计算减半显存、提速；优化器状态仍用 FP32 |
| 梯度累积 | 用多个小 batch 模拟大 batch：`loss / accum_steps` |
| 梯度检查点 | 不存中间激活，反向重算，显存 ↓ 30%，速度 ↓ 20% |
| Warmup + Cosine LR | 先线性升到峰值再余弦衰减，避免初期梯度爆炸 |
| 梯度裁剪（clip=1.0） | 防止偶发大梯度破坏训练 |
| Flash Attention | 精确注意力的 IO 优化实现，长序列提速显著 |
