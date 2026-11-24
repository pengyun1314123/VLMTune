# 03 · 训练数据全流程：从原始网页到可训练张量

> 本文档配套 `data/` 下的样例数据集，前端"数据流水线"页面会逐步展示每一阶段的**输入 → 处理 → 产物（真实文件片段）**。

---

## 0. 全流程总览（12 个阶段）

```
① 原始采集  ② 文本清洗  ③ 语言识别/质量过滤  ④ 去重  ⑤ 安全与去污染
     ⑥ 分词(BPE)  ⑦ 数值化  ⑧ 序列打包(Packing)  ⑨ 构造 Labels
          ⑩ Batch/Collator  ⑪ 训练 & Checkpoint  ⑫ 评估(Perplexity)
```

**最重要的认知**：`数据不是文件夹，而是一条可审计的流水线`。模型出问题，第一反应应该是回溯数据，而不是先改模型。

---

## 1. 样例数据集说明

本工程内置一个** miniature（迷你）教学语料** `data/raw_corpus.jsonl`，刻意加入了真实世界里的各种"脏数据"，方便观察每个阶段的清洗效果：

| 编号 | 类型 | 故意埋设的问题 | 教学目的 |
|---|---|---|---|
| doc-001 | 网页正文 | HTML 标签、导航残留、多余空白 | 演示清洗 |
| doc-002 | 网页正文 | 正常高质量文本 | 对照样本 |
| doc-003 | 网页正文 | 与 doc-002 完全重复 | 演示精确去重 |
| doc-004 | 网页正文 | 与 doc-002 高度相似（仅改几个词） | 演示 MinHash 近似去重 |
| doc-005 | 论坛文本 | 包含邮箱、手机号 | 演示 PII 脱敏 |
| doc-006 | 代码片段 | 中英混合、代码 | 演示语言识别与保留策略 |
| doc-007 | 垃圾文本 | 极短、符号堆砌 | 演示质量过滤 |
| doc-008 | 基准测试题 | 与验证集重合 | 演示 Benchmark 去污染 |

另有 `data/sft_instruct.jsonl`（指令微调样例）与 `data/multimodal_instruct.json`（多模态指令样例，LLaVA 风格），用于展示预训练之后的 SFT 与多模态阶段。

---

## 2. 逐阶段详解（输入 / 处理 / 产物）

### ① 原始采集（Collection）

- **输入**：Common Crawl WARC / 书籍 / 代码仓库 / 维基百科（真实场景 PB 级）
- **处理**：解析 HTML/PDF，抽取正文；打上 metadata（source、url、language、license）
- **产物**：`raw_corpus.jsonl`，每行一个 JSON 对象：`{"id", "source", "text", "meta"}`
- **样例**：
  ```json
  {"id":"doc-001","source":"web","text":"<html><body><div class='nav'>首页 | 登录</div><p>大模型  训练需要  大量  文本数据。</p></body></html>","meta":{"url":"..."}}
  ```
- **要点**：质量上限在这一步就被决定了。直接抽 WARC 正文通常优于用现成的 WET 文本（WET 含大量导航/页脚模板）。

### ② 文本清洗与规范化（Normalization）

- **输入**：上一步的 `text` 原始字符串
- **处理**：去 HTML 标签 → Unicode 规范化（NFC）→ 压缩多余空白 → 去脚本/样式/cookie 横幅 → 统一引号破折号
- **产物**：干净文本
- **样例**：
  ```
  前: <html><body><div class='nav'>首页 | 登录</div><p>大模型  训练需要  大量  文本数据。</p></body></html>
  后: 首页 | 登录
  大模型 训练需要 大量 文本数据。
  ```
- **要点**：便宜的过滤器放前面，昂贵的（分类器、MinHash）放后面。

### ③ 语言识别与质量过滤（LangID + Quality Filter）

- **输入**：清洗后的文档
- **处理**：fastText 语言识别；Gopher 启发式规则（文档长度、平均词长、符号/文字比例、重复行比例）；教育价值分类器（FineWeb-Edu 思路）
- **产物**：`filtered_corpus.jsonl` + 过滤原因日志
- **样例**：doc-007 因"文本长度 < 50 且符号比例 > 0.5"被丢弃，日志记录 `{"id":"doc-007","drop_reason":"low_quality: symbol_ratio=0.72"}`
- **要点**：过滤器会偏置数据分布，要定期抽样"被丢掉的内容"人工检查。

### ④ 去重（Deduplication）

- **输入**：过滤后的文档集合
- **处理**：
  - 精确去重：对文档/段落做哈希（SHA1/MD5），相同哈希只留一份
  - 近似去重：**MinHash + LSH**——把文档转成 n-gram shingle 集合，用 MinHash 签名近似 Jaccard 相似度，LSH 分桶避免 O(n²) 两两比较
  - 行级去重：删除跨文档重复出现的导航行
- **产物**：`deduped_corpus.jsonl` + 重复组报告
- **样例**：doc-003（与 doc-002 完全相同）被精确去重删除；doc-004（相似度高）被 MinHash 判为近重复并删除。
- **数学**：Jaccard 相似度
  $$J(A,B)=\frac{|A\cap B|}{|A\cup B|},\quad \Pr[\min h(A)=\min h(B)] = J(A,B)$$
- **要点**：重复数据会让模型对某些文本"多训几个 epoch"，放大记忆化和基准污染。通常能去掉网页数据的 30%–60%。

### ⑤ 安全、隐私与去污染（Safety / PII / Decontamination）

- **输入**：去重后的语料
- **处理**：正则检测邮箱/电话/IP 并脱敏； toxicity 分类器；与验证集/基准做 n-gram 重叠检测
- **产物**：`safe_corpus.jsonl`
- **样例**：
  ```
  前: 联系方式: alice@example.com, 电话 138-0000-1234
  后: 联系方式: <EMAIL>, 电话 <PHONE>
  ```
  doc-008 因与 benchmark 题目 13-gram 重叠被标记并移除。
- **要点**：合规不是可选项。

### ⑥ 分词（Tokenization）

- **输入**：安全语料中的字符串
- **处理**：BPE / SentencePiece 子词切分 → 词表映射
- **产物**：token 序列（教学 demo 中展示切分结果与 ID）
- **样例**：
  ```
  文本: The model is unfriendly.
  BPE:  ['The', 'model', 'is', 'un', 'friend', 'ly', '.']
  IDs:  [12, 573, 23, 891, 229, 48, 5]
  ```
- **要点**：子词分词解决 OOV 问题，词表通常 50k–150k。BPE 通过不断合并语料中最高频的相邻字符对来构建词表。

### ⑦ 数值化与注意力掩码（Numericalization）

- **输入**：token 列表
- **处理**：token → ID；加特殊 token（BOS/EOS/PAD）；生成 attention_mask
- **产物**：`input_ids`, `attention_mask`
- **样例**：
  ```json
  {"input_ids":[1, 12, 573, ..., 2], "attention_mask":[1,1,1,...,1]}
  ```

### ⑧ 序列打包（Packing）

- **输入**：变长 token 序列
- **处理**：把多条短文本用 EOS 拼接后切成长度恰好为 `seq_len`（如 2048/4096）的定长块
- **产物**：定长张量，GPU 利用率大幅提升
- **样例**：3 篇短文档（300 + 700 + 1100 tokens）→ 拼成一条 `len=2100` → 切成 `2048 + 52`
- **要点**：挤压 padding 浪费。`有效token预算 ≈ 名义预算 ×(1−重复率)×(1−padding率)×解析成功率×质量系数`。

### ⑨ 构造 Labels（因果 LM 的"练习题"）

- **输入**：`input_ids = [t₁, t₂, …, t_T]`
- **处理**：labels 就是 input_ids 右移一位；第一个位置用 −100 忽略（PyTorch cross_entropy 忽略 −100）
- **产物**：`(input_ids, labels)` 对
- **样例**：
  ```
  input_ids: [我, 今, 天, 学, 习, 大, 模, 型]
  labels:    [-100, 今, 天, 学, 习, 大, 模, 型]
  含义: 看到"我"预测"今天"，看到"我今天"预测"学"……一句话产生 T-1 个训练信号
  ```
- **注意**：多模态样本中，图像 token 位置与 prompt 位置的 labels 也设为 −100，只对回答部分计算 loss。

### ⑩ Batch 与 Collator

- **输入**：定长样本
- **处理**：组 batch（如 batch_size=64）；动态 padding；构造 `position_ids`
- **产物**：`{"input_ids": [B, T], "attention_mask": [B, T], "labels": [B, T]}`
- **要点**：DataLoader 的吞吐决定 GPU 是否"饿着"。大规模训练用 Parquet 列存 + 流式 IterableDataset（FineWeb ~45TB 不可能全部读入内存）。

### ⑪ 训练与 Checkpoint

- **输入**：batch 张量
- **处理**：forward → loss → backward → optimizer.step；混合精度、梯度累积、梯度检查点；定期存 checkpoint
- **产物**：
  - `checkpoint-1000/`：`model.safetensors`（权重）、`optimizer.pt`、`scheduler.pt`、`config.json`、训练状态（step、loss 日志）
  - LoRA 场景只存 `adapter_model.safetensors`（几十 MB）
- **要点**：checkpoint 是"可恢复的训练快照"，断电/抢占后能接着训。

### ⑫ 评估（Perplexity）

$$PPL = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{\lt t})\right)$$

- **输入**：留出集（held-out）
- **产物**：loss 曲线、PPL 数值、下游任务指标
- **人话**：PPL 可以理解为"模型在每个位置上平均在多少个词之间犹豫"，越低越好。

---

## 3. 预训练之后：SFT 与多模态数据

### 3.1 SFT 指令数据（`data/sft_instruct.jsonl`）

格式（Alpaca 风格）：

```json
{"instruction":"把下面的话改写成礼貌用语","input":"给我水","output":"请给我一杯水，谢谢。"}
```

训练时套上 chat 模板（如 ChatML / Llama-3 模板）：

```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
把下面的话改写成礼貌用语：给我水<|im_end|>
<|im_start|>assistant
请给我一杯水，谢谢。<|im_end|>
```

**只对 assistant 段计算 loss**（模板和 user 段 labels = −100）。常见开源数据集：FLAN、Dolly 15K、OpenAssistant、ShareGPT；合成数据方法：Self-Instruct、Evol-Instruct（WizardLM）、Orca 推理链蒸馏。

### 3.2 多模态指令数据（`data/multimodal_instruct.json`）

LLaVA-Instruct-150K 风格：

```json
{
  "id": "mm-001",
  "image": "images/cat_on_sofa.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n图里有什么？"},
    {"from": "gpt",   "value": "一只橘猫趴在沙发上。"},
    {"from": "human", "value": "它在做什么？"},
    {"from": "gpt",   "value": "它正闭着眼睛睡觉。"}
  ]
}
```

**多模态数据流水线额外产物**：

| 阶段 | 产物 | 说明 |
|---|---|---|
| 视觉编码 | `image_features [N, d_v]` | ViT 输出的 patch 特征，N = (H/P)·(W/P) |
| 投影对齐 | `image_embeds [N, d_l]` | MLP 投影到 LLM 维度 |
| 序列拼接 | `input_embeds [N+M, d_l]` | 图像 token + 文本 token |
| 训练掩码 | `labels` | 仅回答位置参与 loss |

---

## 4. 各阶段产物速查表

| 阶段 | 输入 | 输出（产物文件 / 张量） | 关键指标 |
|---|---|---|---|
| 采集 | 网页/书/代码 | `raw_corpus.jsonl` | 来源分布、字节量 |
| 清洗 | 原始文本 | 干净文本 | 保留率 |
| 过滤 | 干净文本 | `filtered_corpus.jsonl` | 各类过滤占比 |
| 去重 | 过滤文本 | `deduped_corpus.jsonl` | 重复率 |
| 安全 | 去重文本 | `safe_corpus.jsonl` | PII/污染命中数 |
| 分词 | 字符串 | token 列表 | 压缩率、词表覆盖率 |
| 打包 | token 流 | `packed_2048.bin` | padding 率 |
| Labels | input_ids | `(input_ids, labels)` | 有效监督 token 数 |
| Batch | 样本 | `[B,T]` 张量 | 吞吐 tokens/s |
| 训练 | batch | `checkpoint-*` | loss、显存占用 |
| 评估 | 留出集 | PPL / 指标 | PPL ↓ |
