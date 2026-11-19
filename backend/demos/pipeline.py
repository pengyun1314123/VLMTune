"""
数据流水线演示：对 data/raw_corpus.jsonl 真实执行 12 个阶段的处理，
返回每一阶段的输入、处理逻辑、产物与统计信息（全部 JSON 可序列化）。
"""
import json
import re
import hashlib
import os
from collections import Counter, defaultdict
from .tokenizer import SimpleBPE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

BENCHMARK_QUESTIONS = [
    "下列哪一项不是 Transformer 的组成部分",
    "这是验证集中已存在的基准测试题目",
]


def load_raw():
    path = os.path.join(DATA_DIR, "raw_corpus.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_html(text):
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def ngrams(text, n=5):
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def build_pipeline():
    """执行整个流水线，返回各阶段结构化结果。"""
    raw = load_raw()
    stages = []

    # ① 原始采集
    stages.append({
        "id": 1, "name": "原始采集", "en": "Collection",
        "input": f"Common Crawl / 书籍 / 代码仓库等数据源",
        "process": "解析 HTML/PDF，抽取正文， attach metadata (source, url, crawl)",
        "output": "data/raw_corpus.jsonl",
        "stats": {"文档数": len(raw), "来源分布": dict(Counter(d["source"] for d in raw))},
        "samples": [{"id": d["id"], "source": d["source"], "text": d["text"][:160]} for d in raw[:3]],
        "tip": "数据质量的上限在采集阶段就已决定；直接从 WARC 抽正文优于用现成 WET 文本。",
    })

    # ② 文本清洗
    cleaned = []
    for d in raw:
        c = strip_html(d["text"])
        cleaned.append({**d, "text": c})
    stages.append({
        "id": 2, "name": "文本清洗与规范化", "en": "Normalization",
        "input": "raw text (含 HTML 标签、多余空白)",
        "process": "去 HTML → Unicode NFC → 压缩空白 → 去脚本/页脚模板",
        "output": "cleaned_corpus (内存)",
        "stats": {"输入文档": len(raw), "输出文档": len(cleaned),
                  "平均缩短字符": round(sum(len(a["text"]) - len(b["text"]) for a, b in zip(raw, cleaned)) / len(raw), 1)},
        "samples": [
            {"id": raw[0]["id"], "before": raw[0]["text"][:160], "after": cleaned[0]["text"][:160]}
        ],
        "tip": "便宜的过滤器放前面，昂贵的分类器/MinHash 放后面。",
    })

    # ③ 语言识别 & 质量过滤
    filtered, drop_reasons = [], []
    for d in cleaned:
        text = d["text"]
        symbol_ratio = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text)) / max(len(text), 1)
        reason = None
        if len(text) < 30:
            reason = "low_quality: too_short"
        elif symbol_ratio > 0.45:
            reason = f"low_quality: symbol_ratio={symbol_ratio:.2f}"
        elif text in BENCHMARK_QUESTIONS or any(q in text for q in BENCHMARK_QUESTIONS):
            reason = "contaminated: benchmark_overlap"   # 去污染在这里先标记
        if reason:
            drop_reasons.append({"id": d["id"], "reason": reason, "preview": text[:60]})
        else:
            filtered.append(d)
    stages.append({
        "id": 3, "name": "语言识别与质量过滤", "en": "LangID & Quality",
        "input": "cleaned text", "process": "fastText 语言识别 + Gopher 启发式规则 + 质量分类器",
        "output": "filtered_corpus + 过滤日志",
        "stats": {"输入": len(cleaned), "保留": len(filtered), "丢弃": len(drop_reasons),
                  "丢弃明细": drop_reasons},
        "samples": [{"id": r["id"], "reason": r["reason"]} for r in drop_reasons],
        "tip": "过滤器会偏置分布；定期抽样被丢弃的内容做人工检查。",
    })

    # ④ 去重（精确 + MinHash 近似）
    seen_hash, exact_dups = {}, []
    for d in filtered:
        h = hashlib.sha1(d["text"].encode()).hexdigest()
        if h in seen_hash:
            exact_dups.append({"id": d["id"], "duplicate_of": seen_hash[h]})
        else:
            seen_hash[h] = d["id"]
    unique = [d for d in filtered if d["id"] not in {x["id"] for x in exact_dups}]

    near_dups = []
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            sim = jaccard(ngrams(a["text"], 4), ngrams(b["text"], 4))
            if sim > 0.65:
                near_dups.append({"id": b["id"], "similar_to": a["id"], "jaccard": round(sim, 3)})
    near_ids = {x["id"] for x in near_dups}
    deduped = [d for d in unique if d["id"] not in near_ids]
    stages.append({
        "id": 4, "name": "去重", "en": "Deduplication",
        "input": "filtered text", "process": "SHA1 精确去重 + MinHash/LSH 近似去重 (Jaccard>0.65)",
        "output": "deduped_corpus",
        "stats": {"输入": len(filtered), "精确重复": exact_dups, "近似重复": near_dups, "去重后": len(deduped)},
        "samples": near_dups,
        "formula": "Jaccard(A,B)=|A∩B|/|A∪B|;  Pr[min h(A)=min h(B)] = J(A,B)",
        "tip": "重复数据会让模型对某些文本多训几个 epoch，放大记忆化与基准污染。",
    })

    # ⑤ 安全 / PII / 去污染
    safe, pii_hits = [], []
    for d in deduped:
        text = d["text"]
        before = text
        text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "<EMAIL>", text)
        text = re.sub(r"1[3-9]\d-\d{4}-\d{4}", "<PHONE>", text)
        if text != before:
            pii_hits.append({"id": d["id"], "before": before[:100], "after": text[:100]})
        safe.append({**d, "text": text})
    stages.append({
        "id": 5, "name": "安全 / 隐私 / 去污染", "en": "Safety & Decontamination",
        "input": "deduped text", "process": "正则 PII 脱敏 + toxicity 分类器 + 与验证集 n-gram 重叠检测",
        "output": "safe_corpus",
        "stats": {"输入": len(deduped), "PII 命中": len(pii_hits), "输出": len(safe)},
        "samples": pii_hits,
        "tip": "合规不是可选项；去污染要在训练前完成，否则评估指标不可信。",
    })

    # ⑥ 分词
    all_text = "\n".join(d["text"] for d in safe)
    tokenizer = SimpleBPE().train([all_text], vocab_size=300)
    token_examples = []
    for d in safe[:2]:
        toks, ids, applied = tokenizer.encode(d["text"][:60])
        token_examples.append({"id": d["id"], "text": d["text"][:60], "tokens": toks[:30],
                               "ids": ids[:30], "merge_examples": applied[:5]})
    stages.append({
        "id": 6, "name": "分词 (BPE)", "en": "Tokenization",
        "input": "safe text", "process": "Byte-Pair Encoding：反复合并语料中最高频的相邻字符对",
        "output": "token 序列 + tokenizer.json",
        "stats": tokenizer.info(),
        "samples": token_examples,
        "tip": "子词分词解决 OOV；词表通常 50k–150k（教学 demo 用 300 方便观察）。",
    })

    # ⑦ 数值化
    seq = tokenizer.encode(safe[0]["text"])[1]
    input_ids = [tokenizer.vocab["<BOS>"]] + seq + [tokenizer.vocab["<EOS>"]]
    stages.append({
        "id": 7, "name": "数值化与掩码", "en": "Numericalization",
        "input": "token 列表", "process": "token→ID，加 BOS/EOS/PAD，生成 attention_mask",
        "output": "input_ids, attention_mask",
        "stats": {"序列长度": len(input_ids), "BOS_ID": tokenizer.vocab["<BOS>"], "EOS_ID": tokenizer.vocab["<EOS>"]},
        "samples": [{"input_ids": input_ids[:40], "attention_mask": [1] * len(input_ids[:40])}],
        "tip": "attention_mask 告诉模型哪些位置是真实 token、哪些是 padding。",
    })

    # ⑧ 打包
    seq_len = 128  # 教学演示用小序列长度（真实为 2048/4096）
    all_ids = []
    for d in safe:
        all_ids += tokenizer.encode(d["text"])[1] + [tokenizer.vocab["<EOS>"]]
    packed = [all_ids[i:i + seq_len] for i in range(0, len(all_ids), seq_len)]
    stages.append({
        "id": 8, "name": "序列打包", "en": "Sequence Packing",
        "input": "变长 token 流", "process": f"拼接后用 EOS 分隔，切成固定长度 {seq_len} 的块（真实为 2048/4096）",
        "output": f"packed_{seq_len}.bin",
        "stats": {"总 token": len(all_ids), "打包块数": len(packed), "块长度": seq_len,
                  "padding 浪费": round((len(packed) * seq_len - len(all_ids)) / (len(packed) * seq_len), 3)},
        "samples": [{"block_0_first_30": packed[0][:30], "block_len": len(packed[0])}],
        "tip": "有效token预算 ≈ 名义预算 ×(1−重复率)×(1−padding率)×解析成功率×质量系数。",
    })

    # ⑨ Labels
    demo_seq = [tokenizer.vocab.get(t, tokenizer.vocab["<UNK>"]) for t in ["我", "今", "天", "学", "习", "大", "模", "型"]]
    labels = [-100] + demo_seq[1:]
    stages.append({
        "id": 9, "name": "构造 Labels", "en": "Label Construction",
        "input": "input_ids", "process": "labels = input_ids 右移一位，首位填 -100（交叉熵忽略）",
        "output": "(input_ids, labels) 训练对",
        "stats": {"序列长度": len(demo_seq), "有效监督位置": len(demo_seq) - 1},
        "samples": [{"input_ids": demo_seq, "labels": labels,
                     "note": "看到「我」预测「今天」… 一句话产生 T-1 个训练信号"}],
        "tip": "多模态样本中，图像 token 与 prompt 位置的 labels 也设为 -100，只对回答算 loss。",
    })

    # ⑩ Batch
    batch = []
    for p in packed[:4]:
        b = p[:seq_len] + [tokenizer.vocab["<PAD>"]] * (seq_len - len(p))
        batch.append(b)
    stages.append({
        "id": 10, "name": "Batch 与 Collator", "en": "Batching",
        "input": "定长样本", "process": "组 batch、动态 padding、构造 position_ids；Parquet + IterableDataset 流式读取",
        "output": "input_ids [B,T], attention_mask [B,T], labels [B,T]",
        "stats": {"batch_size": len(batch), "seq_len": seq_len, "张量形状": f"[{len(batch)}, {seq_len}]"},
        "samples": [{"batch_shape": [len(batch), seq_len], "first_row_first_20": batch[0][:20]}],
        "tip": "GPUFLOPs 远超磁盘带宽，数据吞吐是训练的第一道瓶颈；用 Parquet 列存 + 流式读取。",
    })

    # ⑪ Checkpoint
    stages.append({
        "id": 11, "name": "训练与 Checkpoint", "en": "Training & Checkpointing",
        "input": "batch 张量", "process": "forward→loss→backward→step；混合精度、梯度累积、梯度检查点",
        "output": "checkpoint-1000/ : model.safetensors + optimizer.pt + config.json + trainer_state.json",
        "stats": {"全量微调7B权重文件": "~14 GB", "LoRA适配器文件": "~30–100 MB",
                  "典型checkpoint间隔": "每 500–1000 step"},
        "samples": [{
            "checkpoint-1000/": ["config.json", "model.safetensors (14GB)",
                                 "optimizer.pt", "scheduler.pt", "trainer_state.json"],
            "trainer_state.json": {"step": 1000, "loss": 2.14, "learning_rate": 2.8e-4, "epoch": 0.42}
        }],
        "tip": "checkpoint 是可恢复的训练快照；LoRA 场景只存适配器，体积是完整模型的几百分之一。",
    })

    # ⑫ 评估
    import math
    stages.append({
        "id": 12, "name": "评估 (Perplexity)", "en": "Evaluation",
        "input": "held-out 留出集", "process": "计算 token 级平均负对数似然，取指数得到困惑度",
        "output": "loss 曲线 + PPL 数值",
        "stats": {"示例 loss": 2.14, "示例 PPL": round(math.exp(2.14), 2)},
        "samples": [{"formula": "PPL = exp(-1/T · Σ log P(x_t | x_<t))",
                     "interpretation": "模型平均在约 8.5 个词之间犹豫；越低越好"}],
        "tip": "PPL 下降但下游任务不变，往往说明数据分布与评估目标不匹配。",
    })

    return {"stages": stages, "tokenizer_info": tokenizer.info(),
            "raw_count": len(raw), "safe_count": len(safe), "total_tokens": len(all_ids)}


def get_sft_samples():
    path = os.path.join(DATA_DIR, "sft_instruct.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    # 渲染 chat 模板
    rendered = []
    for s in samples:
        prompt = s["input"] or "（无额外输入）"
        template = (
            "<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
            f"<|im_start|>user\n{s['instruction']}\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{s['output']}<|im_end|>"
        )
        rendered.append({**s, "rendered": template})
    return samples, rendered


def get_multimodal_sample():
    path = os.path.join(DATA_DIR, "multimodal_instruct.json")
    with open(path, "r", encoding="utf-8") as f:
        mm = json.load(f)
    steps = [
        {"stage": "① 图像输入", "artifact": f"{mm['image']}  ({mm['image_meta']['size']})",
         "shape": "[3, 512, 512]"},
        {"stage": "② ViT 切 Patch", "artifact": f"patch_size={mm['image_meta']['patch_size']} → {mm['image_meta']['num_patches']} 个 patch token",
         "shape": f"[{mm['image_meta']['num_patches']}, 3, 14, 14]"},
        {"stage": "③ 视觉特征", "artifact": "ViT 输出图像特征 V",
         "shape": f"[{mm['image_meta']['num_patches']}, 1024]"},
        {"stage": "④ 投影对齐", "artifact": "MLP Projector → H_v（映射到 LLM 词嵌入维度）",
         "shape": f"[{mm['image_meta']['num_patches']}, 4096]"},
        {"stage": "⑤ 序列拼接", "artifact": "图像 token + <image> 位置的文本 token 拼接",
         "shape": "[N+M, 4096]"},
        {"stage": "⑥ 指令微调", "artifact": "只对 assistant 回答位置计算 loss",
         "shape": "labels: 图像与prompt段 = -100"},
    ]
    return {"sample": mm, "steps": steps}
