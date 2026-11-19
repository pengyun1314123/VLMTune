"""
微调方法演示：LoRA / QLoRA / 全量微调的参数量与显存精算，
注意力的真实数值计算（numpy），以及各方法对比数据。
"""
import math
import numpy as np


def lora_calculator(d=4096, k=4096, r=16, alpha=32, layers=32,
                    n_attn=4, n_ffn=3, seq_len=2048, batch=1):
    """计算单个 Linear 与整模型级别的 LoRA 参数/显存节省。"""
    full_single = d * k
    lora_single = r * (d + k)
    compression = full_single / lora_single

    # 整模型：每层 n_attn 个 attention 投影 (d×d) + n_ffn 个 FFN 投影
    per_layer_full = (n_attn * d * d + n_ffn * (d * 4 * d))
    per_layer_lora = (n_attn + n_ffn) * r * (d + d)  # 简化估计
    total_full = per_layer_full * layers
    total_lora = per_layer_lora * layers

    # 显存估算 (GB)：权重 bf16=2B/param, 梯度 bf16=2B, AdamW fp32 m+v=8B
    def mem(params, base_bits=16, trainable=True):
        w = params * (base_bits / 8) / 1e9
        g = params * 2 / 1e9 if trainable else 0
        o = params * 8 / 1e9 if trainable else 0
        return w + g + o

    mem_full = mem(total_full, 16, True)
    mem_lora = mem(total_full - total_lora, 16, False) + mem(total_lora, 16, True)
    # QLoRA: 4bit 存储基座 + bf16 LoRA
    mem_qlora = mem(total_full - total_lora, 4, False) + mem(total_lora, 16, True)

    return {
        "single_layer": {
            "full_params": full_single, "lora_params": lora_single,
            "compression_ratio": round(compression, 1),
        },
        "whole_model": {
            "total_full_params": total_full, "total_lora_params": total_lora,
            "trainable_percent": round(total_lora / total_full * 100, 4),
            "compression_ratio": round(total_full / total_lora, 1),
        },
        "memory_gb": {
            "full_finetune": round(mem_full, 1),
            "lora": round(mem_lora, 1),
            "qlora": round(mem_qlora, 1),
        },
        "effective_scaling": round(alpha / r, 4),
        "note": f"α/r = {alpha}/{r} = {alpha/r:.2f}。若只把 r 翻倍而不改 α，缩放会减半，等效学习率被悄悄改变。",
    }


METHODS = [
    {"name": "全量微调", "en": "Full FT", "params": "100%", "latency": "无", "memory": "极高 (~84GB/7B)",
     "pros": "效果上限最高", "cons": "显存巨大、灾难性遗忘、每任务存整模型", "use": "算力充足、追求极致"},
    {"name": "LoRA", "en": "LoRA", "params": "0.1–1%", "latency": "零（可合并）", "memory": "低 (~18GB/7B)",
     "pros": "效果接近全量、零推理延迟、适配器仅几十MB", "cons": "需调 r/α/target", "use": "★ 默认首选"},
    {"name": "QLoRA", "en": "QLoRA", "params": "0.1–1%", "latency": "零（可合并）", "memory": "极低 (~6GB/7B)",
     "pros": "4bit 存储+NF4+双重量化，单卡微调 65B", "cons": "训练略慢、量化有轻微损失", "use": "★ 资源受限"},
    {"name": "Adapter", "en": "Adapter", "params": "0.5–5%", "latency": "有（不可合并）", "memory": "低",
     "pros": "多任务热切换方便", "cons": "推理变慢、无法合并", "use": "多任务服务"},
    {"name": "Prompt Tuning", "en": "Prompt", "params": "<0.01%", "latency": "无", "memory": "极低",
     "pros": "参数极少、实现简单", "cons": "小模型效果差、占上下文", "use": "11B+ 模型简单任务"},
    {"name": "Prefix / P-Tuning v2", "en": "Prefix", "params": "0.1–1%", "latency": "略占上下文", "memory": "低",
     "pros": "表达力强、训练稳定(v2)", "cons": "占上下文窗口", "use": "NLP 理解类任务"},
    {"name": "RLHF / DPO", "en": "Alignment", "params": "取决于底座", "latency": "无", "memory": "中–高",
     "pros": "对齐人类偏好，DPO 更简单", "cons": "RLHF 复杂易崩", "use": "聊天助手对齐"},
]


def attention_demo(tokens=None, d_k=8, seed=42):
    """对一个例句真实计算缩放点积注意力，返回注意力矩阵（热力图数据）。"""
    if tokens is None:
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
    rng = np.random.default_rng(seed)
    n = len(tokens)
    scale = math.sqrt(d_k)
    X = rng.normal(size=(n, d_k))
    Wq, Wk, Wv = (rng.normal(size=(d_k, d_k)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / scale

    # 因果掩码
    mask = np.triu(np.full((n, n), -1e9), k=1)
    scores_masked = scores + mask
    # softmax（数值稳定）
    e = np.exp(scores_masked - scores_masked.max(axis=1, keepdims=True))
    attn = e / e.sum(axis=1, keepdims=True)
    out = attn @ V

    return {
        "tokens": tokens,
        "d_k": d_k,
        "scores": np.round(scores, 3).tolist(),
        "attention": np.round(attn, 3).tolist(),
        "output_shape": list(out.shape),
        "explanation": (
            "每一行代表一个查询 token 对所有位置（含自己）的关注权重，因果掩码保证看不到未来。"
            "softmax(QKᵀ/√d_k)·V 的物理含义：按相关度加权汇总信息。"
        ),
        "formula": "Attention(Q,K,V) = softmax(QKᵀ / √d_k + M) V",
    }


def training_tips():
    return [
        {"name": "混合精度 BF16", "desc": "前向计算减半显存并提速，优化器状态仍保留 FP32 保证稳定"},
        {"name": "梯度累积", "desc": "loss / accum_steps，用多个小 batch 模拟大 batch"},
        {"name": "梯度检查点", "desc": "不存中间激活、反向时重算：显存↓30%，速度↓20%"},
        {"name": "Warmup + Cosine LR", "desc": "先线性升到峰值再余弦衰减，避免初期梯度爆炸"},
        {"name": "梯度裁剪 clip=1.0", "desc": "防止偶发大梯度破坏训练"},
        {"name": "Flash Attention", "desc": "精确注意力的 IO 感知实现，长序列显著提速"},
    ]
