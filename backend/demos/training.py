"""
训练过程模拟器：生成确定性的 loss 曲线、checkpoint 事件与评估指标，
用于前端"训练模拟器"交互演示（真实训练需 GPU，此处为教学模拟）。
"""
import math
import random


def simulate(steps=200, lr=2e-4, warmup=20, seed=42, method="lora", data_quality=0.85):
    rng = random.Random(seed)
    base_loss = 3.2
    final_loss = 1.1 + (1 - data_quality) * 1.5
    losses, lrs, checkpoints = [], [], []

    for step in range(1, steps + 1):
        # 学习率：线性 warmup + 余弦衰减
        if step <= warmup:
            cur_lr = lr * step / warmup
        else:
            progress = (step - warmup) / max(steps - warmup, 1)
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        lrs.append(round(cur_lr, 8))

        # loss：指数衰减 + 噪声
        decay = math.exp(-4.0 * step / steps)
        value = final_loss + (base_loss - final_loss) * decay
        value += rng.gauss(0, 0.04 + 0.06 * decay)
        losses.append(round(max(value, 0.3), 4))

        if step % 50 == 0:
            checkpoints.append({
                "step": step,
                "loss": losses[-1],
                "ppl": round(math.exp(losses[-1]), 2),
                "artifact": (f"adapter_model.safetensors (~40MB)" if method in ("lora", "qlora")
                             else f"model.safetensors (~14GB)"),
            })

    # 方法相关的补充信息
    method_meta = {
        "full": {"trainable": "7.0B (100%)", "mem_gb": 84, "speed": "1.0x", "adapter_mb": 14000},
        "lora": {"trainable": "7.2M (0.10%)", "mem_gb": 18, "speed": "0.95x", "adapter_mb": 40},
        "qlora": {"trainable": "7.2M (0.10%)", "mem_gb": 6, "speed": "0.7x", "adapter_mb": 40},
    }
    return {
        "steps": steps,
        "losses": losses,
        "lrs": lrs,
        "checkpoints": checkpoints,
        "method": method,
        "meta": method_meta.get(method, method_meta["lora"]),
        "summary": (
            f"使用 {method.upper()} 训练 {steps} 步：loss 从 {losses[0]:.3f} 降到 {losses[-1]:.3f}，"
            f"最终困惑度 PPL≈{math.exp(losses[-1]):.2f}。"
            f"可训练参数 {method_meta.get(method, method_meta['lora'])['trainable']}，"
            f"峰值显存约 {method_meta.get(method, method_meta['lora'])['mem_gb']}GB。"
        ),
    }
