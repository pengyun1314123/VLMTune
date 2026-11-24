#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验：浏览器端复刻的 JS 算法是否与 Python 后端输出一致。

用法：python3 tools/verify_static.py
依赖：node（系统 PATH 中可用）、flask、numpy
"""
import os
import sys
import json
import tempfile
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(BASE, "site")
BACKEND = os.path.join(BASE, "backend")
sys.path.insert(0, BACKEND)

from demos import finetuning as ft_demo          # noqa: E402
from demos import training as train_demo         # noqa: E402
from demos.pipeline import load_raw              # noqa: E402
from demos.tokenizer import SimpleBPE            # noqa: E402

JS_RUNNER = r"""
global.window = global;
const fs = require('fs');
const path = require('path');
const siteDir = process.argv[2];
global.fetch = p => Promise.resolve({
  ok: true,
  json: () => Promise.resolve(JSON.parse(fs.readFileSync(path.join(siteDir, p), 'utf8')))
});
require(path.join(siteDir, 'js', 'static-api.js'));
(async () => {
  const out = {
    lora: VLMStaticAPI.lora({r:'16', alpha:'32', d:'4096', layers:'32'}),
    loraCustom: VLMStaticAPI.lora({r:'64', alpha:'128', d:'8192', layers:'16'}),
    sim: VLMStaticAPI.simulate({steps:'100', method:'qlora', quality:'0.9'}),
    tok1: await VLMStaticAPI.tokenize('大模型训练需要大量高质量文本数据。'),
    tok2: await VLMStaticAPI.tokenize('Transformer attention is all you need.')
  };
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
"""


def run_js():
    fd, path = tempfile.mkstemp(suffix=".js")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(JS_RUNNER)
        result = subprocess.run(
            ["node", path, SITE],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError("Node 执行失败：\n" + result.stderr)
        return json.loads(result.stdout)
    finally:
        os.unlink(path)


def expected_tokenize(text, tokenizer):
    tokens, ids, applied = tokenizer.encode(text[:200])
    return {
        "text": text,
        "tokens": tokens,
        "ids": ids,
        "vocab_size": len(tokenizer.vocab),
        "decoded": tokenizer.decode(ids),
        "merge_examples": [list(x) for x in applied],
    }


def compare(label, got, exp, strict=True):
    if strict:
        ok = got == exp
    else:
        ok = True
    print("  %-28s %s" % (label, "PASS" if ok else "FAIL"))
    if not ok:
        print("    期望: %s" % json.dumps(exp, ensure_ascii=False)[:300])
        print("    实际: %s" % json.dumps(got, ensure_ascii=False)[:300])
    return ok


def main():
    print("校验静态站点的浏览器端算法 ...")
    actual = run_js()

    # 直接加载预生成的词表：验证"同一份词表下，浏览器 encode 与 Python encode 一致"
    with open(os.path.join(SITE, "api", "tokenizer.json"), encoding="utf-8") as f:
        tdata = json.load(f)
    tokenizer = SimpleBPE()
    tokenizer.vocab = tdata["vocab"]
    tokenizer.inverse_vocab = {v: k for k, v in tdata["vocab"].items()}
    tokenizer.merges = {tuple(p[0]): p[1] for p in tdata["merges"]}
    tokenizer.special_tokens = tdata.get("special_tokens", tokenizer.special_tokens)

    checks = []
    checks.append(compare(
        "LoRA 默认参数", actual["lora"],
        ft_demo.lora_calculator(r=16, alpha=32, d=4096, layers=32)))
    checks.append(compare(
        "LoRA 自定义参数", actual["loraCustom"],
        ft_demo.lora_calculator(r=64, alpha=128, d=8192, layers=16)))

    sim_exp = train_demo.simulate(steps=100, method="qlora", data_quality=0.9)
    sim_got = actual["sim"]
    for key in ["steps", "lrs", "checkpoints", "method", "meta", "summary"]:
        checks.append(compare("训练模拟 · " + key, sim_got[key], sim_exp[key]))
    checks.append(compare("训练模拟 · losses 长度",
                          len(sim_got["losses"]), len(sim_exp["losses"])))
    max_diff = max(abs(a - b) for a, b in zip(sim_got["losses"], sim_exp["losses"]))
    # 噪声由不同 PRNG 产生，只要求数量级接近
    noise_ok = max_diff < 0.35
    print("  %-28s %s (max diff %.3f)" %
          ("训练模拟 · losses 噪声范围", "PASS" if noise_ok else "FAIL", max_diff))
    checks.append(noise_ok)

    for label, text in [("中文分词", "大模型训练需要大量高质量文本数据。"),
                        ("英文分词", "Transformer attention is all you need.")]:
        checks.append(compare(label, actual["tok1"] if label == "中文分词" else actual["tok2"],
                              expected_tokenize(text, tokenizer)))

    print("-" * 50)
    if all(checks):
        print("全部 %d 项校验通过：静态版数据与后端一致。" % len(checks))
        return 0
    print("存在 %d 项失败，请检查 browser 端实现。" % (len(checks) - sum(checks)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
