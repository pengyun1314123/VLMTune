"""
VLMTune 教学演示后端 (Flask)
启动: python backend/app.py   (默认 http://127.0.0.1:5000)
前端静态文件由本服务一并托管: http://127.0.0.1:5000/
"""
import os
import sys
import json
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demos import pipeline as pipe_demo
from demos import finetuning as ft_demo
from demos import training as train_demo
from demos.tokenizer import SimpleBPE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=None)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# ---------- 静态前端 ----------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ---------- API ----------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "VLMTune teaching backend"})


@app.route("/api/pipeline")
def api_pipeline():
    """完整数据流水线各阶段产物。"""
    return jsonify(pipe_demo.build_pipeline())


@app.route("/api/sft")
def api_sft():
    samples, rendered = pipe_demo.get_sft_samples()
    return jsonify({"samples": samples, "rendered": rendered,
                    "note": "训练时只对 assistant 段计算 loss，模板与 user 段 labels=-100。"})


@app.route("/api/multimodal")
def api_multimodal():
    return jsonify(pipe_demo.get_multimodal_sample())


@app.route("/api/lora")
def api_lora():
    """LoRA 参数量与显存计算器（支持 query 参数定制）。"""
    def gi(name, default):
        try:
            return int(request.args.get(name, default))
        except ValueError:
            return default
    args = dict(
        d=gi("d", 4096), k=gi("k", 4096), r=gi("r", 16), alpha=gi("alpha", 32),
        layers=gi("layers", 32), n_attn=gi("n_attn", 4), n_ffn=gi("n_ffn", 3),
    )
    return jsonify(ft_demo.lora_calculator(**args))


@app.route("/api/methods")
def api_methods():
    return jsonify({"methods": ft_demo.METHODS, "tips": ft_demo.training_tips()})


@app.route("/api/attention")
def api_attention():
    return jsonify(ft_demo.attention_demo())


@app.route("/api/training/simulate")
def api_training():
    try:
        steps = int(request.args.get("steps", 200))
    except ValueError:
        steps = 200
    method = request.args.get("method", "lora")
    quality = float(request.args.get("quality", 0.85))
    steps = max(20, min(steps, 500))
    quality = max(0.1, min(quality, 1.0))
    return jsonify(train_demo.simulate(steps=steps, method=method, data_quality=quality))


@app.route("/api/tokenize", methods=["GET", "POST"])
def api_tokenize():
    """对任意输入文本做 BPE 分词演示。"""
    data = request.get_json(silent=True) or {}
    text = request.args.get("text") or data.get("text") or "大模型训练需要大量高质量文本数据。"
    from demos.pipeline import load_raw
    corpus = "\n".join(d["text"] for d in load_raw())
    tokenizer = SimpleBPE().train([corpus], vocab_size=300)
    tokens, ids, applied = tokenizer.encode(text[:200])
    return jsonify({
        "text": text, "tokens": tokens, "ids": ids,
        "merge_examples": applied, "vocab_size": len(tokenizer.vocab),
        "decoded": tokenizer.decode(ids),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"VLMTune backend running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
