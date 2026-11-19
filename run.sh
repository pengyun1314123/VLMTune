#!/bin/bash
# VLMTune 一键启动脚本
cd "$(dirname "$0")"
echo "Starting VLMTune at http://127.0.0.1:5000"
python3 backend/app.py
