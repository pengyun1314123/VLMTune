#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键重建 VLMTune 静态站并同步发布到主页仓库的 /VLMTune/ 子目录。

用法（在 VLMTune 工程根目录运行）：
    python3 tools/publish_to_homepage.py

流程：
    1. 运行 build_static.py 重建 site/
    2. 运行 verify_static.py 校验浏览器端算法与 Python 后端一致
    3. 把 site/ 内容同步到 ../pengyun1314123.github.io/VLMTune/
    4. git add / commit / push 主页仓库

可用环境变量覆盖：
    VLM_HOMEPAGE_DIR   主页仓库路径（默认 ../pengyun1314123.github.io）
    VLM_SKIP_PUSH=1    只同步文件，不提交推送
"""
import os
import sys
import shutil
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(BASE, "tools")
SITE = os.path.join(BASE, "site")
HOMEPAGE = os.path.abspath(
    os.environ.get("VLM_HOMEPAGE_DIR", os.path.join(BASE, "..", "pengyun1314123.github.io"))
)
TARGET = os.path.join(HOMEPAGE, "VLMTune")
PYTHON = "/usr/local/bin/python3" if os.path.isfile("/usr/local/bin/python3") else sys.executable


def run(cmd, cwd=None):
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit("命令失败（退出码 %d），流程已中止。" % result.returncode)


def main():
    print("=" * 60)
    print("VLMTune -> 主页仓库 /VLMTune/ 一键发布")
    print("=" * 60)

    if not os.path.isdir(HOMEPAGE):
        raise SystemExit("找不到主页仓库：%s\n请用 VLM_HOMEPAGE_DIR 指定路径。" % HOMEPAGE)

    run([PYTHON, os.path.join(TOOLS, "build_static.py")], cwd=BASE)
    run([PYTHON, os.path.join(TOOLS, "verify_static.py")], cwd=BASE)

    # 同步文件：先清空目标，再复制，避免残留旧文件
    if os.path.isdir(TARGET):
        shutil.rmtree(TARGET)
    shutil.copytree(SITE, TARGET)
    print("\n[sync] %s -> %s" % (SITE, TARGET))

    if os.environ.get("VLM_SKIP_PUSH") == "1":
        print("\n已设置 VLM_SKIP_PUSH=1，跳过提交推送。")
        return

    run(["git", "add", "VLMTune"], cwd=HOMEPAGE)
    # 没有改动时 git commit 会失败，不算错误
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=HOMEPAGE)
    if status.returncode == 0:
        print("\n[git] 没有内容变化，跳过提交。")
        return

    run(["git", "commit", "-m", "Update VLMTune interactive lab"], cwd=HOMEPAGE)
    run(["git", "push", "origin", "main"], cwd=HOMEPAGE)
    print("\n完成，约 1 分钟后访问：https://pengyun1314123.github.io/VLMTune/")


if __name__ == "__main__":
    main()
