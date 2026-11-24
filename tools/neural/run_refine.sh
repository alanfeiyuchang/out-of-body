#!/bin/bash
# 在服务器上（~/thirdeye-neural）重启「重画」服务。它只听本机 8770，对外由 predict_server.py 的 /refine 转发。
cd "$(dirname "$0")/.."
pkill -f "[c]ode/refine_server.py" || true
FT=""; [ -s out-ft/ft.pt ] && FT="--ft out-ft/ft.pt"   # 有微调产物就一起加载
setsid nohup .venv/bin/python code/refine_server.py --model difix --src Difix3D/src $FT > refine.log 2>&1 < /dev/null &
