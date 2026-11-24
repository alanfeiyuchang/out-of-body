"""把一批粗渲染图过一遍 Difix（单步扩散，专门修重建渲染出来的毛病），存结果并报每张的耗时。
用法: python difix_run.py <模型目录> <Difix3D/src> <输入目录或 glob> <输出目录> [强度，默认 199]

强度就是去噪的起点（模型训练时固定用 199）。调高到 499 左右它会敢改得多：洞补上、颗粒抹平，
但细节是它编的；到 799 基本是照着轮廓重画一张。"""
import glob
import os
import sys
import time

import torch
from PIL import Image

model_dir, src, pattern, out_dir = sys.argv[1:5]
strength = int(sys.argv[5]) if len(sys.argv) > 5 else 199
sys.path.insert(0, src)
from pipeline_difix import DifixPipeline  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
pipe = DifixPipeline.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
files = sorted(glob.glob(pattern if "*" in pattern else os.path.join(pattern, "*")))
times = []
for i, f in enumerate(files):
    img = Image.open(f).convert("RGB")
    w, h = img.size
    # 模型要 8 的倍数；长边压到 768 以内。
    s = min(1.0, 768 / max(w, h))
    nw, nh = int(w * s) // 8 * 8, int(h * s) // 8 * 8
    img = img.resize((nw, nh), Image.LANCZOS)
    torch.cuda.synchronize()
    t = time.time()
    out = pipe("remove degradation", image=img, num_inference_steps=1, timesteps=[strength],
               guidance_scale=0.0, height=nh, width=nw).images[0]
    torch.cuda.synchronize()
    if i > 0:
        times.append(time.time() - t)
    out.save(os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + ".png"))
if times:
    print(f"{len(files)} 张，{nw}x{nh}，每张 {1000 * sum(times) / len(times):.0f} ms（不含第一张）")
