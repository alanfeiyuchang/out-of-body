"""在测试房间上给扩散重画打分，和小网络用同一把尺子（对标准答案的 PSNR）。
用法: python eval_difix.py <素材目录> <房间,房间> [强度... 或 ft]    需要 refine_server.py 在本机跑着。
强度写 ft 表示用微调过的模型：它的输入是导出时的原图（没扫到的地方中灰），不是填了背景色的。"""
import io
import sys
import urllib.request

import numpy as np
from PIL import Image

from data import list_bases, load_pair, load_raw

root, room = sys.argv[1:3]
strengths = sys.argv[3:] or ["199", "499"]
bg = np.array([0.045, 0.055, 0.075], dtype=np.float32)


def refine(arr, s):
    buf = io.BytesIO()
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(buf, format="JPEG", quality=92)
    query = "m=ft" if s == "ft" else f"s={s}&reset=1"
    req = urllib.request.Request(
        f"http://127.0.0.1:8770/refine?{query}", data=buf.getvalue(), method="POST")
    out = Image.open(io.BytesIO(urllib.request.urlopen(req).read())).convert("RGB")
    return np.asarray(out, dtype=np.float32) / 255


def psnr(a, b):
    return -10 * np.log10(((a - b) ** 2).mean())


for kind, label in (("h", "扣帧"), ("t", "第三人称")):
    bases = list_bases(root, room.split(","), kind)[::3]
    if not bases:
        continue
    scores = {s: [] for s in strengths}
    coarse_scores = []
    for b in bases:
        x, y = load_pair(b)
        x, y = x.permute(1, 2, 0).numpy(), y.permute(1, 2, 0).numpy()
        coarse = x[..., :3] + (1 - x[..., 3:4]) * bg
        coarse_scores.append(psnr(coarse, y))
        gray = load_raw(b)[0].astype(np.float32) / 255
        for s in strengths:
            scores[s].append(psnr(refine(gray if s == "ft" else coarse, s), y))
    line = f"{label}（{len(bases)} 张）粗渲染 {np.mean(coarse_scores):.2f} dB"
    for s in strengths:
        line += f"，强度 {s}：{np.mean(scores[s]):.2f} dB"
    print(line)
