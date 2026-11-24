"""让老师（微调过的扩散模型）给每一张粗渲染出一张图，存成 <名字>_t.png，给手机上的小网络当答案。
用法: python teacher_label.py <素材目录>      需要带微调模型的 refine_server.py 在本机跑着。已经有 _t.png 的跳过。

为什么不直接拿相机照片教小网络：照片和粗渲染有几个像素的错位，小网络只能学出一片糊。
老师的输出和它的输入是逐像素对齐的，而且任何机位都能出（包括没有照片的第三人称机位）。
"""
import io
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

from data import list_bases, load_raw

root = sys.argv[1]
bases = [b for b in list_bases(root, kinds="htc") if not os.path.exists(b + "_t.png")]
print(f"要出 {len(bases)} 张", flush=True)
t0 = time.time()
for i, b in enumerate(bases):
    rgb = load_raw(b, target=None)[0]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
    req = urllib.request.Request("http://127.0.0.1:8770/refine?m=ft", data=buf.getvalue(), method="POST")
    out = Image.open(io.BytesIO(urllib.request.urlopen(req).read())).convert("RGB")
    if out.size != (rgb.shape[1], rgb.shape[0]):
        out = out.resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR)
    out.save(b + "_t.png")
    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{len(bases)}，{time.time() - t0:.0f} 秒", flush=True)
print("LABEL DONE", flush=True)
