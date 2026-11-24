"""蒸馏：训练手机上跑的小网络，让它照着老师（微调过的扩散模型）的输出画。
用法: python train_student.py <素材目录> <输出目录> [--base 32] [--iters 20000]

答案是 teacher_label.py 存的 _t.png。损失：按像素对（老师的图和输入是对齐的，可以直接比）
+ 相邻像素的差（边缘别糊）+ 感知损失 LPIPS（细节要像）。
测试房间（合成 r7、真实 v01~v04）的图训练时不看，报的是「和老师差多少」。
"""
import argparse
import os
import random
import time

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import list_bases, load_raw, to_tensors
from model import Refiner
from train import grad_loss, psnr

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--test", default="r7,v01,v02,v03,v04")
ap.add_argument("--iters", type=int, default=20000)
ap.add_argument("--base", type=int, default=32)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--context", action="store_true")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--crop", type=int, default=320)
ap.add_argument("--lpips", type=float, default=0.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
torch.manual_seed(0)
random.seed(0)

tests = a.test.split(",")
have = [b for b in list_bases(a.data, kinds="htc") if os.path.exists(b + "_t.png")]
room = lambda b: os.path.basename(b).split("p")[0]  # noqa: E731
kind = lambda b: os.path.basename(b).rsplit("_", 1)[1][0]  # noqa: E731
train = [load_raw(b, "t") for b in have if room(b) not in tests]
groups = {
    "跟拍机位": [load_raw(b, "t") for b in have if room(b) in tests and kind(b) == "c"][::4],
    "扣帧机位": [load_raw(b, "t") for b in have if room(b) in tests and kind(b) == "h"][::6],
}
# 真机录像的图单独占三成：手机上的粗渲染和数据集的不一样（没有 LiDAR 帮忙对齐，更毛糙），量又少。
device_items = [load_raw(b, "t") for b in have if room(b).startswith("d")]
print(f"训练 {len(train)} 张（其中真机 {len(device_items)} 张）；测试 "
      + "，".join(f"{k} {len(v)} 张" for k, v in groups.items()), flush=True)

model = Refiner(base=a.base, depth=a.depth, context=a.context).to(dev)
print(f"参数 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=a.iters, pct_start=0.03)
perceptual = lpips.LPIPS(net="vgg").to(dev).eval().requires_grad_(False)


def evaluate(items):
    if not items:
        return float("nan"), float("nan")
    model.eval()
    ps, lp = [], []
    with torch.no_grad():
        for raw in items:
            x, y = to_tensors(raw)
            x, y = x[None].to(dev), y[None].to(dev)
            o = model(x)
            ps.append(psnr(o, y))
            lp.append(float(perceptual(o * 2 - 1, y * 2 - 1)))
    model.train()
    return float(np.mean(ps)), float(np.mean(lp))


t0 = time.time()
for it in range(1, a.iters + 1):
    bx, by = [], []
    for _ in range(a.batch):
        pool = device_items if device_items and random.random() < 0.3 else train
        rgb, gt, dist = random.choice(pool)
        H, W = dist.shape
        top, left = random.randint(0, H - a.crop), random.randint(0, W - a.crop)
        sl = (slice(top, top + a.crop), slice(left, left + a.crop))
        x1, y1 = to_tensors((rgb[sl], gt[sl], dist[sl]))
        bx.append(x1)
        by.append(y1)
    x, y = torch.stack(bx).to(dev), torch.stack(by).to(dev)
    if random.random() < 0.5:
        x, y = x.flip(3), y.flip(3)
    # 一半的批次里再把输入挖掉一大块（整条边或者一个大方块）：刚开始扫的时候画面里大半是空的，
    # 老师在这种图上画的是干净的墙，学生也得学会，而训练图里这种大空洞不多。
    if random.random() < 0.5:
        for b in range(x.shape[0]):
            if random.random() < 0.5:
                cut = random.randint(a.crop // 4, a.crop * 3 // 4)
                if random.random() < 0.5:
                    x[b, :, :, (0 if random.random() < 0.5 else a.crop - cut):][..., :cut] = 0
                else:
                    x[b, :, (0 if random.random() < 0.5 else a.crop - cut):][:, :cut] = 0
            else:
                hh, ww = random.randint(80, 220), random.randint(80, 220)
                t, l = random.randint(0, a.crop - hh), random.randint(0, a.crop - ww)
                x[b, :, t:t + hh, l:l + ww] = 0
    out = model(x)
    loss = F.l1_loss(out, y) + 0.5 * grad_loss(out, y) + a.lpips * perceptual(out * 2 - 1, y * 2 - 1).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if it % 2000 == 0 or it == a.iters:
        parts = []
        for k, items in groups.items():
            p, l = evaluate(items)
            parts.append(f"{k} {p:.2f} dB / LPIPS {l:.3f}")
        print(f"{it:5d} loss {loss.item():.4f} | 和老师比：" + "，".join(parts) + f" | {time.time() - t0:.0f}s", flush=True)
        torch.save({"state": model.state_dict(), "base": a.base, "depth": a.depth, "context": a.context},
                   os.path.join(a.out, "refiner.pt"))

# 对比图：粗渲染 | 学生 | 老师
model.eval()
bg = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
with torch.no_grad():
    for name, items in (("chase", groups["跟拍机位"]), ("held", groups["扣帧机位"])):
        for k in range(0, len(items), max(1, len(items) // 6)):
            x, y = to_tensors(items[k])
            o = model(x[None].to(dev))[0].cpu()
            coarse = x[:3] + (1 - x[3:4]) * bg
            row = torch.cat([coarse, o, y], dim=2).permute(1, 2, 0).numpy()
            Image.fromarray((row * 255).round().clip(0, 255).astype(np.uint8)).save(
                os.path.join(a.out, f"cmp_{name}_{k:03d}.jpg"), quality=90)
