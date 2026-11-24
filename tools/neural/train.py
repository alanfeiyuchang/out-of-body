"""训练 Refiner。用法: python train.py <素材目录> <输出目录> [--test r7,v01,v02] [--iters 8000]

按房间分训练和测试：测试房间的任何一张图训练时都没见过，所以分数说的是
「换一间房还管不管用」，而不是「背下来了没有」。
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import list_bases, load_raw, to_tensors
from model import Refiner


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b)))


def grad_loss(a, b):
    """相邻像素的差也要对上：只用 L1 的话边缘会糊。"""
    out = 0
    for _ in range(3):
        out = out + F.l1_loss(a[..., 1:] - a[..., :-1], b[..., 1:] - b[..., :-1])
        out = out + F.l1_loss(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :])
        a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
    return out


def spectrum_loss(a, b, size=32):
    """一小块一小块地比「各种粗细的纹理各有多少」，不管纹理具体落在哪个像素上。

    真实素材里粗渲染和照片总有几个像素的错位。按像素比的话，网络只能输出周围的平均值（发糊）；
    比每 32×32 小块的频谱幅度就不怕错位，而且糊了会被罚（细纹理的那部分幅度没了）。
    """
    def amp(x):
        # 切成有一半重叠的小块，乘窗函数，取傅里叶变换的幅度。
        p = x.unfold(2, size, size // 2).unfold(3, size, size // 2)
        w = torch.hann_window(size, device=x.device)
        p = p * (w[:, None] * w[None, :])
        return torch.fft.rfft2(p, norm="ortho").abs()
    return F.l1_loss(amp(a), amp(b))


def evaluate(model, items, dev):
    """返回 (粗渲染的 PSNR, 网络输出的 PSNR)。粗渲染里没东西的地方按 app 里的背景色算。"""
    bg = torch.tensor([0.045, 0.055, 0.075], device=dev).view(1, 3, 1, 1)
    pin, pout = [], []
    if not items:
        return float("nan"), float("nan")
    model.eval()
    with torch.no_grad():
        for raw in items:
            x, y = to_tensors(raw)
            x, y = x[None].to(dev), y[None].to(dev)
            coarse = x[:, :3] + (1 - x[:, 3:4]) * bg
            pin.append(psnr(coarse, y))
            pout.append(psnr(model(x), y))
    model.train()
    return float(np.mean(pin)), float(np.mean(pout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--test", default="r7")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--loss", default="pixel", choices=["pixel", "spectrum"])
    ap.add_argument("--spectrum", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0)

    rooms = sorted({os.path.basename(b).split("p")[0] for b in list_bases(a.data)})
    test_rooms = a.test.split(",")
    train_rooms = [r for r in rooms if r not in test_rooms]
    train = [load_raw(b) for b in list_bases(a.data, train_rooms)]
    # 测试分三组：合成房间的扣帧机位和第三人称机位（r 开头），真实扫描的扣帧机位（其他）。
    synth = [r for r in test_rooms if r.startswith("r")]
    real = [r for r in test_rooms if not r.startswith("r")]
    tests = {
        "合成·扣帧": [load_raw(b) for b in list_bases(a.data, synth, "h")][::2],
        "合成·第三人称": [load_raw(b) for b in list_bases(a.data, synth, "t")][::2],
        "真实·扣帧": [load_raw(b) for b in list_bases(a.data, real, "h")][::3],
    }
    n_real = sum(1 for b in list_bases(a.data, train_rooms) if not os.path.basename(b).startswith("r"))
    print(f"训练 {len(train_rooms)} 个房间 {len(train)} 张（其中真实 {n_real} 张）；测试 "
          + "，".join(f"{k} {len(v)} 张" for k, v in tests.items()), flush=True)

    xs = train
    model = Refiner(base=a.base).to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"参数 {n / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=a.iters, pct_start=0.05)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        bx, by = [], []
        for i in torch.randint(0, len(xs), (a.batch,)).tolist():
            rgb, gt, dist = xs[i]
            H, W = dist.shape
            top = torch.randint(0, H - a.crop + 1, (1,)).item()
            left = torch.randint(0, W - a.crop + 1, (1,)).item()
            sl = (slice(top, top + a.crop), slice(left, left + a.crop))
            x1, y1 = to_tensors((rgb[sl], gt[sl], dist[sl]))
            bx.append(x1)
            by.append(y1)
        x = torch.stack(bx).to(dev)
        y = torch.stack(by).to(dev)
        if torch.rand(1).item() < 0.5:
            x, y = x.flip(3), y.flip(3)
        # 随机再挖掉几块，逼它学会补洞（真机上的洞比合成房间里多）。
        if torch.rand(1).item() < 0.5:
            for b in range(a.batch):
                for _ in range(torch.randint(1, 4, (1,)).item()):
                    hh, ww = torch.randint(20, 120, (2,)).tolist()
                    t = torch.randint(0, a.crop - hh, (1,)).item()
                    l = torch.randint(0, a.crop - ww, (1,)).item()
                    x[b, :, t:t + hh, l:l + ww] = 0
        out = model(x)
        if a.loss == "pixel":
            loss = F.l1_loss(out, y) + 0.5 * grad_loss(out, y)
        else:
            # 大色块按像素对（缩小 4 倍之后那点错位不算什么），细节按频谱对。
            loss = (F.l1_loss(F.avg_pool2d(out, 4), F.avg_pool2d(y, 4))
                    + 0.25 * F.l1_loss(out, y) + a.spectrum * spectrum_loss(out, y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == a.iters:
            parts = []
            for k, items in tests.items():
                i0, o0 = evaluate(model, items, dev)
                parts.append(f"{k} {i0:.2f}→{o0:.2f}")
            print(f"{it:5d} loss {loss.item():.4f} | " + "，".join(parts)
                  + f" dB | {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "base": a.base}, os.path.join(a.out, "refiner.pt"))

    # 存几张对比图：粗渲染 | 网络输出 | 标准答案。
    model.eval()
    bg = torch.tensor([0.045, 0.055, 0.075]).view(3, 1, 1)
    with torch.no_grad():
        for name, items in (("synth_t", tests["合成·第三人称"]), ("real", tests["真实·扣帧"])):
            for k in range(0, len(items), max(1, len(items) // 8)):
                x, y = to_tensors(items[k])
                o = model(x[None].to(dev))[0].cpu()
                coarse = x[:3] + (1 - x[3:4]) * bg
                row = torch.cat([coarse, o, y], dim=2).permute(1, 2, 0).numpy()
                Image.fromarray((row * 255).round().astype(np.uint8)).save(
                    os.path.join(a.out, f"cmp_{name}_{k:03d}.png"))


if __name__ == "__main__":
    main()
