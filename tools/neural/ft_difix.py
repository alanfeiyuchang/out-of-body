"""用我们自己的「粗渲染 → 相机照片」素材微调 Difix（单步扩散）。
用法: python ft_difix.py <Difix 模型目录> <Difix3D/src> <素材目录> <输出目录> [--t 499] [--steps 4000]

原版 Difix 是为 NeRF / 3DGS 渲染的毛病训的，在我们这种画面上要么几乎不改（强度 199），
要么自由发挥（499 以上）。这里让它专门学「照着我们的粗渲染画清楚，别乱编」。

只训很少的参数：UNet 上挂 LoRA，外加 VAE 解码器里原有的 LoRA 和跳连卷积，原模型不动。
损失：缩小 4 倍后按像素对（颜色和大结构要对；真实素材里粗渲染和照片有几个像素的错位，
缩小之后就不算什么）+ 感知损失 LPIPS（细节要像照片，但不要求逐像素对齐）。
输入就是导出素材里的 _in.png：没扫到的地方是中灰。
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("src")
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--t", type=int, default=499)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--test", default="r7,v01,v02,v03,v04")
ap.add_argument("--synth-every", type=int, default=6,
                help="每几步里有一步用合成素材。合成房间的墙上有 4 厘米一道的细条纹，"
                     "占一半的时候模型学会了往真实房间的墙上也画竖条纹，所以压到六分之一")
ap.add_argument("--skip", default="d", help="素材名以这些字母开头的房间不参与训练（d = 真机录像，照片是糊的）")
a = ap.parse_args()
sys.path.insert(0, a.src)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lpips  # noqa: E402
from data import list_bases, load_raw  # noqa: E402
from pipeline_difix import DifixPipeline  # noqa: E402

os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)
random.seed(0)
dev = "cuda"

pipe = DifixPipeline.from_pretrained(a.model, torch_dtype=torch.float32)
unet, vae = pipe.unet.to(dev), pipe.vae.to(dev)
text_encoder = pipe.text_encoder.to(dev)
with torch.no_grad():
    ids = pipe.tokenizer("remove degradation", max_length=pipe.tokenizer.model_max_length,
                         padding="max_length", truncation=True, return_tensors="pt").input_ids.to(dev)
    prompt = text_encoder(ids)[0]
del text_encoder
alpha = pipe.scheduler.alphas_cumprod[a.t].item()
timestep = torch.tensor([a.t], device=dev).long()
scale = vae.config.scaling_factor


def forward(x):
    """x: B×3×H×W，取值 -1~1。跟推理时走的是同一条路：编码 → UNet 一步 → 解码（带编码器的跳连）。"""
    z = vae.encode(x).latent_dist.sample() * scale
    eps = unet(z, timestep, encoder_hidden_states=prompt.expand(x.shape[0], -1, -1)).sample
    z0 = (z - (1 - alpha) ** 0.5 * eps) / alpha ** 0.5
    vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks
    return vae.decode(z0 / scale).sample.clamp(-1, 1)


# ---- 哪些参数要训 ----
unet.requires_grad_(False)
vae.requires_grad_(False)
unet.add_adapter(LoraConfig(
    r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "conv1", "conv2"]),
    adapter_name="thirdeye")
trainable = [p for p in unet.parameters() if p.requires_grad]
for n, p in vae.named_parameters():
    if "decoder" in n and ("lora" in n or "skip_conv" in n):
        p.requires_grad_(True)
        trainable.append(p)
print(f"要训的参数 {sum(p.numel() for p in trainable) / 1e6:.1f} M，强度 t={a.t}", flush=True)
opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-2)
# 24 GB 的显卡一批两张放不下（解码器在全分辨率上要留着反传用的中间结果）：
# 一批一张、攒两批再更新，UNet 开梯度检查点省显存。
unet.enable_gradient_checkpointing()
perceptual = lpips.LPIPS(net="vgg").to(dev).eval().requires_grad_(False)

# ---- 素材：竖的和横的分开放，一批里只能是同一种 ----
tests = a.test.split(",")
rooms = sorted({os.path.basename(b).split("p")[0] for b in list_bases(a.data)})
buckets = {}
for b in list_bases(a.data, [r for r in rooms if r not in tests and r[0] not in a.skip]):
    rgb, gt, _ = load_raw(b)
    real = not os.path.basename(b).startswith("r")
    buckets.setdefault((rgb.shape[:2], real), []).append((rgb, gt))
print({f"{k[0][1]}x{k[0][0]} {'真实' if k[1] else '合成'}": len(v) for k, v in buckets.items()}, flush=True)
val = []
for b in list_bases(a.data, [r for r in tests if not r.startswith("r")], "h")[::16] \
        + list_bases(a.data, [r for r in tests if r.startswith("r")], "t")[::12]:
    rgb, gt, _ = load_raw(b)
    val.append((rgb, gt))
print(f"验证 {len(val)} 张", flush=True)


def tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float().div(127.5).sub(1)


def psnr(x, y):
    return float(-10 * torch.log10(F.mse_loss(x * 0.5 + 0.5, y * 0.5 + 0.5)))


def evaluate(save=None):
    unet.eval()
    ps, lp = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k, (rgb, gt) in enumerate(val):
            x, y = tensor(rgb)[None].to(dev), tensor(gt)[None].to(dev)
            o = forward(x).float()
            ps.append(psnr(o, y))
            lp.append(float(perceptual(o, y)))
            if save is not None and k % 2 == 0:
                row = torch.cat([x, o, y], dim=3)[0].permute(1, 2, 0).cpu().numpy()
                Image.fromarray(((row * 0.5 + 0.5) * 255).round().clip(0, 255).astype(np.uint8)).save(
                    os.path.join(a.out, f"{save}_{k:02d}.jpg"), quality=90)
    unet.train()
    return float(np.mean(ps)), float(np.mean(lp))


p0, l0 = evaluate("before")
print(f"微调前：PSNR {p0:.2f} dB，LPIPS {l0:.3f}（越低越像）", flush=True)

# 真实素材为主（教它「照片长什么样」），掺一点合成的（第三人称机位教它「没扫到的地方怎么接」）。
keys = list(buckets)
real_keys = [k for k in keys if k[1]] or keys
synth_keys = [k for k in keys if not k[1]] or keys
t0 = time.time()
unet.train()
for step in range(1, a.steps + 1):
    key = random.choice(synth_keys if (step // a.accum) % a.synth_every == 0 else real_keys)
    picks = random.sample(buckets[key], a.batch)
    x = torch.stack([tensor(p[0]) for p in picks]).to(dev)
    y = torch.stack([tensor(p[1]) for p in picks]).to(dev)
    if random.random() < 0.5:
        x, y = x.flip(3), y.flip(3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward(x)
    out = out.float()
    loss = (F.l1_loss(F.avg_pool2d(out, 4), F.avg_pool2d(y, 4)) + 0.25 * F.mse_loss(out, y)
            + perceptual(out, y).mean())
    (loss / a.accum).backward()
    if step % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    if step % 50 == 0 and step < 300:
        print(f"{step:5d} loss {loss.item():.3f} | {time.time() - t0:.0f}s", flush=True)
    if step % 500 == 0 or step == a.steps:
        p1, l1 = evaluate(f"step{step:05d}" if step % 1000 == 0 or step == a.steps else None)
        print(f"{step:5d} loss {loss.item():.3f} | 验证 PSNR {p1:.2f} dB，LPIPS {l1:.3f} | {time.time() - t0:.0f}s", flush=True)
        state = {"t": a.t, "rank": a.rank,
                 "unet": {k: v.cpu() for k, v in unet.state_dict().items() if "lora" in k},
                 "vae": {k: v.cpu() for k, v in vae.state_dict().items()
                         if "decoder" in k and ("lora" in k or "skip_conv" in k)}}
        torch.save(state, os.path.join(a.out, "ft.pt"))
