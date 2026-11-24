#!/usr/bin/env python3
"""未扫区域的生成式预测服务，跑在有显卡的那台电脑上。

手机发来一张不完整的 360° 全景图和一张遮罩（哪些像素还没扫到），
这里交给本机的 ComfyUI 用 Qwen-Image-Edit 把缺的部分补出来，再把结果（PNG）传回去。

跟「让模型照着图重画一张」不同，这里是带遮罩的补洞：
采样的每一步都把已知区域按原图锁住，模型只能在缺的地方生成，
所以补出来的东西是接着已知部分长出去的；最后再把已知像素原样贴回去。

要用 ComfyUI 那个虚拟环境里的 python（需要 PIL 和 numpy）：

    ~/ComfyUI/venv/bin/python predict_server.py            # 监听 0.0.0.0:8765
    ~/ComfyUI/venv/bin/python predict_server.py --test in.png out.png

请求体：一张 PNG，宽 W、高 2H。上半张是全景图，下半张是遮罩（白 = 缺）。
--test 时也接受普通的 W×H 全景图，缺的地方是纯中灰 (128,128,128)。

前提：ComfyUI 已经在 127.0.0.1:8188 上跑着，并且装好了下面 MODELS 里的四个文件。
"""

import argparse
import io
import json
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageFilter

COMFY = "http://127.0.0.1:8188"

MODELS = {
    "unet": "qwen_image_edit_2511_fp8mixed.safetensors",
    "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
    "lora": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
}

# 要的是「顺着已有的东西往外接」，不是「发挥想象」：缺的是地板就接地板，缺的是墙就接墙。
#
# 缺的地方不留成一块纯灰：试过，模型要么把灰色当成内容原样留着，要么在灰洞里摆一件自己想出来的东西。
# 所以先用周围的颜色把洞糊上（见 blur_fill），再告诉模型「糊的地方是缺的，照着它的颜色画清楚」。
# 这样它拿到的提示是「这里大概是地板的颜色」，补出来的自然就是地板。
PROMPT = (
    "This is a 360-degree equirectangular panorama photo of an indoor room. "
    "Some parts of it are blurry smooth color smears with no detail: "
    "those parts have not been captured yet. "
    "Redraw only the blurry parts as a sharp, photorealistic continuation of the surrounding room: "
    "floor continues as the same floor, walls continue as the same walls, "
    "and the top becomes a ceiling that matches the room. "
    "Complete partially visible objects naturally, but do not invent new furniture or objects. "
    "Keep the same materials, colors and lighting. Do not change the sharp parts."
)

# 显卡一次只跑一张。
gpu_lock = threading.Lock()


def http(method, path, data=None, headers=None, timeout=600):
    req = urllib.request.Request(COMFY + path, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def upload(image, name):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + png_bytes(image) + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    reply = http(
        "POST", "/upload/image", body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return json.loads(reply)["name"]


def workflow(image_name, mask_name, seed, megapixels):
    m = MODELS
    scale = {"upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 16}
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": m["clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "4": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": m["lora"], "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["4", 0], "shift": 3.0}},
        "6": {"class_type": "CFGNorm", "inputs": {"model": ["5", 0], "strength": 1.0}},
        # 让遮罩可以是渐变的：边界上一半保留一半生成，接缝更自然。
        "20": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["6", 0], "strength": 1.0}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["7", 0], **scale}},
        "21": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "22": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["21", 0], **scale}},
        "23": {"class_type": "ImageToMask", "inputs": {"image": ["22", 0], "channel": "red"}},
        "9": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "prompt": PROMPT, "vae": ["3", 0], "image1": ["8", 0]}},
        "10": {"class_type": "TextEncodeQwenImageEditPlus",
               "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["8", 0]}},
        "11": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["9", 0],
                          "reference_latents_method": "index_timestep_zero"}},
        "12": {"class_type": "FluxKontextMultiReferenceLatentMethod",
               "inputs": {"conditioning": ["10", 0],
                          "reference_latents_method": "index_timestep_zero"}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["3", 0]}},
        # 采样时只在遮罩里加噪、生成；遮罩外每一步都按原图锁住。
        "24": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["13", 0], "mask": ["23", 0]}},
        "14": {"class_type": "KSampler",
               "inputs": {"model": ["20", 0], "seed": seed, "steps": 8, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple",
                          "positive": ["11", 0], "negative": ["12", 0],
                          "latent_image": ["24", 0], "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage",
               "inputs": {"images": ["15", 0], "filename_prefix": "thirdeye_pred"}},
    }


def run_comfy(image, mask, megapixels):
    image_name = upload(image, "thirdeye_in.png")
    mask_name = upload(mask, "thirdeye_mask.png")
    body = json.dumps({
        "prompt": workflow(image_name, mask_name, random.randrange(1 << 31), megapixels),
        "client_id": "thirdeye",
    }).encode()
    reply = json.loads(http("POST", "/prompt", body, {"Content-Type": "application/json"}))
    if "prompt_id" not in reply:
        raise RuntimeError(f"ComfyUI 没接受这个任务：{json.dumps(reply)[:800]}")
    pid = reply["prompt_id"]
    t0 = time.time()
    while True:
        time.sleep(0.4)
        hist = json.loads(http("GET", f"/history/{pid}"))
        if pid not in hist:
            if time.time() - t0 > 600:
                raise RuntimeError("等了 10 分钟还没出图")
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI 执行出错：{json.dumps(status)[:800]}")
        images = entry.get("outputs", {}).get("16", {}).get("images", [])
        if not images:
            raise RuntimeError("执行完了但没有输出图")
        img = images[0]
        query = urllib.parse.urlencode({
            "filename": img["filename"], "subfolder": img.get("subfolder", ""),
            "type": img.get("type", "output"),
        })
        return Image.open(io.BytesIO(http("GET", "/view?" + query))).convert("RGB")


def blur_fill(pano, missing):
    """把缺的像素用周围已知的颜色糊上：半径从小到大一圈圈往里填，离已知区域越远越糊。"""
    known = (~missing).astype(np.float32)
    color = pano.astype(np.float32) * known[..., None]
    out = pano.astype(np.float32).copy()
    todo = missing.copy()
    for radius in (3, 6, 12, 24, 48, 96, 192):
        if not todo.any():
            break
        w = np.asarray(Image.fromarray((known * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0
        c = np.stack([
            np.asarray(Image.fromarray(color[..., k].astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)
            for k in range(3)], axis=2)
        ok = todo & (w > 0.04)
        out[ok] = c[ok] / w[ok][:, None]
        todo &= ~ok
    if todo.any():
        out[todo] = pano[~missing].mean(axis=0) if (~missing).any() else 128
    # 不同半径填出来的圈之间有台阶，整体再糊一遍，只作用在缺的地方。
    smooth = np.asarray(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(10))).astype(np.float32)
    out[missing] = smooth[missing]
    return np.clip(out, 0, 255).astype(np.uint8)


def split(png):
    """请求体 → (全景图 RGB 数组, 缺失遮罩 bool 数组)。"""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    a = np.asarray(img)
    h, w = a.shape[:2]
    if h == w:  # 上下叠着的：全景图 + 遮罩
        return a[: h // 2].copy(), a[h // 2:, :, 0] > 127
    missing = np.all(a == 128, axis=2)  # 旧格式 / 手工测试：纯中灰 = 缺
    return a.copy(), missing


def complete(png, megapixels=1.0):
    """补全一张全景图，返回 (PNG 字节, 用了几秒)。"""
    with gpu_lock:
        t0 = time.time()
        pano, missing = split(png)
        h, w = missing.shape

        # 全景图左右是接着的，但模型不知道。把接缝转到「已知最多」的那一列去，
        # 这样接缝两边都是锁住的原图，缺的区域都在画面中间、是连续的一块。
        known_cols = (~missing).sum(axis=0).astype(np.float64)
        k = max(1, w // 32)
        window = np.convolve(np.concatenate([known_cols[-k:], known_cols, known_cols[:k]]),
                             np.ones(2 * k + 1), mode="valid")
        shift = int(np.argmax(window))
        pano = np.roll(pano, -shift, axis=1)
        missing = np.roll(missing, -shift, axis=1)

        # 已知区域的边缘是 splat 的毛边，质量最差：把遮罩往已知区域里扩几个像素，让模型重画这一圈；
        # 再把遮罩边界糊开，过渡带里一半保留一半生成。
        mask = Image.fromarray((missing * 255).astype(np.uint8))
        grown = mask.filter(ImageFilter.MaxFilter(7))
        soft = grown.filter(ImageFilter.GaussianBlur(4))
        soft = Image.fromarray(np.maximum(np.asarray(soft), np.asarray(mask)))

        holes = blur_fill(pano, np.asarray(grown) > 127)
        out = run_comfy(Image.fromarray(holes), soft.convert("RGB"), megapixels)
        out = np.asarray(out.resize((w, h), Image.LANCZOS)).astype(np.float32)

        # 已知像素原样贴回去，只在遮罩里用生成的。
        alpha = (np.asarray(soft).astype(np.float32) / 255.0)[..., None]
        merged = pano.astype(np.float32) * (1 - alpha) + out * alpha
        merged = np.roll(np.clip(merged, 0, 255).astype(np.uint8), shift, axis=1)
        return png_bytes(Image.fromarray(merged)), time.time() - t0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # 连接留着复用，手机每秒发好几张，省掉反复握手

    def log_message(self, fmt, *args):
        # 带上对方的地址：127.0.0.1 是经 Mac 中转来的，192.168.x 是手机直连。
        sys.stderr.write("%s %s %s\n" % (time.strftime("%H:%M:%S"), self.client_address[0], fmt % args))

    def reply(self, code, body, content_type="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, b"ok")
        else:
            self.reply(404, b"not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 20_000_000:
            self.reply(400, b"bad length")
            return
        if self.path.startswith("/refine"):
            # 「把渲染出来的一帧重画一遍」由另一个进程做（tools/neural/refine_server.py，只听本机），
            # 这里原样转过去，手机就不用多连一个端口。
            body = self.rfile.read(length)
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:8770" + self.path, data=body, method="POST")
                with urllib.request.urlopen(req, timeout=20) as r:
                    self.reply(200, r.read(), "image/jpeg")
            except Exception as e:
                self.reply(502, f"refine 服务没起来或出错：{e}".encode())
            return
        if self.path != "/complete":
            self.reply(404, b"not found")
            return
        png = self.rfile.read(length)
        try:
            out, secs = complete(png)
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} 补全完成，{secs:.1f} 秒\n")
            self.reply(200, out, "image/png")
        except Exception as e:  # 把原因原样告诉手机端，方便在日志里看
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} 失败：{e}\n")
            self.reply(500, str(e).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--test", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--megapixels", type=float, default=1.0)
    args = ap.parse_args()
    if args.test:
        with open(args.test[0], "rb") as f:
            out, secs = complete(f.read(), args.megapixels)
        with open(args.test[1], "wb") as f:
            f.write(out)
        print(f"完成，{secs:.1f} 秒，{len(out)} 字节")
        return
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
