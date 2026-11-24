"""「粗渲染 → 生成式画面」的服务：手机把渲染出来的一帧发过来，这里用单步扩散重画一遍发回去。

    POST /refine?s=499[&k=0.3][&reset=1]   请求体 = JPEG，返回 JPEG（原版模型，输入是整帧画面）
    POST /refine?m=ft                      用我们自己微调过的模型（ft_difix.py 的产物）。
                                           输入是「只有扫到的表面、没扫到的地方中灰」的渲染图。
    GET  /health

s 是强度（见 difix_run.py）。k 是把上一帧的输出掺进这一帧输入的比例：扩散模型每帧各画各的，
细节会闪；掺一点上一帧，让它接着上一帧画。只监听本机，对外由 predict_server.py 转发。

用法: .venv/bin/python code/refine_server.py --model difix --src Difix3D/src [--port 8770]
"""
import argparse
import io
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--port", type=int, default=8770)
ap.add_argument("--ft", help="微调产物 ft.pt；给了就多加载一份模型，请求里带 m=ft 时用它")
args = ap.parse_args()
sys.path.insert(0, args.src)
from pipeline_difix import DifixPipeline  # noqa: E402

# 原版模型用到的时候才加载（多占 3.5 GB 显存）：平时只用微调过的那个。
pipe = None


def base_pipe():
    global pipe
    if pipe is None:
        pipe = DifixPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True)
    return pipe


tuned, tuned_t = None, 499
if args.ft:
    from peft import LoraConfig  # noqa: E402
    ck = torch.load(args.ft, map_location="cpu")
    tuned = DifixPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    tuned.unet.add_adapter(LoraConfig(
        r=ck["rank"], lora_alpha=ck["rank"], init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "conv1", "conv2"]),
        adapter_name="thirdeye")
    missing = [k for k in ck["unet"] if k not in tuned.unet.state_dict()]
    assert not missing, f"微调权重对不上：{missing[:3]}"
    tuned.unet.load_state_dict(ck["unet"], strict=False)
    tuned.vae.load_state_dict(ck["vae"], strict=False)
    tuned = tuned.to("cuda", torch.float16)
    tuned.set_progress_bar_config(disable=True)
    tuned_t = ck["t"]
    # 提速（2026-11-23 在 4090 上量的，576×768 一张）：原来 163 ms，其中解码 94、编码 35、主网络只有 23。
    #   - 把微调时挂上去的附加层并进原来的权重：解码 94 → 71；
    #   - 编解码器过一遍 torch.compile：解码 71 → 47，整张 111 ms；主网络编译了没变快，不编；
    #   - 提示词是固定的，文字编码只算一次。
    # channels_last 和 cudnn.benchmark 试过，反而慢。
    tuned.unet.fuse_lora()
    for m in tuned.vae.modules():
        if hasattr(m, "lora_A") and hasattr(m, "merge"):
            m.merge()
    try:
        with torch.no_grad():
            tuned_prompt = tuned.encode_prompt("remove degradation", "cuda", 1, False)[0]
    except Exception as e:  # noqa: BLE001
        print("提示词预先编码不了，每张现算：", str(e)[:120], flush=True)
        tuned_prompt = None
    # 编译是按图的尺寸来的，没见过的尺寸第一次要等几十秒。所以只有启动时热过身的尺寸走编译过的，其它尺寸走原样的。
    plain_vae = (tuned.vae.encoder, tuned.vae.decoder)
    fast_vae = (torch.compile(tuned.vae.encoder), torch.compile(tuned.vae.decoder))
    fast_sizes = set()
    print(f"已加载微调模型（强度 {tuned_t}）", flush=True)
lock = threading.Lock()
previous = None  # 上一帧的输出（PIL）


def refine(jpeg, strength, keep, reset, use_tuned=False):
    global previous
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = img.size
    nw, nh = w // 8 * 8, h // 8 * 8
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.BILINEAR)
    if use_tuned and tuned is not None:
        with lock:
            tuned.vae.encoder, tuned.vae.decoder = fast_vae if (nw, nh) in fast_sizes else plain_vae
            text = {"prompt": "remove degradation"} if tuned_prompt is None else {"prompt_embeds": tuned_prompt}
            out = tuned(image=img, num_inference_steps=1, timesteps=[tuned_t],
                        guidance_scale=0.0, height=nh, width=nw, **text).images[0]
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    with lock:
        if reset or previous is None or previous.size != img.size:
            previous = None
        elif keep > 0:
            a = np.asarray(img, dtype=np.float32)
            b = np.asarray(previous, dtype=np.float32)
            img = Image.fromarray((a * (1 - keep) + b * keep).round().astype(np.uint8))
        out = base_pipe()("remove degradation", image=img, num_inference_steps=1, timesteps=[strength],
                   guidance_scale=0.0, height=nh, width=nw).images[0]
        previous = out
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # 连接留着复用，手机每秒发好几张，省掉反复握手

    def log_message(self, fmt, *a):
        pass

    def reply(self, code, body, kind="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, b"ok") if self.path == "/health" else self.reply(404, b"not found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path != "/refine":
            return self.reply(404, b"not found")
        q = urllib.parse.parse_qs(url.query)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        t = time.time()
        try:
            out = refine(body, int(q.get("s", ["499"])[0]), float(q.get("k", ["0"])[0]),
                         q.get("reset", ["0"])[0] == "1", q.get("m", [""])[0] == "ft")
        except Exception as e:  # noqa: BLE001
            return self.reply(500, str(e).encode())
        print(f"refine {len(body) // 1024} KB → {len(out) // 1024} KB, {1000 * (time.time() - t):.0f} ms", flush=True)
        self.reply(200, out, "image/jpeg")


# 先空跑几张，把第一次的开销吃掉。微调模型：手机会发的两种尺寸都编译好再开始接活（大约一两分钟）。
def warm(w, h, n=1):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(buf, format="JPEG")
    for _ in range(n):
        refine(buf.getvalue(), 499, 0, True, use_tuned=tuned is not None)


warm(384, 512)
if tuned is not None:
    for size in ((480, 640), (576, 768)):
        try:
            fast_sizes.add(size)
            t = time.time()
            warm(*size, n=3)
            print(f"{size[0]}×{size[1]} 编译好了，{time.time() - t:.0f} 秒", flush=True)
        except Exception as e:  # noqa: BLE001
            fast_sizes.discard(size)
            print(f"{size[0]}×{size[1]} 编译失败，这个尺寸走原样的：", str(e)[:160], flush=True)
previous = None
print(f"refine server on 127.0.0.1:{args.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
