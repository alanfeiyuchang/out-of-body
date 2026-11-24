"""量微调后的 Difix 重画一张 576×768 要多久，试几种提速办法。用法: python bench_refine.py <模型目录> <Difix3D/src> <ft.pt>"""
import sys
import time

import torch
from peft import LoraConfig
from PIL import Image

sys.path.insert(0, sys.argv[2])
from pipeline_difix import DifixPipeline  # noqa: E402

ck = torch.load(sys.argv[3], map_location="cpu")
pipe = DifixPipeline.from_pretrained(sys.argv[1], torch_dtype=torch.float32)
pipe.unet.add_adapter(LoraConfig(
    r=ck["rank"], lora_alpha=ck["rank"], init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "conv1", "conv2"]),
    adapter_name="thirdeye")
pipe.unet.load_state_dict(ck["unet"], strict=False)
pipe.vae.load_state_dict(ck["vae"], strict=False)
pipe = pipe.to("cuda", torch.float16)
pipe.set_progress_bar_config(disable=True)
img = Image.new("RGB", (576, 768), (120, 110, 100))


def bench(label, n=12):
    for _ in range(3):
        pipe("remove degradation", image=img, num_inference_steps=1, timesteps=[ck["t"]], guidance_scale=0.0, height=768, width=576)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        pipe("remove degradation", image=img, num_inference_steps=1, timesteps=[ck["t"]], guidance_scale=0.0, height=768, width=576)
    torch.cuda.synchronize()
    print(f"{label}: {1000 * (time.time() - t) / n:.0f} ms", flush=True)


bench("现状（LoRA 挂着）")
try:
    pipe.unet.fuse_lora()
    bench("LoRA 并进权重")
except Exception as e:  # noqa: BLE001
    print("并 LoRA 失败", str(e)[:120])
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
bench("加 channels_last")
torch.backends.cudnn.benchmark = True
bench("加 cudnn.benchmark")
try:
    pipe.unet = torch.compile(pipe.unet, mode="max-autotune-no-cudagraphs")
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="max-autotune-no-cudagraphs")
    bench("加 torch.compile")
except Exception as e:  # noqa: BLE001
    print("compile 失败", str(e)[:160])
