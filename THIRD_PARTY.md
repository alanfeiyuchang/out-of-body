# Third-party models, data and assets

The code in this repository is MIT licensed (see `LICENSE`). **Nothing listed below is included in the
repository.** You download each piece yourself, under its own license. Several of them are
non-commercial, which means the complete system as shown in the demo is for personal and research use.

| What | Used for | License | Where |
|---|---|---|---|
| Depth Anything V2 Small (Core ML) | Per-frame monocular depth on the phone | Apache 2.0 | `tools/fetch_model.sh` downloads it from Apple's Hugging Face repo |
| Difix (NVIDIA) | Base single-step diffusion model that redraws each frame | NVIDIA non-commercial license | `nvidia/difix` on Hugging Face, code in the Difix3D repository |
| ARKitScenes (Apple) | Training data for fine-tuning | Apple's dataset terms, non-commercial | `tools/neural/fetch_arkit.sh` |
| Qwen-Image-Edit via ComfyUI | Optional panorama completion of unscanned areas | See the model's own license | Set up ComfyUI yourself, see `tools/predict_server.py` |
| SkinTokens | Automatic rigging of the avatar mesh | See the project's own license | Used offline, only when baking an avatar |

## Avatar and phone models

The demo video uses these two models. They are **not** in the repository. Without them the app draws a
simple capsule figure and a plain box for the phone.

- **Avatar:** "Tesla Optimus Gen 3" by Michael (@gmmichaelonearth) on Sketchfab, licensed CC BY 4.0.
  <https://sketchfab.com/3d-models/tesla-optimus-gen-3-0c8f83683b094ee4851c30ca1e8cf1c1>
  The license covers the modeller's work. The robot's design belongs to Tesla. This project is not
  affiliated with or endorsed by Tesla.
- **Phone:** "Apple iPhone Duo Fold Star White 2026 (Animated)" on Sketchfab, under Sketchfab's
  standard license, which does not allow redistributing the model.

To use your own avatar, see `docs/AVATAR.md`.

## Fine-tuned weights

The fine-tuned diffusion weights are not published. They derive from a non-commercial model and
non-commercial data. The training scripts are in `tools/neural/`.
