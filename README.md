# Out of Body

A third-person "reality view" of yourself, reconstructed live from a single phone camera. No LiDAR.

You hold the phone and look at the screen. The screen shows you from behind, as a robot avatar standing
where you stand, in a 3D reconstruction of the place you are in. Drag to orbit the camera around
yourself.

https://github.com/user-attachments/assets/5d9a72f2-0895-49da-bfd7-953828707cdd

This is a personal experiment, inspired by the "reality view" visualisations in driver-assistance
systems. It is not affiliated with Tesla or any other company.

## How it works

```
phone camera ──► pose (ARKit) + monocular depth (Depth Anything V2)
                        │
                        ▼
                3D splat map, built on the phone
                        │
                        ▼
        virtual chase camera behind the user renders the map
                        │                         ┌──────────────────────────────┐
                        ├── rough render ────────►│ GPU server: single-step       │
                        │                         │ diffusion model redraws frame │
                        │◄── clean frame ─────────│ (about 0.2 s round trip)      │
                        ▼                         └──────────────────────────────┘
   each returned frame becomes a depth mesh anchored in the world,
   so the phone can re-render it from the current camera at 60 fps
                        │
                        ▼
              avatar + phone drawn on top
```

1. **Tracking and depth.** ARKit gives the phone's pose. A Core ML depth model estimates depth for each
   camera frame, since there is no depth sensor.
2. **Mapping.** Depth frames are fused into a voxel-hashed splat map on the device (`SplatMap.swift`).
3. **Third-person render.** A chase camera behind the user renders the map with a custom Metal renderer
   (`Renderer.swift`, `Shaders.metal`).
4. **Generative redraw.** The splat render is noisy and full of holes. A render of the surfaces is sent
   to a GPU server, where a single-step diffusion model (fine-tuned from NVIDIA's Difix) returns a
   clean image (`NeuralView.swift`, `tools/neural/refine_server.py`).
5. **Staying responsive.** The model is far slower than the display. Each returned image is turned
   into a depth mesh in world space (`GenKeyframe.swift`) and kept. The phone composites the nearest
   ones for the current camera, predicts where the camera will be when a frame comes back, and
   pre-generates views around the user while the camera is still. A user-centred panorama atlas
   (`GenShell.swift`) keeps colors consistent from frame to frame.
6. **Avatar.** The avatar follows the phone's pose, with the head height taken from the front camera
   when a face is visible. Idle, walk and sit are baked vertex animations (`Avatar.swift`,
   `tools/bake_animated.py`).

Without a server the app still works. It shows its own splat render instead of the generated image.

## Try it in the simulator

No phone, server or model download is needed. The simulator runs a synthetic room.

```bash
brew install xcodegen
xcodegen generate
open OutOfBody.xcodeproj      # run the OutOfBody scheme on any iPhone simulator
```

Or from the command line, with a simulator already booted:

```bash
tools/sim.sh
```

## Run it on a phone

1. Download the depth model (about 50 MB, Apache 2.0):

   ```bash
   tools/fetch_model.sh
   ```

2. Copy `Config/Local.example.xcconfig` to `Config/Local.xcconfig` and fill in your signing team, a
   bundle identifier, and the address of your GPU server if you have one.
3. `xcodegen generate`, then build to your device from Xcode, or set `OOB_DEVICE` and run
   `tools/device.sh`.

The on-screen button cycles the generation modes. The eye button hides the interface.

## The GPU server

The scripts in `tools/neural/` and `tools/predict_server.py` are what the author runs on a Linux
machine with an RTX 4090. They are provided as they are. Expect to adapt paths and environments.

- `tools/neural/refine_server.py` loads Difix, optionally with fine-tuned weights, and redraws frames.
- `tools/predict_server.py` is the endpoint the phone talks to (port 8765). It forwards redraw requests
  and can also complete unscanned parts of a panorama through ComfyUI.
- `tools/neural/ft_difix.py`, `prep_arkit.py` and `tools/dataset.sh` build training pairs (a splat
  render of a scene and the real photo from the same pose) and fine-tune the model.

The models, the dataset and the fine-tuned weights are not in this repository. See `THIRD_PARTY.md` for
what each one is and its license. Several are non-commercial.

## Tools

| Script | What it does |
|---|---|
| `tools/sim.sh`, `tools/device.sh` | Build and install to a simulator or a device |
| `tools/replay.sh` | Replay a recorded session and print mapping error against the synthetic ground truth |
| `tools/shot.sh`, `tools/video.sh` | Take a screenshot or a screen recording of a run |
| `tools/smooth.sh`, `tools/rawfrac.sh` | Measure frame-to-frame stability, and how much of the screen the generated layers fail to cover |
| `tools/prep_model.py`, `tools/bake_animated.py` | Prepare and bake an avatar, see `docs/AVATAR.md` |

Launch arguments such as `-yawSwing 0.9`, `-yawFlick 1.0`, `-rawTint YES` and `-hideUI YES` drive the
camera and debugging views for repeatable tests. They are documented where they are read, mostly at the
top of `Renderer.swift`.

## Limitations

- The generated image is soft. The training photos are at most 640×480.
- Monocular depth is unreliable on windows, mirrors and screens.
- Small objects show parallax errors when the camera orbits.
- Sitting versus standing is ambiguous when the front camera cannot see a face.
- Areas the camera has not seen are blank or invented.
- The generated view needs a GPU on the same network.

## Notes on the code

- Comments and commit messages are in Chinese.
- `OutOfBody.xcodeproj` is generated from `project.yml`. Change settings there, not in Xcode.
- The project's working name was Third Eye. A few identifiers still use it, such as the LoRA adapter
  name `thirdeye` (it is part of the checkpoint keys) and the server paths in the scripts.
- Developed on a foldable iPhone. Nothing depends on the hinge, which only poses the phone model.

## License

MIT for the code in this repository. Third-party models, data and assets have their own licenses and
are not included. See `THIRD_PARTY.md`.
