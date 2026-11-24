# Using your own avatar

The app plays a baked vertex animation, not a skinned mesh. Two files go in `OutOfBody/Resources/`:

- `character.bin`: idle, walk and sit frames (format described at the top of `tools/bake_animated.py`)
- `character.png`: the base color texture

If they are missing, the app falls back to a capsule figure. `-capsule YES` forces the fallback.

## Steps

You need Blender (the scripts were written against 5.x) and a humanoid model.

1. **Reduce and re-bake the texture** (skip if your model is already under roughly 80k triangles and
   has a clean UV layout):

   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender -b --python tools/prep_model.py -- \
       input.glb low.glb 60000 2048 preview/
   ```

   AI-generated meshes often have thousands of tiny UV islands. Decimating those directly smears the
   texture, so the script drops the UVs, decimates, unwraps again and bakes the original color onto the
   new layout.

2. **Rig it.** The bake script expects an armature with humanoid bone names (`Hips`, `Spine`, `Chest`,
   `Neck`, `Head`, `LeftUpperArm`, `LeftLowerArm`, `LeftHand`, `LeftUpperLeg`, `LeftLowerLeg`,
   `LeftFoot`, `LeftToes`, and the same for `Right`). A model that is already rigged only needs its
   bones renamed. For an unrigged mesh, the author used SkinTokens to rig automatically and a small
   script to name the bones from the skeleton's shape.

3. **Bake the animation:**

   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender -b --python tools/bake_animated.py -- \
       rigged.glb OutOfBody/Resources/character.bin OutOfBody/Resources/character.png preview/
   ```

   The motion is procedural (foot trajectories, pelvis height, two-bone IK), so it adapts to the model's
   proportions. The right hand is posed to hold the phone.

The phone model works the same way with `tools/bake_phone.py` and `phone.bin` / `phone.png`. Without it
the app draws a box.
