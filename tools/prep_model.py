"""把网上下载的高面数人物模型做成拿去自动绑骨骼的低面数 glb；顺便出几张预览图。
用法: /Applications/Blender.app/Contents/MacOS/Blender -b --python tools/prep_model.py -- \
        <原模型.glb> <输出.glb> <目标三角形数> <贴图边长> [预览图目录]

不能直接减面：这类模型（多半是 AI 生成的）贴图被切成几千块碎片，减面时边一合并，贴图就串到隔壁碎片上，
胸口、护肘上全是黑点。所以走标准做法：
  1. 复制一份，丢掉贴图坐标再减面（没有贴图接缝拦着，形状保得更好）；
  2. 给低面数那份重新展一套贴图坐标；
  3. 把原模型的颜色「照」到低面数模型的新贴图上（Cycles 的 selected-to-active 烘焙）。
原模型的法线贴图几乎是平的、金属/粗糙贴图很碎，都不带了；高光在 app 里按颜色深浅估。
"""
import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
src, out, target, tex = args[0], args[1], int(args[2]), int(args[3])
preview = args[4] if len(args) > 4 else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
scene = bpy.context.scene
meshes = [o for o in bpy.data.objects if o.type == "MESH"]


def only(*objs, active=None):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


if len(meshes) > 1:
    only(*meshes)
    bpy.ops.object.join()
high = bpy.context.view_layer.objects.active if len(meshes) > 1 else meshes[0]


def bounds(o):
    cs = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return Vector(map(min, *cs)), Vector(map(max, *cs))


lo, hi = bounds(high)
H = hi.z - lo.z


def shots(tag, obj):
    if not preview:
        return
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = o is not obj
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x, scene.render.resolution_y = 700, 1000
    mid = (lo + hi) / 2
    cam = bpy.data.objects.get("cam")
    if cam is None:
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        scene.collection.objects.link(cam)
        scene.camera = cam
    cam.data.lens = 60
    views = {
        "front": (0, 0, 2.6, mid), "back": (180, 0, 2.6, mid),
        "quarter": (35, 12, 2.6, mid),
        "chest": (25, 5, 1.0, Vector((mid.x, mid.y, lo.z + H * 0.80))),
        "legs": (25, 0, 1.0, Vector((mid.x, mid.y, lo.z + H * 0.28))),
        "backclose": (200, 5, 1.0, Vector((mid.x, mid.y, lo.z + H * 0.75))),
    }
    for name, (yaw, pitch, dist, look) in views.items():
        a, p = math.radians(yaw), math.radians(pitch)
        # 模型脸朝 -Y（Blender 导入 glb 后的习惯）：yaw 0 = 站在它正前方。
        d = Vector((math.sin(a) * math.cos(p), -math.cos(a) * math.cos(p), math.sin(p)))
        cam.location = look + d * dist * H
        cam.rotation_euler = (look - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = f"{preview}/{tag}_{name}.png"
        bpy.ops.render.render(write_still=True)


tris = sum(len(p.vertices) - 2 for p in high.data.polygons)
print(f"PREP 原模型 {tris} 个三角形，{len(high.data.vertices)} 个顶点，高 {H:.3f}")
shots("orig", high)

# 1. 复制、丢掉贴图坐标、并合重合顶点、减面
low = high.copy()
low.data = high.data.copy()
low.name = "low"
scene.collection.objects.link(low)
only(low)
while low.data.uv_layers:
    low.data.uv_layers.remove(low.data.uv_layers[0])
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=H * 1e-5)
bpy.ops.object.mode_set(mode="OBJECT")
m = low.modifiers.new("dec", "DECIMATE")
m.ratio = min(1.0, target / tris)
bpy.ops.object.modifier_apply(modifier=m.name)
# 外壳的折边保留成硬边，别的地方平滑。
bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))

# 2. 新的贴图坐标
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0015)
bpy.ops.object.mode_set(mode="OBJECT")
print(f"PREP 减面后 {sum(len(p.vertices) - 2 for p in low.data.polygons)} 个三角形，{len(low.data.vertices)} 个顶点")

# 3. 把原模型的颜色照到新贴图上
for mat in high.data.materials:  # 金属度会把「漫反射颜色」烘成黑的，先拔掉
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    for link in list(bsdf.inputs["Metallic"].links):
        mat.node_tree.links.remove(link)
    bsdf.inputs["Metallic"].default_value = 0
img = bpy.data.images.new("character", tex, tex, alpha=False)
mat = bpy.data.materials.new("character")
mat.use_nodes = True
node = mat.node_tree.nodes.new("ShaderNodeTexImage")
node.image = img
bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
mat.node_tree.nodes.active = node
low.data.materials.clear()
low.data.materials.append(mat)

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 4
bake = scene.render.bake
bake.use_selected_to_active = True
bake.cage_extrusion = H * 0.004
bake.max_ray_distance = H * 0.012
bake.use_pass_direct = False
bake.use_pass_indirect = False
bake.use_pass_color = True
bake.margin = 8
high.hide_render = False
low.hide_render = False
only(high, low, active=low)
bpy.ops.object.bake(type="DIFFUSE")
img.pack()
print("PREP 颜色烘焙完成")
shots("low", low)

only(low)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True,
                          export_image_format="AUTO", export_animations=False)
print("PREP 导出", out)
