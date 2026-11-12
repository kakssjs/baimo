"""Stage 4: build the white-model scene: a clay relief of the set plus a figure for each person.

Run: blender -b --factory-startup --python-exit-code 1 -P build_relief.py --
         --depth-dir DIR --figures FIGURES.json --width W --height H --frames N
         --whitebox SCRIPTS_DIR --out SHOT.blend

A grid the shape of the video frame is displaced toward an orthographic camera by the depth
sequence, so every vertex stays on its own pixel and the render lines up with the source video.
Figures are built in the same frame space from 2D poses, at the depth the relief had for that person.
"""
import argparse
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

p = argparse.ArgumentParser()
p.add_argument("--depth-dir", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--height", type=int, required=True)
p.add_argument("--frames", type=int, required=True)
p.add_argument("--figures", help="figures.json from the depth stage")
p.add_argument("--whitebox", required=True)
p.add_argument("--out", required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])

sys.path.insert(0, a.whitebox)
import whitebox as wb  # noqa: E402

FRAME_W = 2.0                           # scene units across the frame
FRAME_H = FRAME_W * a.height / a.width
STRENGTH = 0.5                          # how far the nearest point rises out of the plane
COLS = min(400, a.width // 4)           # mesh density; finer only adds noise from the depth map

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

# --- relief mesh: a UV-mapped grid, smooth shaded, clay material
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=COLS, y_segments=max(2, round(COLS * FRAME_H / FRAME_W)), size=1.0)
extent = max(v.co.x for v in bm.verts)
for v in bm.verts:
    v.co.x *= FRAME_W / 2 / extent
    v.co.y *= FRAME_H / 2 / extent
uv = bm.loops.layers.uv.new("UVMap")    # frame bottom-left = (0, 0), top-right = (1, 1)
for face in bm.faces:
    for loop in face.loops:
        loop[uv].uv = (loop.vert.co.x / FRAME_W + 0.5, loop.vert.co.y / FRAME_H + 0.5)
mesh = bpy.data.meshes.new("Relief")
bm.to_mesh(mesh)
bm.free()
mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
relief = bpy.data.objects.new("Relief", mesh)
wb.collection("WB/Relief").objects.link(relief)
wb.assign_clay(relief)

# --- depth sequence drives a Displace modifier
image = bpy.data.images.load(os.path.join(a.depth_dir, "0001.png"))
image.source = "SEQUENCE"
image.colorspace_settings.name = "Non-Color"
tex = bpy.data.textures.new("WB_Depth", "IMAGE")
tex.image = image
tex.extension = "EXTEND"
tex.image_user.frame_start, tex.image_user.frame_duration = 1, a.frames
tex.image_user.use_auto_refresh = True
mod = relief.modifiers.new("Depth", "DISPLACE")
mod.texture, mod.texture_coords = tex, "UV"
mod.direction, mod.mid_level, mod.strength = "Z", 0.0, STRENGTH

# --- figures: one sphere-and-box mannequin per tracked person, keyframed from the 2D poses
FIGURE_COLOURS = [(0.8, 0.8, 0.8), (0.85, 0.7, 0.1), (0.2, 0.65, 0.6), (0.85, 0.4, 0.15)]
LIMBS = {"Arm_Upper_L": (5, 7), "Arm_Lower_L": (7, 9), "Arm_Upper_R": (6, 8), "Arm_Lower_R": (8, 10),
         "Leg_Upper_L": (11, 13), "Leg_Lower_L": (13, 15), "Leg_Upper_R": (12, 14), "Leg_Lower_R": (14, 16)}
figures = json.loads(open(a.figures, encoding="utf-8").read()) if a.figures else {"tracks": []}
MIN_KEYPOINT = figures.get("min_keypoint", 0.35)


def part_mesh(name, sphere):
    bm = bmesh.new()
    if sphere:
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.5)
    else:
        bmesh.ops.create_cube(bm, size=1.0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if sphere:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def segment(p, q, thickness, z):
    """Placement of a unit cube stretched from scene point p to q: (location, z rotation, scale)."""
    d = q - p
    mid = (p + q) / 2
    return (mid.x, mid.y, z), math.atan2(d.y, d.x) - math.pi / 2, (thickness, d.length, thickness)


def placements(pose):
    """Where each body part goes for one pose; parts that cannot be seen are left out."""
    pts = {i: Vector(((x / a.width - 0.5) * FRAME_W, (0.5 - y / a.height) * FRAME_H))
           for i, (x, y, score) in enumerate(pose["kp"]) if score >= MIN_KEYPOINT}
    face = [pts[i] for i in range(5) if i in pts]
    if 5 not in pts or 6 not in pts:
        return {}
    shoulders = (pts[5] + pts[6]) / 2
    hips = (pts[11] + pts[12]) / 2 if 11 in pts and 12 in pts else None
    size = max([(pts[5] - pts[6]).length]
               + ([0.55 * (shoulders - hips).length] if hips else [])
               + [2.2 * (p - q).length for p in face for q in face])
    if hips is None:        # waist is out of frame or hidden: assume an upright torso
        hips = shoulders - Vector((0, 1.4 * size))
    z = pose["z"] * STRENGTH + 0.3 * size
    out = {"Torso": segment(hips, shoulders, size, z)}
    out["Torso"] = (out["Torso"][0], out["Torso"][1], (size, out["Torso"][2][1], 0.5 * size))
    head = sum(face, Vector((0, 0))) / len(face) if face else shoulders + (shoulders - hips).normalized() * 0.7 * size
    out["Head"] = ((head.x, head.y, z), 0.0, (0.8 * size,) * 3)
    for name, (i, j) in LIMBS.items():
        if i in pts and j in pts:
            out[name] = segment(pts[i], pts[j], (0.32 if name.startswith("Leg") else 0.26) * size, z)
    return out


for number, track_data in enumerate(figures["tracks"], 1):
    colour = FIGURE_COLOURS[(number - 1) % len(FIGURE_COLOURS)]
    mat = wb.clay_material(f"WB_Figure_{number:02d}")
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    mat.diffuse_color = (*colour, 1.0)
    col = wb.collection(f"WB/Figures/Figure_{number:02d}")
    parts = {}
    for name in ["Head", "Torso", *LIMBS]:
        ob = bpy.data.objects.new(f"Fig{number:02d}_{name}", part_mesh(f"Fig{number:02d}_{name}", name == "Head"))
        col.objects.link(ob)
        wb.assign_clay(ob, mat)
        parts[name] = ob
    start = track_data["start"]
    # Hidden (scale 0) on every frame the person is not seen, including just before and after.
    frames = [(start - 1, None)] + list(enumerate(track_data["poses"], start)) + [(start + len(track_data["poses"]), None)]
    for frame, pose in frames:
        if not 1 <= frame <= a.frames:
            continue
        placed = placements(pose) if pose else {}
        for name, ob in parts.items():
            if name in placed:
                ob.location, ob.rotation_euler.z, ob.scale = placed[name]
            else:
                ob.scale = (0, 0, 0)
            for path in ("location", "rotation_euler", "scale"):
                ob.keyframe_insert(path, frame=frame)

# --- orthographic camera looking straight down at the frame
cam = bpy.data.objects.new("WB_Camera", bpy.data.cameras.new("WB_Camera"))
cam.data.type, cam.data.ortho_scale = "ORTHO", max(FRAME_W, FRAME_H)
cam.data.clip_start, cam.data.clip_end = 0.1, 20
cam.location = (0, 0, 5)
wb.collection("WB/Cameras").objects.link(cam)
sc.camera = cam

# --- light from the upper left, soft grey ambient
sun = bpy.data.objects.new("WB_Sun", bpy.data.lights.new("WB_Sun", "SUN"))
sun.data.energy, sun.data.angle = 2.2, 0.3
sun.rotation_euler = Vector((1.0, -1.0, -0.8)).to_track_quat("-Z", "Y").to_euler()
wb.collection("WB/Lighting").objects.link(sun)
world = bpy.data.worlds.new("WB_World")
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
bg.inputs["Strength"].default_value = 0.3
sc.world = world

engines = [e.identifier for e in sc.render.bl_rna.properties["engine"].enum_items]
sc.render.engine = next((e for e in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT") if e in engines), "CYCLES")
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x, sc.render.resolution_y = a.width, a.height
sc.frame_start, sc.frame_end = 1, a.frames

bpy.ops.wm.save_as_mainfile(filepath=a.out)
