"""Stage 3: build the white-model scene for the whole video.

Run: blender -b --factory-startup --python-exit-code 1 -P build_scene.py --
         --scene SCENE.json --width W --height H --whitebox SCRIPTS_DIR --out SCENE.blend

The scene is in metres, z up, with the camera fixed at the origin looking along +Y.
  Room     whitebox boxes for the floor, back wall, side walls and ceiling, parented to an empty
           that turns as the camera pans; each surface is keyframed to its fitted distance.
  Figures  a sphere-and-box mannequin per tracked person, keyframed from 3D joints.
"""
import argparse
import json
import math
import sys

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--height", type=int, required=True)
p.add_argument("--whitebox", required=True)
p.add_argument("--out", required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])

sys.path.insert(0, a.whitebox)
import whitebox as wb  # noqa: E402

data = json.loads(open(a.scene, encoding="utf-8").read())
FRAMES = data["frames"]
SPAN = 60.0             # metres; walls and floor run this far so their ends are never in view
WALL_HEIGHT = 12.0
THICK = 0.2
BEVEL = 0.012
FIGURE_COLOURS = [(0.92, 0.92, 0.9), (0.88, 0.72, 0.1), (0.2, 0.62, 0.6), (0.85, 0.42, 0.15), (0.45, 0.5, 0.85)]
VISOR_COLOUR = (0.05, 0.3, 0.32)
LIMBS = {"Arm_Upper_L": (5, 7), "Arm_Lower_L": (7, 9), "Arm_Upper_R": (6, 8), "Arm_Lower_R": (8, 10),
         "Leg_Upper_L": (11, 13), "Leg_Lower_L": (13, 15), "Leg_Upper_R": (12, 14), "Leg_Lower_R": (14, 16)}
EYE = Vector((0, 0, 0))
DOWN = Vector((0, 0, -1))

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.unit_settings.system, sc.unit_settings.scale_length = "METRIC", 1.0
sc.frame_start, sc.frame_end = 1, FRAMES


def setting(owner, name, value):
    """Set a render option if this Blender version has it (names differ between releases)."""
    try:
        setattr(owner, name, value)
    except (AttributeError, TypeError):
        pass


def material(name, colour, roughness=0.65):
    mat = wb.clay_material(name, roughness=roughness)
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    mat.diffuse_color = (*colour, 1.0)
    return mat


# --- room: five whitebox boxes inside an empty that carries the room's turn
room = data["room"]
pivot = bpy.data.objects.new("Room", None)
wb.collection("WB/Room").objects.link(pivot)
clay = material("WB_Clay", (0.62, 0.62, 0.62), roughness=0.75)
floor_clay = material("WB_Clay_Floor", (0.4, 0.4, 0.4), roughness=0.8)
# name -> (box size, where its inner face goes for a fitted distance d and floor height f)
SURFACES = {
    "floor":   ((SPAN, SPAN, THICK),        lambda d, f: (0, 0, d - THICK)),
    "ceiling": ((SPAN, SPAN, THICK),        lambda d, f: (0, 0, d)),
    "back":    ((SPAN, THICK, WALL_HEIGHT), lambda d, f: (0, d + THICK / 2, f)),
    "left":    ((THICK, SPAN, WALL_HEIGHT), lambda d, f: (d - THICK / 2, 0, f)),
    "right":   ((THICK, SPAN, WALL_HEIGHT), lambda d, f: (d + THICK / 2, 0, f)),
}
surfaces = {}
for key, (size, _) in SURFACES.items():
    ob = wb.box(f"Room_{key.capitalize()}", size, col="WB/Room")
    wb.assign_clay(ob, floor_clay if key == "floor" else clay)
    ob.parent = pivot
    surfaces[key] = ob
for frame in range(1, FRAMES + 1):
    i = frame - 1
    pivot.rotation_euler = (0, 0, room["theta"][i])
    pivot.keyframe_insert("rotation_euler", frame=frame)
    for key, ob in surfaces.items():
        distance = room[key][i]
        if distance is None:        # this shot has no such surface
            ob.scale = (0, 0, 0)
        else:
            ob.scale = (1, 1, 1)
            ob.location = SURFACES[key][1](distance, room["floor"][i])
        ob.keyframe_insert("location", frame=frame)
        ob.keyframe_insert("scale", frame=frame)


# --- figures: sphere head, box torso and limbs, placed between 3D joints
def part_mesh(name, sphere):
    bm = bmesh.new()
    if sphere:
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.5)
    else:
        bmesh.ops.create_cube(bm, size=1.0)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if sphere:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def facing(along, toward):
    """Rotation whose local Y runs along `along` and whose local Z faces `toward` as far as it can."""
    y = along.normalized()
    z = toward - toward.project(y)
    z = z.normalized() if z.length > 1e-6 else y.orthogonal().normalized()
    return Matrix((y.cross(z), y, z)).transposed().to_quaternion()


def stretch(p0, p1, thickness, width=None):
    """Placement of a unit cube stretched from p0 to p1, front face toward the camera."""
    mid = (p0 + p1) / 2
    return mid, facing(p1 - p0, EYE - mid), Vector((width or thickness, (p1 - p0).length, thickness))


def placements(joints, size):
    """Location, rotation and scale of each body part for one pose; unseen parts are left out."""
    pts = {i: Vector(j) for i, j in enumerate(joints) if j}
    if 5 not in pts or 6 not in pts:
        return {}
    shoulders = (pts[5] + pts[6]) / 2
    hips = (pts[11] + pts[12]) / 2 if 11 in pts and 12 in pts else shoulders + DOWN * 1.4 * size
    out = {"Torso": stretch(hips, shoulders, 0.5 * size, width=size)}
    face = [pts[i] for i in range(5) if i in pts]
    spine = (shoulders - hips).normalized()
    head = sum(face, Vector()) / len(face) if face else shoulders + spine * 0.7 * size
    diameter = 0.8 * size
    out["Head"] = (head, Quaternion(), Vector((diameter,) * 3))
    if 0 in pts and (1 in pts or 2 in pts):      # nose and an eye are visible: mark where the face is
        look = (pts[0] - head) + (EYE - head).normalized() * 0.3 * diameter
        centre = head + look.normalized() * 0.49 * diameter
        out["Visor"] = (centre, facing(spine.cross(EYE - centre), EYE - centre),
                        Vector((0.16 * diameter, 0.5 * diameter, 0.1 * diameter)))
    for name, (i, j) in LIMBS.items():
        if i in pts and j in pts:
            out[name] = stretch(pts[i], pts[j], (0.32 if name.startswith("Leg") else 0.26) * size)
    return out


visor_mat = material("WB_Visor", VISOR_COLOUR, roughness=0.4)
for number, fig in enumerate(data["figures"], 1):
    body = material(f"WB_Figure_{number:02d}", FIGURE_COLOURS[(number - 1) % len(FIGURE_COLOURS)])
    col = wb.collection(f"WB/Figures/Figure_{number:02d}")
    parts = {}
    for name in ["Head", "Visor", "Torso", *LIMBS]:
        ob = bpy.data.objects.new(f"Fig{number:02d}_{name}", part_mesh(f"Fig{number:02d}_{name}", name == "Head"))
        col.objects.link(ob)
        wb.assign_clay(ob, visor_mat if name == "Visor" else body)
        ob.rotation_mode = "QUATERNION"
        if name not in ("Head", "Visor"):
            mod = ob.modifiers.new("WB_Bevel", "BEVEL")
            mod.width, mod.segments, mod.limit_method = BEVEL, 2, "ANGLE"
        parts[name] = ob
    start = fig["start"]
    # Hidden (scale 0) on every frame the person is not seen, including just before and after.
    poses = [(start - 1, None), *enumerate(fig["joints"], start), (start + len(fig["joints"]), None)]
    last = {}
    for frame, joints in poses:
        if not 1 <= frame <= FRAMES:
            continue
        placed = placements(joints, fig["size"]) if joints else {}
        for name, ob in parts.items():
            if name in placed:
                loc, rot, scale = placed[name]
                if name in last and rot.dot(last[name]) < 0:
                    rot.negate()    # same hemisphere as the frame before, so nothing spins the long way
                last[name] = rot
                ob.location, ob.rotation_quaternion, ob.scale = loc, rot, scale
            else:
                ob.scale = (0, 0, 0)
            for path in ("location", "rotation_quaternion", "scale"):
                ob.keyframe_insert(path, frame=frame)

# --- camera at the origin, level, looking along +Y
cam = bpy.data.objects.new("WB_Camera", bpy.data.cameras.new("WB_Camera"))
wb.collection("WB/Cameras").objects.link(cam)
cam.rotation_euler = (math.radians(90), 0, 0)
cam.data.sensor_fit, cam.data.sensor_width, cam.data.lens = "HORIZONTAL", 36.0, data["lens"]
cam.data.clip_start, cam.data.clip_end = 0.05, 500.0
sc.camera = cam

# --- light: soft sky plus a sun from over the camera's left shoulder
sun = bpy.data.objects.new("WB_Sun", bpy.data.lights.new("WB_Sun", "SUN"))
sun.data.energy, sun.data.angle = 3.2, math.radians(5)
sun.rotation_euler = Vector((0.8, 0.45, -0.7)).to_track_quat("-Z", "Y").to_euler()
wb.collection("WB/Lighting").objects.link(sun)
world = bpy.data.worlds.new("WB_World")
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.62, 0.64, 0.67, 1)
bg.inputs["Strength"].default_value = 0.45
sc.world = world

# --- render look: clean clay with soft shadows and ambient occlusion
engines = [e.identifier for e in sc.render.bl_rna.properties["engine"].enum_items]
sc.render.engine = next((e for e in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT") if e in engines), "CYCLES")
setting(sc.eevee, "taa_render_samples", 32)
setting(sc.eevee, "use_shadows", True)
setting(sc.eevee, "use_raytracing", True)
setting(sc.eevee, "use_fast_gi", True)
setting(sc.eevee, "fast_gi_method", "AMBIENT_OCCLUSION_ONLY")
setting(sc.eevee, "fast_gi_distance", 1.0)
setting(sc.view_settings, "view_transform", "Standard")
sc.render.resolution_x, sc.render.resolution_y = a.width, a.height

bpy.ops.wm.save_as_mainfile(filepath=a.out)
