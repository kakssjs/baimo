"""Stage 3: build the white-model scene for the whole video.

Run: blender -b --factory-startup --python-exit-code 1 -P build_scene.py --
         --scene SCENE.json --depth-dir DIR --width W --height H --whitebox SCRIPTS_DIR --out SCENE.blend

The scene is in metres, in the camera's own space: the camera sits at the origin looking down -Z.
  Set      one mesh; a Geometry Nodes modifier pushes every vertex out along its line of sight to
           the depth stored in that frame's depth image, so it is a true perspective surface.
  Figures  a sphere-and-box mannequin per tracked person, keyframed from 3D joints.
"""
import argparse
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True)
p.add_argument("--depth-dir", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--height", type=int, required=True)
p.add_argument("--whitebox", required=True)
p.add_argument("--out", required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])

sys.path.insert(0, a.whitebox)
import whitebox as wb  # noqa: E402

data = json.loads(open(a.scene, encoding="utf-8").read())
FRAMES = data["frames"]
FAR = 1e-3              # depth-image value treated as the far limit (keeps the sky at a finite distance)
BEVEL = 0.012
FIGURE_COLOURS = [(0.86, 0.86, 0.86), (0.88, 0.72, 0.1), (0.2, 0.62, 0.6), (0.85, 0.42, 0.15), (0.45, 0.5, 0.85)]
VISOR_COLOUR = (0.05, 0.3, 0.32)
LIMBS = {"Arm_Upper_L": (5, 7), "Arm_Lower_L": (7, 9), "Arm_Upper_R": (6, 8), "Arm_Lower_R": (8, 10),
         "Leg_Upper_L": (11, 13), "Leg_Lower_L": (13, 15), "Leg_Upper_R": (12, 14), "Leg_Lower_R": (14, 16)}
EYE = Vector((0, 0, 0))
DOWN = Vector((0, -1, 0))

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


# --- set: a grid one unit in front of the camera that exactly fills the view
image = bpy.data.images.load(os.path.join(a.depth_dir, "000001.png"))
image.source = "SEQUENCE"
image.colorspace_settings.name = "Non-Color"
half_w = 18.0 / data["lens"]
half_h = half_w * a.height / a.width
cols = min(640, data["depth_width"] // 2)
rows = max(2, round(cols * a.height / a.width))
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=cols, y_segments=rows, size=1.0)
extent = max(v.co.x for v in bm.verts)
uv = bm.loops.layers.uv.new("UVMap")
for v in bm.verts:
    v.co = Vector((v.co.x / extent * half_w, v.co.y / extent * half_h, -1.0))
for face in bm.faces:
    for loop in face.loops:
        loop[uv].uv = (loop.vert.co.x / (2 * half_w) + 0.5, loop.vert.co.y / (2 * half_h) + 0.5)
mesh = bpy.data.meshes.new("Set")
bm.to_mesh(mesh)
bm.free()
set_ob = bpy.data.objects.new("Set", mesh)
wb.collection("WB/Set").objects.link(set_ob)
wb.assign_clay(set_ob, material("WB_Clay", (0.8, 0.8, 0.8), roughness=0.7))

# Geometry Nodes: position = position * (near / depth_image), in the depth model's own units
tree = bpy.data.node_groups.new("WB_DepthToSurface", "GeometryNodeTree")
tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
new, link = tree.nodes.new, tree.links.new


def math_node(operation, first, second):
    node = new("ShaderNodeMath")
    node.operation = operation
    for socket, value in zip(node.inputs, (first, second)):
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            link(value, socket)
    return node.outputs[0]


group_in, group_out = new("NodeGroupInput"), new("NodeGroupOutput")
uv_attr = new("GeometryNodeInputNamedAttribute")
uv_attr.data_type = "FLOAT_VECTOR"
uv_attr.inputs["Name"].default_value = "UVMap"
tex = new("GeometryNodeImageTexture")
tex.interpolation, tex.extension = "Linear", "EXTEND"
tex.inputs["Image"].default_value = image
link(uv_attr.outputs["Attribute"], tex.inputs["Vector"])
link(new("GeometryNodeInputSceneTime").outputs["Frame"], tex.inputs["Frame"])
distance = math_node("DIVIDE", data["near"], math_node("MAXIMUM", tex.outputs["Color"], FAR))
push = new("ShaderNodeVectorMath")
push.operation = "SCALE"
link(new("GeometryNodeInputPosition").outputs["Position"], push.inputs["Vector"])
link(distance, push.inputs["Scale"])
set_position = new("GeometryNodeSetPosition")
link(group_in.outputs["Geometry"], set_position.inputs["Geometry"])
link(push.outputs["Vector"], set_position.inputs["Position"])
# Smooth surfaces, but keep a hard crease wherever the surface folds sharply (depth edges).
smooth_faces = new("GeometryNodeSetShadeSmooth")
link(set_position.outputs["Geometry"], smooth_faces.inputs["Geometry"])
crease = new("GeometryNodeSetShadeSmooth")
crease.domain = "EDGE"
link(smooth_faces.outputs["Geometry"], crease.inputs["Geometry"])
link(math_node("LESS_THAN", new("GeometryNodeInputMeshEdgeAngle").outputs["Unsigned Angle"], math.radians(40)),
     crease.inputs["Shade Smooth"])
link(crease.outputs["Geometry"], group_out.inputs["Geometry"])

modifier = set_ob.modifiers.new("Depth", "NODES")
modifier.node_group = tree
# The camera is at the origin, so scaling the object turns model units into metres. One scale
# per shot, keyed at both ends so it never blends across a cut.
for shot in data["shots"]:
    for frame in (shot["start"], shot["end"]):
        set_ob.scale = (shot["scale"],) * 3
        set_ob.keyframe_insert("scale", frame=frame)


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
        centre = head + look.normalized() * 0.465 * diameter
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
        if name != "Head":
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

# --- camera at the origin, looking down -Z
cam = bpy.data.objects.new("WB_Camera", bpy.data.cameras.new("WB_Camera"))
wb.collection("WB/Cameras").objects.link(cam)
cam.data.sensor_fit, cam.data.sensor_width, cam.data.lens = "HORIZONTAL", 36.0, data["lens"]
cam.data.clip_start, cam.data.clip_end = 0.05, 5000.0
sc.camera = cam

# --- light: soft sky plus a sun from over the camera's left shoulder
sun = bpy.data.objects.new("WB_Sun", bpy.data.lights.new("WB_Sun", "SUN"))
sun.data.energy, sun.data.angle = 2.0, math.radians(6)
sun.rotation_euler = Vector((0.6, -0.7, -0.8)).to_track_quat("-Z", "Y").to_euler()
wb.collection("WB/Lighting").objects.link(sun)
world = bpy.data.worlds.new("WB_World")
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.62, 0.64, 0.67, 1)
bg.inputs["Strength"].default_value = 0.6
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
