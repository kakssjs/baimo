"""Stage 3: build the white-model scene for the whole video.

Run: blender -b --factory-startup --python-exit-code 1 -P build_scene.py --
         --scene SCENE.json --sets-dir DIR --width W --height H --whitebox SCRIPTS_DIR --out SCENE.blend

The scene is in metres, z up; each shot starts with the camera at the origin looking along +Y.
  Sets     a mesh per stretch of a shot, standing still while the camera moves through it: every
           pixel of the set's picture (sets/NNN.npy, inverse depth seen from its keyframe) is a
           vertex out along its line of sight, flat surfaces merged into single faces. Each is
           shown only during its stretch, and over its last frames takes on the next one's shape.
           What was too dark to see (-1 in the picture) is a dark, unlit backdrop behind it.
  Figures  a sphere-and-box mannequin per tracked person: every part keeps one size throughout
           and is placed and turned in every frame as scene.json says.
  Camera   placed and turned in every frame as scene.json says (the camera's path in the video).
"""
import argparse
import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Quaternion, Vector

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True)
p.add_argument("--sets-dir", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--height", type=int, required=True)
p.add_argument("--whitebox", required=True)
p.add_argument("--out", required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])

sys.path.insert(0, a.whitebox)
import whitebox as wb  # noqa: E402

data = json.loads(open(a.scene, encoding="utf-8").read())
FRAMES = data["frames"]
BLOCK = 8                  # pixels: a flat square of a set's picture this size becomes one face ...
FLAT = 1e-4                # ... where its depths lie within this share of one plane (walls, floors, steps)
CREASE = math.radians(40)  # set surfaces folding more sharply than this are shaded as hard edges
EDGE_ON = 0.02             # a square of a set seen this nearly edge-on from its camera (cosine) spans a gap ...
BACK = 4                   # ... and is laid on what lies furthest back within this many pixels, unless it is
LEDGE = 0.1                # the top of something level with the camera (a tread, a landing): near below far,
LEDGE_GAP = 0.4            # the sight line between them at most this steeply down (tangent), and the far side
                           # at most this share further back
HAND_OVER = 8              # frames over which a set takes on the next one's shape before that takes over ...
MATCH = 0.5                # ... where both show the same surface (depths this close)
VOID_MARGIN = 0.03         # share of a set's width its dark backdrop reaches in behind the set's edges ...
VOID_FAR = 2.0             # ... and it stands this many times as far back as the farthest of the set
BEVEL = 0.012
SET_COLOUR = (0.6, 0.6, 0.6)
VOID_COLOUR = (0.006, 0.008, 0.016)     # what was too dark to see: an empty, dark stage (shown as is, unlit)
FIGURE_COLOUR = (0.85, 0.85, 0.84)
VISOR_COLOUR = (0.02, 0.33, 0.35)

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


def unlit_material(name, colour):
    """A material that shows one colour whatever the light falling on it."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()       # (built afresh: a node kept while the others are removed can be left dangling)
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*colour, 1.0)
    mat.node_tree.links.new(emit.outputs[0], out.inputs["Surface"])
    mat.diffuse_color = (*colour, 1.0)
    return mat


def shown(ob, start, end):
    """Show an object only from frame start to frame end (switched, not faded)."""
    for frame, hidden in ((start - 1, True), (start, False), (end, False), (end + 1, True)):
        if 1 <= frame <= FRAMES:
            ob.hide_render = ob.hide_viewport = hidden
            ob.keyframe_insert("hide_render", frame=frame)
            ob.keyframe_insert("hide_viewport", frame=frame)


# --- sets: one mesh per stretch, in its keyframe camera's space (x right, y forward, z up)
def set_mesh(name, q, cx, cy, focal):
    """A set's picture (inverse depth q, > 0 where there is a surface) as a mesh: pixels are
    vertices out along their lines of sight, the squares between them faces; a BLOCK-sized
    square that is flat is one face. (A pixel on the edge of such a face lies on that edge, as
    the face is flat, so no cracks open beside it.)"""
    rows, cols = q.shape
    q = q.astype(np.float64)
    valid = q > 0
    br, bc = (rows - 1) // BLOCK, (cols - 1) // BLOCK      # whole blocks; block (r, c) spans pixels
    # r * BLOCK .. r * BLOCK + BLOCK down and c * BLOCK .. c * BLOCK + BLOCK across, edges included.

    def at(di, dj):
        return q[di:di + br * BLOCK:BLOCK, dj:dj + bc * BLOCK:BLOCK][:br, :bc]

    q00, q01, q10, q11 = at(0, 0), at(0, BLOCK), at(BLOCK, 0), at(BLOCK, BLOCK)
    flat = np.ones((br, bc), bool)
    for di in range(BLOCK + 1):     # on a plane, inverse depth is linear in the pixel position
        for dj in range(BLOCK + 1):
            y, x = di / BLOCK, dj / BLOCK
            v = at(di, dj)
            plane = (1 - y) * ((1 - x) * q00 + x * q01) + y * ((1 - x) * q10 + x * q11)
            flat &= (v > 0) & (np.abs(v - plane) <= FLAT * v)
    # Every pixel square with a surface at all four corners, except those inside flat blocks ...
    fine = valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:] & valid[:-1, :-1]
    fine[:br * BLOCK, :bc * BLOCK] &= ~np.repeat(np.repeat(flat, BLOCK, 0), BLOCK, 1)
    i, j = np.nonzero(fine)
    # ... and one square per flat block; corners in this order face the camera.
    r, c = np.nonzero(flat)
    rows_ = np.concatenate([np.stack((i + 1, i + 1, i, i), 1), np.stack((r + 1, r + 1, r, r), 1) * BLOCK])
    cols_ = np.concatenate([np.stack((j, j + 1, j + 1, j), 1), np.stack((c, c + 1, c + 1, c), 1) * BLOCK])
    # A square seen almost edge-on from the camera is no surface but the gap between something near
    # and something further back (the edge of a pillar, a hanging lamp); seen from anywhere else it
    # would be a sheet stretched across that gap. Its near corners go back onto what lies behind
    # (copies of their vertices, so the near surface keeps its own edge), which still closes the
    # seam as seen from the camera.
    a00, a01, a10, a11 = q[i, j], q[i, j + 1], q[i + 1, j], q[i + 1, j + 1]
    gx, gy = (a01 - a00 + a11 - a10) / 2, (a10 - a00 + a11 - a01) / 2    # per pixel
    mid, x, y = (a00 + a01 + a10 + a11) / 4, j + 1 - cx, i + 1 - cy
    # On a plane 1/z = gx x + gy y + k (x, y in pixels from the centre), the normal is (gx f, gy f, k).
    k = mid - gx * x - gy * y
    cos = mid / (np.sqrt((gx ** 2 + gy ** 2) * focal ** 2 + k ** 2) * np.sqrt((x ** 2 + y ** 2) / focal ** 2 + 1))
    # Such a square stays, though, where it is the top of something the camera sees edge-on, being
    # level with it (the treads of a stair and its landing, the top of a radiator): the near side
    # is below the far side and the sight line between them runs level or a little down, so the
    # square is that top, from its front edge to what stands behind it, and whoever stands there
    # stands on it. (Seen from further above, such a top would show; further back, the far side is
    # what lies behind a lamp or a rail.)
    top = (gy > 2 * np.abs(gx)) & (y <= LEDGE * focal) & (
        np.maximum.reduce([a00, a01, a10, a11]) <= (1 + LEDGE_GAP) * np.minimum.reduce([a00, a01, a10, a11]))
    gap = np.concatenate([(cos < EDGE_ON) & ~top, np.zeros(len(r), bool)])
    back = np.where(valid, q, np.inf)       # the furthest surface (least inverse depth) within BACK pixels
    for axis in (0, 1):
        pad = np.pad(back, [(BACK, BACK) if ax == axis else (0, 0) for ax in (0, 1)], constant_values=np.inf)
        back = np.min([np.take(pad, range(s, s + q.shape[axis]), axis=axis) for s in range(2 * BACK + 1)], 0)
    near = gap[:, None] & (q[rows_, cols_] > 1.05 * back[rows_, cols_])
    pixel = rows_ * cols + cols_ + near * (rows * cols)     # (a copy of the pixel, laid back)
    used, quads = np.unique(pixel, return_inverse=True)
    copy, at_pixel = np.divmod(used, rows * cols)
    ii, jj = np.divmod(at_pixel, cols)
    z = 1.0 / np.where(copy > 0, back[ii, jj], q[ii, jj])
    verts = np.stack(((jj + 0.5 - cx) / focal * z, z, -(ii + 0.5 - cy) / focal * z), axis=1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    me.polygons.add(len(pixel))
    me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=CREASE)
    return me


def backdrop(q):
    """The picture of a set's dark backdrop: where the set was too dark to see (q < 0), and
    VOID_MARGIN of its width around that in behind the set, so that no light gap opens beside it
    as the camera moves; one flat surface facing the camera, VOID_FAR times as far back as the
    farthest of the set. (Not behind the rest: a gap at an edge there shows the sky, which stands
    out far less against the clay than the dark would.)"""
    r = max(1, round(VOID_MARGIN * q.shape[1]))
    grown = q < 0
    for axis in (0, 1):     # (widened along one axis, then the other, by running sums)
        n = q.shape[axis]
        c = np.cumsum(np.pad(grown, [(r + 1, r) if a == axis else (0, 0) for a in (0, 1)]), axis=axis, dtype=np.int32)
        grown = np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=axis) > np.take(c, range(n), axis=axis)
    far = float(q[q > 0].min()) if (q > 0).any() else 0.2
    return np.where(grown, far / VOID_FAR, 0.0)


def hand_over(ob, st, nxt, q):
    """Over the last HAND_OVER frames of its stretch, a set (object ob, scene.json entry st) takes
    on the shape of the next set of its shot (nxt, with picture q) wherever both show the same
    surface, so that the switch from one to the other changes next to nothing."""
    co = np.empty(len(ob.data.vertices) * 3, np.float32)
    ob.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    turn = [np.array(Quaternion(s["rot"]).to_matrix()) for s in (st, nxt)]
    world = np.array(st["loc"]) + st["scale"] * co @ turn[0].T
    there = (world - np.array(nxt["loc"])) @ turn[1] / nxt["scale"]       # in the next set's space
    ahead = there[:, 1] > 1e-6
    depth = np.where(ahead, there[:, 1], 1.0)
    i = np.round(-there[:, 2] / depth * nxt["focal"] + nxt["cy"] - 0.5).astype(np.int64)
    j = np.round(there[:, 0] / depth * nxt["focal"] + nxt["cx"] - 0.5).astype(np.int64)
    inside = ahead & (i >= 0) & (i < q.shape[0]) & (j >= 0) & (j < q.shape[1])
    qn = np.zeros(len(co))
    qn[inside] = q[i[inside], j[inside]]
    same = (qn > 0) & (np.abs(qn * depth - 1) < MATCH)
    there *= np.where(same, 1.0 / np.where(same, qn * depth, 1.0), 1.0)[:, None]   # along its line of sight
    moved = (np.array(nxt["loc"]) + nxt["scale"] * there @ turn[1].T - np.array(st["loc"])) @ turn[0] / st["scale"]
    ob.shape_key_add(name="Basis", from_mix=False)
    key = ob.shape_key_add(name="Next", from_mix=False)
    key.data.foreach_set("co", np.where(same[:, None], moved, co).astype(np.float32).ravel())
    for frame, value in ((max(st["start"], st["end"] - HAND_OVER), 0.0), (st["end"], 1.0)):
        key.value = value
        key.keyframe_insert("value", frame=frame)


clay = material("WB_Clay", SET_COLOUR, roughness=0.75)
void = unlit_material("WB_Void", VOID_COLOUR)
sets = wb.collection("WB/Set")
previous = None
for number, st in enumerate(data["sets"], 1):
    q = np.load(os.path.join(a.sets_dir, st["file"]))
    ob = bpy.data.objects.new(f"Set_{number:03d}", set_mesh(f"Set_{number:03d}", q, st["cx"], st["cy"], st["focal"]))
    parts = [(ob, clay)]
    if (q < 0).any():
        back = bpy.data.objects.new(f"Set_{number:03d}_Void", set_mesh(f"Set_{number:03d}_Void", backdrop(q),
                                                                          st["cx"], st["cy"], st["focal"]))
        setting(back, "visible_shadow", False)      # (it stands for nothing, so it casts nothing)
        parts.append((back, void))
    for part, mat in parts:
        sets.objects.link(part)
        wb.assign_clay(part, mat)
        part.rotation_mode = "QUATERNION"
        part.location, part.rotation_quaternion, part.scale = st["loc"], st["rot"], (st["scale"],) * 3
        shown(part, st["start"], st["end"])
    if previous and previous[1]["shot"] == st["shot"] and previous[1]["end"] + 1 == st["start"]:
        hand_over(*previous, st, q)
    previous = ob, st


# --- figures: sphere head, box torso and limbs, each part one size throughout
def part_mesh(name, size, sphere):
    bm = bmesh.new()
    if sphere:
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.5)
    else:
        bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if sphere:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


edit_prefs = bpy.context.preferences.edit
body_mat = material("WB_Figure", FIGURE_COLOUR)
visor_mat = material("WB_Visor", VISOR_COLOUR, roughness=0.4)
for number, fig in enumerate(data["figures"], 1):
    col = wb.collection(f"WB/Figures/Figure_{number:02d}")
    start = fig["start"]
    for name, track in fig["parts"].items():
        size = [fig["size"] * s for s in data["part_sizes"][name]]
        ob = bpy.data.objects.new(f"Fig{number:02d}_{name}", part_mesh(f"Fig{number:02d}_{name}", size, name == "Head"))
        col.objects.link(ob)
        wb.assign_clay(ob, visor_mat if name == "Visor" else body_mat)
        ob.rotation_mode = "QUATERNION"
        if name != "Head":
            mod = ob.modifiers.new("WB_Bevel", "BEVEL")
            mod.width, mod.segments, mod.limit_method = min(BEVEL, 0.2 * min(size)), 2, "ANGLE"
        edit_prefs.keyframe_new_interpolation_type = "LINEAR"
        for frame, loc, rot in zip(range(start, start + len(track["loc"])), track["loc"], track["rot"]):
            ob.location, ob.rotation_quaternion = loc, rot
            ob.keyframe_insert("location", frame=frame)
            ob.keyframe_insert("rotation_quaternion", frame=frame)
        shown(ob, start, start + len(track["loc"]) - 1)        # only while the person is

# --- camera: placed and turned in every frame
cam = bpy.data.objects.new("WB_Camera", bpy.data.cameras.new("WB_Camera"))
wb.collection("WB/Cameras").objects.link(cam)
cam.rotation_mode = "QUATERNION"
edit_prefs.keyframe_new_interpolation_type = "LINEAR"
for frame, (loc, rot) in enumerate(zip(data["camera"]["loc"], data["camera"]["rot"]), 1):
    cam.location, cam.rotation_quaternion = loc, rot
    cam.keyframe_insert("location", frame=frame)
    cam.keyframe_insert("rotation_quaternion", frame=frame)
edit_prefs.keyframe_new_interpolation_type = "BEZIER"
cam.data.sensor_fit, cam.data.sensor_width, cam.data.lens = "HORIZONTAL", 36.0, data["lens"]
cam.data.clip_start, cam.data.clip_end = 0.05, 5000.0
sc.camera = cam

# --- light: a sun from over the left shoulder of each shot's first camera (so whatever faces the
# camera is lit and every surface turned another way gets its own shade), standing still while the
# camera moves, plus soft light from all round
sun = bpy.data.objects.new("WB_Sun", bpy.data.lights.new("WB_Sun", "SUN"))
sun.data.energy, sun.data.angle = 2.5, math.radians(5)
sun.rotation_euler = Vector((0.35, 1.0, -0.55)).to_track_quat("-Z", "Y").to_euler()
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

bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
