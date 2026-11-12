"""Reopen the saved .blend and check structure, dimensions, materials, openings.
Run: "E:\\blender\\blender.exe" -b output\\WB_SmokeTest_room.blend --factory-startup -P smoke_verify.py
"""
import json
import bpy

o = bpy.data.objects
def dims(n): return [round(d, 3) for d in o[n].dimensions]
def evaluated_faces(n):
    ev = o[n].evaluated_get(bpy.context.evaluated_depsgraph_get())
    return len(ev.to_mesh().polygons)

checks = {
    "table_top_dims_1.8x0.9x0.04": dims("Table_Top") == [1.8, 0.9, 0.04],
    "wall_south_dims_6.2x0.2x2.8": dims("Wall_South") == [6.2, 0.2, 2.8],
    "window_width_1.5": dims("Win_East_01")[0] == 1.5,
    "vase_moved_x_3.6": round(o["Vase_Cyl"].location.x, 3) == 3.6,
    "door_boolean_cuts_wall": evaluated_faces("Wall_South") > len(o["Wall_South"].data.polygons),
    "window_boolean_cuts_wall": evaluated_faces("Wall_East") > len(o["Wall_East"].data.polygons),
    "all_meshes_clay": all(ob.data.materials and ob.data.materials[0].name == "WB_Clay"
                           for ob in o if ob.type == "MESH" and not ob.get("wb_cutter")),
    "scale_is_one": all(tuple(round(s, 6) for s in ob.scale) == (1, 1, 1) for ob in o),
    "has_camera_and_lights": bpy.context.scene.camera is not None
                             and sum(ob.type == "LIGHT" for ob in o) >= 2,
}
tree = {c.name: sorted(ob.name for ob in c.objects) for c in bpy.data.collections}
print("@@JSON@@" + json.dumps({"checks": checks, "all_pass": all(checks.values()), "collections": tree}, indent=1))
