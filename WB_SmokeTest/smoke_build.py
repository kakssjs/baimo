"""Headless smoke test: build a small whitebox room, edit it, render, save, export.
Run: "E:\\blender\\blender.exe" -b --factory-startup -P smoke_build.py
"""
import json, os, sys
sys.path.insert(0, r"C:\Users\kakssjs\.claude\skills\blender-whitebox\scripts")
import bpy
import whitebox as wb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- architecture: 6 x 4 m room, 2.8 m walls
wb.floor("Floor_Slab", (6.4, 4.4), 0.2, (3, 2, 0))
wb.wall("Wall_South", (0, 0), (6, 0), extend=0.1)   # closes corners
wb.wall("Wall_North", (0, 4), (6, 4), extend=0.1)
wb.wall("Wall_West", (0, 0), (0, 4))
wb.wall("Wall_East", (6, 0), (6, 4))
wb.opening("Wall_South", "Door_Main", offset=1.2, width=0.9, height=2.1)
wb.opening("Wall_East", "Win_East_01", offset=2.0, width=1.2, height=1.2, sill=0.9)
wb.stairs("Stair_Step", steps=3, width=1.0, run=0.3, rise=0.17, location=(5.2, 0.4, 0))

# --- products
wb.box("Table_Top", (1.6, 0.8, 0.04), (3, 2, 0.71))
for i, (x, y) in enumerate([(-0.75, -0.35), (0.75, -0.35), (-0.75, 0.35), (0.75, 0.35)]):
    wb.box(f"Table_Leg_{i+1}", (0.05, 0.05, 0.71), (3 + x, 2 + y, 0))
wb.cylinder("Vase_Cyl", 0.12, 0.35, (3.4, 2.1, 0.75))
wb.rounded_box("Speaker_Rounded", (0.25, 0.25, 0.4), 0.04, (2.5, 1.9, 0.75))

# --- incremental edits on existing objects (no rebuild)
before = len(bpy.data.objects)
wb.resize("Table_Top", (1.8, 0.9, 0.04))                         # precise resize
wb.opening("Wall_East", "Win_East_01", offset=2.0, width=1.5, height=1.2, sill=0.9)  # widen window
bpy.data.objects["Vase_Cyl"].location.x = 3.6                    # move
after = len(bpy.data.objects)

wb.clay_all()
wb.preview_setup(target=(3, 2, 0.8), cam_loc=(8.5, -4.5, 8.5), lens=24)
blend = os.path.join(OUT, "WB_SmokeTest_room.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend)
wb.render(os.path.join(OUT, "WB_SmokeTest_preview.png"))
wb.export(os.path.join(OUT, "WB_SmokeTest_room.glb"))
wb.export(os.path.join(OUT, "WB_SmokeTest_room.obj"))

print("@@JSON@@" + json.dumps({"objects_before_edit": before, "objects_after_edit": after,
                                "engine": bpy.context.scene.render.engine}))
