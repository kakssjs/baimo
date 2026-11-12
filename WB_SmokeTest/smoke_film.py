"""White-film pipeline test: animated camera shot -> MP4, then reference-over-blockout comparison MP4.
Run: "E:\\blander\\blender.exe" -b output\\WB_SmokeTest_room.blend --factory-startup -P smoke_film.py
"""
import os, sys
sys.path.insert(0, r"C:\Users\kakssjs\.claude\skills\blender-whitebox\scripts")
import bpy
import whitebox as wb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
REF = r"D:\chatgpt\baimo\3f25cb29474fafc3a7cd1620ecf3ba0f.mp4"
sc = bpy.context.scene
sc.render.fps, sc.frame_start, sc.frame_end = 24, 1, 72

# shot: eye-level dolly in through the door, 35 mm (camera keyframes = editable animation)
cam = wb.preview_setup(target=(1.2, 4.0, 1.4), cam_loc=(1.2, -3.0, 1.6), lens=35, res=(1280, 720))
for f, loc in ((1, (1.2, -3.0, 1.6)), (72, (1.2, 0.8, 1.6))):
    cam.location = loc
    cam.keyframe_insert("location", frame=f)
sc.render.image_settings.media_type = "VIDEO"
sc.render.image_settings.file_format = "FFMPEG"
sc.render.ffmpeg.format, sc.render.ffmpeg.codec = "MPEG4", "H264"
sc.render.filepath = os.path.join(OUT, "WB_Film_shot01.mp4")
bpy.ops.render.render(animation=True)

# comparison: reference footage on top, blockout render below (like the user's sample video)
cmp = bpy.data.scenes.new("Compare")
cmp.render.fps, cmp.render.resolution_x, cmp.render.resolution_y = 24, 1280, 1440
ed = cmp.sequence_editor_create()
ref = ed.strips.new_movie("Ref", REF, 1, 1)
blk = ed.strips.new_movie("Blockout", sc.render.filepath, 2, 1)
for s, y in ((ref, 360), (blk, -360)):
    s.frame_final_duration = 72
    w, h = s.elements[0].orig_width, s.elements[0].orig_height
    s.transform.scale_x = s.transform.scale_y = min(1280 / w, 720 / h)  # fit into its half
    s.transform.offset_y = y
cmp.frame_start, cmp.frame_end = 1, 72
cmp.render.image_settings.media_type = "VIDEO"
cmp.render.image_settings.file_format = "FFMPEG"
cmp.render.ffmpeg.format, cmp.render.ffmpeg.codec = "MPEG4", "H264"
cmp.render.filepath = os.path.join(OUT, "WB_Film_compare.mp4")
with bpy.context.temp_override(scene=cmp):
    bpy.ops.render.render(animation=True)
print("@@FILM@@ done")
