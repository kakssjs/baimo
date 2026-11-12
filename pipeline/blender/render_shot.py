"""Stage 4: render the opened .blend's scene camera to an H.264 MP4 (no audio).

Run: blender -b SHOT.blend --factory-startup --python-exit-code 1 -P render_shot.py --
         --out OUT.mp4 --width W --height H --fps NUM/DEN --frames N
Frames 1..N are rendered. Camera, animation, lights and engine come from the .blend.
"""
import argparse
import sys
from fractions import Fraction

import bpy

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--width", type=int, required=True)
p.add_argument("--height", type=int, required=True)
p.add_argument("--fps", type=Fraction, required=True)
p.add_argument("--frames", type=int, required=True)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])

sc = bpy.context.scene
if sc.camera is None:
    raise RuntimeError("The scene has no active camera.")
r = sc.render
r.resolution_x, r.resolution_y, r.resolution_percentage = a.width, a.height, 100
r.fps, r.fps_base = a.fps.numerator, a.fps.denominator
sc.frame_start, sc.frame_end = 1, a.frames
r.image_settings.media_type = "VIDEO"
r.image_settings.file_format = "FFMPEG"
r.ffmpeg.format, r.ffmpeg.codec = "MPEG4", "H264"
r.ffmpeg.constant_rate_factor = "HIGH"
r.ffmpeg.audio_codec = "NONE"
r.use_file_extension = False
r.filepath = a.out
bpy.ops.render.render(animation=True)
