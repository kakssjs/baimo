"""Stage wrappers: probe, ingest (1), analyse (2), Blender scene build (3) and render (4), compose (5)."""
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from .config import LIMITS, ROOT, WHITEBOX_SCRIPTS, tool

H264 = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
BLENDER_SCRIPTS = Path(__file__).parent / "blender"
BLENDER_FLAGS = ["--factory-startup", "--python-exit-code", "1"]


def _run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        log = (p.stderr.strip() or p.stdout.strip()).splitlines()[-8:]
        raise RuntimeError(f"{Path(cmd[0]).stem} failed:\n" + "\n".join(log))
    return p.stdout


def _ffmpeg(*args):
    _run([tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)])


def probe(path):
    """Return width, height, fps (fraction string), frames, duration and has_audio for a video file."""
    info = json.loads(_run([tool("ffprobe"), "-v", "error", "-show_streams", "-show_format",
                            "-of", "json", str(path)]))
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("The file has no video stream.")
    avg = Fraction(video["avg_frame_rate"]) if video["avg_frame_rate"] != "0/0" else Fraction(0)
    nominal = Fraction(video["r_frame_rate"])
    # r_frame_rate is the nominal rate (24 for a 23.75 fps average clip); on variable-rate
    # files it can be a meaningless timebase value, so fall back to the rounded average.
    fps = nominal if avg and nominal <= avg * 2 else Fraction(round(avg) or 24)
    return {
        "width": video["width"],
        "height": video["height"],
        "fps": str(fps),
        "frames": int(video.get("nb_frames", 0)),
        "duration": float(info["format"].get("duration", 0)),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def ingest(src, job_dir, crop=None):
    """Stage 1: validate the upload and write source.mp4 (constant fps, even size, H.264 + AAC).

    crop is an optional ffmpeg crop expression "w:h:x:y", applied before anything else.
    """
    src = Path(src)
    size_mb = src.stat().st_size / 2**20
    if size_mb > LIMITS["max_upload_mb"]:
        raise ValueError(f"File is {size_mb:.0f} MB; the limit is {LIMITS['max_upload_mb']} MB.")
    try:
        meta = probe(src)
    except RuntimeError:
        raise ValueError("This file could not be read as a video.") from None
    if meta["duration"] > LIMITS["max_duration_s"]:
        raise ValueError(f"Video is {meta['duration']:.0f} s; the limit is {LIMITS['max_duration_s']} s.")
    # Compared long side to long side so portrait videos get the same allowance as landscape.
    size, cap = sorted((meta["width"], meta["height"])), sorted((LIMITS["max_width"], LIMITS["max_height"]))
    if size[0] > cap[0] or size[1] > cap[1]:
        raise ValueError(f"Video is {meta['width']}x{meta['height']}; the limit is "
                         f"{LIMITS['max_width']}x{LIMITS['max_height']}.")
    filters = ([f"crop={crop}"] if crop else []) + ["scale=trunc(iw/2)*2:trunc(ih/2)*2", "setsar=1"]
    out = Path(job_dir) / "source.mp4"
    _ffmpeg("-i", src, "-map", "0:v:0", "-map", "0:a:0?", "-vf", ",".join(filters),
            "-r", meta["fps"], "-fps_mode", "cfr", *H264, "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", out)
    return probe(out)


def output_size(meta):
    """Size of each half of the final video: the source size, enlarged so the long side is at
    least 1280 pixels. Small uploads would otherwise give a coarse white-model render."""
    k = max(1.0, 1280 / max(meta["width"], meta["height"]))
    return {**meta, "width": round(meta["width"] * k / 2) * 2, "height": round(meta["height"] * k / 2) * 2}


def analyse(job_dir):
    """Stage 2: write depth/NNNNNN.png and scene.json for source.mp4 (separate process, see analyse.py)."""
    _run([sys.executable, "-m", "pipeline.analyse", str(job_dir)], cwd=ROOT)


def build_scene(job_dir, meta):
    """Stage 3: build scene.blend: the set surface driven by the depth frames plus a figure per person."""
    job_dir = Path(job_dir)
    _run([tool("blender"), "-b", *BLENDER_FLAGS, "-P", str(BLENDER_SCRIPTS / "build_scene.py"), "--",
          "--scene", str(job_dir / "scene.json"), "--depth-dir", str(job_dir / "depth"),
          "--width", str(meta["width"]), "--height", str(meta["height"]),
          "--whitebox", WHITEBOX_SCRIPTS, "--out", str(job_dir / "scene.blend")])


def render(blend, out, meta):
    """Stage 4: render a .blend with Blender at meta's width, height, fps and frame count."""
    _run([tool("blender"), "-b", str(blend), *BLENDER_FLAGS,
          "-P", str(BLENDER_SCRIPTS / "render_shot.py"), "--", "--out", str(out), "--width", str(meta["width"]),
          "--height", str(meta["height"]), "--fps", meta["fps"], "--frames", str(meta["frames"])])


def compose(job_dir):
    """Stage 5: stack source.mp4 (scaled to the render's size) over white.mp4, keep the audio -> final.mp4."""
    job_dir = Path(job_dir)
    top, bottom = probe(job_dir / "source.mp4"), probe(job_dir / "white.mp4")
    for key in ("fps", "frames"):
        if top[key] != bottom[key]:
            raise RuntimeError(f"white.mp4 does not match source.mp4: {key} {bottom[key]} != {top[key]}")
    out = job_dir / "final.mp4"
    _ffmpeg("-i", job_dir / "source.mp4", "-i", job_dir / "white.mp4", "-filter_complex",
            f"[0:v]scale={bottom['width']}:{bottom['height']}:flags=lanczos,setsar=1[t];[t][1:v]vstack=inputs=2[v]",
            "-map", "[v]", "-map", "0:a?", *H264, "-c:a", "copy", "-movflags", "+faststart", out)
    return probe(out)
